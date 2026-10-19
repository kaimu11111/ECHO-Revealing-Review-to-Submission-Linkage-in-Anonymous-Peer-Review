"""The 46 linkage signals of ECHO Stage 1 (formerly signals.py): chunked dense similarity from six encoder
configurations, BM25 (unigrams, unigrams + bigrams), character 2-4-gram TF-IDF, handcrafted style on content-masked
text and the STAR authorship encoder, each between the review and the candidate paper and/or rebuttal.

  python signals.py features archive     rebuttal-side signals (GPU; Qwen3-Embedding-8B needs F1000_USE_REMOTE_EMB=1 and an endpoint)
  python signals.py combined             adds the paper-side signals -> work/cache/boost_features_combined.npz
Evaluation lives in fusion.py; the LLM reranker in rerank.py.
"""
import collections, hashlib, json, os, re, sys
import numpy as np
from config import ROOT, XDATA, DATA, read_jsonl, read_articles
from benchmark import archive_tasks  # released benchmark loader (uncapped rebuttals)
from fusion import zq, fit_clogit, cv_fusion, rank, metrics, boot, evaluate, evaluate_combined, report, RNG, N_BOOT  # re-exported

OUTDIR = f"{ROOT}/outputs/boost"
STOP = None


def _h(t):
    return hashlib.sha1(t.encode()).hexdigest()



# ---------------------------------------------------------------- tasks

def edition_tasks():
    from config import DATA, XDATA, read_jsonl
    arts = read_articles()
    qs = {q["qid"]: q for q in read_jsonl(f"{XDATA}/private/queries_full.jsonl") if q["split"] == "main"}
    pools = [p for p in json.load(open(f"{XDATA}/private/pools.json"))["pools"]["venue_full"] if p["qid"] in qs]
    return [{"qid": p["qid"], "pid": qs[p["qid"]]["pid"], "pos": p["candidates"].index(p["positive"]),
             "review": qs[p["qid"]]["review_main"], "cands": p["candidates"],
             "texts": [arts[c]["title"] + ". " + arts[c]["abstract"] for c in p["candidates"]],
             "style_texts": [arts[c]["abstract"] for c in p["candidates"]]} for p in pools]



# ---------------------------------------------------------------- features
def dense_features(tasks, name, cache_dir):
    from encoders import encode_cached, chunks
    rchunks = {t["qid"]: chunks(t["review"]) for t in tasks}
    cand_texts = sorted({x for t in tasks for x in t["texts"]})
    cchunks = {x: chunks(x) for x in cand_texts}
    Eq = encode_cached(name, [c for v in rchunks.values() for c in v], True, cache_dir)
    Ed = encode_cached(name, [c for v in cchunks.values() for c in v], False, cache_dir)
    feats = {f"{name}_max": {}, f"{name}_col": {}, f"{name}_mean": {}}
    for t in tasks:
        R = np.stack([Eq[c] for c in rchunks[t["qid"]]])
        rm = R.mean(0)
        rm /= np.linalg.norm(rm)
        mx, col, mean = [], [], []
        for x in t["texts"]:
            D = np.stack([Ed[c] for c in cchunks[x]])
            S = R @ D.T
            mx.append(S.max())
            col.append(S.max(1).mean())
            dm = D.mean(0)
            mean.append(float(rm @ (dm / np.linalg.norm(dm))))
        feats[f"{name}_max"][t["qid"]] = np.array(mx)
        feats[f"{name}_col"][t["qid"]] = np.array(col)
        feats[f"{name}_mean"][t["qid"]] = np.array(mean)
    return feats


def dense_full_features(tasks, name, cache_dir):
    from encoders import encode_cached, REMOTE
    """Whole review vs whole candidate text (long-context embedders only): one cosine per candidate."""
    Eq = encode_cached(name, [t["review"] for t in tasks], True, cache_dir)
    Ed = encode_cached(name, sorted({x for t in tasks for x in t["texts"]}), False, cache_dir)
    return {f"{name}_full": {t["qid"]: np.stack([Ed[x] for x in t["texts"]]) @ Eq[t["review"]] for t in tasks}}


def bm25_features(tasks, k1=1.2, b=0.75):
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
    docs = sorted({x for t in tasks for x in t["texts"]})
    di = {x: i for i, x in enumerate(docs)}
    out = {}
    for name, ngr in (("bm25_uni", (1, 1)), ("bm25_bi", (1, 2))):
        cv = CountVectorizer(lowercase=True, stop_words=list(ENGLISH_STOP_WORDS), ngram_range=ngr, min_df=2,
                             token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]+\b")
        D = cv.fit_transform(docs).tocsr().astype(np.float32)
        N = D.shape[0]
        df = np.asarray((D > 0).sum(0)).ravel()
        idf = np.log(1 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(D.sum(1)).ravel()
        D = D.tocoo()
        denom = D.data + k1 * (1 - b + b * dl[D.row] / dl.mean())
        W = D.copy()
        W.data = (D.data * (k1 + 1) / denom) * idf[D.col]
        W = W.tocsr()
        Q = (cv.transform([t["review"] for t in tasks]) > 0).astype(np.float32)
        out[name] = {}
        for qi, t in enumerate(tasks):
            rows = [di[x] for x in t["texts"]]
            out[name][t["qid"]] = np.asarray(W[rows] @ Q[qi].T.toarray()).ravel()
    return out


def char_style_features(tasks, cache_dir):
    """char n-gram on original (uncapped) text; handcrafted style on aggressively masked text; LUAR on original."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    import style as stylo
    os.environ["F1000_MASK_LEVEL"] = "aggressive"
    from style import Masker, render
    out = {"char": {}, "stylo_masked": {}, "luar": {}}
    docs = sorted({x for t in tasks for x in t["texts"]})
    tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=3, max_features=200000)
    tf.fit(docs + [t["review"] for t in tasks])
    DT = dict(zip(docs, tf.transform(docs)))
    RT = dict(zip([t["qid"] for t in tasks], tf.transform([t["review"] for t in tasks])))
    for t in tasks:
        out["char"][t["qid"]] = np.array([float(DT[x].multiply(RT[t["qid"]]).sum()) for x in t["texts"]])
    # aggressive masking (cached by text hash)
    mpath = f"{cache_dir}/boost_masked_aggr.json"
    M = json.load(open(mpath)) if os.path.exists(mpath) else {}
    need = sorted({x for t in tasks for x in t["style_texts"]} | {t["review"] for t in tasks})
    todo = [x for x in need if _h(x) not in M]
    if todo:
        mk = Masker()
        for i in range(0, len(todo), 1000):
            for x, (m, _) in zip(todo[i:i + 1000], mk.mask_many(todo[i:i + 1000])):
                M[_h(x)] = m
            json.dump(M, open(mpath, "w"))
            print("masked", min(i + 1000, len(todo)), "/", len(todo), flush=True)
    sdocs = sorted({x for t in tasks for x in t["style_texts"]})
    X = stylo.matrix([render(M[_h(x)]) for x in sdocs])
    Z = dict(zip(sdocs, stylo.zscore_by(X, X)))
    rX = stylo.matrix([render(M[_h(t["review"])]) for t in tasks])
    rZ = dict(zip([t["qid"] for t in tasks], stylo.zscore_by(rX, rX)))
    for t in tasks:
        S = np.stack([Z[x] for x in t["style_texts"]])
        r = rZ[t["qid"]]
        out["stylo_masked"][t["qid"]] = (S @ r) / (np.linalg.norm(S, axis=1) * np.linalg.norm(r) + 1e-12)
    # LUAR on original text
    from encoders import Luar, luar_cached
    from config import XDATA  # noqa
    E = luar_cached("luar_boost", Luar, sorted({x for t in tasks for x in t["style_texts"]} | {t["review"] for t in tasks}))
    for t in tasks:
        out["luar"][t["qid"]] = np.stack([E[x] for x in t["style_texts"]]) @ E[t["review"]]
    return out


def baseline_features(tasks, setting):
    """The earlier scores (SPECTER2 on truncated full text; capped char n-gram) for the baseline fusion."""
    from config import DATA, XDATA, read_jsonl
    out = {"topic_old": {}, "char_old": {}}
    if setting == "archive":
        from scores import load_tasks, original_scores
        T0 = load_tasks()
        S0 = original_scores(T0)
        for t in T0:
            out["topic_old"][t["qid"]] = S0[t["qid"]]["topic"]
            out["char_old"][t["qid"]] = S0[t["qid"]]["char"]
    else:
        pe, pk = np.load(f"{DATA}/cache/papers.npy"), json.load(open(f"{DATA}/cache/papers_ids.json"))
        pemb = dict(zip(pk, pe))
        re_, rk = np.load(f"{XDATA}/cache/reviews_main.npy"), json.load(open(f"{XDATA}/cache/reviews_main_ids.json"))
        remb = dict(zip(rk, re_))
        for t in tasks:
            out["topic_old"][t["qid"]] = np.stack([pemb[c] for c in t["cands"]]) @ remb[t["qid"]]
        del out["char_old"]
    return out


def compute_features(setting):
    from config import XDATA
    tasks = archive_tasks() if setting == "archive" else edition_tasks()
    path = f"{XDATA}/cache/boost_features.npz"
    feats = {}
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        feats = z["feats"].item()
    groups = [("bm25", lambda: bm25_features(tasks)),  # the private "baseline" (topic_old/char_old) group is not part of the released method
              ("char_style", lambda: char_style_features(tasks, f"{XDATA}/cache"))]
    groups += [(m, (lambda m=m: dense_features(tasks, m, f"{XDATA}/cache"))) for m in ("specter2", "specter2q", "scincl", "bge", "e5")]
    if os.environ.get("F1000_USE_REMOTE_EMB") == "1":
        groups += [("qwen3emb", lambda: {**dense_features(tasks, "qwen3emb", f"{XDATA}/cache"),
                                         **dense_full_features(tasks, "qwen3emb", f"{XDATA}/cache")})]
    marker = {"baseline": "topic_old", "bm25": "bm25_uni", "char_style": "luar"}
    for g, fn in groups:
        if marker.get(g, f"{g}_max") in feats:  # group already computed
            continue
        print("computing", g, flush=True)
        feats.update(fn())
        np.savez(path, feats=np.array(feats, dtype=object))
    meta = [{"qid": t["qid"], "pid": t["pid"], "pos": t["pos"], "n": len(t["cands"])} for t in tasks]
    json.dump(meta, open(f"{XDATA}/cache/boost_meta.json", "w"))
    print("features:", sorted(feats))



# ---------------------------------------------------------------- combined: paper + author response per candidate
def compute_combined():
    """Rebuttal-archive queries; paper-side scores (title+abstract of the same candidates) + response-side scores."""
    from config import DATA, XDATA, read_jsonl
    from sklearn.feature_extraction.text import TfidfVectorizer
    arts = read_articles()
    tasks = archive_tasks()
    ptasks = [dict(t, texts=[arts[c]["title"] + ". " + arts[c]["abstract"] for c in t["cands"]],
                   style_texts=[arts[c]["abstract"] for c in t["cands"]]) for t in tasks]
    path = f"{XDATA}/cache/boost_features_combined.npz"
    F = np.load(path, allow_pickle=True)["feats"].item() if os.path.exists(path) else {}
    paper = {}
    models = ("specter2", "specter2q", "scincl", "bge", "e5") + (("qwen3emb",) if os.environ.get("F1000_USE_REMOTE_EMB") == "1" else ())
    for m in models:
        if f"paper_{m}_max" not in F:
            print("paper side", m, flush=True)
            paper.update({f"paper_{k}": v for k, v in dense_features(ptasks, m, f"{XDATA}/cache").items()})
            if m in REMOTE:
                paper.update({f"paper_{k}": v for k, v in dense_full_features(ptasks, m, f"{XDATA}/cache").items()})
            F.update(paper)
            np.savez(path, feats=np.array(F, dtype=object))
    if "paper_bm25_uni" not in F:
        F.update({f"paper_{k}": v for k, v in bm25_features(ptasks).items()})
    if "paper_char" not in F:
        docs = sorted({x for t in ptasks for x in t["texts"]})
        tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=3, max_features=200000)
        tf.fit(docs + [t["review"] for t in ptasks])
        DT = dict(zip(docs, tf.transform(docs)))
        RT = dict(zip([t["qid"] for t in ptasks], tf.transform([t["review"] for t in ptasks])))
        F["paper_char"] = {t["qid"]: np.array([float(DT[x].multiply(RT[t["qid"]]).sum()) for x in t["texts"]]) for t in ptasks}
    # response side: everything already computed for the archive setting
    R = np.load(f"{XDATA}/cache/boost_features.npz", allow_pickle=True)["feats"].item()
    F.update({f"resp_{k}": v for k, v in R.items() if not k.endswith("_old")})
    np.savez(path, feats=np.array(F, dtype=object))
    print("combined features:", len(F))



if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "features":
        compute_features(sys.argv[2] if len(sys.argv) > 2 else "archive")
    elif cmd in ("combined", "combined-features"):
        compute_combined()
    else:
        raise SystemExit(__doc__)
