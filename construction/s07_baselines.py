"""Stage-1 baselines (no generative LLMs) over every candidate setting; writes outputs/pilot_results.csv.

Scorers (higher = more likely to be the reviewer's article):
  random             chance
  topic_review       cos(SPECTER2(review), SPECTER2(candidate title+abstract))          [topic baseline 1]
  topic_reviewed     cos(SPECTER2(reviewed paper Q), SPECTER2(candidate))                 [topic baseline 2]
  stylo_cos          cos(z(review style), z(candidate-abstract style)); z-scored within genre
  stylo_euclid       -||z_review - z_candidate||
  stylo_charngram    cos of char 2-4-gram TF-IDF (classic authorship-attribution baseline)
  lr_style           logistic regression on |z_review - z_candidate|          (5-fold CV grouped by person)
  lr_topic           logistic regression on [topic_review, topic_reviewed]    (same folds)
  lr_topic_style     logistic regression on topic scores + |style diffs|      (same folds)
Confound diagnostics (metadata, not linkage methods):
  diag_date_proximity  -|candidate publication date - review date|
  diag_same_type       1 if candidate article type == reviewed article type
"""
import collections, json, os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from common import DATA, OUT, read_jsonl, XDATA, BENCH, CONCURRENT_DAYS, EXP
import stylo
from s06_candidates import opaque

RNG = np.random.default_rng(0)
N_BOOT = 1000


def ranks_from_scores(scores, pos):
    s = np.asarray(scores, float) + RNG.uniform(0, 1e-9, len(scores))  # random tie-breaking
    return int((s > s[pos]).sum()) + 1


def metrics(ranks, sizes=None):
    r = np.asarray(ranks)
    N = np.asarray(sizes if sizes is not None else [50] * len(r), float)
    m = {"n": len(r), "R@1": np.mean(r <= 1), "R@5": np.mean(r <= 5), "R@10": np.mean(r <= 10),
         "MRR": np.mean(1 / r), "median_rank": float(np.median(r)),
         # percentile rank of the positive within its pool (0 = top); chance = 0.5 for any pool size
         "mean_pct_rank": float(np.mean((r - 1) / np.maximum(N - 1, 1))), "pool_size_median": float(np.median(N)),
         "chance_R@1": float(np.mean(np.minimum(1, N) / N)), "chance_R@5": float(np.mean(np.minimum(5, N) / N)),
         "chance_R@10": float(np.mean(np.minimum(10, N) / N)),
         "chance_MRR": float(np.mean([sum(1 / k for k in range(1, int(n) + 1)) / n for n in N]))}
    bs = RNG.integers(0, len(r), (N_BOOT, len(r)))
    for k, f in (("R@10", lambda x: (x <= 10).mean(1)), ("MRR", lambda x: (1 / x).mean(1)), ("R@1", lambda x: (x <= 1).mean(1))):
        v = f(r[bs])
        m[f"{k}_ci_lo"], m[f"{k}_ci_hi"] = np.percentile(v, 2.5), np.percentile(v, 97.5)
    return m


def cos_rows(A, B):
    A = A / (np.linalg.norm(A, axis=-1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=-1, keepdims=True) + 1e-12)
    return (A * B).sum(-1)


class Scorer:
    """Holds per-query review vectors and per-article candidate vectors; produces score arrays."""

    meta = None  # (article key -> (date ordinal, type), qid -> (review date ordinal, reviewed-article type))

    def __init__(self, rev_emb, q_emb, cand_emb, rev_sty, cand_sty, rev_tfidf, cand_tfidf):
        self.rev_emb, self.q_emb, self.cand_emb = rev_emb, q_emb, cand_emb
        self.rev_sty, self.cand_sty = rev_sty, cand_sty
        self.rev_tfidf, self.cand_tfidf = rev_tfidf, cand_tfidf

    def pair_feats(self, qid, cands):
        C = np.stack([self.cand_emb[c] for c in cands])
        t1 = C @ self.rev_emb[qid]
        t2 = C @ self.q_emb[qid]
        S = np.stack([self.cand_sty[c] for c in cands])
        r = self.rev_sty[qid]
        diff = np.abs(S - r)
        sc = cos_rows(S, r[None, :])
        se = -np.linalg.norm(S - r, axis=1)
        ch = np.array([float(self.cand_tfidf[c].multiply(self.rev_tfidf[qid]).sum()) for c in cands])
        am, qm = Scorer.meta
        dp = -np.abs(np.array([am[c][0] for c in cands]) - qm[qid][0])
        st = np.array([float(am[c][1] == qm[qid][1]) for c in cands])
        return {"topic_review": t1, "topic_reviewed": t2, "stylo_cos": sc, "stylo_euclid": se,
                "stylo_charngram": ch, "diag_date_proximity": dp, "diag_same_type": st, "diff": diff}


def run_setting(pools, scorer, groups, with_lr=True):
    """pools: list of {qid, candidates(keys), positive}; returns {method: {qid: rank}}."""
    feats = {p["qid"]: scorer.pair_feats(p["qid"], p["candidates"]) for p in pools}
    pos = {p["qid"]: p["candidates"].index(p["positive"]) for p in pools}
    out = collections.defaultdict(dict)
    for p in pools:
        q, f = p["qid"], feats[p["qid"]]
        out["random"][q] = ranks_from_scores(RNG.uniform(size=len(p["candidates"])), pos[q])
        for m in ("topic_review", "topic_reviewed", "stylo_cos", "stylo_euclid", "stylo_charngram",
                  "diag_date_proximity", "diag_same_type"):
            out[m][q] = ranks_from_scores(f[m], pos[q])
    if not with_lr:
        return out
    qids = [p["qid"] for p in pools]
    g = np.array([groups[q] for q in qids])
    designs = {
        "lr_style": lambda f: f["diff"],
        "lr_topic": lambda f: np.c_[f["topic_review"], f["topic_reviewed"]],
        "lr_topic_style": lambda f: np.c_[f["topic_review"], f["topic_reviewed"], f["diff"]],
    }
    for name, fx in designs.items():
        for tr, te in GroupKFold(5).split(qids, groups=g):
            X = np.concatenate([fx(feats[qids[i]]) for i in tr])
            y = np.concatenate([np.eye(len(pools[i]["candidates"]))[pos[qids[i]]] for i in tr])
            sc = StandardScaler().fit(X)
            clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000).fit(sc.transform(X), y)
            for i in te:
                q = qids[i]
                out[name][q] = ranks_from_scores(clf.decision_function(sc.transform(fx(feats[q]))), pos[q])
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    arts = {a["article_key"]: a for a in read_jsonl(f"{DATA}/articles.jsonl")}
    qs = read_jsonl(f"{XDATA}/private/queries_full.jsonl")
    qmap = {q["qid"]: q for q in qs}
    P = json.load(open(f"{XDATA}/private/pools.json"))
    groups = {q["qid"]: q["pid"] for q in qs}
    import datetime
    from common import pub_date
    od = lambda x: datetime.date.fromisoformat(x).toordinal()
    Scorer.meta = ({k: (od(pub_date(a)), a["type"]) for k, a in arts.items()},
                   {q["qid"]: (od(q["review_date"]), arts[q["Q"]]["type"]) for q in qs})

    pe, pk = np.load(f"{DATA}/cache/papers.npy"), json.load(open(f"{DATA}/cache/papers_ids.json"))
    paper_emb = dict(zip(pk, pe))
    q_emb = {q["qid"]: paper_emb[q["Q"]] for q in qs}

    # style reference populations: all reviews (per text version) and all eligible abstracts
    elig = [k for k in pk if arts[k]["title"] and len(arts[k]["abstract"].split()) >= 50]
    abs_X = stylo.matrix([arts[k]["abstract"] for k in elig])
    abs_Z = dict(zip(elig, stylo.zscore_by(abs_X, abs_X)))
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=3, max_features=200000)
    tfidf.fit([arts[k]["abstract"] for k in elig] + [q["review_main"] for q in qs])
    abs_T = dict(zip(elig, tfidf.transform([arts[k]["abstract"] for k in elig])))

    rows, per_query = [], []
    for version in ("main", "strict"):
        re_, rk = np.load(f"{XDATA}/cache/reviews_{version}.npy"), json.load(open(f"{XDATA}/cache/reviews_{version}_ids.json"))
        rev_emb = dict(zip(rk, re_))
        texts = [q[f"review_{version}"] for q in qs]
        rX = stylo.matrix(texts)
        rev_sty = dict(zip([q["qid"] for q in qs], stylo.zscore_by(rX, rX)))
        rev_T = dict(zip([q["qid"] for q in qs], tfidf.transform(texts)))
        scorer = Scorer(rev_emb, q_emb, paper_emb, rev_sty, abs_Z, rev_T, abs_T)
        for setting, pools in P["pools"].items():
            res = run_setting(pools, scorer, groups)
            ctrl = {p["qid"] for p in pools if (p.get("n_above") or 0) >= 24}
            size = {p["qid"]: len(p["candidates"]) for p in pools}
            subsets = {"main": lambda q: qmap[q]["split"] == "main", "all": lambda q: True}
            if "topic" in setting:
                subsets["main_controlled"] = lambda q: qmap[q]["split"] == "main" and q in ctrl
                subsets["all_controlled"] = lambda q: q in ctrl
            for subset, keep in subsets.items():
                for m, rr in res.items():
                    ids = [q for q in rr if keep(q)]
                    rows.append({"experiment": "review->paper", "query_set": subset, "setting": setting,
                                 "review_text": version, "method": m,
                                 **metrics([rr[q] for q in ids], [size[q] for q in ids])})
            for m, rr in res.items():
                for q, r in rr.items():
                    per_query.append({"experiment": "review->paper", "query_set": "all", "setting": setting, "review_text": version,
                                      "method": m, "qid": q, "rank": r, "pool_size": size[q]})

    # ---------------- optional author-response experiment ----------------
    rt = read_jsonl(f"{BENCH}/response_texts.jsonl")
    resp_text = {(r["qid"], r["article_id"]): r["response_text"] for r in rt}
    import s05_embed as E
    tok, model = E.load()
    rep_texts = {}  # (qid, key, rep) -> text
    for s, pools in P["resp_pools"].items():
        for p in pools:
            for c in p["candidates"]:
                a, r = arts[c], resp_text[(p["qid"], opaque(c))]
                rep_texts[(p["qid"], c, "paper")] = (a["title"] + tok.sep_token + a["abstract"], a["abstract"])
                rep_texts[(p["qid"], c, "response")] = (r, r)
                rep_texts[(p["qid"], c, "paper+response")] = (a["title"] + tok.sep_token + a["abstract"] + " " + r,
                                                               a["abstract"] + "\n" + r)
    keys = sorted(rep_texts)
    embs = E.embed([rep_texts[k][0] for k in keys], tok, model)
    emb_by = dict(zip(keys, embs))
    for version in ("main", "strict"):
        re_, rk = np.load(f"{XDATA}/cache/reviews_{version}.npy"), json.load(open(f"{XDATA}/cache/reviews_{version}_ids.json"))
        rev_emb = dict(zip(rk, re_))
        texts = [q[f"review_{version}"] for q in qs]
        rX = stylo.matrix(texts)
        rev_sty = dict(zip([q["qid"] for q in qs], stylo.zscore_by(rX, rX)))
        rev_T = dict(zip([q["qid"] for q in qs], tfidf.transform(texts)))
        for rep in ("paper", "response", "paper+response"):
            ks = [k for k in keys if k[2] == rep]
            X = stylo.matrix([rep_texts[k][1] for k in ks])
            Z = dict(zip(ks, stylo.zscore_by(X, X)))
            T = dict(zip(ks, tfidf.transform([rep_texts[k][1] for k in ks])))
            for setting, pools in P["resp_pools"].items():
                res = collections.defaultdict(dict)
                for p in pools:
                    q = p["qid"]
                    # candidates keyed per query because the positive's response text is query specific
                    sc = Scorer(rev_emb, q_emb, {c: emb_by[(q, c, rep)] for c in p["candidates"]}, rev_sty,
                                {c: Z[(q, c, rep)] for c in p["candidates"]}, rev_T, {c: T[(q, c, rep)] for c in p["candidates"]})
                    r1 = run_setting([p], sc, groups, with_lr=False)
                    for m, rr in r1.items():
                        res[m].update(rr)
                for m, rr in res.items():
                    sz = {p["qid"]: len(p["candidates"]) for p in pools}
                    rows.append({"experiment": f"review->{rep}", "query_set": "response_subset", "setting": setting,
                                 "review_text": version, "method": m,
                                 **metrics(list(rr.values()), [sz[q] for q in rr])})
                    per_query += [{"experiment": f"review->{rep}", "query_set": "response_subset", "setting": setting,
                                   "review_text": version, "method": m,
                                   "qid": q, "rank": r, "pool_size": sz[q]} for q, r in rr.items()]

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/pilot_results.csv", index=False, float_format="%.4f")
    pd.DataFrame(per_query).to_csv(f"{OUT}/per_query_ranks.csv", index=False)
    show = df[(df.query_set == "main") & (df.review_text == "main")]
    print(show[["setting", "method", "R@1", "R@5", "R@10", "MRR", "median_rank"]].round(3).to_string())


if __name__ == "__main__":
    main()
