"""Per-candidate scores for review -> author-response retrieval on the platform rebuttal-archive experiment.

Run with F1000_CORPUS=platform F1000_EXP=rebuttal_archive F1000_VENUE_K=8 F1000_RESPONSE_ARCHIVE=1.
Score definitions match construction/s07_baselines.py (review->response, setting venue_full):
  topic : cos(SPECTER2(review), SPECTER2(response))
  stylo : cos of handcrafted style vectors, z-scored within genre (reviews vs responses)
  char  : cos of char 2-4-gram TF-IDF
"""
import hashlib, json, os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from config import DATA, XDATA, BENCH, ROOT, read_jsonl
import style as stylo

from config import ART_ROOT


from benchmark import load_tasks  # released benchmark loader

def _h(t):
    return hashlib.sha1(t.encode()).hexdigest()


def embed_cached(texts, name):
    """SPECTER2 embeddings of unique texts, cached by text hash under XDATA/cache/<name>.npz."""
    path = f"{XDATA}/cache/{name}.npz"
    cache = dict(np.load(path)) if os.path.exists(path) else {}
    todo = sorted({t for t in texts if _h(t) not in cache})
    if todo:
        import encoders as E
        tok, model = E.load()
        for t, e in zip(todo, E.embed(todo, tok, model)):
            cache[_h(t)] = e
        np.savez(path, **cache)
    return {t: cache[_h(t)] for t in set(texts)}


def score_tables(tasks, reviews, cand_text, tfidf_fit, with_topic=True, emb_name="resp_emb_original",
                 review_emb=None, render=lambda t: t):
    """reviews: qid -> review text; cand_text: (qid, i) -> text of candidate i.
    render maps texts before style features / char n-grams (used to neutralise mask tags).
    Returns qid -> {"topic", "stylo", "char"} score arrays aligned with task["cands"]."""
    uniq = sorted({cand_text[(t["qid"], i)] for t in tasks for i in range(len(t["cands"]))})
    rev_list = [reviews[t["qid"]] for t in tasks]
    # style: z-score within genre over the review population and the unique-response population
    rX = stylo.matrix([render(x) for x in rev_list])
    rZ = dict(zip([t["qid"] for t in tasks], stylo.zscore_by(rX, rX)))
    cX = stylo.matrix([render(x) for x in uniq])
    cZ = dict(zip(uniq, stylo.zscore_by(cX, cX)))
    tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=3, max_features=200000)
    tf.fit([render(x) for x in tfidf_fit])
    rT = dict(zip([t["qid"] for t in tasks], tf.transform([render(x) for x in rev_list])))
    cT = dict(zip(uniq, tf.transform([render(x) for x in uniq])))
    if with_topic:
        cE = embed_cached(uniq, emb_name)
        if review_emb is None:
            review_emb = embed_cached(rev_list, emb_name + "_reviews")
            review_emb = {t["qid"]: review_emb[reviews[t["qid"]]] for t in tasks}
    out = {}
    for t in tasks:
        q = t["qid"]
        texts = [cand_text[(q, i)] for i in range(len(t["cands"]))]
        S = np.stack([cZ[x] for x in texts])
        r = rZ[q]
        sty = (S @ r) / (np.linalg.norm(S, axis=1) * np.linalg.norm(r) + 1e-12)
        ch = np.array([float(cT[x].multiply(rT[q]).sum()) for x in texts])
        d = {"stylo": sty, "char": ch}
        if with_topic:
            d["topic"] = np.stack([cE[x] for x in texts]) @ review_emb[q]
        out[q] = d
    return out


def original_scores(tasks):
    """Scores on the original (unmasked) texts, matching s07's fit populations."""
    from config import read_articles
    arts = read_articles()
    elig = [k for k, a in arts.items() if a["title"] and len(a["abstract"].split()) >= 50]
    fit = [arts[k]["abstract"] for k in elig] + [t["review"] for t in tasks]
    re_ = embed_cached([t["review"] for t in tasks], "reviews_specter2")
    remb = {t["qid"]: re_[t["review"]] for t in tasks}
    reviews = {t["qid"]: t["review"] for t in tasks}
    cand = {(t["qid"], i): x for t in tasks for i, x in enumerate(t["texts"])}
    return score_tables(tasks, reviews, cand, fit, review_emb={t["qid"]: remb[t["qid"]] for t in tasks})


def rank_of_pos(scores, pos, rng):
    s = np.asarray(scores, float) + rng.uniform(0, 1e-9, len(scores))
    return int((s > s[pos]).sum()) + 1
