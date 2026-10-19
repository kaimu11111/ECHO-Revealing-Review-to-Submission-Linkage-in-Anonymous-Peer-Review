"""Loader for the released R2S-Bench.

Each instance: one sanitized anonymous review, a candidate set of submissions from the same research area, and
one ground-truth submission written by the reviewer. A candidate is represented by its paper (title + abstract,
benchmark/articles.jsonl) and its rebuttal (benchmark/rebuttals.jsonl). For the ground-truth candidate the rebuttal
is the one written by the matched reviewer (benchmark/gold.jsonl, field `positive_rebuttal`); for the two
ground-truth articles whose reviewer-written rebuttal differs from the article's default rebuttal, this is the
only place that text appears.
"""
import json
from config import BENCH, read_jsonl


def load_tasks():
    """One task per instance: qid, pid (pseudonymous reviewer id), review, cands (article ids), pos, texts."""
    queries = {q["qid"]: q for q in read_jsonl(f"{BENCH}/queries.jsonl")}
    gold = {g["qid"]: g for g in read_jsonl(f"{BENCH}/gold.jsonl")}
    reb = {r["article_id"]: r["rebuttal_text"] for r in read_jsonl(f"{BENCH}/rebuttals.jsonl")}
    tasks = []
    for row in read_jsonl(f"{BENCH}/candidates.jsonl"):
        q, g = queries[row["qid"]], gold[row["qid"]]
        cands = row["candidates"]
        pos = cands.index(g["positive"])
        texts = [reb[c] for c in cands]
        texts[pos] = g["positive_rebuttal"]
        tasks.append({"qid": q["qid"], "pid": q["person"], "review": q["review_text"], "cands": cands, "pos": pos,
                      "texts": texts})
    return tasks


def archive_tasks():
    """Task format used by the method scripts (signals.py and later): rebuttal texts for content and style."""
    return [dict(t, texts_capped=t["texts"], style_texts=t["texts"]) for t in load_tasks()]


def subset(name):
    """qids of a released split, e.g. 'subset100_rerankers' or 'subset20_fullpool'."""
    return json.load(open(f"{BENCH}/splits/{name}.json"))


if __name__ == "__main__":
    T = load_tasks()
    import numpy as np
    print(len(T), "instances,", len({t['pid'] for t in T}), "reviewers, median candidates", int(np.median([len(t['cands']) for t in T])))
