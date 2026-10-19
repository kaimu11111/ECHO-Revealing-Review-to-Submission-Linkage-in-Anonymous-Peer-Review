"""Listwise LLM reranking of the combined Top-K: one request per query with all K candidates, the model returns a
full ranking. Candidates are presented in a seeded random order (position-bias control). Candidates the model omits
are appended in fusion order. Score used for fusion = -(listwise rank).

  python rerank_listwise.py trial|run|eval [--model M] [--k 20] [--n 5]
Same environment and key as rerank.py.
"""
import argparse, hashlib, json, os, re, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import requests
from rerank import OUTDIR, URL, cut, load_topk, rank_of

SYSTEM = ("You are an expert in authorship analysis of scientific writing. You will see an anonymous peer review and "
          "a list of candidate submissions, each with its abstract and the author response its authors posted during "
          "review. Exactly one candidate's author response was written by the same researcher who wrote the review. "
          "Rank ALL candidates from most to least likely to be that one, using research expertise and interests "
          "revealed by the texts and writing habits (sentence structure, punctuation, hedging, discourse markers, "
          "politeness formulas, spelling conventions, organisation of points). "
          "Output only a JSON array of candidate labels in ranked order, e.g. [\"C7\",\"C2\",...], nothing else.")
MAX_REVIEW, MAX_ABS, MAX_RESP = 1200, 200, 500


def build_prompt(review, cands, order):
    parts = [f"=== ANONYMOUS PEER REVIEW ===\n{cut(review, MAX_REVIEW)}\n", "=== CANDIDATES ==="]
    for lab, i in enumerate(order, 1):
        title, abstract, resp = cands[i]
        parts.append(f"\n--- C{lab} ---\nTitle: {title}\nAbstract: {cut(abstract, MAX_ABS)}\nAuthor response: {cut(resp, MAX_RESP)}")
    parts.append(f"\nRank all {len(order)} candidates (C1..C{len(order)}) from most to least likely. Output only the JSON array.")
    return "\n".join(parts)


class ListClient:
    def __init__(self, model, cache_path, no_reasoning=True):
        self.model, self.no_reasoning = model, no_reasoning
        self.key = os.environ["OPENROUTER_API_KEY"]
        self.cache_path, self.cache = cache_path, {}
        if os.path.exists(cache_path):
            for line in open(cache_path):
                r = json.loads(line)
                self.cache[r["h"]] = r

    def _h(self, p):
        return hashlib.sha1((self.model + "\n" + p).encode()).hexdigest()

    def ask(self, p, force=False):
        h = self._h(p)
        if h in self.cache and not force:
            return self.cache[h]
        body = {"model": self.model, "temperature": 0, "max_tokens": 200,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
                "usage": {"include": True}}
        if self.no_reasoning:
            body["reasoning"] = {"enabled": False}
        for attempt in range(6):
            try:
                r = requests.post(URL, timeout=300, json=body, headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
                if r.status_code == 429:
                    raise RuntimeError("rate limited")
                r.raise_for_status()
                d = r.json()
                if "error" in d:
                    raise RuntimeError(str(d["error"])[:300])
                text = d["choices"][0]["message"]["content"] or ""
                labels = [int(x) for x in re.findall(r"C(\d+)", text)]
                rec = {"h": h, "text": text[:600], "labels": labels, "served_model": d.get("model"), "usage": d.get("usage")}
                self.cache[h] = rec
                with open(self.cache_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                return rec
            except Exception as e:
                wait = 5 * (attempt + 1)
                print(f"retry {attempt} ({str(e)[:120]}) in {wait}s", flush=True)
                time.sleep(wait)
        raise RuntimeError("request failed repeatedly")


def listwise_scores(t, rec, order):
    """Score per candidate index: higher = ranked earlier by the LLM; omitted candidates after all ranked ones, in fusion order."""
    k = len(order)
    seen, ranked = set(), []
    for lab in rec["labels"]:
        if 1 <= lab <= k and lab not in seen:
            seen.add(lab)
            ranked.append(order[lab - 1])
    for i in t["topk"]:
        if i not in ranked:
            ranked.append(i)
    s = np.full(t["n"], -np.inf)
    for pos_, i in enumerate(ranked):
        s[i] = -pos_
    return s, len(seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["trial", "run", "eval"])
    ap.add_argument("--model", default="deepseek/deepseek-v4-pro")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    tag = a.model.replace("/", "_") + f"_list{a.k}"
    cli = ListClient(a.model, f"{OUTDIR}/cache_{tag}.jsonl")
    T = load_topk(a.k)
    rng = np.random.default_rng(123)
    orders = {t["qid"]: list(rng.permutation(t["topk"])) for t in T}
    prompts = {t["qid"]: build_prompt(t["review"], t["cand"], orders[t["qid"]]) for t in T}

    if a.cmd in ("trial", "run"):
        sel = T[:a.n] if a.cmd == "trial" else T
        t0 = time.time()
        with ThreadPoolExecutor(a.workers) as ex:
            recs = list(ex.map(lambda t: cli.ask(prompts[t["qid"]]), sel))
        cost = sum((r["usage"] or {}).get("cost", 0) or 0 for r in recs)
        toks = sum((r["usage"] or {}).get("prompt_tokens", 0) or 0 for r in recs)
        print(f"{a.cmd}: {len(sel)} queries, {time.time()-t0:.0f}s, prompt tokens {toks:,}, cost ${cost:.3f}, "
              f"served {sorted({r['served_model'] for r in recs})}, complete rankings {np.mean([len(set(r['labels'])) >= a.k for r in recs]):.0%}")
        if a.cmd == "trial":
            for t, r in zip(sel, recs):
                s, nl = listwise_scores(t, r, orders[t["qid"]])
                fr = t["topk"].index(t["pos"]) + 1 if t["pos"] in t["topk"] else None
                lr = int((s > s[t["pos"]]).sum()) + 1 if t["pos"] in t["topk"] else None
                print(f"  {t['qid']}: positive fusion rank {fr} -> listwise rank {lr}; labels returned {nl}; output: {r['text'][:80]!r}")
            r2 = cli.ask(prompts[sel[0]["qid"]], force=True)
            print("stability (first query re-sent):", recs[0]["labels"][:8], "->", r2["labels"][:8])
        return

    import pandas as pd
    from sklearn.model_selection import GroupKFold
    from signals import metrics, boot
    N = [t["n"] for t in T]
    groups = np.array([t["pid"] for t in T])
    L_scores, missing = [], 0
    for t in T:
        h = cli._h(prompts[t["qid"]])
        if h in cli.cache:
            L_scores.append(listwise_scores(t, cli.cache[h], orders[t["qid"]])[0])
        else:
            missing += 1
            L_scores.append(t["fused"].copy())
    if missing:
        print(f"WARNING: {missing} queries without listwise result")

    def z(x):
        sd = x.std()
        return (x - x.mean()) / sd if sd > 0 else x * 0

    def fused_rank(t, s, alpha):
        k = t["topk"]
        comb = alpha * z(t["fused"][k]) + (1 - alpha) * z(s[k])
        full = t["fused"] - 1e6
        full[k] = comb
        return rank_of(full, t["pos"])

    base = [rank_of(t["fused"], t["pos"]) for t in T]
    lw = [fused_rank(t, s, 0.0) for t, s in zip(T, L_scores)]
    pred, chosen = [None] * len(T), []
    for tr, te in GroupKFold(5).split(T, groups=groups):
        best, best_mrr = None, -1
        for al in np.round(np.arange(0, 1.01, 0.1), 1):
            mrr = np.mean([1 / fused_rank(T[i], L_scores[i], al) for i in tr])
            if mrr > best_mrr + 1e-12:
                best, best_mrr = al, mrr
        chosen.append(best)
        for i in te:
            pred[i] = fused_rank(T[i], L_scores[i], best)
    rows = {"fusion baseline": base, f"listwise LLM only within top-{a.k}": lw, f"fusion + listwise within top-{a.k} (alpha by CV)": pred}
    res = pd.DataFrame([{"method": k, **metrics(v, N)} for k, v in rows.items()])
    res.to_csv(f"{OUTDIR}/results_{tag}.csv", index=False, float_format="%.4f")
    cost = sum((r["usage"] or {}).get("cost", 0) or 0 for r in cli.cache.values())
    Lr = [f"# Listwise LLM reranking of the combined Top-{a.k} (model: `{a.model}`)", "",
          f"{len(T)} queries; one request per query with all {a.k} candidates in random order; reported cost ${cost:.2f}; alpha per fold {chosen}.", "",
          "| method | R@1 | R@5 | R@10 | R@20 | R@30 | MRR | median rank |", "|---|---|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        Lr.append(f"| {r.method} | {r['R@1']:.3f} | {r['R@5']:.3f} | {r['R@10']:.3f} | {r['R@20']:.3f} | {r['R@30']:.3f} | {r.MRR:.3f} | {r.median_rank:.0f} |")
    Lr += ["", "Paired bootstrap vs the fusion baseline. **Bold** = 95% CI excludes 0.", "", "| method | ΔMRR | ΔR@1 | ΔR@5 | ΔR@10 |", "|---|---|---|---|---|"]
    for name, v in list(rows.items())[1:]:
        d = boot(v, base)
        cells = []
        for kk in ("dMRR", "dR@1", "dR@5", "dR@10"):
            m, lo, hi = d[kk]
            c = f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            cells.append(f"**{c}**" if lo > 0 or hi < 0 else c)
        Lr.append(f"| {name} | " + " | ".join(cells) + " |")
    open(f"{OUTDIR}/report_{tag}.md", "w").write("\n".join(Lr) + "\n")
    print(res.round(3).to_string(index=False))
    print("alpha per fold", chosen)


if __name__ == "__main__":
    main()
