"""Topic -> style two-stage reranking for review -> author-response linkage (rebuttal archive).

Stage 1: rank every candidate response by topic score; keep Top-K (K = 10, 20, 50, 100, all).
Stage 2: rerank the Top-K by (a) handcrafted style, (b) char n-gram, (c) alpha * z(topic) + (1-alpha) * z(style)
with z-scores computed within the Top-K set. Alpha in {0, 0.1, ..., 1} is chosen on training folds only
(5-fold CV grouped by reviewer, so validation and test reviewers are disjoint).

Outputs: outputs/rerank/results.csv, outputs/rerank/deltas.csv, outputs/rerank/report.md
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import json
from config import XDATA
from scores import ART_ROOT, load_tasks, original_scores, score_tables

KS = [10, 20, 50, 100, None]
ALPHAS = np.round(np.arange(0, 1.01, 0.1), 1)
N_BOOT = 2000
RNG = np.random.default_rng(0)
OUTDIR = f"{ART_ROOT}/rerank"


def z(x):
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else x * 0


def topk_rank(topic, score, pos, K):
    """Rank of the positive after Stage-1 Top-K by topic and Stage-2 ordering by `score`; None if filtered out."""
    jitter = RNG.uniform(0, 1e-9, len(topic))
    order = np.argsort(-(topic + jitter))
    keep = order if K is None else order[:K]
    if pos not in keep:
        return None
    s = score[keep] + jitter[keep]
    return int((s > s[list(keep).index(pos)]).sum()) + 1, keep


def fused_rank(topic, style, pos, K, alpha):
    r = topk_rank(topic, topic, pos, K)
    if r is None:
        return None
    _, keep = r
    f = alpha * z(topic[keep]) + (1 - alpha) * z(style[keep])
    j = RNG.uniform(0, 1e-9, len(keep))
    f = f + j
    return int((f > f[list(keep).index(pos)]).sum()) + 1


def summarize(ranks, K):
    """ranks: list of int or None (None = removed by Stage 1). Returns end-to-end and conditional metrics."""
    surv = np.array([r is not None for r in ranks])
    rr = np.array([r if r is not None else np.inf for r in ranks], float)
    out = {}
    for scope, mask in (("end_to_end", np.ones(len(rr), bool)), ("conditional", surv)):
        x = rr[mask]
        med = float(np.median(x)) if len(x) else np.nan
        out[scope] = {"n": int(mask.sum()), "survival_rate": surv.mean(),
                      "R@1": np.mean(x <= 1), "R@5": np.mean(x <= 5), "R@10": np.mean(x <= 10),
                      "MRR": np.mean(np.where(np.isfinite(x), 1 / x, 0)), "median_rank": med}
    return out


def boot_delta(a, b, mask):
    """Paired bootstrap of (a - b) per-query metric vectors over queries in mask."""
    rows = {}
    for name, f in (("dMRR", lambda r: np.where(np.isfinite(r), 1 / r, 0)), ("dR@1", lambda r: (r <= 1).astype(float)),
                    ("dR@5", lambda r: (r <= 5).astype(float))):
        d = (f(a) - f(b))[mask]
        bs = RNG.integers(0, len(d), (N_BOOT, len(d)))
        m = d[bs].mean(1)
        rows[name] = (d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5), (m <= 0).mean())
    return rows


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    tasks = load_tasks()
    S = original_scores(tasks)
    # content-free style scores: the same style scorers on the aggressively masked texts (from style.py)
    mpath = f"{XDATA}/style_masked_aggressive/masked.json"
    have_masked = os.path.exists(mpath)
    if have_masked:
        from style import render
        M = json.load(open(mpath))
        mrev = {q: v["text"] for q, v in M["reviews"].items()}
        mresp = {u: v["text"] for u, v in M["responses"].items()}
        cand = {(t["qid"], i): mresp[x] for t in tasks for i, x in enumerate(t["texts"])}
        Sm = score_tables(tasks, mrev, cand, list(mrev.values()) + list(mresp.values()), with_topic=False, render=render)
        for q in S:
            S[q]["mstylo"], S[q]["mchar"] = Sm[q]["stylo"], Sm[q]["char"]
    STYLES = [("handcrafted", "stylo"), ("charngram", "char")] + (
        [("masked_handcrafted", "mstylo"), ("masked_charngram", "mchar")] if have_masked else [])
    qids = [t["qid"] for t in tasks]
    pos = {t["qid"]: t["pos"] for t in tasks}
    groups = np.array([t["pid"] for t in tasks])

    # sanity: full-pool single-method numbers (should match pilot_results.csv review->response venue_full)
    sanity = {m: np.mean([1 / topk_rank(S[q]["topic"], S[q][m], pos[q], None)[0] for q in qids]) for m in ("topic", "stylo", "char")}

    rows, deltas, alpha_log = [], [], []
    per_query = {}
    for K in KS:
        Kname = "all" if K is None else str(K)
        ranks = {}
        ranks["topic_only"] = [None if (r := topk_rank(S[q]["topic"], S[q]["topic"], pos[q], K)) is None else r[0] for q in qids]
        for name, key in STYLES:
            m = f"rerank_{name}"
            ranks[m] = [None if (r := topk_rank(S[q]["topic"], S[q][key], pos[q], K)) is None else r[0] for q in qids]
        # fusion with alpha chosen on training folds (reviewer-disjoint)
        for name, key in STYLES:
            m = f"fusion_{name}"
            pred = [None] * len(qids)
            for fold, (tr, te) in enumerate(GroupKFold(5).split(qids, groups=groups)):
                best, best_mrr = None, -1
                for a in ALPHAS:
                    rr = [fused_rank(S[qids[i]]["topic"], S[qids[i]][key], pos[qids[i]], K, a) for i in tr]
                    mrr = np.mean([1 / r if r else 0 for r in rr])
                    if mrr > best_mrr + 1e-12:
                        best, best_mrr = a, mrr
                alpha_log.append({"K": Kname, "method": m, "fold": fold, "alpha": best, "train_MRR": best_mrr})
                for i in te:
                    pred[i] = fused_rank(S[qids[i]]["topic"], S[qids[i]][key], pos[qids[i]], K, best)
            ranks[m] = pred
        surv = np.array([r is not None for r in ranks["topic_only"]])
        for m, rr in ranks.items():
            for scope, met in summarize(rr, K).items():
                rows.append({"K": Kname, "scope": scope, "method": m, **met})
            per_query[(Kname, m)] = np.array([r if r is not None else np.inf for r in rr], float)
        base = per_query[(Kname, "topic_only")]
        for m in ranks:
            if m == "topic_only":
                continue
            for scope, mask in (("end_to_end", np.ones(len(qids), bool)), ("conditional", surv)):
                for metric, (mean, lo, hi, p) in boot_delta(per_query[(Kname, m)], base, mask).items():
                    deltas.append({"K": Kname, "scope": scope, "method": m, "metric": metric, "diff": mean,
                                   "ci_lo": lo, "ci_hi": hi, "p_diff_le_0": p, "n": int(mask.sum())})

    res = pd.DataFrame(rows)
    dl = pd.DataFrame(deltas)
    al = pd.DataFrame(alpha_log)
    res.to_csv(f"{OUTDIR}/results.csv", index=False, float_format="%.4f")
    dl.to_csv(f"{OUTDIR}/deltas.csv", index=False, float_format="%.4f")
    al.to_csv(f"{OUTDIR}/alpha_selection.csv", index=False, float_format="%.4f")

    # ---------------- report ----------------
    n = len(qids)
    pool = int(np.median([len(t["cands"]) for t in tasks]))

    def fmt_table(df, cols):
        h = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
        for _, r in df.iterrows():
            h += "| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) and c not in ("K",) else str(r[c]) for c in cols) + " |\n"
        return h

    L = ["# Topic → style two-stage reranking (review → author-response linkage)", "",
         f"Data: platform rebuttal archive. **{n} queries** (anonymous review → the reviewer's own author response). "
         f"Candidates are the field's author responses public by review date + 182 days; median pool = {pool}. "
         "Scores are the existing ones (SPECTER2 topic, handcrafted style, char 2–4-gram TF-IDF). No new models.", "",
         f"Sanity check (full pool, one method each, should match `platform_rebuttal_archive/pilot_results.csv`): "
         f"MRR topic = {sanity['topic']:.3f}, handcrafted = {sanity['stylo']:.3f}, char-n-gram = {sanity['char']:.3f}.", "",
         "`masked_*` methods apply the same style scorers to the aggressively content-masked texts (`style_masked_aggressive`: every noun, proper noun, adjective and number masked). "
         "They test whether *content-free* style still helps after topic retrieval.", "",
         "Stage 1 keeps the topic Top-K. Stage 2 reorders them by style, or by fusion α·z(topic) + (1−α)·z(style) (z within Top-K). "
         "α is picked from {0, 0.1, …, 1} by MRR on the training folds of a 5-fold, reviewer-disjoint split and applied to the held-out fold. "
         "**End-to-end**: a positive removed in Stage 1 counts as a miss (reciprocal rank 0). "
         "**Conditional**: only queries whose positive survived Stage 1 (the same queries for every method at that K).", "",
         "## Stage-1 survival", "", "| K | positive survives topic Top-K |", "|---|---|"]
    for K in KS:
        Kname = "all" if K is None else str(K)
        s = res[(res.K == Kname) & (res.method == "topic_only") & (res.scope == "end_to_end")].survival_rate.iloc[0]
        L.append(f"| {Kname} | {s:.3f} ({int(round(s * n))}/{n}) |")
    L.append("")
    order = ["topic_only"] + [f"{k}_{n}" for n, _ in STYLES for k in ("rerank", "fusion")]
    for scope in ("end_to_end", "conditional"):
        L += [f"## {'End-to-end' if scope == 'end_to_end' else 'Conditional on Stage-1 survival'}", ""]
        for K in KS:
            Kname = "all" if K is None else str(K)
            d = res[(res.K == Kname) & (res.scope == scope)].set_index("method").loc[order].reset_index()
            d["median_rank"] = d.median_rank.map(lambda v: f">{Kname}" if not np.isfinite(v) else f"{v:.1f}")
            L += [f"### K = {Kname} (n = {int(d.n.iloc[0])})", "", fmt_table(d, ["method", "R@1", "R@5", "R@10", "MRR", "median_rank"])]
    L += ["## Paired bootstrap: method − topic-only", "",
          "2,000 resamples over queries. **Bold** = 95% CI excludes 0.", ""]
    for scope in ("end_to_end", "conditional"):
        L += [f"### {scope}", "", "| K | method | ΔMRR [95% CI] | ΔR@1 [95% CI] | ΔR@5 [95% CI] |", "|---|---|---|---|---|"]
        for K in KS:
            Kname = "all" if K is None else str(K)
            for m in order[1:]:
                cells = []
                for metric in ("dMRR", "dR@1", "dR@5"):
                    r = dl[(dl.K == Kname) & (dl.scope == scope) & (dl.method == m) & (dl.metric == metric)].iloc[0]
                    s = f"{r['diff']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"
                    cells.append(f"**{s}**" if (r.ci_lo > 0 or r.ci_hi < 0) else s)
                L.append(f"| {Kname} | {m} | " + " | ".join(cells) + " |")
        L.append("")
    al_s = al.groupby(["K", "method"]).alpha.agg(lambda a: ", ".join(f"{x:.1f}" for x in a)).reset_index()
    L += ["## Chosen α per fold (1.0 = topic only, 0.0 = style only)", "", "| K | method | α (folds 1–5) |", "|---|---|---|"]
    for _, r in al_s.iterrows():
        L.append(f"| {r.K} | {r.method} | {r.alpha} |")
    L += ["", "## Conclusion", "", "CONCLUSION_PLACEHOLDER", ""]
    concl = f"{OUTDIR}/conclusion.md"
    text = "\n".join(L) + "\n"
    if os.path.exists(concl):
        text = text.replace("CONCLUSION_PLACEHOLDER", open(concl).read().strip())
    open(f"{OUTDIR}/report.md", "w").write(text)
    print(res[res.scope == "end_to_end"].pivot_table(index="method", columns="K", values="MRR").round(3))
    print(res[res.scope == "conditional"].pivot_table(index="method", columns="K", values="MRR").round(3))
    print(dl[(dl.metric == "dMRR")].pivot_table(index=["scope", "method"], columns="K", values="diff").round(3))
    print("sanity", sanity)


if __name__ == "__main__":
    main()
