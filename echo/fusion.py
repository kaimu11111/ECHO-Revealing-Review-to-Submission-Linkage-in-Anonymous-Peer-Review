"""Stage-1 fusion and evaluation: within-instance standardization, conditional logit with L2 regularization chosen by
inner reviewer-grouped cross-validation, reviewer-grouped 5-fold evaluation, retrieval metrics and paired bootstraps
(formerly the fusion part of signals.py).

  python fusion.py evaluate archive            fusion of the rebuttal-side signals
  python fusion.py combined-evaluate [star]   paper + rebuttal signals (the ECHO Stage-1 ranking)
  python fusion.py report
"""
import collections, json, os, sys
import numpy as np
from config import ROOT, XDATA

OUTDIR = f"{ROOT}/outputs/boost"
RNG = np.random.default_rng(0)
N_BOOT = 2000


# ---------------------------------------------------------------- fusion + evaluation
def zq(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x * 0


def fit_clogit(Xs, ps, lam):
    from scipy.optimize import minimize
    from scipy.special import logsumexp, softmax

    def f(b):
        v, g = lam * b @ b, 2 * lam * b
        for X, p in zip(Xs, ps):
            s = X @ b
            v -= s[p] - logsumexp(s)
            g -= X[p] - softmax(s) @ X
        return v, g
    return minimize(f, np.zeros(Xs[0].shape[1]), jac=True, method="L-BFGS-B").x


def cv_fusion(meta, F, names, lams=(0.01, 0.1, 1.0, 10.0)):
    """Out-of-fold fused scores; L2 strength picked by inner reviewer-disjoint CV (mean log-likelihood)."""
    from sklearn.model_selection import GroupKFold
    from scipy.special import logsumexp
    X = [np.stack([zq(F[k][m["qid"]]) for k in names], 1) for m in meta]
    pos = [m["pos"] for m in meta]
    g = np.array([m["pid"] for m in meta])
    oof, chosen = [None] * len(meta), []
    for tr, te in GroupKFold(5).split(X, groups=g):
        best, best_ll = None, -np.inf
        for lam in lams:
            ll = 0
            for itr, ite in GroupKFold(4).split(tr, groups=g[tr]):
                b = fit_clogit([X[tr[i]] for i in itr], [pos[tr[i]] for i in itr], lam)
                ll += sum((X[tr[i]] @ b)[pos[tr[i]]] - logsumexp(X[tr[i]] @ b) for i in ite)
            if ll > best_ll:
                best, best_ll = lam, ll
        b = fit_clogit([X[i] for i in tr], [pos[i] for i in tr], best)
        chosen.append({"lambda": best, "beta": dict(zip(names, np.round(b, 3).tolist()))})
        for i in te:
            oof[i] = X[i] @ b
    return oof, chosen


def rank(s, p):
    s = np.asarray(s, float) + RNG.uniform(0, 1e-9, len(s))
    return int((s > s[p]).sum()) + 1


def metrics(ranks, N, scores=None, pos=None):
    r, N = np.asarray(ranks, float), np.asarray(N, float)
    m = {"R@1": np.mean(r <= 1), "R@5": np.mean(r <= 5), "R@10": np.mean(r <= 10), "R@20": np.mean(r <= 20),
         "R@30": np.mean(r <= 30), "MRR": np.mean(1 / r),
         "median_rank": float(np.median(r)), "mean_pct_rank": float(np.mean((r - 1) / (N - 1)))}
    return {k: float(v) for k, v in m.items()}


def boot(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = {}
    for k, f in (("dMRR", lambda r: 1 / r), ("dR@1", lambda r: (r <= 1) * 1.0), ("dR@5", lambda r: (r <= 5) * 1.0),
                 ("dR@10", lambda r: (r <= 10) * 1.0), ("dR@20", lambda r: (r <= 20) * 1.0), ("dR@30", lambda r: (r <= 30) * 1.0)):
        d = f(a) - f(b)
        m = d[RNG.integers(0, len(d), (N_BOOT, len(d)))].mean(1)
        out[k] = [float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
    return out


def evaluate(setting):
    import pandas as pd
    from config import XDATA
    os.makedirs(OUTDIR, exist_ok=True)
    F = np.load(f"{XDATA}/cache/boost_features.npz", allow_pickle=True)["feats"].item()
    meta = json.load(open(f"{XDATA}/cache/boost_meta.json"))
    N = [m["n"] for m in meta]
    singles = {k: [rank(F[k][m["qid"]], m["pos"]) for m in meta] for k in sorted(F)}
    rows = [{"setting": setting, "kind": "single", "method": k, **metrics(v, N)} for k, v in singles.items()]
    dense = [k for k in F if k.endswith(("_max", "_col", "_mean", "_full"))]
    lex = ["bm25_uni", "bm25_bi", "char"]
    style = ["stylo_masked", "luar"]
    base_set = ["topic_old", "char_old"] if setting == "archive" else ["topic_old"]
    sets = {"baseline (previous best: " + " + ".join(base_set) + ")": base_set,
            "dense ensemble": dense, "dense + lexical": dense + lex,
            "dense + lexical + style (all)": dense + lex + style,
            "lexical + style (no dense)": lex + style,
            "dense + lexical + handcrafted style only": dense + lex + ["stylo_masked"]}
    # alternative neural style models from encoders.py, each replacing LUAR, then all together
    extra = [m for m in ("luarcrud", "styledist", "star") if m in F]
    for m in extra:
        sets[f"dense + lexical + handcrafted + {m} (instead of LUAR)"] = dense + lex + ["stylo_masked", m]
    if extra:
        sets["dense + lexical + handcrafted + all neural style models"] = dense + lex + style + extra
    fused, chosen = {}, {}
    for name, cols in sets.items():
        oof, ch = cv_fusion(meta, F, cols)
        fused[name] = [rank(s, m["pos"]) for s, m in zip(oof, meta)]
        chosen[name] = ch
        rows.append({"setting": setting, "kind": "fusion", "method": name, **metrics(fused[name], N)})
        print(name, rows[-1], flush=True)
    base = fused[list(sets)[0]]
    deltas = [{"setting": setting, "method": k, "vs": "baseline", **{m: v for m, v in boot(r, base).items()}}
              for k, r in fused.items() if r is not base]
    pd.DataFrame(rows).to_csv(f"{OUTDIR}/{setting}_results.csv", index=False, float_format="%.4f")
    json.dump({"deltas": deltas, "chosen": chosen, "n": len(meta), "pool_median": float(np.median(N))},
              open(f"{OUTDIR}/{setting}_fusion.json", "w"), indent=1)



def evaluate_combined(style_models):
    """Same 298 queries and pools; paper-only vs response-only vs paper + response fusion."""
    import pandas as pd
    from config import XDATA
    F = np.load(f"{XDATA}/cache/boost_features_combined.npz", allow_pickle=True)["feats"].item()
    meta = json.load(open(f"{XDATA}/cache/boost_meta.json"))
    N = [m["n"] for m in meta]
    dense = lambda side: [k for k in F if k.startswith(side + "_") and k.endswith(("_max", "_col", "_mean", "_full"))
                          and not any(k.startswith(f"{side}_{s}") for s in ("luar", "luarcrud", "styledist", "star"))]
    lex = lambda side: [f"{side}_{k}" for k in ("bm25_uni", "bm25_bi", "char")]
    style = ["resp_stylo_masked"] + [f"resp_{m}" for m in style_models]
    sets = {"paper only (dense + lexical)": dense("paper") + lex("paper"),
            "response only (dense + lexical + style)": dense("resp") + lex("resp") + style,
            "paper + response (all)": dense("paper") + lex("paper") + dense("resp") + lex("resp") + style,
            "paper + response, no style": dense("paper") + lex("paper") + dense("resp") + lex("resp")}
    rows, fused, chosen = [], {}, {}
    for name, cols in sets.items():
        oof, ch = cv_fusion(meta, F, cols)
        fused[name] = [rank(sc, m["pos"]) for sc, m in zip(oof, meta)]
        chosen[name] = ch
        rows.append({"setting": "combined", "kind": "fusion", "method": name, "n_scores": len(cols), **metrics(fused[name], N)})
        print(name, {k: round(v, 3) for k, v in rows[-1].items() if isinstance(v, float)}, flush=True)
    full = fused["paper + response (all)"]
    deltas = [{"method": "paper + response (all)", "vs": k, **boot(full, r)} for k, r in fused.items() if k != "paper + response (all)"]
    pd.DataFrame(rows).to_csv(f"{OUTDIR}/combined_results.csv", index=False, float_format="%.4f")
    json.dump({"deltas": deltas, "chosen": chosen, "style_models": style_models, "n": len(meta), "pool_median": float(np.median(N)),
               "chance": {f"R@{k}": float(np.mean([min(k, n) / n for n in N])) for k in (1, 5, 10, 20, 30)}},
              open(f"{OUTDIR}/combined_fusion.json", "w"), indent=1)


def report():
    import pandas as pd
    title = {"archive": "Rebuttal archive: review → the reviewer's own author response",
             "edition": "Same edition: review → the reviewer's own paper (title + abstract)"}
    L = ["# Stage-1 accuracy push with existing models (no training)", "",
         "Same queries, pools and metrics as before. Everything is evaluated out of fold: 5-fold CV grouped by reviewer, "
         "with the fusion L2 strength chosen by an inner reviewer-disjoint CV inside each training fold. "
         "No encoder is trained or fine-tuned; only the fusion coefficients (one per score) are fitted.", "",
         "Score names:",
         "- `topic_old` / `char_old`: the earlier scores (SPECTER2 on the review truncated to 512 tokens; char n-gram on responses capped at 600 words).",
         "- `<model>_max|col|mean`: chunked dense matching. `max` = best chunk pair; `col` = mean over review chunks of the best candidate chunk; `mean` = cosine of mean chunk embeddings.",
         "- `bm25_uni|bi`: BM25 over content words, and over uni+bigrams.",
         "- `char`: char 2–4-gram TF-IDF on the uncapped text.",
         "- `stylo_masked`: handcrafted style on aggressively content-masked text.",
         "- `luar`: LUAR author embedding on the original text.", ""]
    for s in ("archive", "edition"):
        if not os.path.exists(f"{OUTDIR}/{s}_results.csv"):
            continue
        d = pd.read_csv(f"{OUTDIR}/{s}_results.csv")
        J = json.load(open(f"{OUTDIR}/{s}_fusion.json"))
        L += [f"## {title[s]}", "", f"{J['n']} queries, median pool {J['pool_median']:.0f}.", "", "### Fusion", "",
              "| method | R@1 | R@5 | R@10 | R@20 | R@30 | MRR | median rank | mean pct rank |", "|---|---|---|---|---|---|---|---|---|"]
        for _, r in d[d.kind == "fusion"].iterrows():
            L.append(f"| {r.method} | {r['R@1']:.3f} | {r['R@5']:.3f} | {r['R@10']:.3f} | {r.get('R@20', float('nan')):.3f} | {r.get('R@30', float('nan')):.3f} | {r.MRR:.3f} | {r.median_rank:.0f} | {r.mean_pct_rank:.3f} |")
        L += ["", "Paired bootstrap vs the baseline (2,000 resamples). **Bold** = 95% CI excludes 0.", "",
              "| method | ΔMRR | ΔR@1 | ΔR@5 | ΔR@10 |", "|---|---|---|---|---|"]
        for x in J["deltas"]:
            cells = []
            for k in ("dMRR", "dR@1", "dR@5", "dR@10"):
                m, lo, hi = x[k]
                c = f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                cells.append(f"**{c}**" if lo > 0 or hi < 0 else c)
            L.append(f"| {x['method']} | " + " | ".join(cells) + " |")
        L += ["", "### Single scores (sorted by MRR)", "", "| score | R@1 | R@5 | R@10 | MRR | median rank |", "|---|---|---|---|---|---|"]
        for _, r in d[d.kind == "single"].sort_values("MRR", ascending=False).iterrows():
            L.append(f"| {r.method} | {r['R@1']:.3f} | {r['R@5']:.3f} | {r['R@10']:.3f} | {r.MRR:.3f} | {r.median_rank:.0f} |")
        top = J["chosen"]["dense + lexical + style (all)"]
        avg = collections.defaultdict(list)
        for f in top:
            for k, v in f["beta"].items():
                avg[k].append(v)
        L += ["", "### Fusion weights of the full model (mean over outer folds; inputs z-scored within query)", "",
              "| score | weight |", "|---|---|"] + [f"| {k} | {np.mean(v):+.2f} |" for k, v in sorted(avg.items(), key=lambda kv: -abs(np.mean(kv[1])))]
        L += [f"\nChosen L2 per fold: {[f['lambda'] for f in top]}", ""]
    if os.path.exists(f"{OUTDIR}/combined_results.csv"):
        d = pd.read_csv(f"{OUTDIR}/combined_results.csv")
        J = json.load(open(f"{OUTDIR}/combined_fusion.json"))
        ch = J["chance"]
        L += ["## Combined: each candidate = its paper (title + abstract) + its author response", "",
              f"Same {J['n']} rebuttal-archive queries and pools (median {J['pool_median']:.0f} candidates). "
              f"Paper-side and response-side scores are fused in one conditional logit. Neural style model(s) on the response side: {', '.join(J['style_models']) or 'none'}.", "",
              "| method | #scores | R@1 | R@5 | R@10 | R@20 | R@30 | MRR | median rank |", "|---|---|---|---|---|---|---|---|---|"]
        for _, r in d.iterrows():
            L.append(f"| {r.method} | {r.n_scores} | {r['R@1']:.3f} | {r['R@5']:.3f} | {r['R@10']:.3f} | {r['R@20']:.3f} | {r['R@30']:.3f} | {r.MRR:.3f} | {r.median_rank:.0f} |")
        L.append(f"| chance | | {ch['R@1']:.3f} | {ch['R@5']:.3f} | {ch['R@10']:.3f} | {ch['R@20']:.3f} | {ch['R@30']:.3f} | | |")
        L += ["", "Paired bootstrap: paper + response (all) minus each alternative. **Bold** = 95% CI excludes 0.", "",
              "| vs | ΔMRR | ΔR@1 | ΔR@10 | ΔR@20 | ΔR@30 |", "|---|---|---|---|---|---|"]
        for x in J["deltas"]:
            cells = []
            for k in ("dMRR", "dR@1", "dR@10", "dR@20", "dR@30"):
                m, lo, hi = x[k]
                c = f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                cells.append(f"**{c}**" if lo > 0 or hi < 0 else c)
            L.append(f"| {x['vs']} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## Interpretation", "", "INTERP_PLACEHOLDER", ""]
    text = "\n".join(L) + "\n"
    if os.path.exists(f"{OUTDIR}/interpretation.md"):
        text = text.replace("INTERP_PLACEHOLDER", open(f"{OUTDIR}/interpretation.md").read().strip())
    open(f"{OUTDIR}/report.md", "w").write(text)
    print("written", f"{OUTDIR}/report.md")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "features":
        compute_features(sys.argv[2])
    elif cmd == "evaluate":
        evaluate(sys.argv[2])
    elif cmd == "combined-features":
        compute_combined()
    elif cmd == "combined-evaluate":
        evaluate_combined(sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] else [])
    else:
        report()


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "evaluate":
        evaluate(sys.argv[2])
    elif cmd == "combined-evaluate":
        evaluate_combined(sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] else [])
    else:
        report()
