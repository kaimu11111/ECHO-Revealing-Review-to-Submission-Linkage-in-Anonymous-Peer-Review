"""Ablations evaluated with the full three-stage pipeline: fusion on a feature subset -> top-20 -> Luna rerank.

Variants: paper only, rebuttal only, content only (no style), writing style only, full. For each: reviewer-disjoint
fusion (nested CV), Luna 0-100 scores on the fusion top-20 (cached), alpha chosen on reviewer-disjoint folds.
Output: outputs/ablation_full/{results.csv, report.md}
Environment as for the combined setting; OpenRouter key in OPENROUTER_API_KEY.
"""
from config import read_articles  # released benchmark: articles keyed by opaque id
import json, os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from config import ROOT
OUTDIR = f"{ROOT}/outputs/ablation_full"
RNG = np.random.default_rng(0)
K = 20


def main():
    from config import DATA, XDATA, read_jsonl
    from signals import archive_tasks, cv_fusion, metrics, boot
    from rerank import Client, prompt, rank_of
    from analysis.rewriting import final_cols
    os.makedirs(OUTDIR, exist_ok=True)
    tasks = archive_tasks()
    meta = json.load(open(f"{XDATA}/cache/boost_meta.json"))
    arts = read_articles()
    F = np.load(f"{XDATA}/cache/boost_features_combined.npz", allow_pickle=True)["feats"].item()
    full = final_cols(F)
    paper = [c for c in full if c.startswith("paper_")]
    resp_content = [c for c in full if c.startswith("resp_") and c not in ("resp_stylo_masked", "resp_star")]
    style = ["resp_stylo_masked", "resp_star"]
    variants = {"paper only": paper, "rebuttal only": resp_content + style,
                "content only": paper + resp_content, "writing style only": style, "full": full}
    N = [m["n"] for m in meta]
    g = np.array([m["pid"] for m in meta])
    cli = Client("openai/gpt-6-luna", f"{ROOT}/outputs/llm_rerank/cache_openai_gpt-6-luna_budget2000_score.jsonl",
                 2400, False, None, "score", None, 2000)

    def z(x):
        sd = x.std()
        return (x - x.mean()) / sd if sd > 0 else x * 0

    rows, ranks_all = [], {}
    for name, cols in variants.items():
        oof, _ = cv_fusion(meta, F, cols)
        fused = [np.asarray(s, float) for s in oof]
        r_fus = [rank_of(s, m["pos"]) for s, m in zip(fused, meta)]
        topk = [[int(i) for i in np.argsort(-(s + RNG.uniform(0, 1e-9, len(s))))[:K]] for s in fused]
        jobs = [(ti, i) for ti, tk in enumerate(topk) for i in tk]
        ps = [prompt(tasks[ti]["review"], arts[tasks[ti]["cands"][i]]["title"], arts[tasks[ti]["cands"][i]]["abstract"],
                     tasks[ti]["texts"][i], "score") for ti, i in jobs]
        new = sum(cli._h(p) not in cli.cache for p in ps)
        print(f"{name}: {len(ps)} judgements, {new} new", flush=True)
        with ThreadPoolExecutor(24) as ex:
            recs = list(ex.map(cli.judge, ps))
        llm = [np.full(len(t["cands"]), -np.inf) for t in tasks]
        for (ti, i), r in zip(jobs, recs):
            llm[ti][i] = r["p_yes"]

        def fr(ti, alpha):
            k = topk[ti]
            comb = alpha * z(fused[ti][k]) + (1 - alpha) * z(llm[ti][k])
            s = fused[ti] - 1e6
            s[k] = comb
            return rank_of(s, tasks[ti]["pos"])
        pred, alphas = [None] * len(tasks), []
        for tr, te in GroupKFold(5).split(tasks, groups=g):
            best, bm = None, -1
            for al in np.round(np.arange(0, 1.01, 0.1), 1):
                mrr = np.mean([1 / fr(i, al) for i in tr])
                if mrr > bm + 1e-12:
                    best, bm = al, mrr
            alphas.append(best)
            for i in te:
                pred[i] = fr(i, best)
        ranks_all[name] = pred
        rows.append({"variant": name, "n_scores": len(cols), "stage": "fusion only", **metrics(r_fus, N)})
        rows.append({"variant": name, "n_scores": len(cols), "stage": "full (fusion + LLM)", "alphas": str(alphas), **metrics(pred, N)})
        print(name, {k: round(v, 3) for k, v in rows[-1].items() if isinstance(v, float) and k in ("R@1", "R@5", "R@10", "MRR")}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTDIR}/results.csv", index=False, float_format="%.4f")
    deltas = {k: boot(v, ranks_all["full"]) for k, v in ranks_all.items() if k != "full"}
    json.dump(deltas, open(f"{OUTDIR}/deltas_vs_full.json", "w"), indent=1)
    L = ["# Ablations with the full three-stage pipeline (fusion subset -> top-20 -> Luna rerank)", "",
         "| variant | #scores | Top-1 | Top-5 | Top-10 | Top-20 | MRR | fusion-only MRR | ΔMRR vs full [95% CI] |", "|---|---|---|---|---|---|---|---|---|"]
    for name in variants:
        f_ = df[(df.variant == name) & (df.stage.str.startswith("full"))].iloc[0]
        s_ = df[(df.variant == name) & (df.stage == "fusion only")].iloc[0]
        d = deltas.get(name)
        dm = f"{d['dMRR'][0]:+.3f} [{d['dMRR'][1]:+.3f}, {d['dMRR'][2]:+.3f}]" if d else "—"
        L.append(f"| {name} | {int(f_.n_scores)} | {f_['R@1']:.3f} | {f_['R@5']:.3f} | {f_['R@10']:.3f} | {f_['R@20']:.3f} | {f_.MRR:.3f} | {s_.MRR:.3f} | {dm} |")
    open(f"{OUTDIR}/report.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
