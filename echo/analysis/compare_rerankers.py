"""Head-to-head comparison of Stage-2 rerankers on the full benchmark (298 queries, fusion top-20): per-query final
ranks for each model (alpha by reviewer-disjoint CV, missing judgements neutral) and paired bootstraps between models.
Output: outputs/llm_rerank/rerankers_compare.{md,json}
"""
import json, os, sys
import numpy as np
from sklearn.model_selection import GroupKFold

from config import ROOT
OUTDIR = f"{ROOT}/outputs/llm_rerank"
MODELS = {  # label -> (cache file, model id, backend)
    "GPT-6 Luna (OpenRouter, budget 2000)": ("cache_openai_gpt-6-luna_budget2000_score.jsonl", "openai/gpt-6-luna"),
    "Claude Opus 5.5 (Bedrock, low)": ("cache_us.anthropic.claude-opus-5-5_bedrock_reason-low_score.jsonl", "us.anthropic.claude-opus-5-5"),
    "GPT-6 Astra (Bedrock, low)": ("cache_us.openai.gpt-6-astra_bedrock_reason-low_score.jsonl", "us.openai.gpt-6-astra"),
}


def main():
    sys.argv = ["x", "eval"]
    import rerank as S
    from signals import metrics, boot
    import hashlib
    T = S.load_topk(20)
    N = [t["n"] for t in T]
    g = np.array([t["pid"] for t in T])

    def z(x):
        x = np.asarray(x, float)
        m, sd = np.nanmean(x), np.nanstd(x)
        out = (x - m) / sd if sd > 0 else x * 0
        return np.nan_to_num(out, nan=0.0)

    def fused_rank(t, s_llm, alpha):
        k = t["topk"]
        comb = alpha * z(t["fused"][k]) + (1 - alpha) * z(s_llm[k])
        full = t["fused"] - 1e6
        full[k] = comb
        return S.rank_of(full, t["pos"])

    ranks = {"fusion only": [S.rank_of(t["fused"], t["pos"]) for t in T]}
    info = {}
    for label, (cf, mid) in MODELS.items():
        cache = {}
        for line in open(f"{OUTDIR}/{cf}"):
            r = json.loads(line)
            cache[r["h"]] = r
        h = lambda p: hashlib.sha1((mid + "\n" + p).encode()).hexdigest()
        llm, missing = [], 0
        for t in T:
            s = np.full(t["n"], np.nan)
            for i in t["topk"]:
                r = cache.get(h(S.prompt(t["review"], *t["cand"][i], "score")))
                if r is None:
                    missing += 1
                else:
                    s[i] = r["p_yes"]
            llm.append(s)
        pred, alphas = [None] * len(T), []
        for tr, te in GroupKFold(5).split(T, groups=g):
            best, bm = None, -1
            for al in np.round(np.arange(0, 1.01, 0.1), 1):
                mrr = np.mean([1 / fused_rank(T[i], llm[i], al) for i in tr])
                if mrr > bm + 1e-12:
                    best, bm = al, mrr
            alphas.append(float(best))
            for i in te:
                pred[i] = fused_rank(T[i], llm[i], best)
        ranks[label] = pred
        info[label] = {"missing_judgements": missing, "alphas": alphas}
        print(label, "missing", missing, "alphas", alphas, flush=True)

    res = {k: metrics(v, N) for k, v in ranks.items()}
    pairs = [("GPT-6 Astra (Bedrock, low)", "GPT-6 Luna (OpenRouter, budget 2000)"),
             ("Claude Opus 5.5 (Bedrock, low)", "GPT-6 Luna (OpenRouter, budget 2000)"),
             ("GPT-6 Astra (Bedrock, low)", "Claude Opus 5.5 (Bedrock, low)")]
    deltas = {f"{a} vs {b}": boot(ranks[a], ranks[b]) for a, b in pairs}
    json.dump({"metrics": res, "info": info, "paired_bootstrap": deltas, "ranks": ranks}, open(f"{OUTDIR}/rerankers_compare.json", "w"), indent=1)
    L = ["# Stage-2 rerankers on the full benchmark (298 queries, fusion top-20, alpha by reviewer-disjoint CV)", "",
         "| reranker | Top-1 | Top-5 | Top-10 | Top-20 | MRR | median rank | missing judgements |", "|---|---|---|---|---|---|---|---|"]
    for k, m in res.items():
        L.append(f"| {k} | {m['R@1']:.3f} | {m['R@5']:.3f} | {m['R@10']:.3f} | {m['R@20']:.3f} | {m['MRR']:.3f} | {m['median_rank']:.0f} | {info.get(k, {}).get('missing_judgements', '—')} |")
    L += ["", "Paired bootstrap over queries (2,000 resamples). **Bold** = 95% CI excludes 0.", "",
          "| comparison | ΔMRR | ΔTop-1 | ΔTop-5 | ΔTop-10 |", "|---|---|---|---|---|"]
    for k, d in deltas.items():
        cells = []
        for kk in ("dMRR", "dR@1", "dR@5", "dR@10"):
            m, lo, hi = d[kk]
            c = f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            cells.append(f"**{c}**" if lo > 0 or hi < 0 else c)
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    open(f"{OUTDIR}/rerankers_compare.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
