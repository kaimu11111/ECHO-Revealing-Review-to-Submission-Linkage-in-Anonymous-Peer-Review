"""Two-channel decomposition of review linkage: topic/expertise channel vs content-free style channel.

Channels (per candidate, per query):
  topic : cos(SPECTER2(review), SPECTER2(candidate)) on ORIGINAL text
  style : handcrafted style cosine on AGGRESSIVELY content-masked text of BOTH sides (s11 masking, tags -> "xx")
Settings:
  edition  F1000_CORPUS=platform F1000_EXP=iclr_like F1000_VENUE_K=8        review -> the reviewer's own paper
           (title+abstract for topic, abstract for style), all articles of the same (field, year) edition
  archive  F1000_CORPUS=platform F1000_EXP=rebuttal_archive F1000_VENUE_K=8 F1000_RESPONSE_ARCHIVE=1
           review -> the reviewer's own author response, the field's response archive
Analyses: conditional-logit information decomposition (bits; reviewer-disjoint 5-fold CV), retrieval metrics,
channel independence, complementarity.

  python -m analysis.decomposition compute edition|archive   (with the matching environment)  -> outputs/decomposition/<s>.json
  python -m analysis.decomposition report                                                    -> outputs/decomposition/report.md
"""
from config import read_articles  # released benchmark: articles keyed by opaque id
import json, os, sys
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

from config import ROOT
OUTDIR = f"{ROOT}/outputs/decomposition"
RNG = np.random.default_rng(0)
N_BOOT = 2000


# ---------------------------------------------------------------- channel scores
def edition_tasks():
    """review -> own paper in the same edition (platform iclr_like, setting venue_full, main queries)."""
    os.environ["F1000_MASK_LEVEL"] = "aggressive"
    from config import DATA, XDATA, read_jsonl
    import style as stylo
    from style import Masker, render
    arts = read_articles()
    qs = {q["qid"]: q for q in read_jsonl(f"{XDATA}/private/queries_full.jsonl") if q["split"] == "main"}
    pools = [p for p in json.load(open(f"{XDATA}/private/pools.json"))["pools"]["venue_full"] if p["qid"] in qs]
    pe, pk = np.load(f"{DATA}/cache/papers.npy"), json.load(open(f"{DATA}/cache/papers_ids.json"))
    pemb = dict(zip(pk, pe))
    re_, rk = np.load(f"{XDATA}/cache/reviews_main.npy"), json.load(open(f"{XDATA}/cache/reviews_main_ids.json"))
    remb = dict(zip(rk, re_))
    # aggressive masking of the reviews and of every candidate abstract (cached)
    mpath = f"{XDATA}/style_masked_aggressive_abstracts.json"
    keys = sorted({c for p in pools for c in p["candidates"]})
    if os.path.exists(mpath):
        M = json.load(open(mpath))
    else:
        mk = Masker()
        qids = sorted(qs)
        M = {"reviews": dict(zip(qids, (m for m, _ in mk.mask_many([qs[q]["review_main"] for q in qids])))),
             "abstracts": dict(zip(keys, (m for m, _ in mk.mask_many([arts[k]["abstract"] for k in keys]))))}
        json.dump(M, open(mpath, "w"))
    rX = stylo.matrix([render(M["reviews"][q]) for q in sorted(qs)])
    rZ = dict(zip(sorted(qs), stylo.zscore_by(rX, rX)))
    aX = stylo.matrix([render(M["abstracts"][k]) for k in keys])
    aZ = dict(zip(keys, stylo.zscore_by(aX, aX)))
    tasks = []
    for p in pools:
        q = p["qid"]
        C = np.stack([pemb[c] for c in p["candidates"]])
        S = np.stack([aZ[c] for c in p["candidates"]])
        r = rZ[q]
        tasks.append({"qid": q, "pid": qs[q]["pid"], "pos": p["candidates"].index(p["positive"]),
                      "topic": (C @ remb[q]).tolist(),
                      "style": ((S @ r) / (np.linalg.norm(S, axis=1) * np.linalg.norm(r) + 1e-12)).tolist()})
    return tasks


def archive_tasks():
    """review -> own author response (platform rebuttal archive, response pools)."""
    from config import XDATA
    from scores import load_tasks, original_scores, score_tables
    from style import render
    T = load_tasks()
    S0 = original_scores(T)
    M = json.load(open(f"{XDATA}/style_masked_aggressive/masked.json"))
    mrev = {q: v["text"] for q, v in M["reviews"].items()}
    mresp = {u: v["text"] for u, v in M["responses"].items()}
    cand = {(t["qid"], i): mresp[x] for t in T for i, x in enumerate(t["texts"])}
    Sm = score_tables(T, mrev, cand, list(mrev.values()) + list(mresp.values()), with_topic=False, render=render)
    return [{"qid": t["qid"], "pid": t["pid"], "pos": t["pos"], "topic": S0[t["qid"]]["topic"].tolist(),
             "style": Sm[t["qid"]]["stylo"].tolist()} for t in T]


# ---------------------------------------------------------------- analysis
def zq(x):
    x = np.asarray(x, float)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else x * 0


def fit_clogit(X_list, pos_list):
    """Conditional logit over candidates: P(pos) = softmax(X beta)[pos]. Returns beta."""
    def nll(b):
        return -sum(X[p] @ b - logsumexp(X @ b) for X, p in zip(X_list, pos_list)) + 1e-3 * b @ b
    return minimize(nll, np.zeros(X_list[0].shape[1]), method="L-BFGS-B").x


def rank(scores, pos):
    s = np.asarray(scores) + RNG.uniform(0, 1e-9, len(scores))
    return int((s > s[pos]).sum()) + 1


def analyse(tasks):
    n = len(tasks)
    N = np.array([len(t["topic"]) for t in tasks], float)
    feats = {t["qid"]: np.c_[zq(t["topic"]), zq(t["style"])] for t in tasks}
    designs = {"topic": [0], "style": [1], "joint": [0, 1]}
    qids = [t["qid"] for t in tasks]
    groups = np.array([t["pid"] for t in tasks])
    logp = {m: np.zeros(n) for m in designs}
    score_oof = {m: [None] * n for m in designs}
    betas = {m: [] for m in designs}
    for tr, te in GroupKFold(5).split(qids, groups=groups):
        for m, cols in designs.items():
            b = fit_clogit([feats[qids[i]][:, cols] for i in tr], [tasks[i]["pos"] for i in tr])
            betas[m].append(b.tolist())
            for i in te:
                s = feats[qids[i]][:, cols] @ b
                logp[m][i] = (s[tasks[i]["pos"]] - logsumexp(s)) / np.log(2)
                score_oof[m][i] = s
    # information gain over uniform guessing, in bits (per query)
    info = {m: logp[m] + np.log2(N) for m in designs}
    bs = RNG.integers(0, n, (N_BOOT, n))

    def ci(v):
        m = v[bs].mean(1)
        return [float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
    dec = {"bits_uniform": ci(np.log2(N)),
           "I_topic": ci(info["topic"]), "I_style": ci(info["style"]), "I_joint": ci(info["joint"]),
           "I_style_given_topic": ci(info["joint"] - info["topic"]), "I_topic_given_style": ci(info["joint"] - info["style"]),
           "redundancy": ci(info["topic"] + info["style"] - info["joint"]),
           # effective candidate-set size: 2 ** (mean surprisal of the positive)
           "eff_N_uniform": float(2 ** np.mean(np.log2(N))),
           "eff_N_topic": float(2 ** np.mean(-logp["topic"])), "eff_N_style": float(2 ** np.mean(-logp["style"])),
           "eff_N_joint": float(2 ** np.mean(-logp["joint"])),
           "beta_joint_folds": betas["joint"]}
    # retrieval metrics
    ranks = {"random": [rank(RNG.uniform(size=int(k)), t["pos"]) for k, t in zip(N, tasks)],
             "topic": [rank(t["topic"], t["pos"]) for t in tasks], "style": [rank(t["style"], t["pos"]) for t in tasks],
             "joint (topic+style)": [rank(score_oof["joint"][i], tasks[i]["pos"]) for i in range(n)]}
    ranks["oracle best-of-two (upper bound)"] = list(np.minimum(ranks["topic"], ranks["style"]))
    met = {}
    for m, r in ranks.items():
        r = np.asarray(r, float)
        met[m] = {"R@1": float(np.mean(r <= 1)), "R@5": float(np.mean(r <= 5)), "R@10": float(np.mean(r <= 10)),
                  "MRR": float(np.mean(1 / r)), "median_rank": float(np.median(r)),
                  "mean_pct_rank": float(np.mean((r - 1) / (N - 1)))}
    d = (1 / np.asarray(ranks["joint (topic+style)"], float)) - (1 / np.asarray(ranks["topic"], float))
    met["joint_minus_topic_dMRR"] = ci(d)
    # independence
    within = [spearmanr(t["topic"], t["style"]).correlation for t in tasks]
    pct_t = (np.asarray(ranks["topic"]) - 1) / (N - 1)
    pct_s = (np.asarray(ranks["style"]) - 1) / (N - 1)
    indep = {"within_query_spearman_mean": float(np.nanmean(within)),
             "within_query_spearman_p10_p90": [float(np.nanpercentile(within, 10)), float(np.nanpercentile(within, 90))],
             "across_query_spearman_pos_pct_rank": float(spearmanr(pct_t, pct_s).correlation),
             "across_query_spearman_p": float(spearmanr(pct_t, pct_s).pvalue)}
    # complementarity: hits@10 cross-table; style by topic-difficulty bucket
    ht, hs = np.asarray(ranks["topic"]) <= 10, np.asarray(ranks["style"]) <= 10
    comp = {"both": int((ht & hs).sum()), "topic_only": int((ht & ~hs).sum()), "style_only": int((~ht & hs).sum()),
            "neither": int((~ht & ~hs).sum())}
    buckets = [("topic top 5%", 0, 0.05), ("topic 5-20%", 0.05, 0.20), ("topic 20-50%", 0.20, 0.50), ("topic bottom 50%", 0.50, 1.01)]
    strat = []
    for name, lo, hi in buckets:
        m = (pct_t >= lo) & (pct_t < hi)
        if m.sum():
            strat.append({"bucket": name, "n": int(m.sum()), "style_mean_pct_rank": float(pct_s[m].mean()),
                          "style_R@10": float(np.mean(np.asarray(ranks["style"])[m] <= 10)),
                          "chance_R@10": float(np.mean(np.minimum(10, N[m]) / N[m]))})
    return {"n": n, "pool_median": float(np.median(N)), "decomposition": dec, "metrics": met, "independence": indep,
            "complementarity": comp, "style_by_topic_bucket": strat}


# ---------------------------------------------------------------- report
def fmt_ci(v, d=2):
    return f"{v[0]:.{d}f} [{v[1]:.{d}f}, {v[2]:.{d}f}]"


def report():
    R = {s: json.load(open(f"{OUTDIR}/{s}.json")) for s in ("edition", "archive") if os.path.exists(f"{OUTDIR}/{s}.json")}
    title = {"edition": "Same edition: review → the reviewer's own paper (ICLR-submission analogue)",
             "archive": "Rebuttal archive: review → the reviewer's own author response (ICLR-rebuttal analogue)"}
    L = ["# Two-channel decomposition of review linkage: topic/expertise vs content-free style", "",
         "**Channels.** *Topic*: SPECTER2 similarity of review and candidate on the original text. "
         "*Style*: handcrafted style similarity after **aggressive content masking of both texts**: every noun, proper noun, adjective and number masked, about 50% of tokens. "
         "On that masked text SPECTER2 is near chance, so this channel carries little topical information by construction.", "",
         "**Method.** A conditional logit over each query's candidates turns channel scores into a probability for the true candidate. "
         "Coefficients are fitted on reviewer-disjoint training folds (5-fold CV) and evaluated out of fold.", "",
         "- **Information gain** (bits) = log₂ P(true candidate) − log₂(1/N). It measures how much anonymity is reduced relative to uniform guessing.",
         "- **Effective candidate-set size** = 2^(mean surprisal of the true candidate).",
         "- **Redundancy** = I(topic) + I(style) − I(joint). Near 0 means the two channels carry separate information.",
         "- 95% CIs are bootstraps over queries.", ""]
    for s, r in R.items():
        d, m, ind, c = r["decomposition"], r["metrics"], r["independence"], r["complementarity"]
        L += [f"## {title[s]}", "", f"{r['n']} queries, median pool {r['pool_median']:.0f} candidates.", "",
              "### Information decomposition", "",
              "| quantity | bits [95% CI] | effective candidate-set size |", "|---|---|---|",
              f"| uniform guessing (log₂ N) | {fmt_ci(d['bits_uniform'])} | {d['eff_N_uniform']:.0f} |",
              f"| I(topic) | {fmt_ci(d['I_topic'])} | {d['eff_N_topic']:.1f} |",
              f"| I(style) | {fmt_ci(d['I_style'])} | {d['eff_N_style']:.1f} |",
              f"| I(topic + style) | {fmt_ci(d['I_joint'])} | {d['eff_N_joint']:.1f} |",
              f"| I(style \\| topic) = joint − topic | {fmt_ci(d['I_style_given_topic'])} | |",
              f"| I(topic \\| style) = joint − style | {fmt_ci(d['I_topic_given_style'])} | |",
              f"| redundancy = topic + style − joint | {fmt_ci(d['redundancy'])} | |", "",
              f"Joint-model coefficients per fold (topic, style): " + "; ".join(f"({b[0]:.2f}, {b[1]:.2f})" for b in d["beta_joint_folds"]), "",
              "### Retrieval", "", "| ranking | R@1 | R@5 | R@10 | MRR | median rank | mean pct rank |", "|---|---|---|---|---|---|---|"]
        for k, v in m.items():
            if isinstance(v, dict):
                L.append(f"| {k} | {v['R@1']:.3f} | {v['R@5']:.3f} | {v['R@10']:.3f} | {v['MRR']:.3f} | {v['median_rank']:.0f} | {v['mean_pct_rank']:.3f} |")
        L += ["", f"ΔMRR joint − topic: {fmt_ci(m['joint_minus_topic_dMRR'], 3)}", "",
              "### Are the channels independent?", "",
              f"- Within-query rank correlation between topic and style scores over candidates: mean ρ = {ind['within_query_spearman_mean']:.3f} "
              f"(10th–90th percentile {ind['within_query_spearman_p10_p90'][0]:.2f} to {ind['within_query_spearman_p10_p90'][1]:.2f}).",
              f"- Across queries, correlation of the true candidate's percentile rank under topic vs under style: ρ = {ind['across_query_spearman_pos_pct_rank']:.3f} "
              f"(p = {ind['across_query_spearman_p']:.2g}).", "",
              "### Complementarity (hit = true candidate in top 10)", "",
              "| | style hit | style miss |", "|---|---|---|",
              f"| **topic hit** | {c['both']} | {c['topic_only']} |", f"| **topic miss** | {c['style_only']} | {c['neither']} |", "",
              "Style channel by how well topic did on the same query:", "",
              "| topic percentile of the true candidate | n | style mean pct rank (0.5 = chance) | style R@10 | chance R@10 |", "|---|---|---|---|---|"]
        for b in r["style_by_topic_bucket"]:
            L.append(f"| {b['bucket']} | {b['n']} | {b['style_mean_pct_rank']:.3f} | {b['style_R@10']:.3f} | {b['chance_R@10']:.3f} |")
        L.append("")
    L += ["## Interpretation", "", "INTERP_PLACEHOLDER", ""]
    text = "\n".join(L) + "\n"
    if os.path.exists(f"{OUTDIR}/interpretation.md"):
        text = text.replace("INTERP_PLACEHOLDER", open(f"{OUTDIR}/interpretation.md").read().strip())
    open(f"{OUTDIR}/report.md", "w").write(text)
    print("written", f"{OUTDIR}/report.md")


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    if sys.argv[1] == "compute":
        s = sys.argv[2]
        res = analyse(edition_tasks() if s == "edition" else archive_tasks())
        json.dump(res, open(f"{OUTDIR}/{s}.json", "w"), indent=1)
        print(json.dumps({k: res[k] for k in ("n", "metrics", "independence", "complementarity")}, indent=1))
        print({k: v for k, v in res["decomposition"].items() if k != "beta_joint_folds"})
    else:
        report()
