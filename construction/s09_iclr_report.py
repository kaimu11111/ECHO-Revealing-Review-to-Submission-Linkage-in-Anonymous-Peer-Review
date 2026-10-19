"""Report for the ICLR-style venue experiment (F1000_EXP=iclr_like F1000_VENUE_K=8) -> outputs/iclr_like/iclr_results.md"""
import collections, json, os
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from common import DATA, OUT, XDATA, VENUE_K, CORPUS, RESPONSE_ARCHIVE, ARCHIVE_LAG_DAYS, read_jsonl, field_clusters, venue_of, pub_date
from s08_report import table, paired, tag_query_sets

METHODS = ["random", "topic_review", "topic_reviewed", "stylo_cos", "stylo_euclid", "stylo_charngram",
           "lr_style", "lr_topic", "lr_topic_style", "diag_date_proximity", "diag_same_type"]


def main():
    arts = {a["article_key"]: a for a in read_jsonl(f"{DATA}/articles.jsonl")}
    cl = field_clusters(VENUE_K)
    qs = [x for x in read_jsonl(f"{XDATA}/private/queries_full.jsonl") if x["split"] == "main"]
    ids = [x["qid"] for x in qs]
    fq = json.load(open(f"{XDATA}/funnel_queries.json"))
    P = json.load(open(f"{XDATA}/private/pools.json"))
    res = pd.read_csv(f"{OUT}/pilot_results.csv")
    pq = tag_query_sets(pd.read_csv(f"{OUT}/per_query_ranks.csv"))

    # field labels from the most frequent keywords
    kw = collections.defaultdict(collections.Counter)
    for k, c in cl.items():
        kw[c].update(w.lower() for w in arts[k]["keywords"])
    sizes = collections.Counter(cl.values())
    venues = collections.Counter(venue_of(arts[k], cl) for k in cl)
    qfield = collections.Counter(cl[x["Q"]] for x in qs)

    def block(setting, query_set="main"):
        d = res[(res.experiment == "review->paper") & (res.query_set == query_set) & (res.setting == setting) & (res.review_text == "main")]
        d = d.set_index("method").loc[[m for m in METHODS if m in set(d.method)]].reset_index()
        d["R@10 95% CI"] = d.apply(lambda r: f"{r['R@10_ci_lo']:.3f}–{r['R@10_ci_hi']:.3f}", axis=1)
        d["MRR 95% CI"] = d.apply(lambda r: f"{r['MRR_ci_lo']:.3f}–{r['MRR_ci_hi']:.3f}", axis=1)
        d["p(R@10>chance)"] = d.apply(lambda r: f"{binomtest(int(round(r['R@10'] * r['n'])), int(r['n']), min(r['chance_R@10'], 1.0), alternative='greater').pvalue:.1e}", axis=1)
        ch = d.iloc[0]
        note = (f"n = {int(ch['n'])}, median pool size = {ch['pool_size_median']:.0f}. Chance: R@1 = {ch['chance_R@1']:.3f}, R@5 = {ch['chance_R@5']:.3f}, "
                f"R@10 = {ch['chance_R@10']:.3f}, MRR = {ch['chance_MRR']:.3f}, mean percentile rank = 0.50.\n\n")
        return note + table(d, ["method", "R@1", "R@5", "R@10", "R@10 95% CI", "MRR", "MRR 95% CI", "median_rank", "mean_pct_rank", "p(R@10>chance)"])

    comps = [("lr_topic_style", "lr_topic"), ("stylo_charngram", "topic_review"), ("stylo_cos", "topic_review"),
             ("lr_style", "random"), ("stylo_cos", "random"), ("stylo_charngram", "random")]

    def comp(setting, qids):
        rows = []
        for a, b in comps:
            for metric in ("MRR", "R@10"):
                m, lo, hi, p = paired(pq, setting, a, b, qids, metric)
                rows.append({"comparison": f"{a} − {b}", "metric": metric, "diff": m, "95% CI": f"{lo:+.3f} to {hi:+.3f}", "p(diff≤0)": f"{p:.3f}"})
        return table(pd.DataFrame(rows), ["comparison", "metric", "diff", "95% CI", "p(diff≤0)"], fmt="{:+.3f}")

    # who wrote the abstract? breakdown on the realistic full-venue ranking
    pos = {x["qid"]: ("sole/first" if x["author_pos"] == 0 else ("last" if x["author_pos"] == x["n_authors"] - 1 else "middle")) for x in qs}
    team = {x["qid"]: ("≤3 authors" if x["n_authors"] <= 3 else ">3 authors") for x in qs}
    d = pq[(pq.setting == "venue_full") & (pq.review_text == "main") & (pq.query_set == "all") & pq.qid.isin(ids)].copy()
    d["pct"] = (d["rank"] - 1) / (d["pool_size"] - 1)
    bd = []
    for name, g in (("author position", pos), ("team size", team)):
        d["grp"] = d.qid.map(g)
        for (grp, m), x in d[d.method.isin(["random", "topic_reviewed", "topic_review", "stylo_cos", "stylo_charngram", "lr_topic_style"])].groupby(["grp", "method"]):
            bd.append({"split": name, "group": grp, "n": len(x), "method": m, "MRR": (1 / x["rank"]).mean()})
    bd = pd.DataFrame(bd).pivot_table(index=["split", "group", "n"], columns="method", values="MRR").reset_index()

    # ---- author-response experiment (review -> the reviewer's own author response on P) ----
    REPS = ["review->paper", "review->response", "review->paper+response"]
    rr = res[(res.query_set == "response_subset") & (res.review_text == "main") & (res.setting == "venue_full")]
    rr_n = int(rr.n.max()) if len(rr) else 0
    rr_pool = rr.pool_size_median.max() if len(rr) else 0
    resp_ids = sorted(pq[(pq.experiment == "review->response") & (pq.setting == "venue_full") & (pq.query_set == "response_subset")].qid.unique())

    def rblock(exp):
        d = rr[rr.experiment == exp].set_index("method")
        d = d.loc[[m for m in ["random", "topic_review", "topic_reviewed", "stylo_cos", "stylo_euclid", "stylo_charngram",
                               "diag_date_proximity", "diag_same_type"] if m in d.index]].reset_index()
        d["R@10 95% CI"] = d.apply(lambda r: f"{r['R@10_ci_lo']:.3f}–{r['R@10_ci_hi']:.3f}", axis=1)
        d["MRR 95% CI"] = d.apply(lambda r: f"{r['MRR_ci_lo']:.3f}–{r['MRR_ci_hi']:.3f}", axis=1)
        return table(d, ["method", "R@1", "R@5", "R@10", "R@10 95% CI", "MRR", "MRR 95% CI", "median_rank", "mean_pct_rank"])

    def rcomp():
        if not resp_ids:
            return "_Per-query ranks for the response experiment were not saved in this run; re-run s07 to get paired tests._\n"
        rows = []
        pairs = [("stylo_cos", "review->response", "stylo_cos", "review->paper"),
                 ("stylo_charngram", "review->response", "stylo_charngram", "review->paper"),
                 ("topic_review", "review->response", "topic_review", "review->paper"),
                 ("stylo_cos", "review->response", "random", "review->response"),
                 ("stylo_charngram", "review->response", "random", "review->response"),
                 ("stylo_cos", "review->paper+response", "stylo_cos", "review->paper"),
                 ("stylo_charngram", "review->paper+response", "topic_review", "review->paper+response")]
        for a, ea, b, eb in pairs:
            for metric in ("MRR", "R@10"):
                m, lo, hi, p = paired(pq, "venue_full", a, b, resp_ids, metric, ea, eb, query_set="response_subset")
                rows.append({"comparison": f"{a} [{ea.split('->')[1]}] − {b} [{eb.split('->')[1]}]", "metric": metric,
                             "diff": m, "95% CI": f"{lo:+.3f} to {hi:+.3f}", "p(diff≤0)": f"{p:.3f}"})
        return table(pd.DataFrame(rows), ["comparison", "metric", "diff", "95% CI", "p(diff≤0)"], fmt="{:+.3f}")

    journal = lambda k: k.split("/")[1].split(".")[0] if "/" in k else "unknown"
    qj = collections.Counter(journal(x["Q"]) for x in qs)
    pj = collections.Counter(journal(x["P"]) for x in qs)
    cross = sum(journal(x["Q"]) != journal(x["P"]) for x in qs)
    aj = collections.Counter(journal(k) for k in cl)
    ctrl = sorted({p["qid"] for p in P["pools"]["venue_topic_matched"] if (p["n_above"] or 0) >= 24} & set(ids))
    verdict = open(f"{OUT}/verdict.md").read() if os.path.exists(f"{OUT}/verdict.md") else "_No verdict written yet._"

    L = ["# ICLR-style test: linking an anonymous review to the reviewer's own article in the same conference edition", "",
         "## Design", "",
         "ICLR is a single-field venue held once a year. Its reviewers are matched to submissions by expertise, and many reviewers also submit to the same edition. "
         "F1000Research has no editions and spans many fields, so this experiment builds the same structure:", "",
         f"- **Field** = one of {VENUE_K} clusters of SPECTER2 article embeddings (KMeans), roughly the granularity of one venue such as ML. Labels below come from each cluster's most frequent keywords.",
         "- **Edition** (one conference year) = (field, publication year of the first version).",
         "- **Query**: a review R of article Q. The reviewer has their own article P **in the same edition** as Q (same field and same year). P ≠ Q, the identity match is confident, and the review is cleaned exactly as in the main pilot.",
         "- **Candidates**:",
         "  - `venue_full` (realistic): every other article of that edition. The attacker ranks the whole edition.",
         "  - `venue_50`: P plus 49 random articles from the edition, comparable with the 50-candidate settings.",
         "  - `venue_topic_matched` (control): within the edition, negatives symmetrically matched on sim(Q,·) to sim(Q,P). Editions are small, so this often needs wide ε.",
         "- **Excluded from the candidates**: Q, and any article that might be by the reviewer (ORCID, full name, surname plus first initial).", "",
         *([f"**Variant: rebuttal archive.** P only needs to be in the same *field* as Q (any year) and public by review date + {ARCHIVE_LAG_DAYS} days, "
            "and the reviewer must have written an author response on P. Candidates (`venue_*` settings) are every article of that field published by the same date. "
            "ICLR analogue: the attacker searches all public OpenReview rebuttals of earlier ICLR editions. "
            "The pools are larger than one edition, so absolute hit rates are lower and percentile ranks are the comparable number.", ""]
           if RESPONSE_ARCHIVE else []),
         f"Corpus: {'all F1000-platform journals with signed open review (F1000Research, Wellcome Open Research, Open Research Europe, HRB, Gates, NIHR, AAS, …); Darwin Tree of Life genome notes excluded as fixed-template papers' if CORPUS == 'platform' else 'F1000Research only'}.", "",
         "## Verdict", "", verdict, "",
         "## Data", "",
         f"- Queries: **{len(qs)}** from **{len({x['pid'] for x in qs})}** people (at most 2 per person). Identity tiers: {dict(collections.Counter(x['tier'] for x in qs))}.",
         f"- Funnel: {fq['persons_after_P_has_abstract']:,} people pass the base filters; **{fq['persons_after_P_same_venue']}** have their own article in the same edition as a paper they reviewed. "
         f"Reviews dropped for lacking that: {fq['review_level_drops'].get('no_P_in_same_venue(field,year)_as_Q', 0) + fq['review_level_drops'].get('no_own_response_article_in_field_archive', 0):,}.",
         f"- Edition sizes (articles with abstract): median {np.median(list(venues.values())):.0f}, range {min(venues.values())}–{max(venues.values())}, over {len(venues)} editions. "
         f"Real ICLR editions are 10–100× larger, so absolute hit rates here are optimistic. Percentile rank is the most size-independent number.",
         f"- Reviewer's position in P's author list: sole/first {sum(v == 'sole/first' for v in pos.values())}, middle {sum(v == 'middle' for v in pos.values())}, last {sum(v == 'last' for v in pos.values())}.", "",
         "| field | articles | queries | top keywords |", "|---|---|---|---|"]
    for c in sorted(sizes):
        L.append(f"| {c} | {sizes[c]} | {qfield.get(c, 0)} | {', '.join(w for w, _ in kw[c].most_common(6))} |")
    L += ["", "## Results", "",
          "Metrics: R@k = share of queries whose positive ranks in the top k; MRR = mean reciprocal rank; "
          "mean_pct_rank = mean of (rank−1)/(pool−1), so 0 = always first and 0.5 = chance. Chance values account for each query's pool size.", "",
          "### 1. Realistic: rank the whole edition (`venue_full`)", "", block("venue_full"),
          "### 2. Fixed 50 candidates from the edition (`venue_50`)", "", block("venue_50"),
          "### 3. Control: topic-matched within the edition (`venue_topic_matched`)", "", "All queries:", "", block("venue_topic_matched"),
          f"Fully controlled subset (n = {len(ctrl)}), where `topic_reviewed` is at chance by construction:", "", block("venue_topic_matched", "main_controlled"),
          "## Does style add to topic? (paired bootstrap over queries)", "",
          "### Realistic full-edition ranking", "", comp("venue_full", ids),
          f"### Topic-matched control, fully controlled (n = {len(ctrl)})", "", comp("venue_topic_matched", ctrl),
          "## Does the signal follow who wrote P's abstract? (full-edition ranking, MRR)", "",
          table(bd, list(bd.columns)), "",
          f"## Author-response experiment: review → the reviewer's own author response (n = {rr_n})", "",
          "ICLR analogue: link a review to the **rebuttal** the reviewer wrote as an author of their own submission in the same edition. "
          "Candidates are the edition's articles that have a usable author response (≥50 words). The positive is represented by the reviewer's own response(s) on P. "
          "Every other candidate is represented by the response(s) of its most prolific responder. Quoted reviewer text is removed and responses are capped at 600 words. "
          f"Median pool size = {rr_pool:.0f}.", "",
          "### Candidate = author response", "", rblock("review->response"),
          "### Candidate = paper (title + abstract), same queries and pools", "", rblock("review->paper"),
          "### Candidate = paper + author response", "", rblock("review->paper+response"),
          "### Paired comparisons (same queries)", "", rcomp(), ""]
    L[L.index("## Results")] = ("## Corpus by journal\n\n| journal | articles | queries (Q) | queries (P) |\n|---|---|---|---|\n" +
        "".join(f"| {j} | {aj[j]} | {qj.get(j, 0)} | {pj.get(j, 0)} |\n" for j, _ in aj.most_common()) +
        f"\n{cross} queries have Q and P in different journals of the platform (same field and year).\n\n## Results")
    open(f"{OUT}/iclr_results.md", "w").write("\n".join(L) + "\n")
    print("written", f"{OUT}/iclr_results.md")


if __name__ == "__main__":
    main()
