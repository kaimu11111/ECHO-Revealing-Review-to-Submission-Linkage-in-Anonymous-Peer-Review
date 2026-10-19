"""Leakage audit + dataset_stats.md, data_quality_report.md, pilot_results.md."""
import collections, json, os, re
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from common import DATA, OUT, ROOT, read_jsonl, aff_tokens, strip_accents, XDATA, BENCH, CONCURRENT_DAYS, EXP
from clean import mask_title_overlap

RNG = np.random.default_rng(1)
CHANCE = {"R@1": 1 / 50, "R@5": 5 / 50, "R@10": 10 / 50, "MRR": sum(1 / k for k in range(1, 51)) / 50, "median_rank": 25.5}
MAIN_METHODS = ["random", "topic_review", "topic_reviewed", "stylo_cos", "stylo_euclid", "stylo_charngram",
                "lr_style", "lr_topic", "lr_topic_style"]
DIAG = ["diag_date_proximity", "diag_same_type"]
ALL_SETTINGS = ["random", "same_field", "topic_matched", "topic_time_matched"]
SETTINGS, TOPIC = [], []  # filled from the pools that exist for this experiment


def q(x, ps=(5, 25, 50, 75, 95)):
    x = np.asarray(x)
    return " / ".join(f"{np.percentile(x, p):.3f}" for p in ps)


def table(df, cols, fmt="{:.3f}"):
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for _, r in df.iterrows():
        body += "| " + " | ".join(fmt.format(r[c]) if isinstance(r[c], (float, np.floating)) else str(r[c]) for c in cols) + " |\n"
    return head + body


def audit(qs, arts, reviews):
    """Residual identity leakage in the shipped review texts (main and strict)."""
    rows = collections.Counter()
    examples = collections.defaultdict(list)
    for x in qs:
        rv = reviews[x["review_id"]]
        c = [c for c in rv["contribs"] if c["role"] == "Referee"][0]
        P = arts[x["P"]]
        for ver in ("main", "strict"):
            t = x[f"review_{ver}"]
            tl = strip_accents(t).lower()
            checks = {
                "surname_any_case": len(c["surname"]) >= 3 and re.search(rf"(?<!\w){re.escape(strip_accents(c['surname']).lower())}(?!\w)", tl),
                "surname_capitalised": len(c["surname"]) >= 3 and re.search(rf"(?<!\w){re.escape(c['surname'])}(?!\w)", t),
                "email": re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", t),
                "orcid_id": re.search(r"\d{4}-\d{4}-\d{4}-\d{3}[\dX]", t),
                "profile_url": re.search(r"orcid\.org|researchgate|linkedin|scholar\.google", t, re.I),
                "reviewer_expertise_label": re.search(r"reviewer expertise", t, re.I),
                "confirmation_boilerplate": re.search(r"confirm that (i|we) have read this submission", t, re.I),
                "checklist_question": re.search(r"Is the work clearly and accurately presented", t),
                "positive_P_doi": x["P"].split("/")[-1] in t,
                "positive_P_title_6gram": mask_title_overlap(t, [P["title"]])[1] > 0,
                "reviewer_aff_distinctive_token": bool({w for w in aff_tokens(c["aff"]) if len(w) >= 6} &
                                                       set(re.findall(r"[a-z]{6,}", tl)) - COMMON_WORDS),
            }
            for k, v in checks.items():
                if v:
                    rows[(ver, k)] += 1
                    if len(examples[(ver, k)]) < 3:
                        examples[(ver, k)].append(x["qid"])
    return rows, examples


COMMON_WORDS = set()


def tag_query_sets(pq):
    """Older per_query_ranks.csv files lack `query_set`; the response experiment's rows come after the main rows,
    so a repeated (experiment, setting, text, method, qid) key marks a response-subset row."""
    if "query_set" not in pq:
        pq = pq.copy()
        dup = pq.duplicated(subset=["experiment", "setting", "review_text", "method", "qid"], keep="first")
        pq["query_set"] = np.where(dup | (pq.experiment != "review->paper"), "response_subset", "all")
    return pq


def paired(pq, setting, a, b, ids, metric="MRR", exp_a="review->paper", exp_b="review->paper", query_set="all"):
    """Paired bootstrap of method a (on experiment exp_a) minus method b (on exp_b) over the same queries."""
    pq = tag_query_sets(pq)
    d = pq[(pq.setting == setting) & (pq.review_text == "main") & (pq.query_set == query_set)]
    if "experiment" in d:
        da, db = d[d.experiment == exp_a], d[d.experiment == exp_b]
    else:
        da = db = d
    ra = da[da.method == a].set_index("qid").loc[ids, "rank"].values
    rb = db[db.method == b].set_index("qid").loc[ids, "rank"].values
    f = (lambda r: 1 / r) if metric == "MRR" else (lambda r: (r <= 10).astype(float))
    diff = f(ra) - f(rb)
    bs = RNG.integers(0, len(diff), (2000, len(diff)))
    m = diff[bs].mean(1)
    return diff.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5), (m <= 0).mean()


def main():
    arts = {a["article_key"]: a for a in read_jsonl(f"{DATA}/articles.jsonl")}
    reviews = {r["review_id"]: r for r in read_jsonl(f"{DATA}/reviews.jsonl")}
    qs_all = read_jsonl(f"{XDATA}/private/queries_full.jsonl")
    qs = [x for x in qs_all if x["split"] == "main"]
    main_ids = [x["qid"] for x in qs]
    ms = json.load(open(f"{DATA}/match_stats.json"))
    fq = json.load(open(f"{XDATA}/funnel_queries.json"))
    P = json.load(open(f"{XDATA}/private/pools.json"))
    res = pd.read_csv(f"{OUT}/pilot_results.csv")
    pq = pd.read_csv(f"{OUT}/per_query_ranks.csv")
    resp_main = [p for p in P["resp_pools"]["topic_matched"]]
    SETTINGS[:] = [s for s in ALL_SETTINGS if s in P["pools"]]
    TOPIC[:] = [s for s in SETTINGS if s.startswith("topic")]
    strictest = TOPIC[-1]
    design = (f"**Design: ICLR-style concurrent setting.** The positive article and all 49 negatives are published within "
              f"±{CONCURRENT_DAYS} days of the review, like submissions to one conference cycle, so date cannot separate them. "
              f"At most 2 queries per person." if CONCURRENT_DAYS else
              "**Design: non-concurrent.** Negatives may come from any earlier period (see topic_time_matched for the time control).")

    # words that are frequent in reviews generally are not identifying even if they appear in an affiliation
    wc = collections.Counter(w for r in list(reviews.values())[:5000] for b in r["blocks"] for w in set(re.findall(r"[a-z]{6,}", b["raw"].lower())))
    COMMON_WORDS.update(w for w, n in wc.items() if n >= 25)

    # ---------------- dataset_stats.md ----------------
    sims = {}
    for s in SETTINGS:
        rows = [r for r in P["pools"][s] if r["qid"] in set(main_ids)]
        sims[s] = (np.array([r["sim_QP"] for r in rows]), np.concatenate([r["sim_Qneg"] for r in rows]),
                   np.array([np.mean(r["sim_Qneg"]) - r["sim_QP"] for r in rows]))
    pe, pk = np.load(f"{DATA}/cache/papers.npy"), json.load(open(f"{DATA}/cache/papers_ids.json"))
    emb = dict(zip(pk, pe))
    rre, rrk = np.load(f"{XDATA}/cache/reviews_main.npy"), json.load(open(f"{XDATA}/cache/reviews_main_ids.json"))
    remb = dict(zip(rrk, rre))
    sim_RP = np.array([float(remb[x["qid"]] @ emb[x["P"]]) for x in qs])
    sim_RQ = np.array([float(remb[x["qid"]] @ emb[x["Q"]]) for x in qs])
    tm = [r for r in P["pools"]["topic_matched"] if r["qid"] in set(main_ids)]
    sim_Rneg_tm = np.concatenate([[float(remb[r["qid"]] @ emb[c]) for c in r["candidates"] if c != r["positive"]] for r in tm])
    rank_P_in_corpus = []
    elig = [k for k in pk if arts[k]["title"] and len(arts[k]["abstract"].split()) >= 50]
    E = np.stack([emb[k] for k in elig])
    for x in qs:
        s = E @ emb[x["Q"]]
        rank_P_in_corpus.append(int((s > float(emb[x["P"]] @ emb[x["Q"]])).sum()))  # includes Q itself at top
    rank_P_in_corpus = np.array(rank_P_in_corpus)

    n_resp_main = sum(1 for p in resp_main if p["qid"] in set(main_ids))
    lines = ["# Dataset statistics — F1000 review→article linkage, Stage-1 pilot", "",
             design, "", "All numbers come from `data/match_stats.json`, `data/funnel_queries.json` and `data/private/pools.json`.", "",
             "## Source corpus", "",
             "| quantity | value |", "|---|---|",
             f"| F1000Research articles in PMC (one record per article, latest version) | {ms['articles_parsed']:,} |",
             f"| articles with ≥1 public review | {ms['articles_with_reviews']:,} |",
             f"| review reports (all versions) | {ms['reviews_parsed']:,} |",
             f"| **distinct F1000 reviewers** (ORCID, else full name) | **{ms['distinct_reviewers']:,}** ({ms['distinct_reviewers_with_orcid']:,} with ORCID) |",
             f"| **distinct F1000 authors** (ORCID, else full name) | **{ms['distinct_authors']:,}** |",
             f"| author-response sub-articles | {sum(1 for _ in open(f'{DATA}/responses.jsonl')):,} |", "",
             "### Seed dataset (raear/indexer-assignment-dataset, F1000 part)", "",
             "| quantity | value |", "|---|---|"]
    for k, v in ms["seed_dataset"].items():
        lines.append(f"| {k} | {v:,} |")
    lines += ["", "The seed dataset alone (2014–2021, ORCID reviewers only) gives roughly 250–300 reviewer∩author people, "
              "which is too few for about 500 queries. So the full PMC F1000Research corpus (7,463 articles through 2026-10) "
              "was parsed with the same JATS logic the seed dataset uses (`sub-article[@article-type=peer-review|reviewer-report]`). "
              "The seed's ORCID works lists are reused as an extra identity tier.", "",
              "## Reviewer ∩ author intersection", "",
              "| best identity tier | people |", "|---|---|"]
    for k in ("orcid", "orcid_works", "name_aff", "name_only"):
        lines.append(f"| {k} | {ms['reviewer_author_intersection_by_best_tier'].get(k, 0):,} |")
    lines += [f"| **usable (orcid + orcid_works + name_aff)** | **{ms['reviewer_author_intersection_usable(orcid,orcid_works,name_aff)']:,}** |",
              "", "`name_only` matches (same full name but no ORCID or affiliation corroboration) are counted but never used.", "",
              "## Filtering funnel (people)", "", "| step | people remaining | removed |", "|---|---|---|"]
    prev = None
    for k, v in fq.items():
        if k.startswith("persons_"):
            lines.append(f"| {k} | {v:,} | {'' if prev is None else f'{prev - v:,}'} |")
            prev = v
    lines += ["", "Review-level drops (one person can lose several reviews):", "", "| reason | reviews dropped |", "|---|---|"]
    for k, v in fq["review_level_drops"].items():
        lines.append(f"| {k} | {v:,} |")
    lines += ["", "## Queries", "", "| quantity | value |", "|---|---|",
              f"| usable (review, article) pairs, at most 2 per person | {fq['eligible_pairs_total(max 2/person)']:,} |",
              f"| eligible people | {fq['persons_eligible']:,} |",
              f"| **final main queries** (1 per person first, ORCID-tier people first; a 2nd per person only if short of 500) | **{fq['selected_queries']}** ({fq['selected_distinct_persons']} people) |",
              f"| extended queries (other eligible people, 1 each; robustness only) | {fq['extended_extra_queries(1/person, beyond main)']} |",
              f"| main queries by identity tier | {dict(collections.Counter(x['tier'] for x in qs))} |",
              f"| main queries where P was published before or within 90 days after the review | {sum(x['P_before_or_around_R'] for x in qs)} |",
              f"| main queries where the review has a single writer | {sum(x['n_review_writers'] == 1 for x in qs)} |",
              f"| reviewer's position in P: first / middle / last / sole | "
              f"{sum(x['author_pos'] == 0 and x['n_authors'] > 1 for x in qs)} / "
              f"{sum(0 < x['author_pos'] < x['n_authors'] - 1 for x in qs)} / "
              f"{sum(x['author_pos'] == x['n_authors'] - 1 and x['n_authors'] > 1 for x in qs)} / "
              f"{sum(x['n_authors'] == 1 for x in qs)} |",
              f"| P authors: median (IQR) | {np.median([x['n_authors'] for x in qs]):.0f} ({np.percentile([x['n_authors'] for x in qs], 25):.0f}–{np.percentile([x['n_authors'] for x in qs], 75):.0f}) |",
              f"| review length after cleaning, words: p5/p25/median/p75/p95 | {q([x['review_words'] for x in qs]).replace('.000', '')} |",
              f"| reviews of version 1 / later versions | {sum('version 1' in x['review_version'] for x in qs)} / {sum('version 1' not in x['review_version'] for x in qs)} |",
              f"| **queries whose positive article has an author response written by the reviewer** | **{n_resp_main} main** ({len(resp_main)} incl. extended) |",
              f"| articles with a usable (≥50-word, English) author response, the negative pool for the response experiment | {P['n_articles_with_response']:,} |",
              "", "## Topic-similarity distributions (SPECTER2 cosine, main queries)", "",
              "Percentiles p5 / p25 / median / p75 / p95.", "",
              "| quantity | distribution |", "|---|---|",
              f"| sim(Q reviewed paper, P+ positive) | {q(sims['random'][0])} |",
              f"| sim(review text, P+) | {q(sim_RP)} |",
              f"| sim(review text, Q) | {q(sim_RQ)} |",
              f"| sim(review text, topic-matched negatives) | {q(sim_Rneg_tm)} |"]
    for s in SETTINGS:
        lines.append(f"| sim(Q, negatives) — {s} | {q(sims[s][1])} |")
    for s in SETTINGS:
        lines.append(f"| mean sim(Q,neg) − sim(Q,P+) per query — {s} | {q(sims[s][2])} |")
    lines += [f"| number of corpus articles more similar to Q than P+ is | {q(rank_P_in_corpus, (5, 25, 50, 75, 95)).replace('.000', '')} |",
              "", f"P+ is among the 25 articles in the corpus closest to Q for **{(rank_P_in_corpus <= 25).mean():.0%}** of queries "
              "(Q itself counts as the first). That is the expertise-matching signal the topic-matched settings are built to remove.", "",
              "### ε used for topic matching (main queries)", "", "| setting | ε=0.03 | 0.05 | 0.075 | 0.10 | 0.15 | fully controlled (≥24 negatives above sim(Q,P+)) |",
              "|---|---|---|---|---|---|---|"]
    for s in TOPIC:
        rows = [r for r in P["pools"][s] if r["qid"] in set(main_ids)]
        c = collections.Counter(r["eps"] for r in rows)
        lines.append(f"| {s} | " + " | ".join(str(c.get(e, 0)) for e in (0.03, 0.05, 0.075, 0.10, 0.15)) +
                     f" | {sum((r['n_above'] or 0) >= 24 for r in rows)} / {len(rows)} |")
    open(f"{OUT}/dataset_stats.md", "w").write("\n".join(lines) + "\n")

    # ---------------- data_quality_report.md ----------------
    au, ex = audit(qs, arts, reviews)
    cs = collections.Counter()
    for x in qs:
        cs.update(x["clean_stats"])
    L = ["# Data quality report", "", design, "",
         "## Identity resolution", "",
         "- **orcid**: the review contributor's ORCID equals the article author's ORCID. All 500 main queries use this tier.",
         "- **orcid_works**: the reviewer's public ORCID works list (seed dataset) contains the article, and an author on it has the same surname and first initial.",
         "- **name_aff**: identical full first given name and surname, no ORCID conflict, the name is not carried by two or more different ORCIDs anywhere in the corpus, "
         "and the two affiliations share at least 2 tokens, at least 1 of them distinctive (in under 1% of affiliations). "
         "A manual check of 8 random matches before tightening found 7 clearly the same person; the eighth (shared only 'health'/'usa') led to the distinctive-token rule.",
         "- **name_only**: counted, never used.",
         "- **Negative exclusion is conservative**: a candidate is dropped if any author shares the reviewer's ORCID, full name, "
         "or **surname + first initial**. This removes some true negatives but avoids unlabeled positives.", "",
         "## De-identification of review text", "",
         "Removed or masked in both versions (counts over the 500 main reviews):", "",
         "| operation | count |", "|---|---|"]
    for k in ("checklist_q", "checklist_a", "confirm_stmt", "expertise", "signature_block", "name", "affiliation", "email", "orcid", "profile_url"):
        L.append(f"| {k} | {cs.get(k, 0)} |")
    L += ["", "Masked in addition in the **strict** version:", "", "| operation | count |", "|---|---|"]
    for k in ("strict_url", "strict_doi", "strict_pmid", "strict_self_citation", "strict_self_title", "strict_self_reference_sentence"):
        L.append(f"| {k} | {cs.get(k, 0)} |")
    L += ["", "Notes:",
          "- Reviewer names, ORCID and affiliation sit in the JATS `front-stub` and are never copied into the text. Masking only catches in-body mentions such as signatures or self-citations.",
          "- F1000's **checklist questions** (template text such as 'Is the study design appropriate…?') and their Yes/Partly/No answers are removed. They are template text, not the reviewer's prose. "
          "A question block counts as template if it recurs in 20 or more reviews.",
          "- 'Reviewer Expertise:' and its content are removed (kept privately as `expertise_removed` for later analysis).",
          "- Name masking is case-sensitive (capitalised and ALL-CAPS forms) so that names which are also common words ('Long', 'Mark') are not masked in running text.",
          "- Strict self-reference masking replaces whole sentences matching patterns such as 'our previous study', 'we have previously shown', 'in our hospital', "
          "plus 6-word overlaps with the titles of any of the reviewer's own F1000 articles.", "",
          "## Residual leakage audit (500 main reviews)", "",
          "Raw-text checks for identity strings that survived cleaning:", "",
          "| check | main | strict | example qids (main) |", "|---|---|---|---|"]
    keys = ["surname_capitalised", "surname_any_case", "email", "orcid_id", "profile_url", "reviewer_expertise_label",
            "confirmation_boilerplate", "checklist_question", "positive_P_doi", "positive_P_title_6gram", "reviewer_aff_distinctive_token"]
    for k in keys:
        L.append(f"| {k} | {au.get(('main', k), 0)} | {au.get(('strict', k), 0)} | {', '.join(ex.get(('main', k), []))} |")
    L += ["", "How to read the audit:",
          "- `surname_any_case` without `surname_capitalised` means the surname appears as a lowercase common word, e.g. a reviewer named *Field* in 'the field'. That is not leakage.",
          "- `reviewer_aff_distinctive_token` is a loose check: it fires when an uncommon word from the reviewer's affiliation (6+ letters, not common in reviews) appears anywhere in the review. "
          "Most hits are topical words such as a disease or region name that also appears in a department name; review by hand before Stage 2.",
          "- `positive_P_title_6gram` in the main version means the review quotes or cites the reviewer's own positive article. Strict masks these.", "",
          "## Author responses (optional experiment)", "",
          f"- Author-response sub-articles parsed: {sum(1 for _ in open(f'{DATA}/responses.jsonl')):,}; articles with a usable response (≥50 words, English): {P['n_articles_with_response']:,}.",
          f"- Cleaning: {P['response_clean_stats'].get('italic_removed', 0):,} blocks had italic (quoted reviewer) spans removed; "
          f"{P['response_clean_stats'].get('quoted_block_dropped', 0):,} blocks were dropped because ≥30% of their 5-grams appear in that article's reviews or they are fully quoted; "
          f"{P['response_clean_stats'].get('name', 0):,} name mentions masked.",
          f"- Queries whose positive article has a response **written by the reviewer**: {n_resp_main} main, {len(resp_main)} including extended. "
          "The response experiment runs on all of these to get usable n.",
          "- The positive's representation is the reviewer's own response(s). A negative's representation is the response(s) of a single responder on that article (whoever wrote the most), truncated to 600 words in both cases.", "",
          "## Known confounds and limitations", "",
          "1. **Publication date.** Positives are usually published near the review date (the query builder prefers P before or within 90 days after R). "
          "Negatives are only required to predate max(R, P) + 90 days. The `diag_date_proximity` diagnostic shows this alone is strongly above chance (see pilot_results.md). "
          "The **topic_time_matched** setting (negatives within ±365 days of P) was added to control for it.",
          "2. **Uncontrollable top-similarity positives.** For about a third of queries, fewer than 24 corpus articles are more similar to Q than P+ is, so symmetric topic matching is impossible. "
          "On those queries `topic_reviewed` still ranks P+ high. Results are reported on all queries and on the *fully controlled* subset.",
          "3. **Same-field setting has a built-in artifact.** Negatives are the 49 nearest neighbours of Q, so P+ is usually the *least* similar candidate (median rank 50 under `topic_reviewed`). "
          "Supervised models (`lr_topic`, `lr_topic_style`) learn to invert this, so do not read their same-field numbers as linkage.",
          "4. **Abstracts are multi-author.** The reviewer is first or sole author of P in only a minority of queries (see dataset_stats.md), so abstract style is often someone else's. "
          "This caps any stylometric signal and is broken down in pilot_results.md.",
          "5. **Genre mismatch.** Reviews and abstracts differ in genre; stylometric features are z-scored within each genre before comparison.",
          "6. **Reviewed-paper metadata.** `benchmark/reviewed_papers.jsonl` maps queries to Q (opaque id → title/abstract via `articles.jsonl`). It is only for the topic baseline. "
          "Do not give it to a Stage-2 LLM: Q's title plus the review text is enough to look the review up on f1000research.com and see the reviewer's name.",
          "7. **Review text is public and signed.** Any model with web access or memorised F1000 pages can recover identity. Stage-2 evaluations must use offline models and treat memorisation as a threat.",
          "8. **Time span.** Reviews run from 2012 to 2026; the main set takes 1 query per person.",
          ]
    open(f"{OUT}/data_quality_report.md", "w").write("\n".join(L) + "\n")

    # ---------------- pilot_results.md ----------------
    def block(query_set, setting, version="main", methods=MAIN_METHODS + DIAG):
        d = res[(res.query_set == query_set) & (res.setting == setting) & (res.review_text == version) & (res.experiment == "review->paper")]
        d = d.set_index("method").loc[[m for m in methods if m in set(d.method)]].reset_index()
        d["R@10 95% CI"] = d.apply(lambda r: f"{r['R@10_ci_lo']:.3f}–{r['R@10_ci_hi']:.3f}", axis=1)
        d["MRR 95% CI"] = d.apply(lambda r: f"{r['MRR_ci_lo']:.3f}–{r['MRR_ci_hi']:.3f}", axis=1)
        d["p(R@10>chance)"] = d.apply(lambda r: f"{binomtest(int(round(r['R@10'] * r['n'])), int(r['n']), 0.2, alternative='greater').pvalue:.1e}", axis=1)
        return table(d, ["method", "n", "R@1", "R@5", "R@10", "R@10 95% CI", "MRR", "MRR 95% CI", "median_rank", "p(R@10>chance)"])

    ctrl_tm = sorted({p["qid"] for p in P["pools"]["topic_matched"] if (p["n_above"] or 0) >= 24} & set(main_ids))
    ctrl_ttm = sorted({p["qid"] for p in P["pools"][strictest] if (p["n_above"] or 0) >= 24} & set(main_ids))
    comps = [("stylo_cos", "topic_review"), ("stylo_charngram", "topic_review"), ("lr_topic_style", "lr_topic"),
             ("stylo_cos", "random"), ("stylo_charngram", "random"), ("lr_style", "random")]

    def comp_table(setting, ids):
        rows = []
        for a, b in comps:
            for metric in ("MRR", "R@10"):
                m, lo, hi, p = paired(pq, setting, a, b, ids, metric)
                rows.append({"comparison": f"{a} − {b}", "metric": metric, "diff": m, "95% CI": f"{lo:+.3f} to {hi:+.3f}", "p(diff≤0)": f"{p:.3f}"})
        return table(pd.DataFrame(rows), ["comparison", "metric", "diff", "95% CI", "p(diff≤0)"], fmt="{:+.3f}")

    # author-position breakdown of the style signal (topic_time_matched, main)
    pos = {x["qid"]: ("sole/first" if x["author_pos"] == 0 else ("last" if x["author_pos"] == x["n_authors"] - 1 else "middle")) for x in qs}
    small = {x["qid"]: ("≤3 authors" if x["n_authors"] <= 3 else ">3 authors") for x in qs}
    bd = []
    for setting in TOPIC:
        d = pq[(pq.setting == setting) & (pq.review_text == "main") & pq.qid.isin(main_ids)].copy()
        for name, g in (("author position", pos), ("team size", small)):
            d["grp"] = d.qid.map(g)
            for (grp, m), x in d[d.method.isin(["random", "topic_review", "stylo_cos", "stylo_charngram", "lr_style"])].groupby(["grp", "method"]):
                bd.append({"setting": setting, "split": name, "group": grp, "method": m, "n": len(x),
                           "MRR": (1 / x["rank"]).mean(), "R@10": (x["rank"] <= 10).mean()})
    bd = pd.DataFrame(bd)
    bdp = bd.pivot_table(index=["setting", "split", "group", "n"], columns="method", values="MRR").reset_index()

    strict_rows = res[(res.query_set == "main") & res.setting.isin(TOPIC) & (res.experiment == "review->paper")
                      & res.method.isin(["topic_review", "stylo_cos", "stylo_charngram", "lr_style", "lr_topic_style"])]
    sp = strict_rows.pivot_table(index=["setting", "method"], columns="review_text", values=["MRR", "R@10"]).reset_index()
    sp.columns = [" ".join(c).strip() for c in sp.columns]

    rr = res[(res.query_set == "response_subset") & (res.review_text == "main")]
    rr_t = rr.pivot_table(index=["setting", "method"], columns="experiment", values="MRR").reset_index()
    rr_n = int(rr.n.max()) if len(rr) else 0

    R = ["# Stage-1 pilot results: linking an anonymous F1000 review to the reviewer's own F1000 article", "",
         f"Each query gives one de-identified review R and 50 candidate F1000 articles (title + abstract): 1 written by an author who wrote R, 49 negatives. "
         f"Main query set: **{len(qs)} queries from {len({x['pid'] for x in qs})} people; identity tiers {dict(collections.Counter(x['tier'] for x in qs))}.** "
         f"Chance level: R@1 = {CHANCE['R@1']:.3f}, R@5 = {CHANCE['R@5']:.3f}, R@10 = {CHANCE['R@10']:.3f}, MRR = {CHANCE['MRR']:.3f}, median rank ≈ 25.5. "
         "", design, "", "No generative LLMs are used. 95% CIs are percentile bootstraps over queries (1,000 resamples). p(R@10>chance) is a one-sided binomial test against 0.2.", "",
         "Full numbers, including strict text, the extended set (if any) and controlled subsets, are in `pilot_results.csv`. Per-query ranks are in `per_query_ranks.csv`.", "",
         "Methods: `topic_review` = cos(review, candidate) [topic baseline 1]; `topic_reviewed` = cos(reviewed paper, candidate) [topic baseline 2, pure expertise matching]; "
         "`stylo_cos` / `stylo_euclid` = 122 hand-crafted style features z-scored within genre; `stylo_charngram` = char 2–4-gram TF-IDF cosine; "
         "`lr_*` = logistic regression over pairs, 5-fold CV grouped by person. `diag_*` are metadata confound probes, not linkage methods.", "",
         "## Verdict", "", open(f"{OUT}/verdict.md").read() if os.path.exists(f"{OUT}/verdict.md") else "_No verdict written yet (outputs/<exp>/verdict.md)._", ""]
    for setting, title in (("topic_matched", "Topic-matched (main setting)"),
                           ("topic_time_matched", "Topic + time matched (stricter control, added in this pilot)"),
                           ("same_field", "Same-field"), ("random", "Random candidates")):
        if setting not in SETTINGS:
            continue
        R += [f"## {title}", "", "### All main queries", "", block("main", setting)]
        if setting.startswith("topic"):
            R += ["### Fully controlled subset (≥24 negatives above sim(Q,P+), so `topic_reviewed` is at chance by construction)", "",
                  block("main_controlled", setting)]
        if setting == "same_field":
            R += ["> Same-field artifact: negatives are Q's 49 nearest neighbours, so P+ is usually the *least* similar candidate. "
                  "`lr_topic` and `lr_topic_style` learn to invert similarity, so their numbers here are not linkage evidence.", ""]
    R += ["## Key comparison: topic vs stylometry", "",
          "Paired bootstrap over queries (2,000 resamples); diff = method A − method B.", "",
          "### Topic-matched, all main queries", "", comp_table("topic_matched", main_ids),
          f"### {strictest}, fully controlled subset (n = {len(ctrl_ttm)})", "", comp_table(strictest, ctrl_ttm),
          "## Main vs strict review text", "",
          "Strict text additionally masks URLs, DOIs, self-citations, the reviewer's own titles and self-referential sentences.", "",
          table(sp, list(sp.columns)), "",
          "## Does the style signal follow who wrote the abstract?", "",
          "MRR by the reviewer's position in P's author list and by P's team size (main queries). "
          "If style really links writers, it should be strongest when the reviewer is first or sole author, or on small teams.", "",
          table(bdp, list(bdp.columns)), "",
          f"## Optional: author-response experiment (n = {rr_n} queries whose positive article has a response written by the reviewer; includes extended queries)", "",
          "MRR for each representation of the candidate. Negatives are drawn from articles that have a usable author response.", "",
          table(rr_t, list(rr_t.columns)), "",
          "Full metrics are in `pilot_results.csv` (experiment = review->response, review->paper+response).", ""]
    open(f"{OUT}/pilot_results.md", "w").write("\n".join(R) + "\n")
    print("reports written")


if __name__ == "__main__":
    main()
