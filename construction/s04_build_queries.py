"""Build (review R, positive article P) queries for matched reviewer-authors, with de-identified review text.

Outputs
  data/private/queries_full.jsonl   identity-bearing (pid, DOIs) - never shipped
  benchmark/queries.jsonl           qid, pseudonymous person id, cleaned review text (main / strict)
  data/funnel_queries.json          counts removed at each filtering step
"""
import collections, datetime, hashlib, json, os, random, re
from common import DATA, ROOT, read_jsonl, write_jsonl, pub_date, name_key, XDATA, BENCH, CONCURRENT_DAYS, EXP, VENUE_K, field_clusters, venue_of, is_template_article, RESPONSE_ARCHIVE, ARCHIVE_LAG_DAYS
from clean import clean_review, english_ok

USABLE = {"orcid", "orcid_works", "name_aff"}
TIER_RANK = {"orcid": 0, "orcid_works": 1, "name_aff": 2}
MIN_REVIEW_WORDS = 100
MIN_ABSTRACT_WORDS = 50
AROUND_DAYS = 90
TARGET = 500
# 1 = when choosing P (and which review to use), prefer articles on which the reviewer wrote an author response,
# so that the review -> author-response experiment has as many queries as possible
PREFER_RESPONSE = os.environ.get("F1000_PREFER_RESPONSE", "0") == "1"
SEED = 13


def d(s):
    return datetime.date.fromisoformat(s) if s else None


def main():
    os.makedirs(f"{BENCH}", exist_ok=True)
    arts = {a["article_key"]: a for a in read_jsonl(f"{DATA}/articles.jsonl")}
    reviews = {r["review_id"]: r for r in read_jsonl(f"{DATA}/reviews.jsonl")}
    persons = read_jsonl(f"{DATA}/private/person_matches.jsonl")
    clusters = field_clusters(VENUE_K) if VENUE_K else None
    # article -> ids (ORCID / full-name key) of people who wrote an author response on it
    responders = collections.defaultdict(set)
    for r in read_jsonl(f"{DATA}/responses.jsonl"):
        for c in r["contribs"]:
            responders[r["article_key"]] |= {x for x in (c["orcid"], name_key(c["given_names"], c["surname"])) if x}

    # template checklist questions: question blocks recurring across >= 20 reviews
    qfreq = collections.Counter()
    for r in reviews.values():
        qfreq.update({b["raw"] for b in r["blocks"] if b["raw"].endswith("?")})

    def has_abs(a):
        return a and a["title"] and len(a["abstract"].split()) >= MIN_ABSTRACT_WORDS and not is_template_article(a)

    funnel = collections.OrderedDict()
    funnel["persons_reviewer_and_author_any_tier"] = len(persons) + 0
    persons = [p for p in persons if any(x["tier"] in USABLE for x in p["authored"])]
    funnel["persons_with_confident_identity_tier"] = len(persons)

    review_drop = collections.Counter()
    stage_persons = collections.Counter()
    pool = []
    for p in persons:
        my_ids = set(p["orcids"]) | set(p["name_keys"])
        authored = [x for x in p["authored"] if x["tier"] in USABLE]
        self_titles = [arts[x["article_key"]]["title"] for x in authored]
        cands = []
        ok_stage = set()
        for rr in p["reviews"]:
            rv = reviews[rr["review_id"]]
            Q = arts.get(rv["article_key"])
            other_P = [x for x in authored if x["article_key"] != rv["article_key"]]
            if not other_P:
                review_drop["no_other_authored_article(P==Q)"] += 1
                continue
            ok_stage.add("has_P_distinct_from_Q")
            if rr["role"] != "Referee":
                review_drop["person_is_co_referee_not_lead"] += 1
                continue
            ok_stage.add("lead_referee")
            if not has_abs(Q):
                review_drop["reviewed_article_missing_title_or_abstract"] += 1
                continue
            ok_stage.add("Q_has_abstract")
            c = clean_review(rv, qfreq, self_titles)
            if not english_ok(c["main"]):
                review_drop["non_english_or_empty_review"] += 1
                continue
            ok_stage.add("english")
            nw = len(c["main"].split())
            if nw < MIN_REVIEW_WORDS:
                review_drop[f"review_under_{MIN_REVIEW_WORDS}_words_after_cleaning"] += 1
                continue
            ok_stage.add("review_long_enough")
            Ps = [x for x in other_P if has_abs(arts[x["article_key"]])]
            if not Ps:
                review_drop["no_P_with_title_and_abstract"] += 1
                continue
            ok_stage.add("P_has_abstract")
            rdate = d(rv["date"])
            if VENUE_K and RESPONSE_ARCHIVE:  # same field, any year, public by review date + lag, with own response
                cq = clusters.get(Q["article_key"])
                Ps = [x for x in Ps if cq is not None and clusters.get(x["article_key"]) == cq
                      and (d(pub_date(arts[x["article_key"]])) - rdate).days <= ARCHIVE_LAG_DAYS
                      and responders.get(x["article_key"], set()) & my_ids]
                if not Ps:
                    review_drop["no_own_response_article_in_field_archive"] += 1
                    continue
                ok_stage.add("P_same_venue")
            elif VENUE_K:  # ICLR-style: the reviewer's own article must be in the same "conference edition" as Q
                vq = venue_of(Q, clusters)
                Ps = [x for x in Ps if vq is not None and venue_of(arts[x["article_key"]], clusters) == vq]
                if not Ps:
                    review_drop["no_P_in_same_venue(field,year)_as_Q"] += 1
                    continue
                ok_stage.add("P_same_venue")
            if CONCURRENT_DAYS:
                Ps = [x for x in Ps if abs((d(pub_date(arts[x["article_key"]])) - rdate).days) <= CONCURRENT_DAYS]
                if not Ps:
                    review_drop[f"no_P_within_{CONCURRENT_DAYS}d_of_review"] += 1
                    continue
                ok_stage.add("P_concurrent")
            opts = []
            for x in Ps:
                pdate = d(pub_date(arts[x["article_key"]]))
                delta = (pdate - rdate).days if (pdate and rdate) else 10 ** 5
                before = delta <= AROUND_DAYS
                own_resp = bool(responders.get(x["article_key"], set()) & my_ids)
                key = (not before, TIER_RANK[x["tier"]], abs(delta))
                opts.append((((not own_resp,) + key) if PREFER_RESPONSE else key, x, delta, own_resp))
            opts.sort(key=lambda t: t[0])
            _, best, delta, own_resp = opts[0]
            cands.append({"pid": p["pid"], "review_id": rv["review_id"], "Q": rv["article_key"],
                          "P": best["article_key"], "tier": best["tier"], "author_pos": best["author_pos"],
                          "n_authors": best["n_authors"], "P_minus_R_days": delta,
                          "P_before_or_around_R": delta <= AROUND_DAYS,
                          "n_review_writers": rr["n_contribs"], "review_date": rv["date"],
                          "review_version": rv["title"], "review_words": nw,
                          "n_P_options": len(Ps), "P_has_own_response": own_resp, "review_main": c["main"], "review_strict": c["strict"],
                          "clean_stats": dict(c["stats"]), "expertise_removed": c["expertise"],
                          "exclude_articles": sorted(set(p["possible_authored"]) | {x["article_key"] for x in p["authored"]})})
        for s in ok_stage:
            stage_persons[s] += 1
        if not cands:
            continue
        # preference: sole writer, P before/around R, strongest tier, longer review
        cands.sort(key=lambda c: ((not c["P_has_own_response"],) if PREFER_RESPONSE else ()) +
                   (c["n_review_writers"] > 1, not c["P_before_or_around_R"], TIER_RANK[c["tier"]], -c["review_words"]))
        chosen, seen_q = [], set()
        for c in cands:
            if c["Q"] in seen_q:
                continue
            chosen.append(c)
            seen_q.add(c["Q"])
            if len(chosen) == 2:
                break
        pool.append(chosen)

    for s in ["has_P_distinct_from_Q", "lead_referee", "Q_has_abstract", "english", "review_long_enough", "P_has_abstract"] + (
            ["P_same_venue"] if VENUE_K else []) + (["P_concurrent"] if CONCURRENT_DAYS else []):
        funnel[f"persons_after_{s}"] = stage_persons[s]
    funnel["persons_eligible"] = len(pool)
    funnel["eligible_pairs_total(max 2/person)"] = sum(len(c) for c in pool)

    # main selection: 1 query per person, strongest identity tier first; second queries only if short of TARGET
    rng = random.Random(SEED)
    rng.shuffle(pool)
    pool.sort(key=lambda cs: (TIER_RANK[cs[0]["tier"]], cs[0]["n_review_writers"] > 1))
    first = [cs[0] for cs in pool]
    selected = first[:TARGET]
    if len(selected) < TARGET:
        seconds = [cs[1] for cs in pool if len(cs) > 1]
        selected += seconds[:TARGET - len(selected)]
    funnel["selected_queries"] = len(selected)
    funnel["selected_distinct_persons"] = len({c["pid"] for c in selected})
    funnel["selected_with_P_own_response"] = sum(c["P_has_own_response"] for c in selected)
    # extended set: every eligible person, first query only (robustness)
    extended_only = first[TARGET:]
    funnel["extended_extra_queries(1/person, beyond main)"] = len(extended_only)

    # pseudonyms: salted hash -> opaque ids (mapping kept private)
    salt = "f1000-linkage-stage1"
    pseudo = {}
    for c in selected + extended_only:
        pseudo.setdefault(c["pid"], "u" + hashlib.sha256((salt + c["pid"]).encode()).hexdigest()[:10])
    rows, pub = [], []
    for split, group in (("main", selected), ("extended", extended_only)):
        for i, c in enumerate(group):
            qid = f"{'q' if split == 'main' else 'x'}{i:04d}"
            c = dict(c, qid=qid, split=split, person=pseudo[c["pid"]])
            rows.append(c)
            pub.append({"qid": qid, "split": split, "person": c["person"], "review_text": c["review_main"],
                        "review_text_strict": c["review_strict"], "review_date": c["review_date"]})
    write_jsonl(f"{XDATA}/private/queries_full.jsonl", rows)
    write_jsonl(f"{BENCH}/queries.jsonl", pub)
    funnel["review_level_drops"] = dict(review_drop)
    json.dump(funnel, open(f"{XDATA}/funnel_queries.json", "w"), indent=2)
    print(json.dumps(funnel, indent=2))


if __name__ == "__main__":
    main()
