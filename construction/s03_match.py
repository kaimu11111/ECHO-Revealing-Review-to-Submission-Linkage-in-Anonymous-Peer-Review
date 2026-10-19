"""Resolve F1000 reviewers against F1000 article authors.

Match tiers (strongest first):
  orcid        review contributor ORCID == article author ORCID
  orcid_works  reviewer ORCID's public works list (seed dataset F1000res_reviewers.json) contains the article,
               and the article has an author with the same surname + first initial
  name_aff     identical full first-given-name + surname, name not carried by >1 ORCID in the corpus, no ORCID
               conflict, >= 2 shared affiliation tokens of which >= 1 is distinctive (in < 1% of affiliations)
  name_only    identical full name, no ORCID conflict, no affiliation evidence (NOT used for queries)
Writes data/private/person_matches.jsonl (identity-bearing; never shipped) and data/match_stats.json.
"""
import collections, json, os, re
from common import DATA, read_jsonl, write_jsonl, name_key, aff_tokens, strip_accents

TIER_RANK = {"orcid": 0, "orcid_works": 1, "name_aff": 2, "name_only": 3}


def sur_init(given, surname):
    s = re.sub(r"[^a-z]", "", strip_accents(surname).lower())
    g = re.sub(r"[^a-z]", "", strip_accents(given).lower())
    return f"{g[:1]}|{s}" if s else None


def main():
    os.makedirs(f"{DATA}/private", exist_ok=True)
    arts = {a["article_key"]: a for a in read_jsonl(f"{DATA}/articles.jsonl")}
    reviews = read_jsonl(f"{DATA}/reviews.jsonl")
    seed = json.load(open(f"{DATA}/raw/F1000res_reviewers.json"))
    seed_assign = json.load(open(f"{DATA}/raw/F1000res_assignments.json"))
    seed_authors = json.load(open(f"{DATA}/raw/F1000res_authors.json"))

    pmid2key = {a["pmid"]: k for k, a in arts.items() if a.get("pmid")}
    # ORCID -> F1000 article keys listed in that person's ORCID works (seed dataset)
    works = collections.defaultdict(set)
    for r in seed:
        for p in r["publications"]:
            doi = (p.get("doi") or "").lower()
            m = re.match(r"(10\.12688/f1000research\.\d+)", doi)
            if m and m.group(1) in arts:
                works[r["orcid"]].add(m.group(1))
            elif p.get("pmid") in pmid2key:
                works[r["orcid"]].add(pmid2key[p["pmid"]])

    # author indexes
    by_orcid, by_name, by_si = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(set)
    for k, a in arts.items():
        n = len(a["authors"])
        for i, au in enumerate(a["authors"]):
            rec = (k, i, n)
            if au["orcid"]:
                by_orcid[au["orcid"]].append(rec)
            nk = name_key(au["given_names"], au["surname"])
            if nk:
                by_name[nk].append(rec)
            si = sur_init(au["given_names"], au["surname"])
            if si:
                by_si[si].add(k)

    # affiliation token document frequency -> distinctive tokens; name keys carried by >1 ORCID are ambiguous
    aff_df, n_aff, nk2o = collections.Counter(), 0, collections.defaultdict(set)
    for a in arts.values():
        for au in a["authors"]:
            for af in au["aff"]:
                n_aff += 1
                aff_df.update(aff_tokens([af]))
            nk = name_key(au["given_names"], au["surname"])
            if nk and au["orcid"]:
                nk2o[nk].add(au["orcid"])
    for rv in reviews:
        for c in rv["contribs"]:
            nk = name_key(c["given_names"], c["surname"])
            if nk and c["orcid"]:
                nk2o[nk].add(c["orcid"])
    common_tok = {t for t, n in aff_df.items() if n > 0.01 * n_aff}
    ambiguous = {k for k, v in nk2o.items() if len(v) > 1}

    # person id for review contributors: ORCID if present, else full-name key
    persons = {}
    reviewer_ids = set()
    for rv in reviews:
        for ci, c in enumerate(rv["contribs"]):
            nk = name_key(c["given_names"], c["surname"])
            pid = c["orcid"] or (f"name:{nk}" if nk else f"raw:{c['surname']}|{c['given_names']}")
            reviewer_ids.add(pid)
            p = persons.setdefault(pid, {"pid": pid, "orcids": set(), "name_keys": set(), "si": set(),
                                         "affs": set(), "reviews": [], "authored": {}})
            if c["orcid"]:
                p["orcids"].add(c["orcid"])
            if nk:
                p["name_keys"].add(nk)
            si = sur_init(c["given_names"], c["surname"])
            if si:
                p["si"].add(si)
            p["affs"] |= aff_tokens(c["aff"])
            p["reviews"].append({"review_id": rv["review_id"], "contrib_idx": ci, "role": c["role"],
                                 "n_contribs": len(rv["contribs"])})

    def add(p, key, i, n, tier):
        old = p["authored"].get(key)
        if old is None or TIER_RANK[tier] < TIER_RANK[old["tier"]]:
            p["authored"][key] = {"article_key": key, "author_pos": i, "n_authors": n, "tier": tier}

    for p in persons.values():
        for o in p["orcids"]:
            for k, i, n in by_orcid.get(o, []):
                add(p, k, i, n, "orcid")
            for k in works.get(o, ()):
                for i, au in enumerate(arts[k]["authors"]):
                    if sur_init(au["given_names"], au["surname"]) in p["si"]:
                        add(p, k, i, len(arts[k]["authors"]), "orcid_works")
                        break
        for nk in p["name_keys"]:
            if nk in ambiguous:
                continue
            for k, i, n in by_name.get(nk, []):
                au = arts[k]["authors"][i]
                if au["orcid"] and p["orcids"] and au["orcid"] not in p["orcids"]:
                    continue  # ORCID conflict: different person with same name
                shared = p["affs"] & aff_tokens(au["aff"])
                strong = len(shared) >= 2 and len(shared - common_tok) >= 1
                add(p, k, i, n, "name_aff" if strong else "name_only")
        # conservative exclusion set for negatives: any article with an author sharing surname + first initial
        p["possible_authored"] = sorted(set().union(*[by_si.get(s, set()) for s in p["si"]]) if p["si"] else set())

    # global counts
    author_ids = set()
    for a in arts.values():
        for au in a["authors"]:
            author_ids.add(au["orcid"] or name_key(au["given_names"], au["surname"]) or f"raw:{au['surname']}|{au['given_names']}")
    tiers = collections.Counter()
    matched = []
    for p in persons.values():
        if not p["authored"]:
            continue
        best = min((v["tier"] for v in p["authored"].values()), key=TIER_RANK.get)
        tiers[best] += 1
        matched.append(p)

    # same intersection restricted to the seed dataset (its ORCID reviewers x its article author lists)
    seed_rev_orcids = {r["orcid"] for r in seed}
    seed_auth_orcids = {x["orcid"] for v in seed_authors.values() for x in v if x["orcid"]}
    seed_auth_names = {name_key(x["given_names"] or "", x["surname"] or "") for v in seed_authors.values() for x in v}
    seed_rev_names = {name_key(r["given_names"], r["surname"]) for r in seed}
    seed_works_hit = {o for o in seed_rev_orcids if any(
        (p.get("pmid") in seed_authors) for r in seed if r["orcid"] == o for p in r["publications"])}

    stats = {
        "articles_parsed": len(arts),
        "articles_with_reviews": len({r["article_key"] for r in reviews}),
        "reviews_parsed": len(reviews),
        "distinct_reviewers": len(reviewer_ids),
        "distinct_reviewers_with_orcid": sum(1 for x in reviewer_ids if not x.startswith(("name:", "raw:"))),
        "distinct_authors": len(author_ids),
        "reviewer_author_intersection_by_best_tier": dict(tiers),
        "reviewer_author_intersection_usable(orcid,orcid_works,name_aff)": sum(v for k, v in tiers.items() if k != "name_only"),
        "seed_dataset": {
            "assignment_articles": len(seed_assign), "orcid_reviewers": len(seed_rev_orcids),
            "author_lists": len(seed_authors), "distinct_author_orcids": len(seed_auth_orcids),
            "reviewers_with_orcid_in_seed_authors": len(seed_rev_orcids & seed_auth_orcids),
            "reviewers_with_fullname_in_seed_authors": len((seed_rev_names - {None}) & seed_auth_names),
            "reviewers_whose_orcid_works_include_a_seed_article": len(seed_works_hit),
        },
    }
    json.dump(stats, open(f"{DATA}/match_stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))

    out = []
    for p in matched:
        out.append({"pid": p["pid"], "orcids": sorted(p["orcids"]), "name_keys": sorted(p["name_keys"]),
                    "si": sorted(p["si"]), "reviews": p["reviews"],
                    "authored": sorted(p["authored"].values(), key=lambda v: v["article_key"]),
                    "possible_authored": p["possible_authored"]})
    write_jsonl(f"{DATA}/private/person_matches.jsonl", out)


if __name__ == "__main__":
    main()
