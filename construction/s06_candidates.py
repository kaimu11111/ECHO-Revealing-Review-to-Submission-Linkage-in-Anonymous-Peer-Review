"""Candidate pools (1 positive + 49 negatives) for each query, in three settings, plus author-response pools.

Negatives are never: the reviewed article Q, the positive P, any article with an author matching the reviewer
(ORCID / full name / surname+first initial - conservative), or an article published > 90 days after
max(review date, P date) (so publication date cannot separate positives from negatives).
With F1000_CONCURRENT_DAYS=N (ICLR-style), all candidates must instead be published within +-N days of the review.

random        : uniform over eligible articles
same_field    : the 49 eligible articles most similar to Q (SPECTER2 cosine)
topic_time_matched : as topic_matched, restricted to articles published within +-365 days of P
topic_matched : negatives with |sim(Q,c) - sim(Q,P)| < eps, sampled symmetrically (24/25 above, 25/24 below
                sim(Q,P)); eps starts at 0.03 and is relaxed per side (0.05, 0.075, 0.10, 0.15) when needed
"""
import collections, datetime, hashlib, json, random
import numpy as np
from common import DATA, ROOT, read_jsonl, write_jsonl, pub_date, name_key, XDATA, BENCH, CONCURRENT_DAYS, EXP, VENUE_K, field_clusters, venue_of, is_template_article, RESPONSE_ARCHIVE, ARCHIVE_LAG_DAYS
from clean import clean_response, english_ok

N_NEG = 49
EPS = [0.03, 0.05, 0.075, 0.10, 0.15]
AROUND_DAYS = 90
SEED = 7
RESP_MAX_WORDS = 600
RESP_MIN_WORDS = 50


def d(s):
    return datetime.date.fromisoformat(s)


def opaque(key):
    return "a" + hashlib.sha256(("f1000-art:" + key).encode()).hexdigest()[:10]


def load_emb(name):
    return np.load(f"{DATA}/cache/{name}.npy"), json.load(open(f"{DATA}/cache/{name}_ids.json"))


def topic_matched(rng, simQ, elig_idx, sp):
    """Symmetric matching: half the negatives just above sim(Q,P+), half just below, so that ranking by
    sim(Q,c) cannot separate the positive (it lands at rank 25 or 26). Each side uses the smallest eps in
    EPS with enough candidates; if none suffices, the nearest candidates on that side are taken."""
    s = simQ[elig_idx]
    n_above = int(rng.integers(24, 26))
    need = {"above": n_above, "below": N_NEG - n_above}
    side_idx = {"above": elig_idx[s >= sp], "below": elig_idx[s < sp]}
    if len(side_idx["above"]) < need["above"]:  # positive at the very top of the distribution
        need = {"above": len(side_idx["above"]), "below": N_NEG - len(side_idx["above"])}
    chosen, eps_used = [], {}
    for side, idx in side_idx.items():
        gap = np.abs(simQ[idx] - sp)
        for eps in EPS:
            pool = idx[gap < eps]
            if len(pool) >= need[side]:
                chosen += [int(i) for i in rng.choice(pool, need[side], replace=False)]
                eps_used[side] = eps
                break
        else:
            order = idx[np.argsort(gap)][:need[side]]
            chosen += [int(i) for i in order]
            eps_used[side] = float(np.abs(simQ[order] - sp).max()) if len(order) else 0.0
    return chosen, max(eps_used.values()), need["above"]


VENUE = {}


def build_pools(qs, keys, emb, kidx, elig_mask, dates, rng, settings, time_window_days=365):
    pools = {s: [] for s in settings}
    for q in qs:
        simQ = emb @ emb[kidx[q["Q"]]]
        sp = float(simQ[kidx[q["P"]]])
        excl = {kidx[k] for k in q["exclude_articles"] if k in kidx} | {kidx[q["Q"]], kidx[q["P"]]}
        m = elig_mask.copy()
        m[list(excl)] = False
        if VENUE_K and RESPONSE_ARCHIVE:  # field archive: same field as Q, public by review date + lag
            fq = VENUE[q["Q"]][0]
            lim = d(q["review_date"]) + datetime.timedelta(days=ARCHIVE_LAG_DAYS)
            m &= np.array([(VENUE.get(k) or (None,))[0] == fq and dt <= lim for k, dt in zip(keys, dates)])
        elif VENUE_K:  # ICLR-style venue: candidates are the other articles of Q's (field, year) edition
            vq = VENUE[q["Q"]]
            m &= np.array([VENUE.get(k) == vq for k in keys])
        elif CONCURRENT_DAYS:  # ICLR-style: every candidate comes from the same "cycle" as the review
            rdate = d(q["review_date"])
            m &= np.array([abs((dt - rdate).days) <= CONCURRENT_DAYS for dt in dates])
        else:
            limit = max(d(q["review_date"]), dates[kidx[q["P"]]]) + datetime.timedelta(days=AROUND_DAYS)
            m &= np.array([dt <= limit for dt in dates])
        elig_idx = np.where(m)[0]
        pdate = dates[kidx[q["P"]]]
        near_idx = np.array([i for i in elig_idx if abs((dates[i] - pdate).days) <= time_window_days])
        for s in settings:
            n_above = None
            if s == "random":
                neg, eps = [int(i) for i in rng.choice(elig_idx, N_NEG, replace=False)], None
            elif s == "same_field":
                neg, eps = [int(i) for i in elig_idx[np.argsort(-simQ[elig_idx])[:N_NEG]]], None
            elif s == "venue_full":  # rank against every article of the edition
                neg, eps = [int(i) for i in elig_idx], None
            elif s == "venue_50":
                neg, eps = [int(i) for i in rng.choice(elig_idx, min(N_NEG, len(elig_idx)), replace=False)], None
            elif s in ("topic_matched", "venue_topic_matched"):
                neg, eps, n_above = topic_matched(rng, simQ, elig_idx, sp)
            elif s == "topic_time_matched":
                neg, eps, n_above = topic_matched(rng, simQ, near_idx, sp)
            cands = [kidx[q["P"]]] + neg
            rng.shuffle(cands)
            pools[s].append({"qid": q["qid"], "candidates": [keys[i] for i in cands], "positive": q["P"],
                             "eps": eps, "n_above": n_above, "n_eligible": int(len(elig_idx)),
                             "n_eligible_time_window": int(len(near_idx)), "sim_QP": sp,
                             "sim_Qneg": [float(simQ[i]) for i in neg]})
    return pools


def responder_texts(arts, reviews_by_art, persons):
    """Per article: {responder_key: cleaned concatenated response text}."""
    resps = read_jsonl(f"{DATA}/responses.jsonl")
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    stats = collections.Counter()
    for r in resps:
        if not r["contribs"]:
            continue
        revs = reviews_by_art.get(r["article_key"], [])
        rev_texts = [" ".join(b["raw"] for b in rv["blocks"]) for rv in revs]
        mask = r["contribs"] + [c for rv in revs for c in rv["contribs"]]
        text, st = clean_response(r, rev_texts, mask)
        stats.update(st)
        c = r["contribs"][0]
        rk = c["orcid"] or name_key(c["given_names"], c["surname"]) or c["surname"]
        if text.strip():
            out[r["article_key"]][rk].append(text)
    final = {}
    for k, by in out.items():
        final[k] = {rk: " ".join(" ".join(ts).split()[:RESP_MAX_WORDS]) for rk, ts in by.items()}
    return final, stats


def main():
    rng = np.random.default_rng(SEED)
    arts = {a["article_key"]: a for a in read_jsonl(f"{DATA}/articles.jsonl")}
    emb, keys = load_emb("papers")
    kidx = {k: i for i, k in enumerate(keys)}
    dates = [d(pub_date(arts[k])) for k in keys]
    elig = np.array([bool(arts[k]["title"]) and len(arts[k]["abstract"].split()) >= 50 and not is_template_article(arts[k])
                     for k in keys])
    qs = read_jsonl(f"{XDATA}/private/queries_full.jsonl")

    # in the concurrent design every pool is already time-restricted, so topic_time_matched would be redundant
    settings = ["random", "same_field", "topic_matched"] + ([] if CONCURRENT_DAYS else ["topic_time_matched"])
    if VENUE_K:
        global VENUE
        cl = field_clusters(VENUE_K)
        VENUE = {k: venue_of(a, cl) for k, a in arts.items()}
        settings = ["venue_full", "venue_50", "venue_topic_matched"]
    pools = build_pools(qs, keys, emb, kidx, elig, dates, rng, settings)

    # ---------- optional author-response experiment ----------
    reviews = read_jsonl(f"{DATA}/reviews.jsonl")
    by_art = collections.defaultdict(list)
    for r in reviews:
        by_art[r["article_key"]].append(r)
    persons = {p["pid"]: p for p in read_jsonl(f"{DATA}/private/person_matches.jsonl")}
    rtexts, rstats = responder_texts(arts, by_art, persons)
    resp_rows, resp_qs = [], []
    art_resp = {}  # article -> representative response text (single responder, most text) for negatives
    for k, by in rtexts.items():
        rk, t = max(by.items(), key=lambda kv: len(kv[1].split()))
        if len(t.split()) >= RESP_MIN_WORDS and english_ok(t):
            art_resp[k] = t
    for q in qs:
        p = persons[q["pid"]]
        ids = set(p["orcids"]) | set(p["name_keys"])
        mine = [t for rk, t in rtexts.get(q["P"], {}).items() if rk in ids]
        if mine and len(mine[0].split()) >= RESP_MIN_WORDS and english_ok(mine[0]):
            resp_qs.append(dict(q))
            resp_rows.append({"qid": q["qid"], "positive_response": mine[0]})
    resp_elig = elig & np.array([k in art_resp for k in keys])
    resp_pools = build_pools(resp_qs, keys, emb, kidx, resp_elig, dates, rng,
                             ["venue_full"] if VENUE_K else
                             ["random", "topic_matched"] + ([] if CONCURRENT_DAYS else ["topic_time_matched"]))
    # the positive uses the response written by the matched reviewer-author; negatives use their own article's response
    pos_resp = {r["qid"]: r["positive_response"] for r in resp_rows}

    # ---------- write benchmark files (opaque ids, no author metadata) ----------
    used = set()
    for s, rows in list(pools.items()) + [("resp_" + s, r) for s, r in resp_pools.items()]:
        for r in rows:
            used.update(r["candidates"])
    used |= {q["Q"] for q in qs}
    write_jsonl(f"{BENCH}/articles.jsonl",
                [{"article_id": opaque(k), "title": arts[k]["title"], "abstract": arts[k]["abstract"]} for k in sorted(used)])
    write_jsonl(f"{BENCH}/reviewed_papers.jsonl",
                [{"qid": q["qid"], "reviewed_article_id": opaque(q["Q"])} for q in qs])
    for s, rows in pools.items():
        write_jsonl(f"{BENCH}/candidates_{s}.jsonl",
                    [{"qid": r["qid"], "candidates": [opaque(c) for c in r["candidates"]]} for r in rows])
    for s, rows in resp_pools.items():
        write_jsonl(f"{BENCH}/response_candidates_{s}.jsonl",
                    [{"qid": r["qid"], "candidates": [opaque(c) for c in r["candidates"]]} for r in rows])
    write_jsonl(f"{BENCH}/gold.jsonl", [{"qid": q["qid"], "positive": opaque(q["P"])} for q in qs])
    # response texts keyed by (qid, candidate) since the positive's text is query specific
    rt = []
    for r in [r for rows in resp_pools.values() for r in rows]:
        for c in r["candidates"]:
            rt.append({"qid": r["qid"], "article_id": opaque(c),
                       "response_text": pos_resp[r["qid"]] if c == r["positive"] else art_resp[c]})
    write_jsonl(f"{BENCH}/response_texts.jsonl", rt)
    # private (keys) for evaluation / reporting
    json.dump({"pools": pools, "resp_pools": resp_pools, "response_clean_stats": dict(rstats),
               "n_articles_with_response": len(art_resp), "n_resp_queries": len(resp_qs)},
              open(f"{XDATA}/private/pools.json", "w"))
    for s, rows in pools.items():
        eps = collections.Counter(r["eps"] for r in rows)
        print(s, len(rows), "eps:", dict(eps) if "topic" in s else "-",
              "fully_controlled:", sum((r["n_above"] or 0) >= 24 for r in rows) if "topic" in s else "-",
              "pool size median:", int(np.median([len(r["candidates"]) for r in rows])))
    print("response queries:", len(resp_qs), "articles with usable response:", len(art_resp), dict(rstats))


if __name__ == "__main__":
    main()
