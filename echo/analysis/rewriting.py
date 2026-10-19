"""Effect of AI rewriting of the review on the full linkage pipeline (rebuttal-archive, combined setting).

  python -m analysis.rewriting rewrite     Luna rewrites each sanitized review at two levels (light / heavy), cached
  python -m analysis.rewriting features    recompute all review-side scores (44 features, same set as the final system)
  python -m analysis.rewriting evaluate    Stage 1-2: attacker with original weights (transfer) and refit attacker; channel bits
  python -m analysis.rewriting rerank      Stage 3: Luna rerank of the top-20 under both attackers; final numbers
  python -m analysis.rewriting report
Environment: F1000_CORPUS=platform F1000_EXP=rebuttal_archive F1000_VENUE_K=8 F1000_RESPONSE_ARCHIVE=1 F1000_USE_REMOTE_EMB=1
"""
from config import read_articles  # released benchmark: articles keyed by opaque id
import hashlib, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import requests

from config import ROOT
OUTDIR = f"{ROOT}/outputs/rewrite"
LEVELS = ["light", "heavy"]
REWRITE_MODEL = "openai/gpt-6-luna"
RNG = np.random.default_rng(0)
PROMPTS = {
    "light": ("Lightly polish the following peer review: correct grammar, spelling and punctuation and improve clarity, "
              "but keep the wording, sentence order, paragraph structure and every point, claim and request exactly as they are. "
              "Do not add, remove or reorder content. Output only the polished review."),
    "heavy": ("Rewrite the following peer review in your own words using neutral, formal academic English. Preserve every point, "
              "claim, question and request and keep them in the same order, but change the phrasing, sentence structure and word "
              "choice throughout. Do not add or remove content and do not comment on the review. Output only the rewritten review."),
}


def _h(t):
    return hashlib.sha1(t.encode()).hexdigest()


# ---------------------------------------------------------------- rewriting
def rewrite_all(reviews):
    key = os.environ["OPENROUTER_API_KEY"]
    path = f"{OUTDIR}/rewrites.json"
    R = json.load(open(path)) if os.path.exists(path) else {}

    def one(level, qid, text):
        k = f"{level}:{qid}"
        if k in R:
            return
        body = {"model": REWRITE_MODEL, "temperature": 0, "max_tokens": 4000, "reasoning": {"max_tokens": 1000},
                "messages": [{"role": "system", "content": PROMPTS[level]}, {"role": "user", "content": text}],
                "usage": {"include": True}}
        for attempt in range(6):
            try:
                r = requests.post("https://openrouter.ai/api/v1/chat/completions", timeout=300, json=body,
                                  headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
                r.raise_for_status()
                d = r.json()
                out = (d["choices"][0]["message"]["content"] or "").strip()
                if len(out.split()) < 0.4 * len(text.split()):
                    raise RuntimeError(f"rewrite too short ({len(out.split())} vs {len(text.split())} words)")
                R[k] = {"text": out, "cost": (d.get("usage") or {}).get("cost"), "words_in": len(text.split()), "words_out": len(out.split())}
                return
            except Exception as e:
                print("retry", attempt, k, str(e)[:100], flush=True)
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(k)
    jobs = [(lv, q, t) for lv in LEVELS for q, t in reviews.items()]
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda j: one(*j), jobs))
    json.dump(R, open(path, "w"), indent=0)
    cost = sum((v["cost"] or 0) for v in R.values())
    print(f"rewrites: {len(R)}; cost ${cost:.2f}")
    return R


# ---------------------------------------------------------------- features (same 44-score set as the final system)
def combined_feats(tasks, cache_dir):
    """paper_* and resp_* scores for tasks whose 'review' may be rewritten; candidate-side caches are reused."""
    from config import DATA, read_jsonl
    from sklearn.feature_extraction.text import TfidfVectorizer
    import signals as B
    from encoders import embed_cached as star_cached, Star
    arts = read_articles()
    ptasks = [dict(t, texts=[arts[c]["title"] + ". " + arts[c]["abstract"] for c in t["cands"]],
                   style_texts=[arts[c]["abstract"] for c in t["cands"]]) for t in tasks]
    F = {}
    for side, T in (("paper", ptasks), ("resp", tasks)):
        for m in ("specter2", "specter2q", "scincl", "bge", "e5", "qwen3emb"):
            F.update({f"{side}_{k}": v for k, v in B.dense_features(T, m, cache_dir).items()})
            if m == "qwen3emb":
                F.update({f"{side}_{k}": v for k, v in B.dense_full_features(T, m, cache_dir).items()})
        F.update({f"{side}_{k}": v for k, v in B.bm25_features(T).items()})
        docs = sorted({x for t in T for x in t["texts"]})
        tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=3, max_features=200000)
        tf.fit(docs + [t["review"] for t in T])
        DT = dict(zip(docs, tf.transform(docs)))
        RT = dict(zip([t["qid"] for t in T], tf.transform([t["review"] for t in T])))
        F[f"{side}_char"] = {t["qid"]: np.array([float(DT[x].multiply(RT[t["qid"]]).sum()) for x in t["texts"]]) for t in T}
    cs = B.char_style_features(tasks, cache_dir)  # stylo_masked (masks new reviews), luar, char (dup)
    F["resp_stylo_masked"] = cs["stylo_masked"]
    E = star_cached("star", Star, sorted({x for t in tasks for x in t["style_texts"]} | {t["review"] for t in tasks}), cache_dir)
    F["resp_star"] = {t["qid"]: np.stack([E[x] for x in t["style_texts"]]) @ E[t["review"]] for t in tasks}
    return F


def final_cols(F):
    dense = lambda side: [k for k in F if k.startswith(side + "_") and k.endswith(("_max", "_col", "_mean", "_full"))
                          and not any(k.startswith(f"{side}_{s}") for s in ("luar", "luarcrud", "styledist", "star"))]
    lex = lambda side: [f"{side}_{k}" for k in ("bm25_uni", "bm25_bi", "char")]
    return dense("paper") + lex("paper") + dense("resp") + lex("resp") + ["resp_stylo_masked", "resp_star"]


# ---------------------------------------------------------------- fusion with transferred weights
def cv_transfer(meta, F_train, F_test, cols, lams=(0.01, 0.1, 1.0, 10.0)):
    """Outer folds grouped by reviewer; weights fitted on F_train (original reviews), applied to F_test (rewritten)."""
    from sklearn.model_selection import GroupKFold
    from scipy.special import logsumexp
    from signals import zq, fit_clogit
    Xtr = [np.stack([zq(F_train[k][m["qid"]]) for k in cols], 1) for m in meta]
    Xte = [np.stack([zq(F_test[k][m["qid"]]) for k in cols], 1) for m in meta]
    pos = [m["pos"] for m in meta]
    g = np.array([m["pid"] for m in meta])
    oof = [None] * len(meta)
    for tr, te in GroupKFold(5).split(Xtr, groups=g):
        best, best_ll = None, -np.inf
        for lam in lams:
            ll = 0
            for itr, ite in GroupKFold(4).split(tr, groups=g[tr]):
                b = fit_clogit([Xtr[tr[i]] for i in itr], [pos[tr[i]] for i in itr], lam)
                ll += sum((Xtr[tr[i]] @ b)[pos[tr[i]]] - logsumexp(Xtr[tr[i]] @ b) for i in ite)
            if ll > best_ll:
                best, best_ll = lam, ll
        b = fit_clogit([Xtr[i] for i in tr], [pos[i] for i in tr], best)
        for i in te:
            oof[i] = Xte[i] @ b
    return oof


def rank_of(s, p):
    s = np.asarray(s, float) + RNG.uniform(0, 1e-9, len(s))
    return int((s > s[p]).sum()) + 1


def main():
    from config import DATA, XDATA, read_jsonl
    from signals import archive_tasks, cv_fusion, metrics, boot
    os.makedirs(OUTDIR, exist_ok=True)
    cmd = sys.argv[1]
    tasks = archive_tasks()
    meta = json.load(open(f"{XDATA}/cache/boost_meta.json"))
    reviews = {t["qid"]: t["review"] for t in tasks}
    cache_dir = f"{XDATA}/cache"

    if cmd == "rewrite":
        rewrite_all(reviews)
        return
    R = json.load(open(f"{OUTDIR}/rewrites.json"))
    rew = {lv: {q: R[f"{lv}:{q}"]["text"] for q in reviews} for lv in LEVELS}

    if cmd == "features":
        for lv in LEVELS:
            path = f"{cache_dir}/rewrite_features_{lv}.npz"
            if os.path.exists(path):
                continue
            T = [dict(t, review=rew[lv][t["qid"]]) for t in tasks]
            F = combined_feats(T, cache_dir)
            np.savez(path, feats=np.array(F, dtype=object))
            print(lv, "features", len(F), flush=True)
        return

    F0 = np.load(f"{cache_dir}/boost_features_combined.npz", allow_pickle=True)["feats"].item()
    cols = final_cols(F0)
    N = [m["n"] for m in meta]
    Fl = {lv: np.load(f"{cache_dir}/rewrite_features_{lv}.npz", allow_pickle=True)["feats"].item() for lv in LEVELS}

    if cmd == "evaluate":
        import encoders as E5
        import style as stylo
        from style import render
        from analysis.decomposition import analyse
        from scores import embed_cached as s10_cached, load_tasks
        res = {"cols": len(cols)}
        # quality of rewriting
        tok, model = E5.load()
        orig_e = E5.embed([reviews[q] for q in sorted(reviews)], tok, model)
        qual = {}
        M = json.load(open(f"{cache_dir}/boost_masked_aggr.json"))
        for lv in LEVELS:
            new_e = E5.embed([rew[lv][q] for q in sorted(reviews)], tok, model)
            sem = (orig_e * new_e).sum(1)
            lr = [len(rew[lv][q].split()) / len(reviews[q].split()) for q in sorted(reviews)]
            X = stylo.matrix([render(M[_h(reviews[q])]) for q in sorted(reviews)] + [render(M[_h(rew[lv][q])]) for q in sorted(reviews)])
            Z = stylo.zscore_by(X, X)
            n = len(reviews)
            sty = [float(Z[i] @ Z[n + i] / (np.linalg.norm(Z[i]) * np.linalg.norm(Z[n + i]) + 1e-12)) for i in range(n)]
            qual[lv] = {"semantic_sim_mean": float(sem.mean()), "semantic_sim_p10": float(np.percentile(sem, 10)),
                        "length_ratio_mean": float(np.mean(lr)), "masked_style_sim_mean": float(np.mean(sty)),
                        "cost": sum((R[f"{lv}:{q}"]["cost"] or 0) for q in reviews)}
        res["quality"] = qual
        # stage 1-2 rankings
        base_oof, _ = cv_fusion(meta, F0, cols)
        ranks = {"original": [rank_of(s, m["pos"]) for s, m in zip(base_oof, meta)]}
        oofs = {"original": base_oof}
        for lv in LEVELS:
            oofs[f"{lv}_transfer"] = cv_transfer(meta, F0, Fl[lv], cols)
            oofs[f"{lv}_refit"], _ = cv_fusion(meta, Fl[lv], cols)
            for k in (f"{lv}_transfer", f"{lv}_refit"):
                ranks[k] = [rank_of(s, m["pos"]) for s, m in zip(oofs[k], meta)]
        res["stage2"] = {k: metrics(v, N) for k, v in ranks.items()}
        res["stage2_delta_vs_original"] = {k: boot(v, ranks["original"]) for k, v in ranks.items() if k != "original"}
        np.savez(f"{OUTDIR}/stage2_oof.npz", **{k: np.array(v, dtype=object) for k, v in oofs.items()})
        # channel bits: topic = SPECTER2(review) vs SPECTER2(response) raw; style = masked handcrafted
        T0 = load_tasks()
        resp_e = s10_cached(sorted({x for t in T0 for x in t["texts"]}), "resp_emb_original")
        bits = {}
        for name, texts in [("original", reviews)] + [(lv, rew[lv]) for lv in LEVELS]:
            rv_e = dict(zip(sorted(texts), E5.embed([texts[q] for q in sorted(texts)], tok, model)))
            Fs = F0 if name == "original" else Fl[name]
            dt = [{"qid": t["qid"], "pid": t["pid"], "pos": t["pos"],
                   "topic": (np.stack([resp_e[x] for x in t["texts"]]) @ rv_e[t["qid"]]).tolist(),
                   "style": Fs["resp_stylo_masked"][t["qid"]].tolist()} for t in T0]
            a = analyse(dt)
            bits[name] = {k: a["decomposition"][k] for k in ("I_topic", "I_style", "I_joint", "redundancy")}
        res["bits"] = bits
        json.dump(res, open(f"{OUTDIR}/evaluate.json", "w"), indent=1)
        for k, v in res["stage2"].items():
            print(k, {kk: round(vv, 3) for kk, vv in v.items() if kk in ("R@1", "R@5", "R@10", "R@20", "MRR")})
        print("bits", {k: {kk: round(vv[0], 3) for kk, vv in v.items()} for k, v in bits.items()})
        print("quality", json.dumps(qual, indent=1))
        return

    if cmd == "rerank":
        from rerank import Client, prompt
        from sklearn.model_selection import GroupKFold
        arts = read_articles()
        oofs = np.load(f"{OUTDIR}/stage2_oof.npz", allow_pickle=True)
        K = 20
        cli = Client(REWRITE_MODEL, f"{ROOT}/outputs/llm_rerank/cache_openai_gpt-6-luna_budget2000_score.jsonl", 2400, False, None, "score", None, 2000)
        out = {}

        def z(x):  # NaN = missing judgement -> neutral (mean) score, as in rerank.py
            x = np.asarray(x, float)
            m, sd = np.nanmean(x), np.nanstd(x)
            return np.nan_to_num((x - m) / sd if sd > 0 else x * 0, nan=0.0)

        def run_level(name, review_texts, fused_key, alpha_mode):
            fused = [np.asarray(s, float) for s in oofs[fused_key]]
            topk = [[int(i) for i in np.argsort(-(s + RNG.uniform(0, 1e-9, len(s))))[:K]] for s in fused]
            jobs = [(ti, i) for ti, tk in enumerate(topk) for i in tk]
            ps = [prompt(review_texts[tasks[ti]["qid"]], arts[tasks[ti]["cands"][i]]["title"], arts[tasks[ti]["cands"][i]]["abstract"],
                         tasks[ti]["texts"][i], "score") for ti, i in jobs]
            with ThreadPoolExecutor(24) as ex:
                recs = list(ex.map(cli.judge, ps))
            llm = [np.full(len(t["cands"]), np.nan) for t in tasks]
            for (ti, i), r in zip(jobs, recs):
                llm[ti][i] = r["p_yes"]

            def fr(ti, alpha):
                k = topk[ti]
                comb = alpha * z(fused[ti][k]) + (1 - alpha) * z(llm[ti][k])
                full = fused[ti] - 1e6
                full[k] = comb
                return rank_of(full, tasks[ti]["pos"])
            if alpha_mode == "fixed":
                rk = [fr(i, 0.5) for i in range(len(tasks))]
                alphas = [0.5]
            else:
                g = np.array([t["pid"] for t in tasks])
                rk, alphas = [None] * len(tasks), []
                for tr, te in GroupKFold(5).split(tasks, groups=g):
                    best, bm = None, -1
                    for al in np.round(np.arange(0, 1.01, 0.1), 1):
                        mrr = np.mean([1 / fr(i, al) for i in tr])
                        if mrr > bm + 1e-12:
                            best, bm = al, mrr
                    alphas.append(best)
                    for i in te:
                        rk[i] = fr(i, best)
            out[name] = {"metrics": metrics(rk, N), "alphas": alphas, "ranks": rk}
            print(name, {k: round(v, 3) for k, v in out[name]["metrics"].items() if k in ("R@1", "R@5", "R@10", "R@20", "MRR")}, alphas, flush=True)

        run_level("original", reviews, "original", "fixed")
        for lv in LEVELS:  # attacker unchanged: weights fitted on original reviews, alpha = 0.5
            run_level(f"{lv}_transfer", rew[lv], f"{lv}_transfer", "fixed")
        base = out["original"]["ranks"]
        for k, v in out.items():
            v["delta_vs_original"] = boot(v["ranks"], base) if k != "original" else None
            v.pop("ranks")
        json.dump(out, open(f"{OUTDIR}/stage3.json", "w"), indent=1)
        return

    if cmd == "report":
        ev = json.load(open(f"{OUTDIR}/evaluate.json"))
        s3 = json.load(open(f"{OUTDIR}/stage3.json")) if os.path.exists(f"{OUTDIR}/stage3.json") else {}
        L = ["# Effect of AI rewriting of the review", "",
             f"Reviews rewritten by {REWRITE_MODEL} (temperature 0) at two levels; candidates unchanged; same 298 queries and pools; "
             f"same {ev['cols']} fusion scores as the final system. The attacker is unchanged: fusion weights and alpha = 0.5 "
             "fitted on original reviews are applied to the rewritten reviews (reviewer-disjoint CV).", "",
             "## Rewrite quality", "", "| level | semantic sim to original (mean / p10) | length ratio | masked-style sim to original | cost |", "|---|---|---|---|---|"]
        for lv, q in ev["quality"].items():
            L.append(f"| {lv} | {q['semantic_sim_mean']:.3f} / {q['semantic_sim_p10']:.3f} | {q['length_ratio_mean']:.2f} | {q['masked_style_sim_mean']:.3f} | ${q['cost']:.2f} |")
        L += ["", "## Stage 1–2 (fusion) and channel information", "",
              "| condition | Top-1 | Top-5 | Top-10 | Top-20 | MRR | ΔMRR vs original [95% CI] | topic bits | style bits | joint bits |", "|---|---|---|---|---|---|---|---|---|---|"]
        for k, m in ev["stage2"].items():
            if k.endswith("_refit"):
                continue
            d = ev["stage2_delta_vs_original"].get(k)
            lv = k.split("_")[0]
            b = ev["bits"].get(lv, {})
            dm = f"{d['dMRR'][0]:+.3f} [{d['dMRR'][1]:+.3f}, {d['dMRR'][2]:+.3f}]" if d else "—"
            bt = " | ".join(f"{b[x][0]:.2f}" if x in b else "—" for x in ("I_topic", "I_style", "I_joint"))
            L.append(f"| {k.replace('_transfer', ' rewrite')} | {m['R@1']:.3f} | {m['R@5']:.3f} | {m['R@10']:.3f} | {m['R@20']:.3f} | {m['MRR']:.3f} | {dm} | {bt} |")
        if s3:
            L += ["", "## Full system (Stage 3: Luna rerank of top-20)", "",
                  "| condition | Top-1 | Top-5 | Top-10 | Top-20 | MRR | ΔMRR vs original [95% CI] | α |", "|---|---|---|---|---|---|---|---|"]
            for k, v in s3.items():
                m, d = v["metrics"], v["delta_vs_original"]
                dm = f"{d['dMRR'][0]:+.3f} [{d['dMRR'][1]:+.3f}, {d['dMRR'][2]:+.3f}]" if d else "—"
                L.append(f"| {k.replace('_transfer', ' rewrite')} | {m['R@1']:.3f} | {m['R@5']:.3f} | {m['R@10']:.3f} | {m['R@20']:.3f} | {m['MRR']:.3f} | {dm} | {v['alphas']} |")
        open(f"{OUTDIR}/report.md", "w").write("\n".join(L) + "\n")
        print("\n".join(L))


if __name__ == "__main__":
    main()
