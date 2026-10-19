"""Pretrained authorship-representation models as style scorers on the rebuttal archive (review -> author response).

Models (encoders, not generative):
  luar     rrivera1849/LUAR-MUD (Rivera-Soto et al., EMNLP 2021; Reddit MUD), pinned revision. A text is split into
           consecutive 30-token chunks, which form one LUAR "episode" (all text used, up to 64 chunks).
  wegmann  AnnaWegmann/Style-Embedding (Wegmann et al., RepL4NLP 2022; content-controlled training). Each sentence is
           embedded and the sentence embeddings are averaged.
Text variants: original, standard-masked, aggressive-masked (`style_masked_text` from s11; tags rendered as "xx").
Evaluation: (1) full-pool retrieval vs handcrafted / char n-gram / random; (2) topic -> style rerank and topic+style
fusion at K = 10, 20, 50, 100, all, alpha chosen on reviewer-disjoint folds (same protocol as analysis/topic_style_rerank.py).
Outputs: outputs/authorship_models/{results.csv, rerank_results.csv, rerank_deltas.csv, report.md}
"""
import hashlib, json, os, re
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from config import XDATA
from scores import ART_ROOT, load_tasks, original_scores, score_tables, rank_of_pos
from analysis.topic_style_rerank import KS, ALPHAS, topk_rank, fused_rank, summarize, boot_delta
from style import render

OUTDIR = f"{ART_ROOT}/authorship_models"
LUAR_REV = "f1db50251805ed69b43cf4f72ea2f0e231f36a1c"
RNG = np.random.default_rng(0)
SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def _h(t):
    return hashlib.sha1(t.encode()).hexdigest()


class Luar:
    def __init__(self):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained("rrivera1849/LUAR-MUD", revision=LUAR_REV)
        self.model = AutoModel.from_pretrained("rrivera1849/LUAR-MUD", revision=LUAR_REV, trust_remote_code=True).cuda().eval()

    @torch.no_grad()
    def embed(self, texts):
        out = []
        for t in texts:
            ids = self.tok(t, add_special_tokens=False)["input_ids"]
            chunks = [ids[i:i + 30] for i in range(0, max(1, len(ids)), 30)][:64]
            docs = [self.tok.decode(c) for c in chunks]
            b = self.tok(docs, max_length=32, padding="max_length", truncation=True, return_tensors="pt")
            b = {k: v.reshape(1, len(docs), -1).cuda() for k, v in b.items() if k in ("input_ids", "attention_mask")}
            out.append(self.model(**b)[0].float().cpu().numpy())
        e = np.stack(out)
        return e / np.linalg.norm(e, axis=1, keepdims=True)


class Wegmann:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("AnnaWegmann/Style-Embedding", device="cuda")

    def embed(self, texts):
        sents = [[s for s in SENT.split(t) if s.strip()] or [t] for t in texts]
        flat = [s for ss in sents for s in ss]
        E = self.model.encode(flat, batch_size=256, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        out, i = [], 0
        for ss in sents:
            v = E[i:i + len(ss)].mean(0)
            out.append(v / np.linalg.norm(v))
            i += len(ss)
        return np.stack(out)


def embed_cached(model_name, model_obj, texts):
    path = f"{XDATA}/cache/auth_{model_name}.npz"
    cache = dict(np.load(path)) if os.path.exists(path) else {}
    todo = sorted({t for t in texts if _h(t) not in cache})
    if todo:
        m = model_obj()
        for i in range(0, len(todo), 500):
            for t, e in zip(todo[i:i + 500], m.embed(todo[i:i + 500])):
                cache[_h(t)] = e
            np.savez(path, **cache)
            print(model_name, min(i + 500, len(todo)), "/", len(todo), flush=True)
        del m
        torch.cuda.empty_cache()
    return {t: cache[_h(t)] for t in set(texts)}


def metrics(r, N):
    r, N = np.asarray(r, float), np.asarray(N, float)
    return {"n": len(r), "R@1": np.mean(r <= 1), "R@5": np.mean(r <= 5), "R@10": np.mean(r <= 10), "MRR": np.mean(1 / r),
            "median_rank": float(np.median(r)), "mean_pct_rank": float(np.mean((r - 1) / (N - 1)))}


def boot(a, b, n=2000):
    out = {}
    for name, f in (("dMRR", lambda r: 1 / r), ("dR@1", lambda r: (r <= 1) * 1.0), ("dR@5", lambda r: (r <= 5) * 1.0),
                    ("dR@10", lambda r: (r <= 10) * 1.0)):
        d = f(np.asarray(a, float)) - f(np.asarray(b, float))
        m = d[RNG.integers(0, len(d), (n, len(d)))].mean(1)
        out[name] = (d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5))
    return out


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    tasks = load_tasks()
    qids = [t["qid"] for t in tasks]
    pos = {t["qid"]: t["pos"] for t in tasks}
    sizes = [len(t["cands"]) for t in tasks]
    S = original_scores(tasks)  # topic, stylo, char on original text

    # text variants: original + the two masked levels produced by style.py
    variants = {"original": ({t["qid"]: t["review"] for t in tasks}, {x: x for t in tasks for x in t["texts"]})}
    for lvl, d in (("masked", "style_masked"), ("masked_aggr", "style_masked_aggressive")):
        M = json.load(open(f"{XDATA}/{d}/masked.json"))
        variants[lvl] = ({q: render(v["text"]) for q, v in M["reviews"].items()},
                         {u: render(v["text"]) for u, v in M["responses"].items()})
        if lvl == "masked_aggr":  # handcrafted / char-n-gram on aggressive masking, for comparison
            rev_m, resp_m = M["reviews"], M["responses"]
            cand = {(t["qid"], i): resp_m[x]["text"] for t in tasks for i, x in enumerate(t["texts"])}
            mr = {q: v["text"] for q, v in rev_m.items()}
            Sm = score_tables(tasks, mr, cand, list(mr.values()) + [v["text"] for v in resp_m.values()], with_topic=False, render=render)
            for q in S:
                S[q]["stylo_masked_aggr"], S[q]["char_masked_aggr"] = Sm[q]["stylo"], Sm[q]["char"]

    for mname, mobj in (("luar", Luar), ("wegmann", Wegmann)):
        for vname, (revs, resp_map) in variants.items():
            texts = list(revs.values()) + [resp_map[x] for t in tasks for x in t["texts"]]
            E = embed_cached(f"{mname}_{vname}", mobj, texts)
            for t in tasks:
                q = t["qid"]
                C = np.stack([E[resp_map[x]] for x in t["texts"]])
                S[q][f"{mname}_{vname}"] = C @ E[revs[q]]

    # ---------- (1) full-pool retrieval ----------
    R = {"random": [rank_of_pos(RNG.uniform(size=len(t["cands"])), t["pos"], RNG) for t in tasks]}
    keys = ["topic", "stylo", "char", "stylo_masked_aggr", "char_masked_aggr"] + [
        f"{m}_{v}" for m in ("luar", "wegmann") for v in variants]
    for k in keys:
        R[k] = [rank_of_pos(S[t["qid"]][k], t["pos"], RNG) for t in tasks]
    res = pd.DataFrame([{"scorer": k, **metrics(v, sizes)} for k, v in R.items()])
    res.to_csv(f"{OUTDIR}/results.csv", index=False, float_format="%.4f")
    comps = []
    for m in ("luar", "wegmann"):
        comps += [(f"{m}_original", "random"), (f"{m}_masked_aggr", "random"), (f"{m}_masked_aggr", f"{m}_original"),
                  (f"{m}_original", "stylo"), (f"{m}_masked_aggr", "stylo_masked_aggr"), (f"{m}_original", "char")]
    B = {(a, b): boot(R[a], R[b]) for a, b in comps}

    # ---------- (2) topic -> style rerank / fusion ----------
    styles = [k for k in keys if k.startswith(("luar", "wegmann"))] + ["stylo_masked_aggr"]
    groups = np.array([t["pid"] for t in tasks])
    rows, deltas = [], []
    for K in KS:
        Kname = "all" if K is None else str(K)
        ranks = {"topic_only": [None if (r := topk_rank(S[q]["topic"], S[q]["topic"], pos[q], K)) is None else r[0] for q in qids]}
        for key in styles:
            ranks[f"rerank_{key}"] = [None if (r := topk_rank(S[q]["topic"], S[q][key], pos[q], K)) is None else r[0] for q in qids]
            pred = [None] * len(qids)
            for tr, te in GroupKFold(5).split(qids, groups=groups):
                best, best_mrr = None, -1
                for a in ALPHAS:
                    mrr = np.mean([1 / r if r else 0 for r in (fused_rank(S[qids[i]]["topic"], S[qids[i]][key], pos[qids[i]], K, a) for i in tr)])
                    if mrr > best_mrr + 1e-12:
                        best, best_mrr = a, mrr
                for i in te:
                    pred[i] = fused_rank(S[qids[i]]["topic"], S[qids[i]][key], pos[qids[i]], K, best)
            ranks[f"fusion_{key}"] = pred
        surv = np.array([r is not None for r in ranks["topic_only"]])
        base = np.array([r if r is not None else np.inf for r in ranks["topic_only"]], float)
        for m, rr in ranks.items():
            for scope, met in summarize(rr, K).items():
                rows.append({"K": Kname, "scope": scope, "method": m, **met})
            if m != "topic_only":
                arr = np.array([r if r is not None else np.inf for r in rr], float)
                for scope, mask in (("end_to_end", np.ones(len(qids), bool)), ("conditional", surv)):
                    for metric, (mean, lo, hi, p) in boot_delta(arr, base, mask).items():
                        deltas.append({"K": Kname, "scope": scope, "method": m, "metric": metric, "diff": mean,
                                       "ci_lo": lo, "ci_hi": hi, "n": int(mask.sum())})
    rr_df, dl = pd.DataFrame(rows), pd.DataFrame(deltas)
    rr_df.to_csv(f"{OUTDIR}/rerank_results.csv", index=False, float_format="%.4f")
    dl.to_csv(f"{OUTDIR}/rerank_deltas.csv", index=False, float_format="%.4f")

    # ---------- report ----------
    label = {"random": "random", "topic": "topic (SPECTER2)", "stylo": "handcrafted style", "char": "char n-gram",
             "stylo_masked_aggr": "handcrafted style, aggressive mask", "char_masked_aggr": "char n-gram, aggressive mask",
             "luar_original": "LUAR", "luar_masked": "LUAR, standard mask", "luar_masked_aggr": "LUAR, aggressive mask",
             "wegmann_original": "Wegmann style emb.", "wegmann_masked": "Wegmann, standard mask",
             "wegmann_masked_aggr": "Wegmann, aggressive mask"}
    L = ["# Pretrained authorship models on review → author-response linkage", "",
         f"Same {len(tasks)} queries and pools as the rebuttal archive (median {int(np.median(sizes))} candidate responses; chance MRR ≈ "
         f"{np.mean([sum(1 / k for k in range(1, n + 1)) / n for n in sizes]):.3f}). "
         "Each candidate is one submission's author response, so the attacker cannot pool a person's responses across papers, as on OpenReview. "
         "Unlike ICLR rebuttals, every F1000 response here has a single named author, so the positive is guaranteed to be written by the reviewer. "
         "On ICLR, a co-author may write the rebuttal, which would weaken the signal.", "",
         "Models: **LUAR** (`rrivera1849/LUAR-MUD`, pinned revision; Reddit-trained author embeddings; text split into 30-token chunks used as one episode). "
         "**Wegmann style embedding** (`AnnaWegmann/Style-Embedding`, trained to be content-independent; mean of sentence embeddings). "
         "Both are encoders, not generative LLMs. Masked variants use the `style_masked_text` from s11 with tags rendered as `xx`.", "",
         "## 1. Full-pool retrieval", "",
         "| scorer | R@1 | R@5 | R@10 | MRR | median rank | mean pct rank |", "|---|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        L.append(f"| {label.get(r.scorer, r.scorer)} | {r['R@1']:.3f} | {r['R@5']:.3f} | {r['R@10']:.3f} | {r['MRR']:.3f} | {r.median_rank:.0f} | {r.mean_pct_rank:.3f} |")
    L += ["", "Paired bootstrap (2,000 resamples). **Bold** = 95% CI excludes 0.", "",
          "| comparison | ΔMRR | ΔR@1 | ΔR@5 | ΔR@10 |", "|---|---|---|---|---|"]
    for (a, b), d in B.items():
        cells = []
        for k in ("dMRR", "dR@1", "dR@5", "dR@10"):
            m, lo, hi = d[k]
            s = f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            cells.append(f"**{s}**" if lo > 0 or hi < 0 else s)
        L.append(f"| {label.get(a, a)} − {label.get(b, b)} | " + " | ".join(cells) + " |")
    L += ["", "## 2. Topic → style rerank and topic + style fusion (end-to-end MRR; α on reviewer-disjoint folds)", "",
          "| method | " + " | ".join(f"K={'all' if K is None else K}" for K in KS) + " |", "|---|" + "---|" * len(KS)]
    e2e = rr_df[rr_df.scope == "end_to_end"]
    for m in ["topic_only"] + [f"{p}_{k}" for k in styles for p in ("rerank", "fusion")]:
        L.append(f"| {m} | " + " | ".join(f"{e2e[(e2e.method == m) & (e2e.K == ('all' if K is None else str(K)))].MRR.iloc[0]:.3f}" for K in KS) + " |")
    L += ["", "ΔMRR vs topic-only, end-to-end [95% CI]; **bold** = CI excludes 0:", "",
          "| method | " + " | ".join(f"K={'all' if K is None else K}" for K in KS) + " |", "|---|" + "---|" * len(KS)]
    for m in [f"fusion_{k}" for k in styles] + [f"rerank_{k}" for k in styles]:
        cells = []
        for K in KS:
            r = dl[(dl.method == m) & (dl.K == ("all" if K is None else str(K))) & (dl.scope == "end_to_end") & (dl.metric == "dMRR")].iloc[0]
            s = f"{r['diff']:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]"
            cells.append(f"**{s}**" if r.ci_lo > 0 or r.ci_hi < 0 else s)
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    L += ["", "## Interpretation", "", "INTERP_PLACEHOLDER", ""]
    text = "\n".join(L) + "\n"
    if os.path.exists(f"{OUTDIR}/interpretation.md"):
        text = text.replace("INTERP_PLACEHOLDER", open(f"{OUTDIR}/interpretation.md").read().strip())
    open(f"{OUTDIR}/report.md", "w").write(text)
    print(res.round(3).to_string(index=False))
    print(e2e.pivot_table(index="method", columns="K", values="MRR").round(3).to_string())


if __name__ == "__main__":
    main()
