"""Data for the two-channel figure: information (bits) carried by the content channel (SPECTER2 similarity) and the
content-free style channel (handcrafted features on aggressively masked text) under four conditions of the review:
original, content-masked (both texts masked), light AI rewrite, heavy AI rewrite.
Original/light/heavy are taken from outputs/rewrite/evaluate.json (same `analyse` as s13); the masked condition is
computed here with the cached masked SPECTER2 embeddings. Output: outputs/figures/channel_bits.{json,md}
"""
import json, os, sys
import numpy as np

from config import ROOT
OUT = f"{ROOT}/outputs/figures"


def main():
    from config import XDATA
    from scores import load_tasks, embed_cached
    from analysis.decomposition import analyse
    T = load_tasks()
    F0 = np.load(f"{XDATA}/cache/boost_features_combined.npz", allow_pickle=True)["feats"].item()
    style = {t["qid"]: F0["resp_stylo_masked"][t["qid"]].tolist() for t in T}

    # original (sanity check against evaluate.json)
    re_ = embed_cached([t["review"] for t in T], "reviews_specter2")
    remb = {t["qid"]: re_[t["review"]] for t in T}
    cE = embed_cached(sorted({x for t in T for x in t["texts"]}), "resp_emb_original")
    orig = [{"qid": t["qid"], "pid": t["pid"], "pos": t["pos"], "style": style[t["qid"]],
             "topic": (np.stack([cE[x] for x in t["texts"]]) @ remb[t["qid"]]).tolist()} for t in T]

    # content-masked: review and every candidate replaced by the aggressively masked text
    M = json.load(open(f"{XDATA}/style_masked_aggressive/masked.json"))
    mrev = {q: v["text"] for q, v in M["reviews"].items()}
    mresp = {u: v["text"] for u, v in M["responses"].items()}
    mE = embed_cached(sorted({mresp[x] for t in T for x in t["texts"]}), "resp_emb_masked_aggressive")
    mR = embed_cached([mrev[t["qid"]] for t in T], "resp_emb_masked_aggressive_reviews")
    masked = [{"qid": t["qid"], "pid": t["pid"], "pos": t["pos"], "style": style[t["qid"]],
               "topic": (np.stack([mE[mresp[x]] for x in t["texts"]]) @ mR[mrev[t["qid"]]]).tolist()} for t in T]

    keys = ("I_topic", "I_style", "I_joint", "redundancy", "I_style_given_topic", "I_topic_given_style")
    res = {}
    for name, dt in (("original_recomputed", orig), ("masked", masked)):
        a = analyse(dt)["decomposition"]
        res[name] = {k: a[k] for k in keys}
    ev = json.load(open(f"{ROOT}/outputs/rewrite/evaluate.json"))["bits"]
    out = {"original": ev["original"], "masked": res["masked"], "light": ev["light"], "heavy": ev["heavy"],
           "_original_recomputed_here": res["original_recomputed"],
           "_note": "bits = log2 N + log2 P(true candidate) under a reviewer-disjoint conditional logit; "
                    "topic = SPECTER2 review/response similarity, style = handcrafted features on aggressively masked text; "
                    "[mean, 2.5%, 97.5%] bootstrap over 298 queries"}
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(f"{OUT}/channel_bits.json", "w"), indent=1)
    L = ["| condition | content bits | style bits | joint bits | redundancy |", "|---|---|---|---|---|"]
    for c in ("original", "masked", "light", "heavy", "_original_recomputed_here"):
        b = out[c]
        L.append(f"| {c} | " + " | ".join(f"{b[k][0]:.2f} [{b[k][1]:.2f}, {b[k][2]:.2f}]" for k in ("I_topic", "I_style", "I_joint", "redundancy")) + " |")
    open(f"{OUT}/channel_bits.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
