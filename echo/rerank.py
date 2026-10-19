"""LLM reranking of the Top-K candidates from the combined (paper + response) fusion, via OpenRouter.

For each query, the K highest-scoring candidates of the out-of-fold fusion are each judged by the LLM:
"Were this review and this candidate's author response written by the same researcher?" The answer is
restricted to Yes/No and the score is P(Yes) / (P(Yes) + P(No)) from the first-token logprobs.
Final ranking: (a) LLM score alone within Top-K, (b) alpha * z(fusion) + (1 - alpha) * z(llm) with alpha chosen
on reviewer-disjoint folds. Candidates outside Top-K keep their fusion order (end-to-end evaluation).

  python rerank.py trial  [--model M] [--n 5]      small run: served model per request, stability, cost
  python rerank.py run    [--model M] [--k 30]     full run (responses cached, resumable)
  python rerank.py eval   [--model M] [--k 30]     metrics + report section
Environment as for the combined setting (F1000_CORPUS=platform F1000_EXP=rebuttal_archive F1000_VENUE_K=8
F1000_RESPONSE_ARCHIVE=1 F1000_USE_REMOTE_EMB=1). Key in OPENROUTER_API_KEY.
"""
from config import read_articles  # released benchmark: articles keyed by opaque id
import argparse, collections, hashlib, json, math, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import requests

from config import ROOT
OUTDIR = f"{ROOT}/outputs/llm_rerank"
URL = "https://openrouter.ai/api/v1/chat/completions"
LOCAL_URL = os.environ.get("LLM_BASE_URL")  # e.g. http://127.0.0.1:8001/v1 -> vLLM; key from LLM_API_KEY
BEDROCK = os.environ.get("LLM_BACKEND") == "bedrock"  # AWS Bedrock Converse API with a Bedrock API key (bearer)


def _bedrock_env():
    """AWS Bedrock credentials from the environment: AWS_REGION (default us-east-1) and AWS_BEARER_TOKEN_BEDROCK."""
    return os.environ.get("AWS_REGION", "us-east-1"), os.environ.get("AWS_BEARER_TOKEN_BEDROCK")  # only needed for new requests

DEFAULT_MODEL = "openai/gpt-6-luna"
RNG = np.random.default_rng(0)
MAX_REVIEW_WORDS, MAX_ABS_WORDS, MAX_RESP_WORDS = 1500, 300, 1200

SYSTEM = ("You are an expert in authorship analysis of scientific writing. You compare an anonymous peer review with a "
          "candidate submission (its abstract and the author response its authors posted during review) and judge "
          "whether the review and the author response were written by the same researcher. Consider research "
          "expertise and interests revealed by the texts, and writing habits: sentence structure, punctuation, "
          "hedging, discourse markers, politeness formulas, spelling conventions, how points are organised. "
          "Answer with exactly one word: Yes or No.")


def cut(t, n):
    w = t.split()
    return " ".join(w[:n]) + (" [...]" if len(w) > n else "")


SCORE_SUFFIX = ("Question: How likely is it that the AUTHOR RESPONSE was written by the same researcher who wrote the "
                "ANONYMOUS PEER REVIEW? Reply with a single integer from 0 (certainly a different person) to 100 "
                "(certainly the same person), and nothing else.")
SYSTEM_SCORE = SYSTEM.replace("Answer with exactly one word: Yes or No.",
                              "Reply with a single integer from 0 to 100 giving the likelihood that they were written by the same person.")


def prompt(review, title, abstract, response, mode="yesno"):
    if mode == "score":
        return (f"=== ANONYMOUS PEER REVIEW (written by reviewer R about some other paper) ===\n{cut(review, MAX_REVIEW_WORDS)}\n\n"
                f"=== CANDIDATE SUBMISSION ===\nTitle: {title}\nAbstract: {cut(abstract, MAX_ABS_WORDS)}\n\n"
                f"=== AUTHOR RESPONSE posted by the candidate submission's authors ===\n{cut(response, MAX_RESP_WORDS)}\n\n" + SCORE_SUFFIX)
    return (f"=== ANONYMOUS PEER REVIEW (written by reviewer R about some other paper) ===\n{cut(review, MAX_REVIEW_WORDS)}\n\n"
            f"=== CANDIDATE SUBMISSION ===\nTitle: {title}\nAbstract: {cut(abstract, MAX_ABS_WORDS)}\n\n"
            f"=== AUTHOR RESPONSE posted by the candidate submission's authors ===\n{cut(response, MAX_RESP_WORDS)}\n\n"
            "Question: Was the AUTHOR RESPONSE written by the same researcher who wrote the ANONYMOUS PEER REVIEW? "
            "Answer Yes or No.")


class Client:
    def __init__(self, model, cache_path, max_tokens=1, no_reasoning=False, reasoning_effort=None, mode="yesno", providers=None,
                 reasoning_budget=None):
        self.model, self.max_tokens, self.no_reasoning = model, max_tokens, no_reasoning
        self.reasoning_effort, self.mode, self.providers = reasoning_effort, mode, providers
        self.reasoning_budget = reasoning_budget
        self.local = bool(LOCAL_URL)
        if BEDROCK:
            self.region, self.key = _bedrock_env()
        else:
            self.key = os.environ.get("LLM_API_KEY" if self.local else "OPENROUTER_API_KEY")  # only needed for new requests
        self.thinking = os.environ.get("LLM_THINKING", "0") == "1"  # vLLM/Qwen3 enable_thinking
        self.cache_path = cache_path
        self.cache = {}
        if os.path.exists(cache_path):
            for line in open(cache_path):
                r = json.loads(line)
                if not r.get("has_logprobs") and not r.get("text"):
                    continue  # truncated (reasoning hit max_tokens) -> no answer; re-request
                self.cache[r["h"]] = r

    def _h(self, p):
        return hashlib.sha1((self.model + "\n" + p).encode()).hexdigest()

    def _bedrock(self, p, h):
        """One Converse call; returns a cache record in the same format as the OpenRouter path (score mode only)."""
        body = {"system": [{"text": SYSTEM_SCORE if self.mode == "score" else SYSTEM}],
                "messages": [{"role": "user", "content": [{"text": p}]}],
                "inferenceConfig": {"maxTokens": self.max_tokens}}  # this model rejects `temperature`
        if self.reasoning_effort:
            if "anthropic" in self.model:  # Claude 4.6+: adaptive thinking + output_config.effort
                body["additionalModelRequestFields"] = {"thinking": {"type": "adaptive"}, "output_config": {"effort": self.reasoning_effort}}
            else:  # OpenAI models on Bedrock
                body["additionalModelRequestFields"] = {"reasoning": {"effort": self.reasoning_effort}}
        r = requests.post(f"https://bedrock-runtime.{self.region}.amazonaws.com/model/{self.model}/converse", timeout=300, json=body,
                          headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        if r.status_code == 429 or "Throttling" in r.text[:200]:
            raise RuntimeError("rate limited")
        r.raise_for_status()
        d = r.json()
        content = d["output"]["message"]["content"]
        text = " ".join(x["text"] for x in content if "text" in x).strip()
        m = re.search(r"\d+(\.\d+)?", text)
        if not m:
            raise RuntimeError(f"no score in answer: {text[:60]!r} stop={d.get('stopReason')}")
        v = min(100.0, float(m.group()))
        u = d.get("usage", {})
        return {"h": h, "p_yes": v / 100, "p_yes_raw": v / 100, "p_no_raw": 1 - v / 100, "text": text[:40],
                "reasoning_tokens": None, "served_model": self.model, "provider": "bedrock", "has_logprobs": True,
                "usage": {"prompt_tokens": (u.get("inputTokens") or 0) + (u.get("cacheReadInputTokens") or 0),
                          "cached_prompt_tokens": u.get("cacheReadInputTokens"), "completion_tokens": u.get("outputTokens"), "cost": None}}

    def judge(self, p, force=False):
        h = self._h(p)
        if h in self.cache and not force:
            return self.cache[h]
        body = {"model": self.model, "temperature": 0, "max_tokens": self.max_tokens,
                **({} if self.mode == "score" else {"logprobs": True, "top_logprobs": 10}),  # reasoning models reject logprobs
                "messages": [{"role": "system", "content": SYSTEM_SCORE if self.mode == "score" else SYSTEM}, {"role": "user", "content": p}],
                "provider": {"require_parameters": self.mode != "score", **({"only": self.providers} if self.providers else {})},
                "usage": {"include": True}}
        if self.local:  # vLLM: no OpenRouter-specific fields; thinking via chat_template_kwargs
            body.pop("provider", None); body.pop("usage", None)
            body["chat_template_kwargs"] = {"enable_thinking": self.thinking}
        elif self.no_reasoning:
            body["reasoning"] = {"enabled": False}
        elif self.reasoning_budget:
            body["reasoning"] = {"max_tokens": self.reasoning_budget}
        elif self.reasoning_effort:
            body["reasoning"] = {"effort": self.reasoning_effort}
        for attempt in range(6):
            try:
                if BEDROCK:
                    rec = self._bedrock(p, h)
                    self.cache[h] = rec
                    with open(self.cache_path, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    return rec
                r = requests.post((LOCAL_URL.rstrip("/") + "/chat/completions") if self.local else URL, timeout=600, json=body,
                                  headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
                if r.status_code == 429:
                    raise RuntimeError("rate limited")
                r.raise_for_status()
                d = r.json()
                if "error" in d:
                    raise RuntimeError(str(d["error"])[:300])
                ch = d["choices"][0]
                tops = ((ch.get("logprobs") or {}).get("content") or [{}])[0].get("top_logprobs", []) if ch.get("logprobs") else []
                py = pn = 0.0
                for t in tops:
                    tok = t["token"].strip().lower()
                    if tok.startswith("yes"):
                        py += math.exp(t["logprob"])
                    elif tok.startswith("no"):
                        pn += math.exp(t["logprob"])
                text = (ch["message"]["content"] or "").strip()
                if self.mode == "score":
                    m = re.search(r"\d+(\.\d+)?", text)
                    if not m:
                        raise RuntimeError(f"no score in answer: {text[:60]!r}")
                    py, pn = min(100.0, float(m.group())) / 100, 1 - min(100.0, float(m.group())) / 100
                    tops = tops or [None]  # mark as valid
                elif py + pn == 0:  # no usable logprobs: fall back to the text answer
                    py, pn = (1.0, 0.0) if text.lower().startswith("yes") else (0.0, 1.0)
                rec = {"h": h, "p_yes": py / max(py + pn, 1e-12), "p_yes_raw": py, "p_no_raw": pn, "text": text[:40],
                       "reasoning_tokens": ((d.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens"),
                       "served_model": d.get("model"), "provider": d.get("provider") or ("local" if self.local else None),
                       "usage": d.get("usage"), "has_logprobs": bool(tops)}
                self.cache[h] = rec
                with open(self.cache_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                return rec
            except Exception as e:
                wait = 5 * (attempt + 1)
                print(f"retry {attempt} ({str(e)[:120]}) in {wait}s", flush=True)
                time.sleep(wait)
        raise RuntimeError("LLM request failed repeatedly")


def load_topk(k):
    """Combined tasks, out-of-fold fusion scores (paper + response, all), and each query's Top-K candidate indices."""
    from config import DATA, XDATA, read_jsonl
    from signals import archive_tasks, cv_fusion
    arts = read_articles()
    tasks = archive_tasks()
    F = np.load(f"{XDATA}/cache/boost_features_combined.npz", allow_pickle=True)["feats"].item()
    meta = json.load(open(f"{XDATA}/cache/boost_meta.json"))
    dense = lambda side: [x for x in F if x.startswith(side + "_") and x.endswith(("_max", "_col", "_mean", "_full"))
                          and not any(x.startswith(f"{side}_{s}") for s in ("luar", "luarcrud", "styledist", "star"))]
    lex = lambda side: [f"{side}_{x}" for x in ("bm25_uni", "bm25_bi", "char")]
    cols = dense("paper") + lex("paper") + dense("resp") + lex("resp") + ["resp_stylo_masked", "resp_star"]
    oof, _ = cv_fusion(meta, F, cols)
    out = []
    for t, m, s in zip(tasks, meta, oof):
        assert t["qid"] == m["qid"]
        order = np.argsort(-(s + RNG.uniform(0, 1e-9, len(s))))
        out.append({"qid": t["qid"], "pid": t["pid"], "pos": t["pos"], "review": t["review"], "fused": np.asarray(s),
                    "topk": [int(i) for i in order[:k]], "n": len(t["cands"]),
                    "cand": [(arts[c]["title"], arts[c]["abstract"], t["texts"][i]) for i, c in enumerate(t["cands"])]})
    return out


def rank_of(scores, pos):
    s = np.asarray(scores, float) + RNG.uniform(0, 1e-9, len(scores))
    return int((s > s[pos]).sum()) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["trial", "run", "eval"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1)
    ap.add_argument("--no-reasoning", action="store_true")
    ap.add_argument("--reasoning-effort", default=None, choices=[None, "low", "medium", "high"])
    ap.add_argument("--mode", default="yesno", choices=["yesno", "score"])
    ap.add_argument("--providers", default=None, help="comma-separated OpenRouter provider names to pin")
    ap.add_argument("--reasoning-budget", type=int, default=None, help="reasoning token budget (OpenRouter reasoning.max_tokens)")
    ap.add_argument("--complete-only", action="store_true", help="eval: only queries whose Top-K judgements are all cached")
    ap.add_argument("--qids", default=None, help="JSON file with a list of qids: restrict run/eval to this subset")
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    tag = a.model.replace("/", "_").replace(":", "_") + ("_bedrock" if BEDROCK else "") + (("_think" if os.environ.get("LLM_THINKING") == "1" else "_nothink") if LOCAL_URL else "") + ("_noreason" if a.no_reasoning else (f"_budget{a.reasoning_budget}" if a.reasoning_budget else (f"_reason-{a.reasoning_effort}" if a.reasoning_effort else ""))) + ("_score" if a.mode == "score" else "")
    cli = Client(a.model, f"{OUTDIR}/cache_{tag}.jsonl", a.max_tokens, a.no_reasoning, a.reasoning_effort, a.mode,
                 a.providers.split(",") if a.providers else None, a.reasoning_budget)
    T = load_topk(a.k)
    if a.qids:
        keep = set(json.load(open(a.qids)))
        T = [t for t in T if t["qid"] in keep]
        print(f"subset: {len(T)} of {len(keep)} qids from {a.qids}", flush=True)
        tag += f"_sub{len(T)}"
    if a.cmd == "eval" and a.complete_only:
        T = [t for t in T if all(cli._h(prompt(t["review"], *t["cand"][i], a.mode)) in cli.cache for i in t["topk"])]
        print(f"complete-only: {len(T)} queries with all top-{a.k} judgements cached")
        tag += f"_partial{len(T)}"

    if a.cmd == "trial":
        sel = T[:a.n]
        jobs = [(t["qid"], i, prompt(t["review"], *t["cand"][i], a.mode)) for t in sel for i in t["topk"]]
        print(f"trial: {len(sel)} queries, {len(jobs)} requests, model {a.model}", flush=True)
        t0 = time.time()
        with ThreadPoolExecutor(a.workers) as ex:
            recs = list(ex.map(lambda j: cli.judge(j[2]), jobs))
        dt = time.time() - t0
        served = {}
        for (q, i, _), r in zip(jobs, recs):
            served.setdefault(q, set()).add(r["served_model"])
        cost = sum((r["usage"] or {}).get("cost", 0) or 0 for r in recs)
        toks = sum((r["usage"] or {}).get("prompt_tokens", 0) or 0 for r in recs)
        rt = [r.get("reasoning_tokens") or 0 for r in recs]
        print(f"time {dt:.0f}s, prompt tokens {toks:,}, reported cost ${cost:.4f}, logprobs present: {np.mean([r['has_logprobs'] for r in recs]):.0%}, "
              f"reasoning tokens mean {np.mean(rt):.0f} max {max(rt)}, answers: {collections.Counter(r['text'][:5] for r in recs).most_common(4)}")
        print("served model(s) per query:", {q: sorted(x for x in s if x) for q, s in served.items()})
        print("providers:", sorted({str(r['provider']) for r in recs}))
        print("P(yes) distribution: min %.3f  median %.3f  max %.3f" % tuple(np.percentile([r["p_yes"] for r in recs], [0, 50, 100])))
        for t in sel:
            sc = {i: cli.judge(prompt(t["review"], *t["cand"][i], a.mode))["p_yes"] for i in t["topk"]}
            fr = t["topk"].index(t["pos"]) + 1 if t["pos"] in t["topk"] else None
            lr = sorted(sc, key=lambda i: -sc[i]).index(t["pos"]) + 1 if t["pos"] in t["topk"] else None
            print(f"  {t['qid']}: positive fusion rank {fr}, LLM rank within top-{a.k} {lr}, P(yes) of positive {sc.get(t['pos'], float('nan')):.3f}, "
                  f"max P(yes) {max(sc.values()):.3f}")
        # stability: re-send 5 requests and compare
        old_vals = [cli.cache[cli._h(j[2])]["p_yes"] for j in jobs[:5]]  # values before re-sending
        rep = [cli.judge(j[2], force=True) for j in jobs[:5]]
        print("stability (first 5 requests re-sent): " + ", ".join(f"{o:.3f}->{r['p_yes']:.3f}" for o, r in zip(old_vals, rep)))
        return

    if a.cmd == "run":
        jobs = [(t["qid"], i, prompt(t["review"], *t["cand"][i], a.mode)) for t in T for i in t["topk"]]
        todo = [j for j in jobs if cli._h(j[2]) not in cli.cache]
        print(f"run: {len(jobs)} requests, {len(todo)} not cached", flush=True)
        done = 0
        with ThreadPoolExecutor(a.workers) as ex:
            for _ in ex.map(lambda j: cli.judge(j[2]), todo):
                done += 1
                if done % 200 == 0:
                    print(done, "/", len(todo), flush=True)
        cost = sum((r["usage"] or {}).get("cost", 0) or 0 for r in cli.cache.values())
        print(f"done; cumulative reported cost in cache ${cost:.3f}")
        return

    # ---------------- eval ----------------
    import pandas as pd
    from sklearn.model_selection import GroupKFold
    from signals import metrics, boot
    N = [t["n"] for t in T]
    pos = [t["pos"] for t in T]
    groups = np.array([t["pid"] for t in T])
    llm = []
    missing = 0
    for t in T:
        s = np.full(t["n"], np.nan)  # NaN = no judgement; treated as a neutral (mean) LLM score inside the Top-K
        for i in t["topk"]:
            h = cli._h(prompt(t["review"], *t["cand"][i], a.mode))
            if h in cli.cache:
                s[i] = cli.cache[h]["p_yes"]
            else:
                missing += 1
        llm.append(s)
    if missing:
        print(f"WARNING: {missing} judgements missing from cache (treated as neutral LLM scores); run `run` first")

    def z(x):
        x = np.asarray(x, float)
        m, sd = np.nanmean(x), np.nanstd(x)
        out = (x - m) / sd if sd > 0 else x * 0
        return np.nan_to_num(out, nan=0.0)

    def fused_rank(t, s_llm, alpha):
        """Within Top-K: alpha*z(fusion)+(1-alpha)*z(llm); outside Top-K: fusion order below all Top-K."""
        k = t["topk"]
        f = z(t["fused"][k])
        l = z(s_llm[k])
        comb = alpha * f + (1 - alpha) * l
        full = t["fused"] - 1e6  # push non-top-K below
        full[k] = comb
        return rank_of(full, t["pos"])

    base = [rank_of(t["fused"], t["pos"]) for t in T]
    llm_only = [fused_rank(t, s, 0.0) for t, s in zip(T, llm)]
    pred = [None] * len(T)
    alphas = np.round(np.arange(0, 1.01, 0.1), 1)
    chosen = []
    for tr, te in GroupKFold(5).split(T, groups=groups):
        best, best_mrr = None, -1
        for al in alphas:
            mrr = np.mean([1 / fused_rank(T[i], llm[i], al) for i in tr])
            if mrr > best_mrr + 1e-12:
                best, best_mrr = al, mrr
        chosen.append(best)
        for i in te:
            pred[i] = fused_rank(T[i], llm[i], best)
    rows = {"fusion (paper + response, all) — baseline": base, f"LLM only within top-{a.k}": llm_only,
            f"fusion + LLM within top-{a.k} (alpha by CV)": pred}
    surv = np.mean([t["pos"] in t["topk"] for t in T])
    res = pd.DataFrame([{"method": k, **metrics(v, N)} for k, v in rows.items()])
    res.to_csv(f"{OUTDIR}/results_{tag}.csv", index=False, float_format="%.4f")
    d1, d2 = boot(llm_only, base), boot(pred, base)
    served = {}
    for r in cli.cache.values():
        served[r["served_model"]] = served.get(r["served_model"], 0) + 1
    cost = sum((r["usage"] or {}).get("cost", 0) or 0 for r in cli.cache.values())
    L = [f"# LLM reranking of the combined Top-{a.k} (model: `{a.model}`)", "",
         f"{len(T)} queries. Stage 1 = out-of-fold fusion of paper + response scores (the current best). "
         f"The positive is inside the fusion Top-{a.k} for {surv:.1%} of queries; the LLM cannot recover the rest. "
         f"Judgements: {sum(len(t['topk']) for t in T):,} (cached {len(cli.cache):,}); served models: {served}; reported cost ${cost:.2f}. "
         f"Chosen alpha per fold (1 = fusion only): {chosen}.", "",
         "| method | R@1 | R@5 | R@10 | R@20 | R@30 | MRR | median rank |", "|---|---|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        L.append(f"| {r.method} | {r['R@1']:.3f} | {r['R@5']:.3f} | {r['R@10']:.3f} | {r['R@20']:.3f} | {r['R@30']:.3f} | {r.MRR:.3f} | {r.median_rank:.0f} |")
    L += ["", "Paired bootstrap vs the fusion baseline. **Bold** = 95% CI excludes 0.", "",
          "| method | ΔMRR | ΔR@1 | ΔR@5 | ΔR@10 |", "|---|---|---|---|---|"]
    for name, d in ((f"LLM only within top-{a.k}", d1), (f"fusion + LLM within top-{a.k}", d2)):
        cells = []
        for kk in ("dMRR", "dR@1", "dR@5", "dR@10"):
            m, lo, hi = d[kk]
            c = f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            cells.append(f"**{c}**" if lo > 0 or hi < 0 else c)
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    open(f"{OUTDIR}/report_{tag}.md", "w").write("\n".join(L) + "\n")
    print(res.round(3).to_string(index=False))
    print("alpha per fold", chosen, "| survival", round(surv, 3))


if __name__ == "__main__":
    main()
