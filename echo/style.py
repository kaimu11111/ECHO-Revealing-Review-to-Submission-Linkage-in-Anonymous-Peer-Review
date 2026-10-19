"""Writing-style evidence: handcrafted stylometric features (first part) and deterministic content masking
(second part, formerly style.py). Run `python style.py` for the masked-retrieval diagnostic.

Masking docstring:Content-masked review -> author-response retrieval (`style_masked_text`).

Deterministic masking (regex + spaCy en_core_web_sm POS/NER + wordfreq), no paraphrasing, no LLM. Each masked token is
replaced 1:1 by a tag so sentence length, punctuation and paragraph breaks are preserved:
  [URL] [DOI] [EMAIL] [CITATION] [EQUATION] [NUMBER]           regex, before tagging
  [TECH_TERM]  tokens mixing letters and digits (GPT-4, COVID-19, IL-6), CamelCase, snake_case, dotted names;
               field-specific nouns (not review-genre words); rare (zipf < 4) adjectives / verbs / adverbs
  [ACRONYM]    two or more capitals (BERT, RNA, NIH) unless a common word in caps
  [ENTITY]     named-entity tokens (PERSON, ORG, GPE, LOC, NORP, FAC, PRODUCT, EVENT, WORK_OF_ART, LAW, LANGUAGE)
  [PROPER_NOUN] remaining PROPN tokens
Kept: function words, pronouns, auxiliaries, hedges, discourse markers, conjunctions, punctuation, question marks,
and "genre" content words used in >= 1% of reviews in at least 7 of the 8 fields (e.g. manuscript, authors, figure).

Model-, dataset- and software-name categories cannot be told apart deterministically without lexicons; they fall under
[TECH_TERM] / [ACRONYM] / [PROPER_NOUN] / [ENTITY].

Outputs: data .../style_masked/*.jsonl (originals untouched), outputs/style_masked/{examples.md, stats.json,
results.csv, report.md}
"""

import re
import numpy as np

FUNCTION_WORDS = """the of and to a in that is for it as with be on by this are not or from at which an have has
was were but can these their its also more than may such been there other between only both into would
should could however all any each most some so very well what when where while who how if then thus
because about after before over under through during without within our we i my me us you they them
his her one two first""".split()
HEDGES = """may might could possibly perhaps likely unlikely unclear appears appear seem seems seemingly suggest
suggests suggested somewhat rather potentially probably presumably apparently arguably relatively generally
largely tend tends""".split()
DISCOURSE = ["however", "moreover", "furthermore", "therefore", "thus", "additionally", "although",
             "nevertheless", "nonetheless", "in addition", "overall", "firstly", "secondly", "finally",
             "in particular", "for example", "for instance", "e.g.", "i.e.", "in contrast", "on the other hand",
             "consequently", "hence", "indeed", "specifically", "importantly"]
PUNCT = [",", ";", ":", "(", "-", "–", "—", "\"", "“", "'", "’", "!", "?", "/", "%", "[", "…"]
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\"“0-9])|\n+")
WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")
MASK = re.compile(r"\[(NAME|EMAIL|ORCID|PROFILE-URL|AFFILIATION|URL|DOI|PMID|SELF-[A-Z]+)\]")


def mattr(words, w=100):
    if len(words) < w:
        return len(set(words)) / max(1, len(words))
    return float(np.mean([len(set(words[i:i + w])) / w for i in range(0, len(words) - w + 1, 25)]))


def features(text):
    text = MASK.sub(" ", text)
    sents = [s for s in SENT.split(text) if s.strip()]
    words = WORD.findall(text)
    lw = [w.lower() for w in words]
    nw = max(1, len(words))
    nc = max(1, len(text))
    sl = [len(WORD.findall(s)) for s in sents] or [0]
    low = " " + " ".join(lw) + " "
    f = {
        "sent_len_mean": np.mean(sl), "sent_len_std": np.std(sl),
        "word_len_mean": np.mean([len(w) for w in words]) if words else 0,
        "long_word_frac": sum(len(w) >= 7 for w in words) / nw,
        "ttr_mattr100": mattr(lw),
        "question_per_sent": text.count("?") / max(1, len(sents)),
        "fp_sing_rate": sum(w in ("i", "me", "my", "mine", "myself") for w in lw) / nw,
        "fp_plur_rate": sum(w in ("we", "us", "our", "ours", "ourselves") for w in lw) / nw,
        "hedge_rate": sum(w in HEDGES for w in lw) / nw,
        "discourse_rate": sum(low.count(" " + m + " ") for m in DISCOURSE) / nw,
        "upper_char_frac": sum(c.isupper() for c in text) / nc,
        "digit_char_frac": sum(c.isdigit() for c in text) / nc,
        "contraction_rate": len(re.findall(r"\b\w+[’'](t|s|re|ve|ll|d|m)\b", text)) / nw,
        "brit_ise_rate": len(re.findall(r"\b\w+is(e|ed|es|ing|ation)\b", low)) / nw,
        "amer_ize_rate": len(re.findall(r"\b\w+iz(e|ed|es|ing|ation)\b", low)) / nw,
        "paren_rate": text.count("(") / nw,
    }
    for p in PUNCT:
        f[f"punct_{p}"] = text.count(p) / nw
    for w in FUNCTION_WORDS:
        f[f"fw_{w}"] = lw.count(w) / nw
    return f


FEATURE_NAMES = list(features("A sample sentence. Another one?").keys())


def matrix(texts):
    return np.array([[f[k] for k in FEATURE_NAMES] for f in map(features, texts)], dtype=float)


def zscore_by(X, ref):
    mu, sd = ref.mean(0), ref.std(0)
    sd[sd == 0] = 1
    return (X - mu) / sd


# ====================================================================================================
# content masking

import collections, json, os, random, re
import numpy as np
import pandas as pd
import spacy
from wordfreq import zipf_frequency
from config import DATA, XDATA, read_jsonl, write_jsonl, field_clusters
from config import ART_ROOT

# F1000_MASK_LEVEL: "standard" (default) or "aggressive" (every noun / proper noun / adjective masked, and verbs /
# adverbs unless frequent (zipf >= 5) or a hedge / discourse marker) - a dose-response check on residual content
LEVEL = os.environ.get("F1000_MASK_LEVEL", "standard")
_sfx = "" if LEVEL == "standard" else f"_{LEVEL}"
OUTDIR = f"{ART_ROOT}/style_masked{_sfx}"
MDIR = f"{XDATA}/style_masked{_sfx}"
RNG = np.random.default_rng(0)
N_BOOT = 2000

PRE = [  # (pattern, tag) applied in order on raw text
    (re.compile(r"\[(NAME|AFFILIATION|EMAIL|ORCID|PROFILE-URL|URL|DOI|PMID|SELF-[A-Z]+)\]"), None),  # existing masks
    (re.compile(r"(https?://|www\.)\S+", re.I), "URL"),
    (re.compile(r"\b(doi:\s*)?10\.\d{4,9}/\S+", re.I), "DOI"),
    (re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+"), "EMAIL"),
    # citations need an author-list marker or a parenthesised year, so "Since 2020" is not a citation
    (re.compile(r"\b[A-Z][\w'’-]+(\s+(and|&)\s+[A-Z][\w'’-]+|\s+et\s+al\.?)\s*,?\s*(\((19|20)\d{2}[a-z]?\)|(19|20)\d{2}[a-z]?)"), "CITATION"),
    (re.compile(r"\(\s*[A-Z][\w'’-]+,\s*(19|20)\d{2}[a-z]?\s*\)"), "CITATION"),
    (re.compile(r"\b[A-Z][\w'’-]+\s*\((19|20)\d{2}[a-z]?\)"), "CITATION"),
    (re.compile(r"\b[A-Z][\w'’-]+\s+et\s+al\.?"), "CITATION"),
    (re.compile(r"\[\d+([,–-]\s*\d+)*\]"), "CITATION"),
    (re.compile(r"\S*[=<>≤≥±×÷^√∑]\S*"), "EQUATION"),
    (re.compile(r"\b(?=[\w-]*[A-Za-z])(?=[\w-]*\d)[\w-]+\b"), "TECH_TERM"),
    (re.compile(r"(?<![\w\[])[-+]?\d+([.,]\d+)*\s?%?"), "NUMBER"),
]
ENT_TYPES = {"PERSON", "ORG", "GPE", "LOC", "NORP", "FAC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE"}
CONTENT = {"NOUN", "ADJ", "VERB", "ADV"}
PH = re.compile(r"^MASKx([A-Z_]+)$")
KEEP_STYLE = set(HEDGES) | {w for m in DISCOURSE for w in m.split()}
TAG = re.compile(r"\[([A-Z_]+)\]")


def genre_words():
    """Lower-case words used in >= 1% of reviews in at least 7 of 8 fields (review vocabulary, not topic)."""
    p = f"{XDATA}/cache/genre_words.json"
    if os.path.exists(p):
        return set(json.load(open(p)))
    from config import BENCH
    if os.path.exists(f"{BENCH}/review_genre_words.json"):  # released with the benchmark (needs the full review corpus to recompute)
        return set(json.load(open(f"{BENCH}/review_genre_words.json")))
    cl = field_clusters(8)
    df, n = collections.defaultdict(collections.Counter), collections.Counter()
    for r in read_jsonl(f"{DATA}/reviews.jsonl"):
        c = cl.get(r["article_key"])
        if c is None:
            continue
        n[c] += 1
        df[c].update(set(re.findall(r"[a-z]+", " ".join(b["raw"] for b in r["blocks"]).lower())))
    words = {w for w in set().union(*[set(d) for d in df.values()])
             if sum(df[c][w] >= 0.01 * n[c] for c in n) >= 7}
    json.dump(sorted(words), open(p, "w"))
    return words


class Masker:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        self.genre = genre_words()

    def pre(self, text):
        """Regex masks -> single-token placeholders (MASKxTAG); a space is added only where the placeholder would
        otherwise touch a letter or digit, so punctuation spacing is unchanged."""
        for pat, tag in PRE:
            def rep(m, src=text, tag=tag):
                t = m.group(1).replace("-", "_") if tag is None else tag
                left = " " if m.start() > 0 and src[m.start() - 1].isalnum() else ""
                right = " " if m.end() < len(src) and src[m.end()].isalnum() else ""
                return f"{left}MASKx{t}{right}"
            text = pat.sub(rep, text)
        return text

    def classify(self, tok):
        t = tok.text
        if PH.match(t):
            return PH.match(t).group(1)
        if tok.is_punct or tok.is_space or not any(ch.isalpha() for ch in t):
            return None
        low = t.lower()
        if re.search(r"[a-z][A-Z]|_|\w\.\w", t):  # CamelCase, snake_case, dotted names
            return "TECH_TERM"
        if sum(ch.isupper() for ch in t) >= 2 and not (t.isupper() and zipf_frequency(low, "en") >= 4.0 and len(t) > 3):
            return "ACRONYM"
        if LEVEL == "aggressive":
            if tok.pos_ in ("NOUN", "PROPN", "ADJ", "NUM") or tok.ent_type_ in ENT_TYPES:
                return "CONTENT"
            if tok.pos_ in ("VERB", "ADV") and low not in KEEP_STYLE and zipf_frequency(low, "en") < 5.0:
                return "CONTENT"
            return None
        if low in self.genre:  # review-genre words (figure, methods, table, ...) survive even when tagged PROPN
            return None
        if tok.ent_type_ in ENT_TYPES:
            return "ENTITY"
        if tok.pos_ == "PROPN":
            return "PROPER_NOUN"
        if tok.pos_ == "NOUN" and low not in self.genre:
            return "TECH_TERM"
        if tok.pos_ in CONTENT and low not in self.genre and zipf_frequency(low, "en") < 4.0:
            return "TECH_TERM"
        return None

    def mask_many(self, texts):
        """Returns list of (masked_text, stats Counter)."""
        out = []
        for doc in self.nlp.pipe([self.pre(t) for t in texts], batch_size=64):
            st = collections.Counter()
            parts = []
            for tok in doc:
                tag = self.classify(tok)
                is_ph = bool(PH.match(tok.text))
                if not (tok.is_punct or tok.is_space):
                    st["tokens"] += 1
                    if tag:
                        st["masked"] += 1
                        st[f"tag_{tag}"] += 1
                    if not is_ph and tok.pos_ in ("NOUN", "PROPN"):
                        st["nouns_propn"] += 1
                        st["nouns_propn_masked"] += bool(tag)
                parts.append((f"[{tag}]" if tag else tok.text) + tok.whitespace_)
            m = "".join(parts)
            m = re.sub(r" *\n *", "\n", m).strip()
            out.append((m, st))
        return out


def render(t):
    """Neutralise mask tags for style features / char n-grams: every tag becomes the same short word."""
    return TAG.sub("xx", t)


def words(t):
    return re.findall(r"\[[A-Z_]+\]|[A-Za-z]+(?:['’][a-z]+)?", t)


def metrics(ranks, sizes):
    r, N = np.asarray(ranks, float), np.asarray(sizes, float)
    return {"n": len(r), "R@1": np.mean(r <= 1), "R@5": np.mean(r <= 5), "R@10": np.mean(r <= 10), "MRR": np.mean(1 / r),
            "median_rank": float(np.median(r)), "mean_pct_rank": float(np.mean((r - 1) / (N - 1)))}


def boot(a, b):
    out = {}
    for name, f in (("dMRR", lambda r: 1 / r), ("dR@1", lambda r: (r <= 1) * 1.0), ("dR@5", lambda r: (r <= 5) * 1.0),
                    ("dR@10", lambda r: (r <= 10) * 1.0)):
        d = f(np.asarray(a, float)) - f(np.asarray(b, float))
        m = d[RNG.integers(0, len(d), (N_BOOT, len(d)))].mean(1)
        out[name] = (d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5))
    return out


def main():
    from scores import load_tasks, original_scores, score_tables, rank_of_pos  # imported here: scores.py imports this module
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(MDIR, exist_ok=True)
    tasks = load_tasks()
    reviews = {t["qid"]: t["review"] for t in tasks}
    uniq = sorted({x for t in tasks for x in t["texts"]})

    # ---------- masking (cached) ----------
    mpath = f"{MDIR}/masked.json"
    if os.path.exists(mpath):
        M = json.load(open(mpath))
    else:
        mk = Masker()
        rv = mk.mask_many([reviews[q] for q in sorted(reviews)])
        rs = mk.mask_many(uniq)
        M = {"reviews": {q: {"text": m, "stats": dict(s)} for q, (m, s) in zip(sorted(reviews), rv)},
             "responses": {u: {"text": m, "stats": dict(s)} for u, (m, s) in zip(uniq, rs)}}
        json.dump(M, open(mpath, "w"))
        write_jsonl(f"{MDIR}/reviews.jsonl", [{"qid": q, "style_masked_text": v["text"]} for q, v in M["reviews"].items()])
    mrev = {q: v["text"] for q, v in M["reviews"].items()}
    mresp = {u: v["text"] for u, v in M["responses"].items()}

    # ---------- validation stats ----------
    def agg(kind, originals):
        st = collections.Counter()
        for v in M[kind].values():
            st.update(v["stats"])
        orig_vocab = {w.lower() for t in originals for w in words(t)}
        mask_vocab = {w.lower() for v in M[kind].values() for w in words(v["text"])}
        return {"texts": len(M[kind]), "pct_tokens_masked": st["masked"] / st["tokens"],
                "pct_nouns_propn_masked": st["nouns_propn_masked"] / max(1, st["nouns_propn"]),
                "vocab_before": len(orig_vocab), "vocab_after": len(mask_vocab),
                "avg_words_before": np.mean([len(words(t)) for t in originals]),
                "avg_words_after": np.mean([len(words(v["text"])) for v in M[kind].values()]),
                "avg_chars_before": np.mean([len(t) for t in originals]),
                "avg_chars_after": np.mean([len(v["text"]) for v in M[kind].values()]),
                "tag_counts": {k[4:]: v for k, v in st.items() if k.startswith("tag_")}}
    stats = {"reviews": agg("reviews", [reviews[q] for q in M["reviews"]]),
             "responses": agg("responses", list(M["responses"].keys()))}
    json.dump(stats, open(f"{OUTDIR}/stats.json", "w"), indent=2, default=float)

    # ---------- 50 random examples (review + its positive rebuttal) ----------
    random.seed(0)
    ex = random.sample(tasks, 50)

    def cut(t, n=220):
        w = t.split(" ")
        return " ".join(w[:n]) + (" …" if len(w) > n else "")
    E = ["# style_masked_text — 50 random review / rebuttal pairs", "",
         "Each pair is an anonymous review and the author response written by the same person (the positive). "
         "Texts are truncated to about 220 words for readability; masking is applied to the full text.", ""]
    for i, t in enumerate(ex, 1):
        pos_text = t["texts"][t["pos"]]
        E += [f"## Example {i} ({t['qid']})", "", "**ORIGINAL REVIEW**", "", "> " + cut(t["review"]).replace("\n", "\n> "), "",
              "**MASKED REVIEW**", "", "> " + cut(mrev[t["qid"]]).replace("\n", "\n> "), "",
              "**ORIGINAL REBUTTAL**", "", "> " + cut(pos_text).replace("\n", "\n> "), "",
              "**MASKED REBUTTAL**", "", "> " + cut(mresp[pos_text]).replace("\n", "\n> "), ""]
    open(f"{OUTDIR}/examples.md", "w").write("\n".join(E) + "\n")

    # ---------- retrieval: original vs masked ----------
    S0 = original_scores(tasks)
    cand_m = {(t["qid"], i): mresp[x] for t in tasks for i, x in enumerate(t["texts"])}
    fit_m = list(mrev.values()) + list(mresp.values())
    S1 = score_tables(tasks, mrev, cand_m, fit_m, with_topic=True, emb_name=f"resp_emb_masked{_sfx}", render=render)
    sizes = [len(t["cands"]) for t in tasks]
    R = collections.defaultdict(list)
    for t in tasks:
        q, p = t["qid"], t["pos"]
        R["random"].append(rank_of_pos(RNG.uniform(size=len(t["cands"])), p, RNG))
        for m in ("stylo", "char", "topic"):
            R[f"original_{m}"].append(rank_of_pos(S0[q][m], p, RNG))
            R[f"masked_{m}"].append(rank_of_pos(S1[q][m], p, RNG))
    rows = [{"representation": k, **metrics(v, sizes)} for k, v in R.items()]
    chance = {"R@1": np.mean([1 / n for n in sizes]), "R@5": np.mean([5 / n for n in sizes]),
              "R@10": np.mean([10 / n for n in sizes]), "MRR": np.mean([sum(1 / k for k in range(1, n + 1)) / n for n in sizes])}
    rows.append({"representation": "chance_expected", **chance})
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTDIR}/results.csv", index=False, float_format="%.4f")
    comps = [("masked_stylo", "original_stylo"), ("masked_char", "original_char"),
             ("masked_stylo", "random"), ("masked_char", "random"), ("masked_char", "masked_stylo"),
             ("masked_topic", "original_topic")]
    B = {(a, b): boot(R[a], R[b]) for a, b in comps}
    pd.DataFrame([{"a": a, "b": b, "metric": k, "diff": v[0], "ci_lo": v[1], "ci_hi": v[2]}
                  for (a, b), d in B.items() for k, v in d.items()]).to_csv(f"{OUTDIR}/paired_bootstrap.csv", index=False, float_format="%.4f")

    # ---------- report ----------
    def row(name, label):
        r = df[df.representation == name].iloc[0]
        return f"| {label} | {r['R@1']:.3f} | {r['R@5']:.3f} | {r['R@10']:.3f} | {r['MRR']:.3f} | {r['median_rank']:.0f} | {r['mean_pct_rank']:.3f} |"
    sr, ss = stats["reviews"], stats["responses"]
    L = [f"# Content-masked review → author-response linkage (mask level: {LEVEL})", "",
         f"Same {len(tasks)} queries and candidate pools as the rebuttal-archive experiment (median pool {int(np.median(sizes))} responses). "
         "Both the review and every candidate response are replaced by `style_masked_text`. The masking is deterministic (regex + spaCy POS/NER + wordfreq; no paraphrase, no LLM). "
         "Masked texts are written to a separate file; the original data are unchanged.", "",
         "## Masking validation", "",
         "| | reviews | responses |", "|---|---|---|",
         f"| texts | {sr['texts']} | {ss['texts']} |",
         f"| tokens masked | {sr['pct_tokens_masked']:.1%} | {ss['pct_tokens_masked']:.1%} |",
         f"| nouns + proper nouns masked | {sr['pct_nouns_propn_masked']:.1%} | {ss['pct_nouns_propn_masked']:.1%} |",
         f"| vocabulary size (word types) before → after | {sr['vocab_before']:,} → {sr['vocab_after']:,} | {ss['vocab_before']:,} → {ss['vocab_after']:,} |",
         f"| mean length, words before → after | {sr['avg_words_before']:.0f} → {sr['avg_words_after']:.0f} | {ss['avg_words_before']:.0f} → {ss['avg_words_after']:.0f} |",
         f"| mean length, characters before → after | {sr['avg_chars_before']:.0f} → {sr['avg_chars_after']:.0f} | {ss['avg_chars_before']:.0f} → {ss['avg_chars_after']:.0f} |",
         "", "Mask tags used (reviews / responses): " +
         ", ".join(f"{k} {sr['tag_counts'].get(k, 0):,} / {ss['tag_counts'].get(k, 0):,}" for k in sorted(set(sr["tag_counts"]) | set(ss["tag_counts"]))), "",
         "Word counts stay about the same because each masked token becomes one tag. The character count changes because tags are longer or shorter than the words they replace. "
         "For style features and char n-grams every tag is rendered as the same placeholder word `xx`, so tag spelling carries no information. "
         "50 random review/rebuttal pairs are in `examples.md`.", "",
         "**Content-removal check (diagnostic, not a linkage method):** SPECTER2 topic similarity on the masked texts. If masking removed topic, this should fall towards chance.", "",
         "| representation | R@1 | R@5 | R@10 | MRR | median rank | mean percentile rank |", "|---|---|---|---|---|---|---|",
         row("original_topic", "topic (SPECTER2), original"), row("masked_topic", "topic (SPECTER2), masked"), "",
         "## Linkage after masking", "",
         "| representation | R@1 | R@5 | R@10 | MRR | median rank | mean percentile rank |", "|---|---|---|---|---|---|---|",
         row("original_stylo", "handcrafted style — original rebuttal"), row("masked_stylo", "handcrafted style — masked rebuttal"),
         row("original_char", "char n-gram — original rebuttal"), row("masked_char", "char n-gram — masked rebuttal"),
         row("random", "random (observed)"),
         f"| chance (expected) | {chance['R@1']:.3f} | {chance['R@5']:.3f} | {chance['R@10']:.3f} | {chance['MRR']:.3f} | — | 0.500 |", "",
         "## Paired bootstrap (2,000 resamples over queries)", "", "**Bold** = 95% CI excludes 0.", "",
         "| comparison | ΔMRR | ΔR@1 | ΔR@5 | ΔR@10 |", "|---|---|---|---|---|"]
    for (a, b), d in B.items():
        cells = []
        for k in ("dMRR", "dR@1", "dR@5", "dR@10"):
            m, lo, hi = d[k]
            s = f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            cells.append(f"**{s}**" if lo > 0 or hi < 0 else s)
        L.append(f"| {a} − {b} | " + " | ".join(cells) + " |")
    L += ["", "## Interpretation", "", "INTERP_PLACEHOLDER", ""]
    text = "\n".join(L) + "\n"
    if os.path.exists(f"{OUTDIR}/interpretation.md"):
        text = text.replace("INTERP_PLACEHOLDER", open(f"{OUTDIR}/interpretation.md").read().strip())
    open(f"{OUTDIR}/report.md", "w").write(text)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "tag_counts"} for k, v in stats.items()}, indent=1, default=float))
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
