"""Paths and small helpers for the released R2S-Bench / ECHO code.

Layout (relative to the release root):
  benchmark/   the public benchmark (queries, candidates, gold, rebuttals, articles, fields, splits)
  work/cache/  feature and embedding caches written by the scripts (shipped caches let you skip GPU work)
  outputs/     reports and LLM judgement caches

No environment variables are needed. The private construction pipeline (construction/) used a different
layout; the names below keep the method scripts unchanged.
"""
import json, os, re, unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = f"{ROOT}/benchmark"
DATA = XDATA = f"{ROOT}/work"
OUT = f"{ROOT}/outputs"
ART_ROOT = OUT  # analysis reports (style.py, scores.py, analysis/)
CORPUS, EXP = "platform", "rebuttal_archive"
CONCURRENT_DAYS, VENUE_K, RESPONSE_ARCHIVE, ARCHIVE_LAG_DAYS = 0, 8, True, 182
for _d in (f"{XDATA}/cache", f"{XDATA}/private", OUT):
    os.makedirs(_d, exist_ok=True)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_articles():
    """article_id -> {article_key, article_id, title, abstract}. `article_key` mirrors the id so that code written
    against the construction pipeline (which keyed articles by DOI) works unchanged on the released opaque ids."""
    return {a["article_id"]: {"article_key": a["article_id"], **a} for a in read_jsonl(f"{BENCH}/articles.jsonl")}


def field_clusters(k=8):
    """article_id -> research area (0..7), as released in benchmark/fields.json."""
    assert k == 8, "the released benchmark uses the 8-area partition"
    return json.load(open(f"{BENCH}/fields.json"))


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def name_key(given, surname):
    g = re.sub(r"[^a-z]", "", strip_accents(given or "").lower())
    s = re.sub(r"[^a-z]", "", strip_accents(surname or "").lower())
    return f"{g}|{s}" if g and s else None
