import json, os, re, unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Corpus: "" = F1000Research only (original); "platform" = all F1000-platform journals with signed public review
CORPUS = os.environ.get("F1000_CORPUS", "")
DATA = f"{ROOT}/data_{CORPUS}" if CORPUS else f"{ROOT}/data"
JOURNALS = {"": ["F1000Res"],
            "platform": ["F1000Res", "Wellcome Open Res", "Gates Open Res", "HRB Open Res", "Open Res Eur",
                         "AAS Open Res", "NIHR Open Res", "Open Res Afr", "Emerald Open Res", "Health Open Res",
                         "MNI Open Res"]}[CORPUS]
# Experiment variants write their queries / pools / review embeddings / benchmark / outputs to separate
# directories so that the original (non-concurrent) results stay reproducible. Default "" = original design.
EXP = os.environ.get("F1000_EXP", "")
# >0: ICLR-style concurrent setting - P and every candidate must be published within +-N days of the review
CONCURRENT_DAYS = int(os.environ.get("F1000_CONCURRENT_DAYS", "0"))
# >0: ICLR-style venue mode - a "conference" is (field cluster, publication year); K = number of field clusters
VENUE_K = int(os.environ.get("F1000_VENUE_K", "0"))
# 1 (with VENUE_K): "rebuttal archive" - the reviewer's own article / author response may come from any year of the same
# field, as long as it was public by review date + ARCHIVE_LAG_DAYS; candidates are that field's archive up to the same date
RESPONSE_ARCHIVE = os.environ.get("F1000_RESPONSE_ARCHIVE", "0") == "1"
ARCHIVE_LAG_DAYS = 182
XDATA = f"{DATA}/exp/{EXP}" if EXP else DATA
_tag = "_".join(x for x in (CORPUS, EXP) if x)
BENCH = f"{ROOT}/benchmark/{_tag}" if _tag else f"{ROOT}/benchmark"
OUT = f"{ROOT}/outputs/{_tag}" if _tag else f"{ROOT}/outputs"
for _d in (f"{DATA}/xml", f"{DATA}/raw", f"{XDATA}/private", f"{XDATA}/cache", BENCH, OUT):
    os.makedirs(_d, exist_ok=True)

STOP_AFF = set("""of the and for de la des du di da del der und at in university universidad universite università
universitat institute institut department dept school college faculty center centre hospital research
laboratory lab sciences science medical medicine national state""".split())


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f]


def write_jsonl(p, rows):
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s or "") if not unicodedata.combining(c))


def name_key(given, surname):
    """'first-given-token|surname' with accents/punctuation removed; None if the given name is only an initial."""
    g = re.sub(r"[^a-z\s-]", " ", strip_accents(given).lower()).replace("-", " ").split()
    s = re.sub(r"[^a-z]", "", strip_accents(surname).lower())
    if not g or not s or len(g[0]) < 2:
        return None
    return f"{g[0]}|{s}"


def aff_tokens(affs):
    toks = set()
    for a in affs or []:
        for t in re.findall(r"[a-z]{3,}", strip_accents(a).lower()):
            if t not in STOP_AFF:
                toks.add(t)
    return toks


def pub_date(art):
    return art.get("v1_date") or art.get("record_date")


def field_clusters(k):
    """KMeans(k) over SPECTER2 paper embeddings of articles with title + >= 50-word abstract; cached."""
    import numpy as np
    from sklearn.cluster import KMeans
    p = f"{DATA}/cache/field_clusters_k{k}.json"
    if os.path.exists(p):
        return json.load(open(p))
    arts = {a["article_key"]: a for a in read_jsonl(f"{DATA}/articles.jsonl")}
    E = np.load(f"{DATA}/cache/papers.npy")
    keys = json.load(open(f"{DATA}/cache/papers_ids.json"))
    ok = [i for i, kk in enumerate(keys) if arts[kk]["title"] and len(arts[kk]["abstract"].split()) >= 50]
    lab = KMeans(k, n_init=5, random_state=0).fit(E[ok]).labels_
    out = {keys[i]: int(l) for i, l in zip(ok, lab)}
    json.dump(out, open(p, "w"))
    return out


def venue_of(art, clusters):
    """(field cluster, publication year) - the ICLR-style 'conference edition' an article belongs to."""
    c = clusters.get(art["article_key"])
    return None if c is None else (c, pub_date(art)[:4])


GENOME_NOTE = re.compile(r"(the )?(chromosome-level |reference )?genome (sequence|assembly) of", re.I)


def is_template_article(art):
    """Darwin Tree of Life genome notes (Wellcome Open Research): fixed-template consortium abstracts, unusable for
    topic or style comparison. Excluded only in the platform corpus so that F1000-only results stay reproducible."""
    return CORPUS == "platform" and bool(GENOME_NOTE.match(art["title"] or ""))
