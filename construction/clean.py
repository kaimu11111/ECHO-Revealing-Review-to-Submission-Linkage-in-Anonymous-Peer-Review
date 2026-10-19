"""Review / author-response de-identification.

main   : drop template checklist, confirmation statement, 'Reviewer Expertise' metadata, signatures;
         mask contributor names, emails, ORCIDs, identity/profile URLs, affiliation strings.
strict : main + mask all URLs/DOIs/PMIDs, citations to the reviewer's name, titles of the reviewer's own
         F1000 articles (>= 6-word overlap) and self-referential sentences ("our previous study ...").
"""
import collections, re
from common import strip_accents

ANSWERS = re.compile(r"^(yes|no|partly|not applicable|n/?a|i cannot comment.*|no source data required|"
                     r"not applicable.*|unclear|i don.t know)\.?$", re.I)
CONFIRM = re.compile(r"^(i|we) (confirm that (i|we) have read this submission|have read this submission)", re.I)
EXPERTISE = re.compile(r"^reviewer expertise\s*:?\s*(.*)$", re.I)
SIGNOFF = re.compile(r"^(sincerely|best( wishes| regards)?|kind regards|regards|with (best|kind) regards|yours( sincerely| faithfully)?|thanks?( you)?|cheers)\b[,.!]?", re.I)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")
ORCID = re.compile(r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b")
URL = re.compile(r"(https?://|www\.)\S+", re.I)
ID_URL = re.compile(r"(https?://)?(www\.)?(orcid\.org|researchgate\.net|linkedin\.com|scholar\.google\.[a-z.]+|twitter\.com|x\.com|"
                    r"facebook\.com|academia\.edu|publons\.com|webofscience\.com|loop\.frontiersin\.org|scopus\.com/authid)\S*", re.I)
DOI = re.compile(r"\b(doi:\s*|https?://(dx\.)?doi\.org/)?10\.\d{4,9}/\S+", re.I)
PMID = re.compile(r"\bPMID:?\s*\d+|\bPMC\d+", re.I)
SELF_REF = re.compile(
    r"\b(our|my)\s+(own\s+|previous\s+|prior\s+|recent\s+|earlier\s+|published\s+|past\s+)*(work|works|study|studies|paper|papers|"
    r"article|articles|group|lab|laboratory|team|research|findings|data|experience|publication|publications|results|review|"
    r"manuscript|tool|software|package|method|cohort|trial)\b"
    r"|\bwe\s+(have\s+)?(previously|recently|also)?\s*(shown|showed|reported|published|described|demonstrated|developed|proposed)\b"
    r"|\bin\s+our\s+(hands|hospital|institution|centre|center|clinic|unit|department|country|setting)\b"
    r"|\bi\s+(have\s+)?(previously|recently)?\s*(published|developed|reported|proposed|authored|co-authored)\b"
    r"|\b(the\s+)?(present|current)\s+reviewer'?s?\s+(own\s+)?(work|study|paper|group)\b", re.I)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\"“])")


def is_template_question(b, qfreq):
    return b.endswith("?") and qfreq.get(b, 0) >= 20


def name_patterns(contribs):
    """Regexes for contributor names: full names, surnames (>= 3 chars), given names (>= 3 chars)."""
    pats = []
    for c in contribs:
        sur, giv = c.get("surname") or "", c.get("given_names") or ""
        for s in {sur, strip_accents(sur)}:
            if len(s) >= 3:
                pats.append(re.escape(s))
        for g in set(giv.split()) | set(strip_accents(giv).split()):
            g = g.strip(".")
            if len(g) >= 3:
                pats.append(re.escape(g))
    if not pats:
        return None
    pats = sorted(set(pats) | {p.upper() for p in pats}, key=len, reverse=True)
    # case-sensitive so that names that are also common words ("Mark", "Long", "Brown") only match capitalised
    return re.compile(r"(?<![\w])(" + "|".join(pats) + r")(?![\w])")


def aff_patterns(contribs):
    segs = set()
    for c in contribs:
        for a in c.get("aff") or []:
            for s in a.split(","):
                s = s.strip()
                if len(s) >= 8 and not re.fullmatch(r"[A-Z]{2,3}|USA|UK", s):
                    segs.add(s)
    if not segs:
        return None
    return re.compile("|".join(re.escape(s) for s in sorted(segs, key=len, reverse=True)), re.I)


def english_ok(text):
    letters = [ch for ch in text if ch.isalpha()]
    if len(letters) < 50:
        return False
    ascii_frac = sum(ch.isascii() for ch in letters) / len(letters)
    words = re.findall(r"[a-z]+", text.lower())
    common = sum(w in {"the", "and", "of", "to", "is", "in", "that", "this"} for w in words)
    return ascii_frac > 0.9 and common / max(1, len(words)) > 0.08


def mask_title_overlap(text, titles, n=6):
    """Mask spans of >= n consecutive words shared with any of the given titles."""
    toks = list(re.finditer(r"\S+", text))
    norm = [re.sub(r"[^a-z0-9]", "", t.group().lower()) for t in toks]
    hit = [False] * len(toks)
    count = 0
    for title in titles:
        tw = [w for w in (re.sub(r"[^a-z0-9]", "", x.lower()) for x in title.split()) if w]
        grams = {tuple(tw[i:i + n]) for i in range(len(tw) - n + 1)}
        for i in range(len(norm) - n + 1):
            if grams and tuple(norm[i:i + n]) in grams:
                hit[i:i + n] = [True] * n
                count += 1
    if not count:
        return text, 0
    out, prev_end, prev_hit = [], 0, False
    for t, h in zip(toks, hit):
        if h and prev_hit:
            prev_end = t.end()
            continue
        out.append(text[prev_end:t.start()])
        out.append("[SELF-TITLE]" if h else t.group())
        prev_end, prev_hit = t.end(), h
    out.append(text[prev_end:])
    return "".join(out), count


def clean_review(rv, qfreq, self_titles=()):
    """Return dict(main=..., strict=..., stats=Counter, expertise=str)."""
    st = collections.Counter()
    kept, skip_next, expertise = [], False, ""
    bl = [b["raw"] for b in rv["blocks"]]
    i = 0
    while i < len(bl):
        b = bl[i]
        if is_template_question(b, qfreq):
            st["checklist_q"] += 1
            if i + 1 < len(bl) and (ANSWERS.match(bl[i + 1]) or len(bl[i + 1].split()) <= 3):
                st["checklist_a"] += 1
                i += 1
            i += 1
            continue
        if CONFIRM.match(b):
            st["confirm_stmt"] += 1
            i += 1
            continue
        m = EXPERTISE.match(b)
        if m:
            st["expertise"] += 1
            expertise = m.group(1)
            if not expertise and i + 1 < len(bl):
                expertise = bl[i + 1]
                i += 1
            i += 1
            continue
        kept.append(b)
        i += 1

    # signatures: trailing short blocks with sign-offs or contributor names
    npat = name_patterns(rv["contribs"])
    while kept:
        last = kept[-1]
        short = len(last.split()) <= 12
        if short and (SIGNOFF.match(last) or (npat and npat.search(last)) or re.match(r"^(dr\.?|prof\.?)\s", last, re.I)):
            st["signature_block"] += 1
            kept.pop()
        else:
            break

    text = "\n".join(kept)

    def sub(pat, rep, s, key):
        s2, n = pat.subn(rep, s)
        st[key] += n
        return s2

    text = sub(EMAIL, "[EMAIL]", text, "email")
    text = sub(ORCID, "[ORCID]", text, "orcid")
    text = sub(ID_URL, "[PROFILE-URL]", text, "profile_url")
    apat = aff_patterns(rv["contribs"])
    if apat:
        text = sub(apat, "[AFFILIATION]", text, "affiliation")
    if npat:
        text = sub(npat, "[NAME]", text, "name")
    main = text

    s = text
    s = sub(URL, "[URL]", s, "strict_url")
    s = sub(DOI, "[DOI]", s, "strict_doi")
    s = sub(PMID, "[PMID]", s, "strict_pmid")
    # citations naming the (masked) reviewer: "[NAME] et al. (2019)", "([NAME], 2019)", "[NAME] and X, 2018"
    s = sub(re.compile(r"\(?\[NAME\](\s+(et al\.?|and|&)\s*[\w\[\].]*)?,?\s*\(?\d{4}[a-z]?\)?\)?"), "[SELF-CITATION]", s, "strict_self_citation")
    s, n = mask_title_overlap(s, self_titles)
    st["strict_self_title"] += n
    paras = []
    for para in s.split("\n"):
        out = []
        for se in SENT_SPLIT.split(para):
            if SELF_REF.search(se):
                st["strict_self_reference_sentence"] += 1
                out.append("[SELF-REFERENCE]")
            else:
                out.append(se)
        paras.append(" ".join(out))
    strict = "\n".join(paras)
    return {"main": main, "strict": strict, "stats": st, "expertise": expertise}


def ngrams(text, n=5):
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def clean_response(resp, review_texts, contribs_to_mask):
    """Author response with italic (quoted) spans removed, plus blocks overlapping reviewer text dropped."""
    st = collections.Counter()
    rev_grams = set()
    for t in review_texts:
        rev_grams |= ngrams(t)
    kept = []
    for b in resp["blocks"]:
        t = b["kept"]
        if b["raw"] != b["kept"]:
            st["italic_removed"] += 1
        if not t or len(t.split()) < 3:
            continue
        g = ngrams(t)
        if g and len(g & rev_grams) / len(g) >= 0.3:
            st["quoted_block_dropped"] += 1
            continue
        if re.match(r'^\s*["“]', t) and re.search(r'["”]\s*$', t):
            st["quoted_block_dropped"] += 1
            continue
        kept.append(t)
    npat = name_patterns(contribs_to_mask)
    while kept and len(kept[-1].split()) <= 12 and (SIGNOFF.match(kept[-1]) or (npat and npat.search(kept[-1]))):
        kept.pop()
        st["signature_block"] += 1
    text = "\n".join(kept)
    for pat, rep in ((EMAIL, "[EMAIL]"), (ORCID, "[ORCID]"), (ID_URL, "[PROFILE-URL]")):
        text = pat.sub(rep, text)
    if npat:
        text, n = npat.subn("[NAME]", text)
        st["name"] += n
    return text, st
