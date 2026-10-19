"""Parse PMC F1000Research JATS XML into articles / authors / reviews / author-responses (JSONL).

Identity fields (names, ORCIDs, affiliations) are kept here for matching only; they never reach the benchmark files.
"""
import glob, json, os, re
import xml.etree.ElementTree as ET

from common import ROOT, DATA
XLINK = "{http://www.w3.org/1999/xlink}href"
REVIEW_TYPES = {"peer-review", "reviewer-report", "referee-report"}


def text(e):
    return re.sub(r"\s+", " ", "".join(e.itertext())).strip() if e is not None else ""


def text_skip(e, skip_tags):
    """Text of e, excluding the subtrees of any descendant whose tag is in skip_tags (tails kept)."""
    out = [e.text or ""]
    for c in e:
        if c.tag not in skip_tags:
            out.append(text_skip(c, skip_tags))
        out.append(c.tail or "")
    return "".join(out)


def blocks(body, skip_tags=()):
    """Paragraph-level blocks in document order. Nested <p> are emitted separately (parent excludes them)."""
    res = []
    if body is None:
        return res
    for p in body.iter():
        if p.tag not in ("p", "title"):
            continue
        has_inner = any(d.tag == "p" for d in p.iter() if d is not p)
        raw = text_skip(p, {"p", "list"} if has_inner else set())
        kept = text_skip(p, ({"p", "list"} if has_inner else set()) | set(skip_tags))
        raw = re.sub(r"\s+", " ", raw).strip()
        kept = re.sub(r"\s+", " ", kept).strip()
        if raw:
            res.append({"raw": raw, "kept": kept})
    return res


def norm_orcid(s):
    if not s:
        return None
    m = re.search(r"(\d{4}-\d{4}-\d{4}-\d{3}[\dX])", s)
    return m.group(1) if m else None


def contribs(group_parent, scope):
    """Contributors under group_parent/contrib-group with resolved affiliations."""
    affs = {a.get("id"): re.sub(r"^\s*\d+\s*", "", text_skip(a, {"label"})).strip() for a in scope.iter("aff")}
    out = []
    for cg in group_parent.findall("contrib-group"):
        for c in cg.findall("contrib"):
            if c.get("contrib-type") not in (None, "author"):
                continue
            nm = c.find("name")
            sur = text(nm.find("surname")) if nm is not None else ""
            giv = text(nm.find("given-names")) if nm is not None else ""
            if nm is None and c.find("collab") is not None:
                continue
            orcid = None
            for cid in c.findall("contrib-id"):
                if cid.get("contrib-id-type") == "orcid":
                    orcid = norm_orcid(cid.text)
            aff = [affs.get(x.get("rid"), "") for x in c.findall("xref") if x.get("ref-type") == "aff"]
            aff += [text(a) for a in c.findall("aff")]
            out.append({"surname": sur, "given_names": giv, "orcid": orcid,
                        "role": text(c.find("role")), "aff": [a for a in aff if a],
                        "email": [text(x) for x in c.iter("email")]})
    return out


def ymd(d):
    y, m, dd = text(d.find("year")), text(d.find("month")) or "1", text(d.find("day")) or "1"
    return f"{int(y):04d}-{int(m):02d}-{int(dd):02d}" if y else None


def first_version_date(m):
    """Earliest version publication date from pub-history 'original-publication' events."""
    ds = [ymd(ev.find("pub-date")) for ev in m.iter("event")
          if ev.get("event-type") == "original-publication" and ev.find("pub-date") is not None]
    ds = [d for d in ds if d]
    return min(ds) if ds else None


def date_of(e):
    best = None
    for d in e.findall("pub-date") + e.findall("front-stub/pub-date"):
        if d.get("pub-type") == "collection":
            continue
        y, m, dd = text(d.find("year")), text(d.find("month")) or "1", text(d.find("day")) or "1"
        if y:
            s = f"{int(y):04d}-{int(m):02d}-{int(dd):02d}"
            best = min(best, s) if best else s
    return best


def parse_article(a):
    m = a.find("front/article-meta")
    ids = {i.get("pub-id-type"): (i.text or "").strip() for i in m.findall("article-id")}
    doi = ids.get("doi", "")
    mm = re.match(r"(10\.12688/[a-z0-9]+\.\d+)\.(\d+)", doi)  # any F1000-platform journal
    key = mm.group(1) if mm else doi
    ver = int(mm.group(2)) if mm else None
    abstract = " ".join(b["raw"] for ab in m.findall("abstract") if ab.get("abstract-type") not in ("graphical", "teaser")
                        for b in blocks(ab))
    kw = [text(k) for k in m.iter("kwd")]
    subj = [text(s) for s in m.iter("subject")]
    # record pub-date is the latest version; the earliest review date approximates v1 lower bound
    art = {"article_key": key, "doi": doi, "version": ver, "pmcid": ids.get("pmcid"), "pmid": ids.get("pmid"),
           "type": a.get("article-type"), "title": text(m.find("title-group/article-title")),
           "abstract": abstract, "keywords": kw, "subjects": subj, "record_date": date_of(m), "v1_date": first_version_date(m),
           "authors": contribs(m, m)}
    reviews, responses = [], []

    def walk(node, parent_review):
        for s in node.findall("sub-article"):
            t = s.get("article-type")
            fs = s.find("front-stub")
            if t in REVIEW_TYPES:
                rel = [r.get(XLINK) for r in fs.findall("related-article")] if fs is not None else []
                rec = [text(cm.find("meta-value")) for cm in fs.iter("custom-meta") if text(cm.find("meta-name")) == "recommendation"] if fs is not None else []
                reviews.append({"review_id": s.get("id"), "article_key": key, "reviewed_doi": rel[0] if rel else None,
                                "date": date_of(fs) if fs is not None else None,
                                "title": text(fs.find("title-group/article-title")) if fs is not None else "",
                                "contribs": contribs(fs, s) if fs is not None else [],
                                "coi": text(fs.find("author-notes")) if fs is not None else "",
                                "recommendation": rec[0] if rec else None,
                                "blocks": blocks(s.find("body"))})
                walk(s, s.get("id"))
            elif t == "response":
                responses.append({"response_id": s.get("id"), "article_key": key, "parent_review_id": parent_review,
                                  "date": date_of(fs) if fs is not None else None,
                                  "contribs": contribs(fs, s) if fs is not None else [],
                                  "blocks": blocks(s.find("body"), skip_tags=("italic",))})
                walk(s, parent_review)
            else:
                walk(s, parent_review)
    walk(a, None)
    return art, reviews, responses


def main():
    out = {k: open(f"{DATA}/{k}.jsonl", "w") for k in ("articles", "reviews", "responses")}
    seen = set()
    for f in sorted(glob.glob(f"{DATA}/xml/batch_*.xml")):
        root = ET.parse(f).getroot()
        for a in root.iter("article"):
            if a.find("front/article-meta") is None:
                continue
            art, revs, resps = parse_article(a)
            if art["article_key"] in seen:
                continue
            seen.add(art["article_key"])
            out["articles"].write(json.dumps(art, ensure_ascii=False) + "\n")
            for r in revs:
                out["reviews"].write(json.dumps(r, ensure_ascii=False) + "\n")
            for r in resps:
                out["responses"].write(json.dumps(r, ensure_ascii=False) + "\n")
    print("articles", len(seen))


if __name__ == "__main__":
    main()
