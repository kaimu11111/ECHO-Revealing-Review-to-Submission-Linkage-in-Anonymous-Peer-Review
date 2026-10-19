"""Download all F1000-platform (F1000_CORPUS) full-text JATS XML from PMC (includes peer-review and response sub-articles)."""
import json, os, time, urllib.request, urllib.parse

from common import DATA, JOURNALS
XML_DIR = f"{DATA}/xml"
EU = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
BATCH = 100


def get(url, tries=5):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except Exception as e:
            print("retry", i, e)
            time.sleep(3 * (i + 1))
    raise RuntimeError(url)


def main():
    ids_path = f"{DATA}/raw/pmc_ids.json"
    ids = json.load(open(ids_path)) if os.path.exists(ids_path) else []
    # one search per journal: esearch returns at most 9,999 ids per query. New ids are appended so that
    # already-downloaded batch files keep their contents.
    seen = set(ids)
    for j in JOURNALS:
        term = urllib.parse.quote(f'"{j}"[jour]')
        for start in range(0, 100000, 9999):
            r = json.loads(get(f"{EU}/esearch.fcgi?db=pmc&term={term}&retmax=9999&retstart={start}&format=json"))["esearchresult"]
            new = [i for i in r["idlist"] if i not in seen]
            ids += new
            seen.update(new)
            if start + 9999 >= int(r["count"]):
                break
            time.sleep(0.4)
        time.sleep(0.4)
    json.dump(ids, open(ids_path, "w"))
    print("ids", len(ids))
    for b in range(0, len(ids), BATCH):
        out = f"{XML_DIR}/batch_{b // BATCH:04d}.xml"
        if os.path.exists(out) and os.path.getsize(out) > 1000:
            continue
        chunk = ids[b:b + BATCH]
        data = get(f"{EU}/efetch.fcgi?db=pmc&id={','.join(chunk)}")
        open(out, "wb").write(data)
        print(out, len(data), flush=True)
        time.sleep(0.5)


if __name__ == "__main__":
    main()
