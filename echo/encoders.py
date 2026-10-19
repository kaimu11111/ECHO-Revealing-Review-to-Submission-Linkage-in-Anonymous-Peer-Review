"""Text encoders used by the signals: SPECTER2 (paper/review embeddings), the chunked dense encoders of the content
channel (SPECTER2, SPECTER2 adhoc-query, SciNCL, BGE-large, E5-large, Qwen3-Embedding-8B through an OpenAI-compatible
endpoint), and the authorship encoders tested for the style channel (STAR, LUAR, LUAR-CRUD, StyleDistance, Wegmann).
Embeddings are cached under work/cache by text hash. `python encoders.py archive` runs the style-model comparison.
"""
import collections, hashlib, json, os, re, sys
import numpy as np
import torch
from config import XDATA

# ====================================================================================================
# SPECTER2 (paper embeddings; formerly encoders.py)

import json, os
import numpy as np
import torch
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from config import DATA, read_jsonl, XDATA, BENCH

CACHE = f"{DATA}/cache"
BS = 32


def load():
    tok = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)
    return tok, model.cuda().eval().half()


@torch.no_grad()
def embed(texts, tok, model):
    out = []
    for i in range(0, len(texts), BS):
        b = tok(texts[i:i + BS], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
        out.append(model(**b).last_hidden_state[:, 0, :].float().cpu().numpy())
    e = np.concatenate(out)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def cached(name, ids, texts, tok, model, cache=CACHE):
    p_ids, p_emb = f"{cache}/{name}_ids.json", f"{cache}/{name}.npy"
    if os.path.exists(p_ids) and json.load(open(p_ids)) == ids:
        print(name, "cached")
        return
    e = embed(texts, tok, model)
    np.save(p_emb, e)
    json.dump(ids, open(p_ids, "w"))
    print(name, e.shape)


def main():
    os.makedirs(CACHE, exist_ok=True)
    tok, model = load()
    arts = read_jsonl(f"{DATA}/articles.jsonl")
    cached("papers", [a["article_key"] for a in arts],
           [a["title"] + tok.sep_token + a["abstract"] for a in arts], tok, model)
    qs = read_jsonl(f"{XDATA}/private/queries_full.jsonl")
    cached("reviews_main", [q["qid"] for q in qs], [q["review_main"] for q in qs], tok, model, f"{XDATA}/cache")
    cached("reviews_strict", [q["qid"] for q in qs], [q["review_strict"] for q in qs], tok, model, f"{XDATA}/cache")




# ====================================================================================================
# ---------------------------------------------------------------- chunking
def chunks(text, lo=120, hi=200):
    """Paragraph-aware chunks of ~lo-hi words; long paragraphs are split on sentence boundaries."""
    paras = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    units = []
    for p in paras:
        if len(p.split()) <= hi:
            units.append(p)
        else:
            cur = []
            for s in re.split(r"(?<=[.!?])\s+", p):
                cur.append(s)
                if len(" ".join(cur).split()) >= lo:
                    units.append(" ".join(cur))
                    cur = []
            if cur:
                units.append(" ".join(cur))
    out, cur = [], []
    for u in units:
        cur.append(u)
        if len(" ".join(cur).split()) >= lo:
            out.append(" ".join(cur))
            cur = []
    if cur:
        if out and len(" ".join(cur).split()) < lo / 2:
            out[-1] += " " + " ".join(cur)
        else:
            out.append(" ".join(cur))
    return out or [text]



# ---------------------------------------------------------------- encoders (query side = review chunks)
REMOTE = {"qwen3emb": {"model": "Qwen/Qwen3-Embedding-8B", "url": os.environ.get("EMB_URL", "http://127.0.0.1:8000/v1/embeddings")}}
QWEN_INSTRUCT = ("Instruct: Given a peer review of a scientific manuscript, retrieve texts (papers or author responses) "
                 "written by the same researcher who wrote the review\nQuery: ")


class RemoteEncoder:
    """Calls an OpenAI-compatible embeddings endpoint (vLLM). Query side gets the Qwen3 instruction prefix."""

    def __init__(self, name):
        self.cfg = REMOTE[name]
        self.key = os.environ.get("EMB_API_KEY", "")  # bearer token of your embeddings endpoint, if any

    def _post(self, batch):
        import time, requests
        for attempt in range(6):
            try:
                r = requests.post(self.cfg["url"], timeout=300, headers={"Authorization": f"Bearer {self.key}"},
                                  json={"model": self.cfg["model"], "input": batch, "truncate_prompt_tokens": 8000})
                r.raise_for_status()
                return np.array([d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])], np.float32)
            except Exception as e:
                print("retry", attempt, str(e)[:200], flush=True)
                time.sleep(5 * (attempt + 1))
        raise RuntimeError("embedding request failed")

    def encode(self, texts, query):
        from concurrent.futures import ThreadPoolExecutor
        texts = [(QWEN_INSTRUCT + t) if query else t for t in texts]
        batches = [texts[i:i + 32] for i in range(0, len(texts), 32)]
        with ThreadPoolExecutor(4) as ex:
            out = list(ex.map(self._post, batches))
        e = np.concatenate(out)
        return e / np.linalg.norm(e, axis=1, keepdims=True)


class Encoder:
    def __new__(cls, name):
        if name in REMOTE:
            return RemoteEncoder(name)
        return super().__new__(cls)

    def __init__(self, name):
        import torch
        self.name, self.torch = name, torch
        if name in ("specter2", "specter2q"):
            from transformers import AutoTokenizer
            from adapters import AutoAdapterModel
            self.tok = AutoTokenizer.from_pretrained("allenai/specter2_base")
            m = AutoAdapterModel.from_pretrained("allenai/specter2_base")
            m.load_adapter("allenai/specter2", source="hf", load_as="proximity")
            if name == "specter2q":
                m.load_adapter("allenai/specter2_adhoc_query", source="hf", load_as="adhoc_query")
            self.model = m.cuda().eval().half()
        elif name == "scincl":
            from transformers import AutoTokenizer, AutoModel
            self.tok = AutoTokenizer.from_pretrained("malteos/scincl")
            self.model = AutoModel.from_pretrained("malteos/scincl").cuda().eval().half()
        else:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer({"bge": "BAAI/bge-large-en-v1.5", "e5": "intfloat/e5-large-v2"}[name], device="cuda")
            self.model.half()

    def encode(self, texts, query):
        torch = self.torch
        if self.name in ("specter2", "specter2q", "scincl"):
            if self.name == "specter2q":
                self.model.set_active_adapters("adhoc_query" if query else "proximity")
            elif self.name == "specter2":
                self.model.set_active_adapters("proximity")
            out = []
            with torch.no_grad():
                for i in range(0, len(texts), 64):
                    b = self.tok(texts[i:i + 64], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
                    out.append(self.model(**b).last_hidden_state[:, 0, :].float().cpu().numpy())
            e = np.concatenate(out)
            return e / np.linalg.norm(e, axis=1, keepdims=True)
        if self.name == "bge":
            texts = [("Represent this sentence for searching relevant passages: " + t) if query else t for t in texts]
        else:
            texts = [("query: " if query else "passage: ") + t for t in texts]
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)


def encode_cached(name, texts, query, cache_dir):
    path = f"{cache_dir}/boost_emb_{name}_{'q' if query else 'd'}.npz"
    cache = dict(np.load(path)) if os.path.exists(path) else {}
    todo = sorted({t for t in texts if _h(t) not in cache})
    if todo:
        enc = Encoder(name)
        for i in range(0, len(todo), 4000):
            for t, e in zip(todo[i:i + 4000], enc.encode(todo[i:i + 4000], query)):
                cache[_h(t)] = e.astype(np.float16)
            np.savez(path, **cache)
            print(name, "q" if query else "d", min(i + 4000, len(todo)), "/", len(todo), flush=True)
        del enc
        import torch
        torch.cuda.empty_cache()
    return {t: cache[_h(t)].astype(np.float32) for t in set(texts)}




# ====================================================================================================
# authorship encoders (formerly encoders.py)

import hashlib, json, os, re, sys
import numpy as np
import torch

LUAR_CRUD_REV = "8b36ce7f289ee91900a7dff0eaccf5cdca745ab1"
SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def _h(t):
    return hashlib.sha1(t.encode()).hexdigest()


def _norm(e):
    return e / np.linalg.norm(e, axis=-1, keepdims=True)


class LuarCrud:
    def __init__(self):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained("rrivera1849/LUAR-CRUD", revision=LUAR_CRUD_REV)
        self.model = AutoModel.from_pretrained("rrivera1849/LUAR-CRUD", revision=LUAR_CRUD_REV, trust_remote_code=True).cuda().eval()

    @torch.no_grad()
    def embed(self, texts):
        out = []
        for t in texts:
            ids = self.tok(t, add_special_tokens=False)["input_ids"]
            docs = [self.tok.decode(ids[i:i + 30]) for i in range(0, max(1, len(ids)), 30)][:64]
            b = self.tok(docs, max_length=32, padding="max_length", truncation=True, return_tensors="pt")
            b = {k: v.reshape(1, len(docs), -1).cuda() for k, v in b.items() if k in ("input_ids", "attention_mask")}
            out.append(self.model(**b)[0].float().cpu().numpy())
        return _norm(np.stack(out))


class StyleDist:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("StyleDistance/styledistance", device="cuda")

    def embed(self, texts):
        sents = [[s for s in SENT.split(t) if s.strip()] or [t] for t in texts]
        flat = [s for ss in sents for s in ss]
        E = self.model.encode(flat, batch_size=256, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        out, i = [], 0
        for ss in sents:
            out.append(E[i:i + len(ss)].mean(0))
            i += len(ss)
        return _norm(np.stack(out))


class Star:
    def __init__(self):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained("roberta-large")
        self.model = AutoModel.from_pretrained("AIDA-UPM/star").cuda().eval().half()

    @torch.no_grad()
    def embed(self, texts):
        ch = [chunks(t) for t in texts]
        flat = [c for cs in ch for c in cs]
        E = []
        for i in range(0, len(flat), 64):
            b = self.tok(flat[i:i + 64], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
            E.append(self.model(**b).pooler_output.float().cpu().numpy())
        E = _norm(np.concatenate(E))
        out, i = [], 0
        for cs in ch:
            out.append(E[i:i + len(cs)].mean(0))
            i += len(cs)
        return _norm(np.stack(out))


MODELS = {"luarcrud": LuarCrud, "styledist": StyleDist, "star": Star}


def embed_cached(name, cls, texts, cache_dir):
    path = f"{cache_dir}/boost_style_{name}.npz"
    cache = dict(np.load(path)) if os.path.exists(path) else {}
    todo = sorted({t for t in texts if _h(t) not in cache})
    if todo:
        m = cls()
        for i in range(0, len(todo), 1000):
            for t, e in zip(todo[i:i + 1000], m.embed(todo[i:i + 1000])):
                cache[_h(t)] = e.astype(np.float16)
            np.savez(path, **cache)
            print(name, min(i + 1000, len(todo)), "/", len(todo), flush=True)
        del m
        torch.cuda.empty_cache()
    return {t: cache[_h(t)].astype(np.float32) for t in set(texts)}


def main(setting):
    from config import XDATA
    from signals import archive_tasks, edition_tasks
    os.environ["F1000_MASK_LEVEL"] = "aggressive"
    from style import render
    tasks = archive_tasks() if setting == "archive" else edition_tasks()
    cache_dir = f"{XDATA}/cache"
    M = json.load(open(f"{cache_dir}/boost_masked_aggr.json"))  # written by s14 (aggressive masking by text hash)
    fpath = f"{cache_dir}/boost_features.npz"
    F = np.load(fpath, allow_pickle=True)["feats"].item()
    for name, cls in MODELS.items():
        for variant in ("", "_masked"):
            key = name + variant
            if key in F:
                continue
            conv = (lambda x: x) if not variant else (lambda x: render(M[_h(x)]))
            texts = sorted({conv(x) for t in tasks for x in t["style_texts"]} | {conv(t["review"]) for t in tasks})
            E = embed_cached(key, cls, texts, cache_dir)
            F[key] = {t["qid"]: np.stack([E[conv(x)] for x in t["style_texts"]]) @ E[conv(t["review"])] for t in tasks}
            np.savez(fpath, feats=np.array(F, dtype=object))
            print("added", key, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])


# ====================================================================================================
# LUAR / Wegmann (formerly in analysis/authorship_models.py)
LUAR_REV = "f1db50251805ed69b43cf4f72ea2f0e231f36a1c"
RNG = np.random.default_rng(0)
SENT = re.compile(r"(?<=[.!?])\s+|\n+")


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


def luar_cached(model_name, model_obj, texts):
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
