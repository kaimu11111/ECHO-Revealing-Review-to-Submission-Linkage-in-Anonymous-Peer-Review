"""SPECTER2 (allenai/specter2_base + proximity adapter) embeddings, cached under data/cache/.

papers : title [SEP] abstract   (all parsed articles)
reviews: cleaned review text (main and strict), truncated to 512 tokens
"""
import json, os
import numpy as np
import torch
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from common import DATA, read_jsonl, XDATA, BENCH, CONCURRENT_DAYS, EXP

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


if __name__ == "__main__":
    main()
