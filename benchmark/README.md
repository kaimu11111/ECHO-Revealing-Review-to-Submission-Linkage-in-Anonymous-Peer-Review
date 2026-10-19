# R2S-Bench files

All identifiers are opaque. `qid` identifies an instance, `person` a reviewer (pseudonymous, consistent across
instances), `article_id` a submission (salted hash; the salt is not released).

| file | one row per | fields |
|---|---|---|
| `queries.jsonl` | instance | `qid`, `person`, `review_text` (sanitized anonymous review) |
| `candidates.jsonl` | instance | `qid`, `candidates` (list of `article_id`, the candidate set, in evaluation order) |
| `gold.jsonl` | instance | `qid`, `positive` (the reviewer's own submission, one of `candidates`), `positive_rebuttal` (the rebuttal written by the reviewer) |
| `rebuttals.jsonl` | submission | `article_id`, `rebuttal_text` (the submission's cleaned rebuttal, as seen when it is a non-gold candidate) |
| `articles.jsonl` | submission | `article_id`, `title`, `abstract` (every candidate and every reviewed paper; e-mail addresses in abstracts replaced by `[EMAIL]`) |
| `reviewed_papers.jsonl` | instance | `qid`, `reviewed_article_id` (the paper the review is about; never used as evidence) |
| `fields.json` | submission | `article_id` → research area 0–7 (KMeans on SPECTER2 embeddings) |
| `splits/` | — | `subset100_rerankers.json` (100 instances with the positive inside the fusion top-20, used for the reranker comparison), `subset20_fullpool.json` (20 random instances used for full-candidate-set scoring), with `*_meta.json` describing how they were drawn |
| `review_genre_words.json` | — | words used in at least 1% of reviews in at least 7 of the 8 areas; kept unmasked by the style masking (Section 3.4) |
| `manifest.json` | — | SHA-1 of every review text, for integrity checks |

Candidate sets contain every submission of the reviewed paper's research area that was public by the review
date plus 182 days and has a rebuttal, minus the reviewed paper, every other article by the reviewer, and every
article with an author who shares the reviewer's surname and first initial. For two ground-truth submissions the
reviewer-written rebuttal differs from the article's default rebuttal; `gold.jsonl` always carries the
reviewer-written one. Statistics: 298 instances, 256 reviewers, 5,061 distinct candidate submissions, candidate
sets of 17 to 1,181 submissions (median 414).

Loading in Python:

```python
import sys; sys.path.insert(0, "../echo")
from benchmark import load_tasks
for t in load_tasks():      # qid, pid, review, cands, pos, texts
    ...
```
