# Agastya Embeddings System

ToC-navigated retrieval. A decision model (Jev, TypeSafe) walks a document's table of contents section by section, and local embeddings (EmbeddingGemma 2) rank the leaves it keeps. No training, no labels, no fine-tuning: point it at a structured document and it navigates.

The name follows the rishi, fitting a workspace where the hypothesis chain already runs as Bramha, Vishnu, and Maheshwara.

## Headline result

**89.1% of gold sections returned at 3.6 leaves per query.** Measured on all 6,005 test queries of the public SearchTome release (6 domains), training-free, on one 8GB consumer GPU. Zero empty results.

| system | R@1 | R@3 | recall of gold at the system's budget |
| --- | --- | --- | --- |
| local BM25 (same-data control) | 47.8 | 68.1 | |
| Jev only (flat noul ranking) | 54.2 | 75.7 | |
| EmbeddingGemma only | 64.2 | 84.5 | 84.5% at 3 leaves |
| **Agastya (Jev filter, Gemma rank)** | **64.2** | **84.5** | **89.1% at 3.6 leaves** |
| published R@1 (paper, 18 books) | STAIR 82.6, DSI 76.9, DPR 68.7, BM25 59.5, Mistral 13.8 | | |

How to read this honestly:

- Agastya's rank metrics equal the dense retriever's inside its candidate set, by construction: the Jev walk supplies a recall floor (89.1% at 3.6 leaves) and a guaranteed non-empty result, while Gemma does the ordering. The Jev stage is a safety net and a scale mechanism, not a better ranker at this size.
- The published rows are a reference point, not a reproduction target. The public release has 6 domains with 8 to 121 sections each, the paper reports 18 books, and our local BM25 control scores 47.8 against the published 59.5, so the public split is measurably harder than the paper setup. Training-based retrievers (STAIR, DSI) beat the dense retriever here, and therefore beat Agastya's ranking: that is the honest state of things.
- Against same-data BM25 (47.8 R@1), the training-free dense stack is +16.4 R@1 and the walk adds 4.6 points of recall over the dense retriever at a comparable budget (89.1% at 3.6 leaves vs 84.5% at 3).

## How it works

```
question
   |
   v
Jev walk: one noul probability per candidate section at each ToC level,
keep branches within 15% of the best score at that level
   |
   v
candidate leaves = walked leaves
                 + top-3 leaves by EmbeddingGemma similarity
                 + top-5 embedding fallback when the walk keeps nothing
   |
   v
EmbeddingGemma ranks the candidate leaves and the chunks inside them
```

Design notes:

- The walk gives Jev up to 3 levels of ToC lineage per candidate (Section > Chapter > Subtopic) plus the current path, so each decision is local and cheap. The state never grows with corpus size.
- Leaf and chunk vectors are embedded once at startup. The request path does exactly one embedding call (the query) plus one Jev call per walk level.
- EmbeddingGemma is never run in float16: it overflows and silently returns garbage. bfloat16 only on compute capability >= 8, float32 on Turing and CPU.

## Results in detail

### 1. Deep single book: 389 ToC leaves (SLP3)

| pipeline | leaf hit rate | leaves/query | p50 latency |
| --- | --- | --- | --- |
| Jev walk only, strict relative cutoff | 45% | 1.5 | 0.54s |
| walk + embedding union + fallback | 95% | 4.2 | 0.42s |

This is the workload the system was built for: a deep hierarchy where the walk prunes 389 leaves with 2 to 4 Jev calls per query, the embedding union converts near-miss walk decisions into hits, and the fallback removes the zero-result failure mode. Questions here are generated per leaf by a separate LLM, so this validates the plumbing and the navigation, not retrieval quality against gold queries.

### 2. Library scale: what the architecture does not solve (limit case)

All six SearchTome domains pooled into one 413-section corpus with a 3-level tree (book > chapter > section), same 6,005 gold queries, chapter nodes untitled ("Chapter 9"):

| system | result |
| --- | --- |
| Gemma flat over all 413 sections | R@1 63.0, R@3 83.0 |
| Jev walk only | set-hit 44.5%, zero empty results, 8.7 calls/query |
| Agastya (walk + union) | R@1 63.0, R@3 83.0, set-hit 88.0% at 11.8 leaves |

The walk loses its footing here: cross-domain routing across 6 books and untitled chapter nodes prunes the right branch too often, and the union rescue costs 11.8 leaves per query, worse than simply taking a deeper dense top-k. Conclusion, measured rather than assumed: use per-book trees with real chapter titles. Do not merge unrelated books into one walkable tree and expect the navigator to help.

## When to use it, and when not

Use it when the document is deep and structured (3+ ToC levels with titled chapters, a few hundred to a few thousand leaves), when the corpus changes often and you cannot retrain, and when recall of a small returned set matters more than rank-1. It is training-free and auditable: every kill is a logged section with its probability.

Do not use it for flat corpora under a few hundred sections (a plain dense retriever wins R@1 at zero marginal cost), cross-domain trees with untitled chapters (measured above), high-QPS serving (one external call per walk level), or leaderboard chasing against trained retrievers on flat data.

## Components

| file | role |
| --- | --- |
| `core.py` | the product: FastAPI `/ask` endpoint, walk + union + chunk ranking |
| `gemma.py` | chunk, embed, and rank with EmbeddingGemma 2 (text-only) |
| `jev.py` | TypeSafe Jev client, standard typed-answer quickstart |
| `split_pdf.py` | split a PDF into ToC leaves (outline, headings, leaf text) |
| `bench_book.py` | embedding-only benchmark over one book: R@1/R@3/R@5, nDCG, MRR |
| `bench_core.py` | end-to-end benchmark of the `/ask` pipeline: leaf hit rate, leaves/query, latency |
| `bench_searchtome.py` | external comparison on the public SearchTome release |
| `bench_library.py` | library-scale limit test: 6 books pooled into one tree |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# put your TypeSafe key in .env: jev_key=apikey_...
```

PyPI `torch` wheels on Linux are CUDA builds. On a Turing card (RTX 20xx) the code falls back to float32 automatically; do not force float16.

## Usage

Prepare a book (PDF with an embedded outline works best):

```bash
python split_pdf.py --help
# writes data/books/<book>/toc.json and leaves.jsonl
```

Serve the pipeline:

```bash
python core.py --serve          # POST /ask {"question": "..."} on 127.0.0.1:8000
python core.py "What is TTS?"   # one-shot from the CLI
```

Benchmarks:

```bash
python bench_book.py --leaves data/books/<book>/leaves.jsonl --out data/books/<book>/bench
python bench_core.py --n 20 --queries data/books/<book>/bench_core.json
python bench_searchtome.py --limit 120   # sampled, ~120 queries per domain
python bench_searchtome.py --full        # all 6,005 test queries
python bench_library.py --workers 8      # pooled 413-section limit test
```

`bench_searchtome.py` caches Jev scores per (domain, query) and `bench_library.py` caches walk results, so reruns never pay twice.

## Data

`data/` is not shipped: it holds book PDFs and extracted text that are not mine to redistribute. `results/` carries the measured benchmark outputs referenced above.

## Reproducing the SearchTome numbers

1. `load_dataset("meghanadh/SearchTome", "<domain>_corpus", split="corpus")` and the matching `_queries` test split.
2. Embed `title + "\n" + content[:8000]` with `prompt_name="Document"`, queries with `prompt_name="SearchQuery"`, normalize both.
3. Dot product, rank, top 3. Metrics: R@1, R@3, nDCG@3 where nDCG@3 = 1/log2(rank+1) for a hit in the top 3, else 0. Macro-average the 6 domains.
4. Record dataset revision, model revision, device, dtype, prompts, and truncation length when citing numbers.

## License

MIT
