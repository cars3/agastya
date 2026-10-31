# Agastya Embeddings System

ToC-navigated retrieval. A decision model (Jev, TypeSafe) walks a document's table of contents section by section, and local embeddings (EmbeddingGemma 2) rank the leaves it keeps. No training, no labels, no fine-tuning: point it at a structured document and it navigates.

The name follows the rishi, fitting a workspace where the hypothesis chain already runs as Bramha, Vishnu, and Maheshwara.

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

## Measured results

### SearchTome public release, all 6 domains, full test splits (6,005 gold queries)

Macro average over domains. Measured on an RTX 2060 SUPER 8GB, float32 EmbeddingGemma 2.

| system | R@1 | R@3 | nDCG@3 |
| --- | --- | --- | --- |
| local BM25 (same-data control) | 47.8 | 68.1 | 59.6 |
| Jev only (flat noul ranking) | 54.2 | 75.7 | 66.8 |
| EmbeddingGemma only | **64.2** | **84.5** | **76.1** |
| Agastya (Jev filter, Gemma rank) | **64.2** | **84.5** | **76.1** |
| published R@1 (paper, 18 books) | STAIR 82.6, DSI 76.9, DPR 68.7, BM25 59.5, Mistral 13.8 | | |

Agastya's returned set: 89.1% set-hit at 3.6 leaves on average.

Reading this honestly:

- The product's rank metrics equal EmbeddingGemma's by construction: the candidate set always contains Gemma's global top-3, so a Gemma ordering reproduces them. Jev contributes set recall (89.1% at 3.6 leaves) and a guaranteed non-empty result, not rank-1 lift.
- Published rows are a reference point, not a reproduction target: the paper reports 18 books, the public release has 6 domains with 8 to 121 sections each. The local BM25 control scores 47.8 against the published 59.5, so the public split is measurably harder than the paper setup.

### One book, 389 ToC leaves (SLP3)

| pipeline | leaf hit rate | leaves/query | p50 latency |
| --- | --- | --- | --- |
| Jev walk only, strict relative cutoff | 45% | 1.5 | 0.54s |
| walk + embedding union + fallback | 95% | 4.2 | 0.42s |

Questions here are generated per leaf by a separate LLM, so this validates the plumbing rather than retrieval quality. The walk is where Agastya earns its keep: on deep hierarchies it prunes a 389-leaf space with 2 to 4 Jev calls per query, and the embedding union guarantees the returned set is never empty.

## When to use it, and when not

Use it when the document is deep and structured (3+ ToC levels, a few hundred to a few thousand leaves), when the corpus changes often and you cannot retrain, and when recall of a small returned set matters more than rank-1. It is training-free and auditable: every kill is a logged section with its probability.

Do not use it for flat corpora under a few hundred sections (a plain dense retriever wins R@1 at zero marginal cost), high-QPS serving (one external call per walk level), or leaderboard chasing against trained retrievers on flat data.

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
```

`bench_searchtome.py` caches Jev scores per (domain, query) in `data/searchtome/jevalks_<domain>.json`, so reruns never pay for the same query twice.

## Data

`data/` is not shipped: it holds book PDFs and extracted text that are not mine to redistribute. `results/` carries the measured benchmark outputs referenced above.

## Reproducing the SearchTome numbers

1. `load_dataset("meghanadh/SearchTome", "<domain>_corpus", split="corpus")` and the matching `_queries` test split.
2. Embed `title + "\n" + content[:8000]` with `prompt_name="Document"`, queries with `prompt_name="SearchQuery"`, normalize both.
3. Dot product, rank, top 3. Metrics: R@1, R@3, nDCG@3 where nDCG@3 = 1/log2(rank+1) for a hit in the top 3, else 0. Macro-average the 6 domains.

Record dataset revision, model revision, device, dtype, prompts, and truncation length when citing numbers.

## License

MIT
