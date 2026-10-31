# Agastya Embeddings System

ToC-navigated retrieval: a decision model (Jev, TypeSafe) walks a document's table of contents section by section, and local embeddings (EmbeddingGemma 2) rank the leaves it keeps. No training, no labels, no fine-tuning.

## What it is good at

- **89.1% of gold sections returned at 3.6 leaves per query** on all 6,005 test queries of the public SearchTome release, training-free, on one 8GB consumer GPU.
- **Zero empty results.** The embedding fallback removes the silent-failure mode of walk-based retrievers.
- **No training, ever.** A new document or a new revision costs a re-split, not a training run.
- **Auditable navigation.** Every prune is a named section with its probability in the kill log.
- **Fast on consumer hardware.** 0.42s p50 on a 389-leaf book, 0.40s on a 331-control standards document, RTX 2060 SUPER 8GB.
- **Deep documents are the point.** 45% to 95% leaf hit on the SLP3 book once the walk and the embedding union combine.

## Headline benchmark

SearchTome public release, all 6 domains, full test splits (6,005 gold queries), macro average:

| system | R@1 | R@3 | recall of gold at the system's budget |
| --- | --- | --- | --- |
| local BM25 (same-data control) | 47.8 | 68.1 | |
| Jev only (flat noul ranking) | 54.2 | 75.7 | |
| EmbeddingGemma only | 64.2 | 84.5 | 84.5% at 3 leaves |
| **Agastya (Jev filter, Gemma rank)** | **64.2** | **84.5** | **89.1% at 3.6 leaves** |

Published R@1 for reference (paper setup, 18 books): STAIR 82.6, DSI 76.9, DPR 68.7, BM25 59.5, Mistral 13.8. See Limitations before comparing.

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

- The walk feeds Jev up to 3 levels of ToC lineage per candidate, so each decision is local and the state never grows with corpus size.
- Leaf and chunk vectors are embedded once at startup. The request path is one query embedding plus one Jev call per walk level.
- EmbeddingGemma is never run in float16 (it overflows silently): fp32 on Turing and CPU, bf16 on compute capability >= 8.

## Proof points

### Deep book: SLP3, 389 ToC leaves

| pipeline | leaf hit rate | leaves/query | p50 latency |
| --- | --- | --- | --- |
| Jev walk only, strict relative cutoff | 45% | 1.5 | 0.54s |
| walk + embedding union + fallback | 95% | 4.2 | 0.42s |

### Industry: enterprise compliance, NIST SP 800-53 Rev 5

331 control sections, 3 levels (Introduction > control family > control), 60 questions generated with the section title withheld.

| system | result |
| --- | --- |
| flat dense (EmbeddingGemma) over all 331 controls | R@1 81.7, R@3 93.3, R@5 95.0 |
| Agastya (`/ask` pipeline) | 93.3% leaf hit at 5.5 leaves, p50 0.40s, zero empty results |

Why this use case matters: control catalogs get revised on a schedule and every retrieval needs a reviewable justification. Agastya answers at 93.3% hit with a 5-leaf review set, per revision, with zero training spend and an auditable path.

## Usage

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # put your TypeSafe key: jev_key=apikey_...

python split_pdf.py data/books/<book>.pdf --out data/books/<book>   # PDF outline -> ToC leaves
RAG_DATA=data/books/<book> python core.py --serve                   # POST /ask {"question": "..."}
python core.py "What is TTS?"                                       # one-shot CLI
```

Benchmarks:

```bash
python bench_book.py --leaves data/books/<book>/leaves.jsonl --out data/books/<book>/bench [--blind]
python bench_core.py --n 20 --queries data/books/<book>/bench_core.json
python bench_searchtome.py --full    # 6,005-query external comparison
python bench_library.py              # pooled 6-book limit test
```

## Components

`core.py` is the product (FastAPI `/ask`, walk + union + chunk ranking). `gemma.py` chunks, embeds, and ranks with EmbeddingGemma 2. `jev.py` is the TypeSafe client. `split_pdf.py` turns a PDF outline into ToC leaves. `bench_book.py`, `bench_core.py`, `bench_searchtome.py`, and `bench_library.py` are the benchmarks above. Raw numbers live in `results/`; `data/` is not shipped (book PDFs are not mine to redistribute).

## Limitations, honestly

- Rank metrics equal the dense retriever's inside the candidate set, by construction: the walk supplies the recall floor and the guaranteed non-empty result, Gemma does the ordering. Jev is a safety net and a scale mechanism, not a better ranker at these sizes.
- The published rows are a reference point, not a reproduction target. The public release has 6 domains with 8 to 121 sections each, the paper reports 18 books, and our same-data BM25 scores 47.8 against the published 59.5. Trained STAIR and DSI beat this stack on ranking, and this README does not pretend otherwise.
- Library scale is the failure case: pooling six books into one tree with untitled chapter nodes drops the walk to 44.5% set-hit, and the union rescue (88.0% at 11.8 leaves) is worse than a deeper dense top-k. Use per-book trees with titled chapters.
- The book benchmarks generate their own questions (blind in the NIST run), so they validate plumbing and navigation; SearchTome's gold queries are the external check.
- One Jev API call per walk level: not built for high-QPS serving, and cost scales with ToC branching, not corpus size.
- Do not use it for flat corpora under a few hundred sections; a plain dense retriever wins R@1 at zero marginal cost.

## License

MIT
