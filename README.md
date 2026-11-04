# Agastya RAG

Structure-aware retrieval for AI agents. No fine-tuning. One 8GB GPU.

Agastya RAG is a training-free retrieval system that helps AI agents navigate large, structured documents without relying entirely on traditional fixed-chunk vector search.

Agastya combines intelligent table-of-contents navigation with dense embedding retrieval to locate relevant sections in textbooks, technical manuals, regulatory documents, and other hierarchical knowledge sources.

Instead of treating a 500-page document as thousands of disconnected text chunks, Agastya uses the document's existing structure to guide retrieval.

The result is a system that can retrieve relevant evidence from complex documents with high recall, low latency, no fine-tuning pipeline, and hardware requirements accessible to individual developers.

**89.1% gold-section recall across 6,005 benchmark queries, averaging just 3.6 candidate sections per query.**

## Why Agastya?

Traditional Retrieval-Augmented Generation (RAG) typically follows a simple process:

1. Split documents into fixed-size chunks.
2. Convert those chunks into vector embeddings.
3. Retrieve the closest matches to a user's question.
4. Pass the retrieved context to an LLM.

This works well for many applications, but it discards valuable information: how the document is organized.

Consider an AI agent answering a question from a 600-page engineering manual.

The answer might be located under:

Chapter 8 → Memory Systems → Cache Organization → Set Associativity

A conventional vector retriever compares the question against chunks throughout the manual. It may retrieve related passages from other chapters while missing the section that directly answers the question.

Agastya approaches retrieval differently.

It first uses the document's hierarchy to identify promising branches, then combines those results with dense embedding search to recover relevant sections that structural navigation might miss.

The goal isn't simply to retrieve the single highest-ranking result. It's to give an AI agent the right evidence within a small, manageable context budget.

This makes Agastya particularly useful for agents working with long technical documents where accuracy, source traceability, and frequent document updates matter.

## Performance at a Glance

| Metric | Measured result |
| --- | --- |
| SearchTome benchmark | 6,005 queries across 6 domains |
| Gold-section recall | 89.1% |
| Average retrieval budget | 3.6 sections/query |
| Deep textbook retrieval | 95% leaf hit rate |
| NIST SP 800-53 retrieval | 93.3% leaf hit rate |
| Median latency on deep textbook | 0.42 seconds |
| Tested hardware | RTX 2060 SUPER, 8GB VRAM |
| Model fine-tuning | None |
| Empty retrieval results | Zero in evaluated runs |

These measurements cover distinct evaluation settings. SearchTome provides an external benchmark, while the deep textbook and NIST evaluations test navigation over larger hierarchical documents.

## Why these numbers matter

An AI agent does not necessarily need the correct passage ranked first. It needs that passage to be present in the context it receives.

On SearchTome:

- EmbeddingGemma alone retrieves the correct section within its top three results for 84.5% of queries.
- Agastya expands the candidate set through structural navigation, reaching 89.1% gold-section recall while examining an average of only 3.6 sections.
- This improves evidence coverage without requiring model training or a large retrieval context.

Agastya is designed around Recall@Budget: maximizing the probability that the information an agent needs is present within a limited number of retrieved sections.

## Benchmarks

### SearchTome: 6,005 Queries

Evaluation on the complete public SearchTome release across six domains.

| Retrieval system | Recall@1 | Recall@3 | Gold recall at retrieval budget |
| --- | --- | --- | --- |
| BM25 | 47.8% | 68.1% | — |
| Jev-only ranking | 54.2% | 75.7% | — |
| EmbeddingGemma 2 | 64.2% | 84.5% | 84.5% at 3 leaves |
| Agastya RAG | 64.2% | 84.5% | 89.1% at 3.6 leaves |

Agastya does not improve the top-one ranking over EmbeddingGemma in this evaluation. Its improvement comes from expanding the relevant candidate set while maintaining a small retrieval budget.

### Deep Document Evaluation

Tested on a 389-leaf textbook.

| Configuration | Leaf hit rate | Leaves/query | Median latency |
| --- | --- | --- | --- |
| Strict Jev hierarchy walk | 45% | 1.5 | 0.54s |
| Agastya combined pipeline | 95% | 4.2 | 0.42s |

This demonstrates why combining semantic retrieval with hierarchical navigation matters: a structural walk alone can prune away the correct answer, while the embedding union can recover it.

### Enterprise Compliance Evaluation

Tested using NIST SP 800-53 Revision 5, organized into 331 control sections.

| System | Result |
| --- | --- |
| Dense EmbeddingGemma | 81.7% Recall@1 |
| Dense EmbeddingGemma | 93.3% Recall@3 |
| Agastya RAG | 93.3% leaf hit at 5.5 leaves |
| Agastya median latency | 0.40s |

These results demonstrate practical retrieval on a structured regulatory corpus without domain-specific model fine-tuning.

## Agastya vs. IBM STAIR

Both recognize that document hierarchies contain useful information that traditional flat retrieval systems often ignore.

However, they make different engineering tradeoffs.

| Capability | Agastya RAG | IBM STAIR |
| --- | --- | --- |
| Document navigation | Inference-time hierarchy walk | Fine-tuned generative index |
| Training required | No | Yes |
| New document ingestion | Split and embed | Requires index-model preparation/training |
| Corpus updates | Reprocess affected documents | Update trained index |
| Structural traceability | Explicit navigation and pruning log | Model-generated section identifiers |
| Local hardware | Tested on 8GB GPU | Training infrastructure required |
| Primary strength | Training-free, adaptable retrieval | High top-one retrieval accuracy |

Its advantage is offering practical hierarchical retrieval without requiring a new training pipeline for each document collection.

For applications with stable corpora and strict top-one retrieval requirements, a trained system such as STAIR may be preferable.

## Getting Started

### Requirements

- Python 3
- Dependencies from `requirements.txt`
- A TypeSafe API key for Jev
- A supported environment for EmbeddingGemma 2

The system was evaluated on an NVIDIA RTX 2060 SUPER with 8GB VRAM.

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure the environment

```bash
cp .env.example .env
```

Add your TypeSafe API key to `.env`:

```
jev_key=apikey_...
```

### 3. Process a PDF

```bash
python split_pdf.py data/books/<book>.pdf --out data/books/<book>
```

Agastya extracts the document outline and generates the structured leaves used by the retrieval pipeline.

### 4. Start the retrieval API

```bash
RAG_DATA=data/books/<book> python core.py --serve
```

The service exposes a FastAPI `/ask` endpoint.

Send a question:

```http
POST /ask
Content-Type: application/json

{
  "question": "What is TTS?"
}
```

Alternatively, run a one-shot query from the command line:

```bash
python core.py "What is TTS?"
```

## Running the Benchmarks

The repository includes reproducible evaluation scripts.

Evaluate document navigation:

```bash
python bench_book.py \
  --leaves data/books/<book>/leaves.jsonl \
  --out data/books/<book>/bench
```

Evaluate retrieval latency and quality:

```bash
python bench_core.py \
  --n 20 \
  --queries data/books/<book>/bench_core.json
```

Run the complete SearchTome benchmark:

```bash
python bench_searchtome.py --full
```

Evaluate multi-document limitations:

```bash
python bench_library.py
```

Investigate hybrid retrieval and reranking:

```bash
python bench_hybrid.py
```

Raw benchmark measurements are stored in `results/`.

## Project Structure

| Component | Responsibility |
| --- | --- |
| `core.py` | Main retrieval pipeline and FastAPI endpoint |
| `gemma.py` | Embedding, chunking, and semantic ranking |
| `jev.py` | TypeSafe Jev integration and structural navigation |
| `split_pdf.py` | PDF outline extraction and document splitting |
| `bench_book.py` | Book-level retrieval evaluations |
| `bench_core.py` | Core pipeline benchmarking |
| `bench_searchtome.py` | External SearchTome evaluation |
| `bench_library.py` | Multi-document retrieval evaluation |
| `bench_hybrid.py` | Hybrid search and reranking experiments |
| `results/` | Recorded benchmark results |

Book PDFs are not distributed with the repository because redistribution rights are not held for those documents.

## Design Tradeoffs and Limitations

Agastya is optimized for structured retrieval, not every possible RAG workload.

1. **Dense retrieval can be sufficient for smaller documents.**
   For flat document collections containing fewer than a few hundred sections, the hierarchy walk may add latency without improving top-one retrieval accuracy. Dense retrieval alone is often the simpler choice.

2. **Multi-document routing needs special attention.**
   Combining six books into a single hierarchy with untitled chapter nodes reduced the structural walk's set-hit rate to 44.5%.
   The combined retrieval system recovered to 88.0% recall, but required an average of 11.8 sections per query.
   A better architecture for larger libraries is to maintain separate, titled document hierarchies and route questions to the appropriate document before running Agastya.

3. **Agastya is not optimized for extremely high query throughput.**
   The hierarchy walk makes one Jev API call per traversed level, introducing an external dependency and additional latency.
   High-throughput applications may benefit from a simpler local dense index or a trained generative retrieval model.

4. **The hierarchy is a retrieval aid, not a replacement for semantic search.**
   Structural navigation contributes additional candidates and an auditable decision path. EmbeddingGemma remains responsible for ranking those candidates.

5. **Retrieval recall is not the same as answer accuracy.**
   The reported benchmarks evaluate whether the correct section is retrieved. They do not establish downstream answer correctness, hallucination rates, or production-level reliability for complete AI agents.

These limitations are intentional parts of the evaluation rather than hidden implementation details.

## License

MIT License.
