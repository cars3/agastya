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

## Getting Started

Install from a checkout into a virtual environment (Python 3.10 or newer):

```bash
cd /path/to/agastya
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install .
```

On Windows use `.venv\Scripts\python` and `.venv\Scripts\agastya.exe`. The
installation downloads PyTorch and other model dependencies. The first query
also downloads the `google/embeddinggemma-2` model; access to the model and
Hugging Face authentication may be required. A CUDA GPU is recommended;
`GEMMA_DEVICE=cpu` selects CPU inference, which can be slow.

Register the installed command with an agent, using the **absolute path** to
its executable:

```bash
# Claude Code
claude mcp add agastya -- /absolute/path/to/agastya/.venv/bin/agastya --mcp
```

```toml
# Codex: ~/.codex/config.toml
[mcp_servers.agastya]
command = "/absolute/path/to/agastya/.venv/bin/agastya"
args = ["--mcp"]
```

```json
{
  "mcpServers": {
    "agastya": {
      "command": "/absolute/path/to/agastya/.venv/bin/agastya",
      "args": ["--mcp"]
    }
  }
}
```

The JSON format is used by clients such as Claude Desktop and Cursor. Restart
the agent after changing its configuration. Ask it to call `config` with the
absolute path of a folder containing PDFs, then call `query` with a question.
PDFs need embedded bookmarks for meaningful section splitting. The index is
written to `agastya_index` in that folder; the selected index persists across
server restarts in `~/.config/agastya/active-index.json` (or under
`XDG_CONFIG_HOME`). Set `AGASTYA_INDEX` in the agent's server environment to
select a particular existing index instead. The first query embeds the corpus
in memory, and repeats that work after a server restart.

| Tool | Purpose |
| --- | --- |
| `config(folder, model, out, parquet)` | Build and select an index; Parquet is off by default. |
| `query(question, leaf, relevant_match)` | Retrieve relevant sections or chunks. |
| `decide(state, questions, backend)` | Run a typed decision (default: the local decision model). |

Per-agent registration guides — exact config file, snippet, and verification
step for each client — live in
[agent-instructions/agents/](agent-instructions/agents/).

The MCP query currently ranks sections and chunks with dense embeddings; it
does not perform the hierarchy walk used in the earlier benchmarks. Scores and
the optional relevance verdict are heuristics, not calibrated probabilities.

For Parquet output install `pip install '.[parquet]'`.

### Decision model

`decide` runs on the decision model by default: it answers typed questions
(choice / score / noul) with probabilities through one of two backends that
share identical output shapes.

- **Local (default):** the checkpoint
  [amartya16/Agastya-RAG-2B](https://huggingface.co/amartya16/Agastya-RAG-2B),
  a Qwen3.5-2B with an Unsloth decision head (Apache-2.0, about 4.6GB),
  downloads from Hugging Face on the first local `decide` call and is cached
  by `huggingface_hub`. Requires the decision extra plus a compatible Unsloth
  install: `pip install '.[decision]' 'unsloth==2026.11.2'`. To prefetch the
  weights:

  ```bash
  hf download amartya16/Agastya-RAG-2B
  ```

- **Cloud:** TypeSafe Jev. Set `TYPESAFE_API_KEY` (or `jev_key=apikey_...` in
  `.env`) in the server environment.

`DECISION_MODEL_ID` selects a different Hugging Face checkpoint;
`DECISION_MODEL_DIR` points at a local checkpoint directory. See
`requirements-agastya-rag.txt` for the original machine-specific GPU
environment.

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

## For AI agents

This repository carries native instructions for coding agents:

- `agent-instructions/` — all agent-facing material in one folder
- `agent-instructions/AGENTS.md` — the canonical guide, the only instruction
  file; point your agent at it directly
- `agent-instructions/SKILL.md` — the canonical skill, the only skill file
  in the repo; `AGENTS.md` tells every agent to read it
- `agent-instructions/agents/` — per-agent connection guides

Point your agent at `agent-instructions/AGENTS.md`, and it will know how to
set agastya up, including the decision model download.

## Running the Benchmarks

The repository includes reproducible evaluation scripts.

Evaluate document navigation:

```bash
python benchmarks/bench_book.py \
  --leaves data/books/<book>/leaves.jsonl \
  --out data/books/<book>/bench
```

Evaluate retrieval latency and quality:

```bash
python benchmarks/bench_core.py \
  --n 20 \
  --queries data/books/<book>/bench_core.json
```

Run the complete SearchTome benchmark:

```bash
python benchmarks/bench_searchtome.py --full
```

Evaluate multi-document limitations:

```bash
python benchmarks/bench_library.py
```

Investigate hybrid retrieval and reranking:

```bash
python benchmarks/bench_hybrid.py
```

These scripts still import the removed `jev` module, so restore or repoint that
import before running them.

Raw benchmark measurements are stored in `results/`.

## Project Structure

| Component | Responsibility |
| --- | --- |
| `main.py` | Corpus build from every PDF, plus the MCP server (`config`, `query`, `decide`) |
| `agent-instructions/` | Agent readmes, the agent skill, and per-agent MCP connection guides |
| `decision_model.py` | One `/decide` endpoint over cloud Jev or the local Hugging Face checkpoint (`amartya16/Agastya-RAG-2B`), identical output |
| `split_pdf.py` | PDF outline extraction and document splitting |
| `gemma.py` | Embedding, chunking, and semantic ranking |
| `core.py` | Original retrieval pipeline and FastAPI `/ask` endpoint |
| `benchmarks/` | Book, core, SearchTome, library and hybrid evaluations |
| `results/` | Recorded benchmark results |
| `requirements.txt` | Retrieval stack |
| `requirements-agastya-rag.txt` | Pinned unsloth/torch stack for the local decision model |

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
