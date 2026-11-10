# Agastya RAG

**Structure-aware retrieval for AI agents. Training-free, one 8GB GPU.**

Agastya builds a table-of-contents index from your PDFs, walks it to find the sections that can answer a question, and ranks the candidates with EmbeddingGemma 2. Every answer ships with the section paths it came from, so an agent can cite what it read.

**89.1% gold-section recall across 6,005 benchmark queries, averaging 3.6 candidate sections per query.**

A flat vector index treats a 600-page manual as thousands of disconnected chunks. Agastya keeps the shape of the book: it splits along the embedded bookmarks, retrieves structure-first, and uses dense similarity to widen the candidate set without blowing the context budget.

## Install

From a checkout (the PyPI release is prepared but not published yet):

```sh
git clone https://github.com/cars3/agastya
cd agastya
python3 -m venv .venv
.venv/bin/python -m pip install .
```

On Windows use `.venv\Scripts\python`. Python 3.10 or newer. The first query downloads `google/embeddinggemma-2` (about 1.5GB; Hugging Face access may be required). A CUDA GPU is recommended; Apple Silicon runs on MPS; AMD GPUs work through ROCm builds of torch. `GEMMA_DEVICE=cpu` forces CPU inference, which is slow.

## Verify

```sh
agastya doctor
```

Checks python, torch, the chosen device, the transformers version, the embedder (loads it and embeds one string), and the decision backends. About 30 seconds once the model is cached, and it names the fix when something is off.

## Build an index

```sh
agastya init ~/papers
```

Every PDF in the folder is split along its bookmarks into `~/papers/agastya_index/<book>/`. A `config.json` manifest is written and the index becomes active in `~/.config/agastya/active-index.json`, so agents do not need to run `config` first. PDFs without bookmarks fail with an explanation. Books whose text extracts empty are flagged as warnings, not green ticks. `--out`, `--model` and `--skip-unchanged` adjust the run.

Restarts are cheap: unchanged books are skipped and vectors are cached in `leaf_vecs.npz` inside the index, invalidated by the sha256 of each book.

## Connect your agent

Every client runs the same server, `agastya --mcp`, over stdio. `agastya install <client>` prints the exact snippet for this machine, resolved to absolute paths. Add `--write` where writing the config file directly is safe (Cursor, Claude Desktop). The full per-client guides live in [`agent-instructions/agents/`](agent-instructions/agents/).

### Claude Code

```sh
claude mcp add agastya -- /absolute/path/.venv/bin/agastya --mcp
```

Guide: [claude-code.md](agent-instructions/agents/claude-code.md)

### Codex

`agastya install codex` prints this block for `~/.codex/config.toml`:

```toml
[mcp_servers.agastya]
command = "/absolute/path/.venv/bin/agastya"
args = ["--mcp"]
```

Guide: [codex.md](agent-instructions/agents/codex.md)

### Claude Desktop

```sh
agastya install claude-desktop --write
```

Then restart the app. Guide: [claude-desktop.md](agent-instructions/agents/claude-desktop.md)

### Cursor

```sh
agastya install cursor --write
```

Writes `~/.cursor/mcp.json`. Guide: [cursor.md](agent-instructions/agents/cursor.md)

### opencode

```sh
agastya install opencode
```

Prints the JSON block for opencode's MCP config. Guide: [opencode.md](agent-instructions/agents/opencode.md)

### Gemini CLI, Copilot CLI, Antigravity, Grok, Kimi, Hermes

`agastya install gemini-cli` (or `copilot-cli`, `antigravity`, `grok`, `kimi`, `hermes`) prints the client's config block plus one line naming the matching guide in `agent-instructions/agents/`.

### Any other MCP client

```json
{
  "mcpServers": {
    "agastya": {
      "command": "/absolute/path/.venv/bin/agastya",
      "args": ["--mcp"]
    }
  }
}
```

After connecting, ask the agent to call `query` with a question; the index built by `agastya init` is already active. Calling `config` with a folder builds a new index from inside the chat and switches to it.

| Tool | Purpose |
| --- | --- |
| `config(folder, model, out, parquet)` | Build an index from an agent session and make it active. |
| `query(question, leaf, relevant_match)` | Retrieve the strongest chunks or whole leaves, with section paths. |
| `decide(state, questions, backend)` | Run a typed decision (default: the local decision model). |

The MCP query ranks sections and chunks with dense embeddings; the hierarchy walk from the benchmarks is not part of the server yet. Scores and the optional relevance verdict are heuristics, not calibrated probabilities.

For Parquet output install `pip install '.[parquet]'`.

## Decision model

`decide` answers typed questions (choice / score / noul) with probabilities. The default backend is the local model, the checkpoint [amartya16/Agastya-RAG-2B](https://huggingface.co/amartya16/Agastya-RAG-2B), a Qwen3.5-2B with an Unsloth decision head (Apache-2.0, about 4.6GB). It downloads on first use and is cached by `huggingface_hub`. To prefetch:

```sh
hf download amartya16/Agastya-RAG-2B
```

The local stack pins `transformers<=5.17.0`, while the embedder needs `>=5.19`, so the two cannot share one environment. Run the decision service from its own pinned environment and point the MCP server at it:

```sh
# decision environment (pinned stack in requirements-agastya-rag.txt)
python decision_model.py --serve --port 8000

# MCP server environment
export DECISION_URL=http://127.0.0.1:8000
```

When `DECISION_URL` is unset, `decide` runs in-process, which works anywhere unsloth is importable. The cloud backend is TypeSafe Jev: set `TYPESAFE_API_KEY` (or `jev_key=apikey_...` in `.env`). `DECISION_MODEL_ID` selects a different Hugging Face checkpoint and `DECISION_MODEL_DIR` points at a local one.

## Results

Measured performance (SearchTome, the deep textbook, NIST SP 800-53), the reranker and fusion attempts that did not work, and the raw numbers live in [benchmarks/README.md](benchmarks/README.md) and `results/`.

## Agastya vs IBM STAIR

| Capability | Agastya RAG | IBM STAIR |
| --- | --- | --- |
| Document navigation | Inference-time hierarchy walk | Fine-tuned generative index |
| Training required | No | Yes |
| New document ingestion | Split and embed | Index-model preparation/training |
| Corpus updates | Reprocess affected documents | Update the trained index |
| Structural traceability | Explicit navigation and pruning log | Model-generated section identifiers |
| Local hardware | Tested on 8GB GPU | Training infrastructure required |
| Primary strength | Training-free, adaptable retrieval | High top-one accuracy |

For stable corpora with strict top-one requirements, a trained system may be preferable. Agastya's case is practical hierarchical retrieval with no training pipeline per collection.

## For AI agents

- `agent-instructions/AGENTS.md`: the canonical guide; point a coding agent at it first
- `agent-instructions/SKILL.md`: the skill file agents load
- `agent-instructions/agents/`: per-client connection guides

## Project structure

| Component | Responsibility |
| --- | --- |
| `main.py` | CLI (`init`, `doctor`, `install`), the build loop, and the MCP server |
| `device.py` | One device decision for the embedder and the decision model |
| `split_pdf.py` | PDF outline extraction and document splitting |
| `gemma.py` | Embedding, chunking, and semantic ranking |
| `decision_model.py` | One `/decide` endpoint over cloud Jev or the local checkpoint |
| `core.py` | Original retrieval pipeline and FastAPI `/ask` endpoint |
| `agent-instructions/` | Agent readmes, the skill, and per-agent MCP guides |
| `benchmarks/` | Book, core, SearchTome, library and hybrid evaluations |
| `results/` | Recorded benchmark results |
| `requirements-agastya-rag.txt` | Pinned decision environment (transformers 5.17 + unsloth) |

Book PDFs are not distributed because redistribution rights are not held.

## Design tradeoffs and limitations

1. **Dense retrieval can be sufficient for smaller documents.** Under a few hundred sections, the hierarchy walk may add work without improving top-one accuracy.
2. **Multi-document routing needs care.** Six books in one hierarchy with untitled chapter nodes dropped the walk's set-hit rate to 44.5% (combined pipeline: 88.0% recall at 11.8 sections/query). Separate titled hierarchies plus routing work better.
3. **Not built for high query throughput.** One Jev call per traversed level adds an external dependency and latency.
4. **The hierarchy is an aid, not a replacement for semantic search.** Structural navigation adds candidates and an audit trail; EmbeddingGemma ranks them.
5. **Retrieval recall is not answer accuracy.** The benchmarks measure section retrieval, not downstream correctness or hallucination rates.
6. **Two environments for two models.** The embedder (transformers >= 5.19) and the local decision stack (<= 5.17) cannot coexist in one environment; `DECISION_URL` bridges them over HTTP.

These limitations are part of the evaluation, not hidden implementation details.

## Running the benchmarks

See [`benchmarks/README.md`](benchmarks/README.md) for each script, its environment, and first-run downloads. Quick starts:

```sh
python benchmarks/bench_book.py --leaves data/books/<book>/leaves.jsonl --out data/books/<book>/bench
python benchmarks/bench_core.py --n 20 --queries data/books/<book>/bench_core.json
python benchmarks/bench_searchtome.py --full
```

`bench_searchtome.py` and `bench_library.py` additionally need a `jev.py` client, which is not part of this release; they exit with that explanation.

## License

MIT.
