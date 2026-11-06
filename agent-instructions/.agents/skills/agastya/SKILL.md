---
name: agastya
description: >-
  Structure-aware retrieval over PDF books and manuals: build a table-of-contents
  index, retrieve the strongest sections or chunks for a question, and run typed
  decisions through a local or cloud backend. Use when the user asks a question
  over a PDF corpus, needs cited evidence from a long technical document, or
  wants a typed decision over retrieved state.
license: MIT
---

# agastya — structure-aware retrieval

Agastya is an MCP server that indexes PDFs by their table of contents and
retrieves evidence from the active index. No fine-tuning is required. A CUDA
GPU is recommended (the embedding model runs on 8GB); CPU works but is slow.

## Tools

### config(folder, model="local", out=None, parquet=False) -> dict
Builds an index for every PDF in `folder` and makes it the active index.
Output goes to `<folder>/agastya_index/` unless `out` is given. `model` is
`"local"` or `"cloud"` and records which decision backend the index is
configured for; it does not affect retrieval.

- PDFs need embedded bookmarks. A PDF without an outline fails or indexes
  poorly.
- The call is slow: it splits and embeds the whole corpus. Say so before
  running it on a large folder.
- The selected index persists in `~/.config/agastya/active-index.json` (or
  under `XDG_CONFIG_HOME`), so later sessions reuse it.

### query(question, leaf=False, relevant_match=False) -> dict
Retrieves evidence from the active index.

- `leaf=False` (default) returns the strongest chunks, the R@5 evidence set.
- `leaf=True` returns whole leaf sections with their chapter path, which is
  usually what a citation needs.
- `relevant_match=True` adds a relevance verdict computed from the best leaf
  similarity.

Scores and the relevance verdict are heuristics, not calibrated
probabilities. Do not present them as confidence.

### decide(state, questions, backend="local") -> dict
Runs a typed decision (choice / score / noul) through the local unsloth model
or the cloud Jev backend. Optional: the `decision` extra and a configured
backend are required.

## Recommended workflow

1. If no index is active, ask for the absolute path of the PDF folder and call
   `config(folder=...)` once. Do not guess a path.
2. Call `query` with the user's question. Prefer `leaf=True` when the answer
   needs a section citation; use the default chunk mode when the agent needs
   compact evidence.
3. Read the returned sections and answer only from them. If the result is
   empty or clearly off-topic, rephrase the question before rebuilding
   anything.
4. Call `decide` only when the user asks for a typed decision, not as a
   substitute for reading the retrieved evidence.

## Caveats

- The MCP `query` path ranks with dense embeddings only; it does not run the
  hierarchy walk used in the published benchmarks. Describe it that way.
- The first query after a restart re-embeds the corpus in memory and is slow.
- Book PDFs are not distributed with the repository. Never fabricate a corpus.
- `AGASTYA_INDEX` in the server environment selects a specific existing index
  and overrides the persisted selection.

## Connecting agastya to an agent

Registration guides for every supported agent live in
`agent-instructions/agents/` of this repository. Canonical instructions:
`agent-instructions/AGENTS.md`.
