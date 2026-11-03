# Agastya RAG

**An IBM STAIR-inspired retrieval system that uses a single thinking process to walk a document's table of contents and reach the right gold-leaf section, fast and efficiently, on one 8GB GPU. No training, no labels, no fine-tuning.**

As an Agent Architect choosing between **Agastya** and **IBM's STAIR** to replace conventional fixed-chunk RAG, **I would choose Agastya (or its hybrid dense + outline-walk pattern) for 95% of real-world production agent architectures**, and reserve **STAIR** strictly for static, high-QPS, closed-world knowledge bases where top-1 precision justifies an ongoing fine-tuning MLOps pipeline.

The reason comes down to a fundamental tension in agent systems: **STAIR optimizes for a static benchmark metric (`Recall@1`), whereas Agastya optimizes for how an LLM agent actually consumes context (`Recall@Budget`, zero-training agility, and auditability).**

---

### 1. Why Agastya Wins for Production Agent Systems

#### A. Agents Care About *Set Recall at a Small Budget*, Not `Recall@1`

STAIR's headline victory is **82.6% R@1** vs. Agastya's **64.2% R@1** (though note the dataset split discrepancy: the public SearchTome release has a lower baseline floor where BM25 hits 47.8% vs. the paper's 59.5%).
However, in an agentic workflow, you almost never feed just **one** leaf section into the synthesizer LLM. Modern context windows effortlessly handle 3 to 5 leaf sections (or their top chunks).

* At an average budget of **3.6 leaves per query**, Agastya's union pipeline (`walked leaves` $\cup$ `top-3 EmbeddingGemma` + `top-5 fallback`) delivers **89.1% gold recall** on SearchTome, **95% leaf hit at 4.2 leaves** on a deep 389-leaf textbook (*SLP3*), and **93.3% hit at 5.5 leaves** on NIST SP 800-53.
* Once the gold leaf is inside that ~4-leaf candidate set, the downstream reasoning agent will find the answer. Paying the massive operational tax of fine-tuning a DSI model just to move the gold leaf from rank #3 to rank #1 yields almost zero marginal gain for the final agent response.

#### B. The "Document Revision" Problem (Zero-Shot Agility vs. Parametric Lock-in)

STAIR is built on a **Differentiable Search Index (DSI)**—it bakes the corpus and its Table of Contents directly into the LLM's weights via fine-tuning.

* **In STAIR:** When NIST releases a revision to SP 800-53, a legal team updates a policy manual, or an engineering team commits new documentation, your retriever is instantly stale. Updating it requires generating synthetic queries for the new sections, running a GPU fine-tuning job, validating that catastrophic forgetting hasn't degraded older chapters, and redeploying model weights.
* **In Agastya:** Updating a document costs a `split_pdf.py` execution and a single startup pass over the leaves with `EmbeddingGemma 2`. It runs on a single 8GB consumer GPU (RTX 2060 SUPER) with **0.40s–0.42s p50 latency** and zero training spend.

#### C. Auditability and Deterministic Fallback ("Kill Logs" vs. Black-Box Weights)

Enterprise agents (especially in compliance, finance, or legal domains) require explainable retrieval traces:

* **STAIR** is a black-box parametric generator. If it outputs section ID `4.2.1` instead of `4.3.2`, you cannot inspect *why* the weights favored one path over the other—and while its hallucination rate is low (less than 0.05%), non-zero structural hallucination is still a risk.
* **Agastya** produces an explicit, auditable **kill log**: every pruned branch has a named ToC section and a calibrated `noul` probability from `Jev` (TypeSafe). Furthermore, because pure tree-walks fail silently when a top-level branch is wrongly pruned (e.g., 45% hit rate on *SLP3* with a strict walk alone), Agastya's **union with top-3 dense embeddings + top-5 fallback** guarantees **zero empty results** and rescues the hit rate to **95%**.

---

### 2. Where Agastya's Architecture Breaks (And How to Design Around It)

The engineering honesty in Agastya's README actually gives an architect the exact blueprint for how to deploy it safely:

1. **The Multi-Book "Library Scale" Bottleneck:**
* *The Problem:* Pooling 6 books into one master tree with untitled chapter nodes tanks the `Jev` walk to **44.5% set-hit**, making the union rescue (88.0% at 11.8 leaves) worse than just taking a deeper flat dense `top-k`.
* *The Agent Fix:* **Never merge multiple books into a single untitled ToC tree.** Instead, give the agent a top-level **Document Router Tool** (or metadata filter) where each book/manual is registered as a distinct, titled entity with its own clean ToC tree. Let the agent pick the book first, then invoke Agastya's `/ask` pipeline on that specific book's tree.


2. **Flat Corpora Under ~300 Sections:**
* *The Problem:* As the README notes, `EmbeddingGemma` alone gets **64.2% R@1 / 84.5% R@3**, and on flat corpora under a few hundred sections, the `Jev` walk adds API calls without improving ranking inside the candidate set.
* *The Agent Fix:* Add a structural gate at ingestion. If a document's ToC depth is less than 2 levels or has fewer than 250 leaves, bypass the `Jev` walk entirely and route directly to flat `EmbeddingGemma` (in `bf16`/`fp32`). Save the `Jev` walk + union strictly for deep, multi-level hierarchies (like 389-leaf textbooks or 331-control compliance catalogs) where expanding the candidate pool from 3.0 to 3.6–4.2 leaves rescues orthogonal sections that dense similarity misses.


3. **Skip the Standard RAG "Best Practices" (BM25 RRF & Cross-Encoders):**
* One of the most valuable findings in Agastya's README is the negative ablation table. Conventional RAG wisdom says *"always add BM25 hybrid RRF and a BGE cross-encoder reranker."* Here, `bge-reranker-v2-m3` **crashed R@1 from 63.3% down to 42.4%** because ToC leaves are long, multi-topic sections rather than short MS-MARCO passages. Accepting Agastya's architecture means trusting the measured data: keep pure `EmbeddingGemma 2` ordering over the unioned leaf/chunk candidates and drop the cross-encoder.



---

### 3. When Would an Architect Actually Choose IBM's STAIR?

You would only choose **STAIR** over Agastya if **all four** of the following conditions are true:

1. **Your corpus is frozen for months at a time** (e.g., canonical medical textbooks, fixed historical tax codes, or static hardware manuals).
2. **You operate at high QPS** where making one external `Jev` API call per ToC level introduces unacceptable network dependency or per-level token/request overhead.
3. **You have a strict `k=1` budget constraint** where downstream compute only permits retrieving a single section, making STAIR's 82.6% R@1 mandatory.
4. **You have dedicated GPU MLOps infrastructure** to generate synthetic query-to-ToC training pairs and fine-tune DSI models every time the knowledge base changes.

---

### Architectural Verdict

| Decision Criterion | Winner | Architectural Rationale |
| --- | --- | --- |
| **Day-1 Production Readiness** | **Agastya** | Zero fine-tuning; ingest a PDF outline and serve `/ask` in minutes on an 8GB GPU. |
| **Dynamic / Living Documents** | **Agastya** | Document edits only require re-splitting and re-embedding leaves, not retraining LLM weights. |
| **End-to-End Agent Answer Quality** | **Agastya** | 89.1%–95% recall within ~4 leaves gives the reasoning LLM everything it needs without DSI training complexity. |
| **Compliance & Auditability** | **Agastya** | Explicit section-level `Jev` prune probabilities in the kill log + zero empty results. |
| **Strict Top-1 Accuracy (`R@1`)** | **IBM STAIR** | Parametric memorization of the ToC beats zero-shot dense/decision-walk ranking at `k=1` (82.6% vs 64.2%). |
| **High-Throughput / Zero External API** | **IBM STAIR** | Single local forward pass of a fine-tuned model vs. sequential per-level `Jev` API calls. |

**My Implementation Plan:** I would deploy **Agastya's `walk + dense union` architecture** for any deep, hierarchical document (>250 leaves, $\ge 3$ ToC levels with clean section titles), pair it with a lightweight **Book/Document Router** at the top level so trees stay per-document rather than pooled, and let **EmbeddingGemma 2 (`fp32`/`bf16`)** handle flat documents directly when the hierarchy isn't deep enough to warrant the `Jev` walk.

---

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

### Attempted upgrades that did not work (measured)

The obvious next moves were built and run on the same 6,005 queries:

| configuration | R@1 | R@3 | R@5 |
| --- | --- | --- | --- |
| dense EmbeddingGemma (deployed ordering) | **63.3** | **83.4** | **89.8** |
| BM25 alone | 47.1 | 66.4 | 74.5 |
| hybrid RRF (BM25 + dense) | 59.6 | 80.0 | 86.7 |
| dense + bge-reranker-v2-m3 | 42.4 | 62.6 | 72.7 |
| hybrid + reranker | 42.1 | 62.8 | 73.2 |
| full stack + reranker | 42.0 | 61.8 | 71.7 |

Root causes, probed rather than assumed: BM25 is 16 points behind dense on this corpus, so a 50/50 RRF fusion drags dense down, and reweighting toward dense just returns dense. The cross-encoder scored gold sections below their neighbors on direct inspection (gold 0.002 against 0.04 for an adjacent section), because these "documents" are long multi-topic sections while the model judges short-passage relevance. Fusing the rerank order back with dense recovered only partway (45% vs 57.5% dense R@1 on a 40-query probe). The deployed ordering therefore stays dense; `bench_hybrid.py` ships with the repo so the same test can be rerun on corpora where these upgrades might actually pay.

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

`core.py` is the product (FastAPI `/ask`, walk + union + chunk ranking). `gemma.py` chunks, embeds, and ranks with EmbeddingGemma 2. `jev.py` is the TypeSafe client. `split_pdf.py` turns a PDF outline into ToC leaves. `bench_book.py`, `bench_core.py`, `bench_searchtome.py`, and `bench_library.py` are the benchmarks above, and `bench_hybrid.py` carries the hybrid/reranker evaluation. Raw numbers live in `results/`; `data/` is not shipped (book PDFs are not mine to redistribute).

## Limitations, honestly

- Rank metrics equal the dense retriever's inside the candidate set, by construction: the walk supplies the recall floor and the guaranteed non-empty result, Gemma does the ordering. Jev is a safety net and a scale mechanism, not a better ranker at these sizes.
- The published rows are a reference point, not a reproduction target. The public release has 6 domains with 8 to 121 sections each, the paper reports 18 books, and our same-data BM25 scores 47.8 against the published 59.5. Trained STAIR and DSI beat this stack on ranking, and this README does not pretend otherwise.
- Library scale is the failure case: pooling six books into one tree with untitled chapter nodes drops the walk to 44.5% set-hit, and the union rescue (88.0% at 11.8 leaves) is worse than a deeper dense top-k. Use per-book trees with titled chapters.
- The book benchmarks generate their own questions (blind in the NIST run), so they validate plumbing and navigation; SearchTome's gold queries are the external check.
- One Jev API call per walk level: not built for high-QPS serving, and cost scales with ToC branching, not corpus size.
- Do not use it for flat corpora under a few hundred sections; a plain dense retriever wins R@1 at zero marginal cost.
- The obvious ranking upgrades were measured on this data and rejected: hybrid fusion (BM25+dense) scored 59.6 R@1 against dense's 63.3, and an off-the-shelf cross-encoder reranker collapsed to 42.4 by mis-ranking long multi-topic sections. Always re-measure on your own corpus with `bench_hybrid.py` before assuming a reranker helps.

## License

MIT
