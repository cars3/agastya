# Benchmarks

| Script | Measures | Environment |
| --- | --- | --- |
| book | Book retrieval quality | Serving stack from pyproject |
| core | Core retrieval performance | Serving stack from pyproject |
| searchtome | SearchTome retrieval quality | Serving stack from pyproject and a jev.py client |
| library | Library retrieval quality | Serving stack from pyproject and a jev.py client |
| hybrid | Hybrid retrieval quality | Serving stack from pyproject; downloads BAAI/bge-reranker-v2-m3, about 2.3GB, on first run |

Benchmarks are separate from the MCP install. Use `pip install '.[bench]'` when requests is missing.

## Performance at a glance

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

These measurements cover distinct settings: SearchTome is an external benchmark, while the deep textbook and NIST runs test navigation over larger hierarchical documents.

## SearchTome: 6,005 queries

| Retrieval system | Recall@1 | Recall@3 | Gold recall at retrieval budget |
| --- | --- | --- | --- |
| BM25 | 47.8% | 68.1% | - |
| Jev-only ranking | 54.2% | 75.7% | - |
| EmbeddingGemma 2 | 64.2% | 84.5% | 84.5% at 3 leaves |
| Agastya RAG | 64.2% | 84.5% | 89.1% at 3.6 leaves |

Agastya does not improve top-one ranking over EmbeddingGemma here. Its gain is widening the relevant candidate set while keeping the retrieval budget small.

## Deep document

Tested on a 389-leaf textbook.

| Configuration | Leaf hit rate | Leaves/query | Median latency |
| --- | --- | --- | --- |
| Strict Jev hierarchy walk | 45% | 1.5 | 0.54s |
| Agastya combined pipeline | 95% | 4.2 | 0.42s |

A structural walk alone can prune the correct answer; the embedding union recovers it.

## Enterprise compliance

NIST SP 800-53 Revision 5, 331 control sections.

| System | Result |
| --- | --- |
| Dense EmbeddingGemma | 81.7% Recall@1 |
| Dense EmbeddingGemma | 93.3% Recall@3 |
| Agastya RAG | 93.3% leaf hit at 5.5 leaves |
| Agastya median latency | 0.40s |

## Attempted upgrades that did not work (measured)

| Attempt | R@1 / R@3 / gold recall (dense: 63.3 / 83.4 / 89.8) |
| --- | --- |
| BM25 alone | 47.1 / 66.4 / 74.5 |
| Hybrid RRF (dense + BM25) | 59.6 / 80.0 / 86.7 |
| Dense + cross-encoder rerank | 42.4 / 62.6 / 72.7 |
| Hybrid + rerank | 42.1 / 62.8 / 73.2 |
| Full stack + rerank | 42.0 / 61.8 / 71.7 |

The reranker loses because a cross-encoder judges short passages while SearchTome documents are long multi-topic sections; probes show the gold section scored 0.002 against 0.04 for an unrelated neighbour. Fusion recovers only partway. Dense ranks, the walk widens the set, and that is the shipped design.

Raw measurements are stored in `results/`.
