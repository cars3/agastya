#!/usr/bin/env python3
"""bench_hybrid.py -- does hybrid fusion + a cross-encoder reranker beat dense
EmbeddingGemma on the public SearchTome release?

One shared rerank pass over a per-query candidate pool (dense top-20, RRF hybrid
top-20, Jev walk leaves, Jev top-3) yields every row from the same scores:

  - bm25            lexical baseline
  - dense           EmbeddingGemma alone
  - hybrid_rrf      BM25 + dense, reciprocal rank fusion
  - dense_rerank    dense top-20 reordered by the cross-encoder
  - hybrid_rerank   RRF top-20 reordered by the cross-encoder
  - agastya         full pool (walk + hybrid + dense) reordered by the cross-encoder

Gold queries are the dataset's own test queries. Reranker: BAAI/bge-reranker-v2-m3,
off-the-shelf, zero fine-tuning on this data.

    .venv/bin/python bench_hybrid.py --workers 1
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from gemma import embed_documents, load_model

DOMAINS = ["education", "finance", "law", "medicine", "natural_sciences", "social_sciences"]
CACHE_DIR = Path(__file__).resolve().parent / "data" / "searchtome"
DOC_CHAR_LIMIT = 8000
RERANK_CHAR_LIMIT = 800
JEV_JOIN = 0.85
TOP_RETRIEVE = 50
TOP_POOL = 20
POOL_CAP = 28
RRF_K = 60


def rrf_score(rank_maps: list[dict[str, int]], title: str) -> float:
    score = 0.0
    for rm in rank_maps:
        if title in rm:
            score += 1.0 / (RRF_K + rm[title] + 1)
    return score


def rank_of(gold: str, ordered: list[str]) -> int:
    try:
        return ordered.index(gold)
    except ValueError:
        return len(ordered) + 1000


def metrics(ranks: list[int]) -> dict:
    arr = np.asarray(ranks)
    return {
        "R@1": float(np.mean(arr < 1)),
        "R@3": float(np.mean(arr < 3)),
        "R@5": float(np.mean(arr < 5)),
        "nDCG@3": float(np.mean([1 / math.log2(r + 2) if r < 3 else 0.0 for r in arr])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all test queries")
    ap.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3")
    args = ap.parse_args()

    t_start = time.time()
    print("loading EmbeddingGemma ...", flush=True)
    gemma = load_model()

    jobs = []  # per query: dict(domain, gold, bm25_order, dense_order, rrf_order, pool)
    for domain in DOMAINS:
        corpus = load_dataset("meghanadh/SearchTome", f"{domain}_corpus", split="corpus")
        tests = load_dataset("meghanadh/SearchTome", f"{domain}_queries", split="test")
        titles = [row["title"] for row in corpus]
        contents = [row["content"] or "" for row in corpus]
        title_set = set(titles)
        rows = [r for r in tests if r["gold_doc_title"] in title_set]
        if args.limit:
            rows = rows[: args.limit]

        print(f"{domain}: {len(titles)} sections, {len(rows)} queries", flush=True)
        doc_vecs = embed_documents(gemma, [f"{t}\n{c[:DOC_CHAR_LIMIT]}" for t, c in zip(titles, contents)],
                                   batch_size=4)
        q_vecs = gemma.encode([r["query"] for r in rows], prompt_name="SearchQuery",
                              normalize_embeddings=True, batch_size=16,
                              convert_to_numpy=True, show_progress_bar=False)
        bm25 = BM25Okapi([f"{t} {c[:DOC_CHAR_LIMIT]}".lower().split()
                          for t, c in zip(titles, contents)])

        jev_cache_path = CACHE_DIR / f"jevalks_{domain}.json"
        jev_cache = json.loads(jev_cache_path.read_text()) if jev_cache_path.exists() else {}

        for qi, row in enumerate(rows):
            query, gold = row["query"], row["gold_doc_title"]
            sims = doc_vecs @ q_vecs[qi]
            dense_order = [titles[int(i)] for i in np.argsort(-sims)[:TOP_RETRIEVE]]

            bm_scores = bm25.get_scores(query.lower().split())
            bm25_order = [titles[int(i)] for i in np.argsort(-bm_scores)[:TOP_RETRIEVE]]

            # full rankings for the plain rows use the same top-50 (sufficient for R@5)
            dense_rank_map = {t: i for i, t in enumerate(dense_order)}
            bm25_rank_map = {t: i for i, t in enumerate(bm25_order)}
            union = list(dict.fromkeys(dense_order + bm25_order))
            rrf_order = sorted(union, key=lambda t: -rrf_score([dense_rank_map, bm25_rank_map], t))

            # Jev candidates from the cached flat noul scores
            walked: list[str] = []
            jev_top: list[str] = []
            entry = jev_cache.get(str(row["query_id"]))
            if entry:
                nouls = np.asarray(entry)
                order = np.argsort(-nouls)
                jev_top = [titles[int(i)] for i in order[:3]]
                best = float(nouls.max()) if nouls.size else 0.0
                walked = [titles[int(i)] for i in order if best > 0 and nouls[i] >= JEV_JOIN * best]

            pool = list(dict.fromkeys(walked + jev_top + rrf_order[:TOP_POOL] + dense_order[:TOP_POOL]))
            if len(pool) > POOL_CAP:
                pool = pool[:POOL_CAP]

            jobs.append({
                "domain": domain,
                "query": query,
                "gold": gold,
                "bm25_order": bm25_order,
                "dense_order": dense_order,
                "rrf_order": rrf_order,
                "dense20": dense_order[:TOP_POOL],
                "rrf20": rrf_order[:TOP_POOL],
                "pool": pool,
                "passage": {t: f"{t}\n{contents[titles.index(t)][:RERANK_CHAR_LIMIT]}" for t in pool},
            })

    del gemma
    torch.cuda.empty_cache()
    print(f"retrieval done in {time.time()-t_start:.0f}s, loading reranker {args.reranker} ...", flush=True)

    reranker = CrossEncoder(args.reranker, device="cuda" if torch.cuda.is_available() else "cpu",
                            max_length=320)
    print("reranking ...", flush=True)
    t0 = time.time()
    BATCH = 16
    for done, job in enumerate(jobs, 1):
        pairs = [(job["query"], job["passage"][t]) for t in job["pool"]]
        scores = reranker.predict(pairs, batch_size=BATCH, show_progress_bar=False)
        by_score = [t for t, _ in sorted(zip(job["pool"], scores), key=lambda x: -x[1])]
        score_of = {t: float(s) for t, s in zip(job["pool"], scores)}
        job["pool_rerank"] = by_score
        job["dense_rerank"] = sorted(job["dense20"], key=lambda t: -score_of.get(t, -1e9))
        job["rrf_rerank"] = sorted(job["rrf20"], key=lambda t: -score_of.get(t, -1e9))
        if done % 250 == 0 or done == len(jobs):
            print(f"  rerank {done}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)

    rows = {
        "bm25": [], "dense": [], "hybrid_rrf": [],
        "dense_rerank": [], "hybrid_rerank": [], "agastya": [],
    }
    per_domain = {}
    for job in jobs:
        for key, order_key in [
            ("bm25", "bm25_order"), ("dense", "dense_order"), ("hybrid_rrf", "rrf_order"),
            ("dense_rerank", "dense_rerank"), ("hybrid_rerank", "rrf_rerank"),
            ("agastya", "pool_rerank"),
        ]:
            rows[key].append((job["domain"], rank_of(job["gold"], job[order_key])))

    print("\n=== SearchTome full test split ===", flush=True)
    summary = {}
    for key, pairs in rows.items():
        ranks = [r for _, r in pairs]
        summary[key] = metrics(ranks)
        dom_names = sorted({d for d, _ in pairs})
        per_domain[key] = {
            d: metrics([r for dd, r in pairs if dd == d]) for d in dom_names
        }
    for key in rows:
        m = summary[key]
        print(f"  {key:14s} R@1={m['R@1']:.3f}  R@3={m['R@3']:.3f}  R@5={m['R@5']:.3f}  nDCG@3={m['nDCG@3']:.3f}",
              flush=True)
    print(f"\ntotal wall time: {(time.time()-t_start)/60:.1f} min", flush=True)

    out = CACHE_DIR / "results_hybrid.json"
    out.write_text(json.dumps({
        "queries": len(jobs),
        "reranker": args.reranker,
        "summary": summary,
        "per_domain": per_domain,
        "published_r1": {"STAIR": 82.6, "DSI": 76.9, "DPR": 68.7, "BM25": 59.5, "Mistral": 13.8},
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
