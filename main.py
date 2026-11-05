#!/usr/bin/env python3
"""Agastya entry point: build a corpus from PDFs, then serve it over MCP.

Build (every *.pdf in the folder, timed, per book):
    /usr/bin/python3.14 main.py
    /usr/bin/python3.14 main.py --data data --out data/out --no-parquet
    /usr/bin/python3.14 main.py --skip-unchanged

Serve as an MCP server over stdio (Claude Code, Codex, Hermes, Cursor, ...):
    /usr/bin/python3.14 main.py --mcp

MCP tools:
    config(folder, model, out, parquet)
        Split every PDF in `folder` into <folder>/agastya_index/<book>/ and make
        that index active. `model` picks the decision backend ("local" | "cloud")
        recorded with the index. Returns the books that were processed, skipped
        or failed.
    query(question, leaf, relevant_match)
        Ask the active index. leaf=False returns the strongest chunks, leaf=True
        returns the leaves themselves. relevant_match=True adds a relevance
        verdict from the top similarity. Evidence in, answer written by the
        calling model.
    decide(state, questions, backend)
        Typed decision through decision_model.py (cloud Jev or local unsloth).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from split_pdf import split_pdf

BASE = Path(__file__).resolve().parent
INDEX_NAME = "agastya_index"
DOC_CHAR_LIMIT = 8000      # same cap core.py embeds
LEAF_TOP_K = 3             # leaves considered before chunk expansion
CHUNKS_PER_LEAF = 5        # chunks returned per candidate leaf
RELEVANCE_FLOOR = 0.50     # cosine floor for "this looks relevant at all"
BACKENDS = ("local", "cloud")


def state_file() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "agastya" / "active-index.json"


def clock(seconds: float) -> str:
    """Human-readable duration: 8.4 s, 2 min 04.1 s, 1 h 03 min."""
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, rest = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)} min {rest:04.1f} s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours} h {minutes} min"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _unchanged(pdf: Path, target: Path) -> bool:
    book = target / "book.json"
    if not book.exists():
        return False
    try:
        return json.loads(book.read_text(encoding="utf-8"))["sha256"] == sha256(pdf)
    except (json.JSONDecodeError, KeyError):
        return False


# --------------------------------------------------------------------------
# Build: the loop main.py always wanted.
# --------------------------------------------------------------------------
def build(
    data: Path,
    out: Path,
    *,
    parquet: bool = True,
    skip_unchanged: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    data = Path(data).expanduser().resolve()
    out = Path(out).expanduser().resolve()

    if not data.is_dir():
        raise FileNotFoundError(f"no such folder: {data}")
    pdfs = sorted(p for p in data.iterdir() if p.suffix.lower() == ".pdf")
    if not pdfs:
        raise FileNotFoundError(f"no PDFs in {data}")

    if verbose:
        print(f"{len(pdfs)} PDF(s) in {data} -> {out}"
              f"{' (parquet)' if parquet else ''}\n")

    started = time.perf_counter()
    books, skipped, failed = [], [], []

    for pdf in pdfs:
        target = out / pdf.stem
        if skip_unchanged and _unchanged(pdf, target):
            if verbose:
                print(f"  skip {pdf.name} (unchanged)")
            skipped.append(pdf.name)
            continue

        began = time.perf_counter()
        try:
            result = split_pdf(pdf, target, parquet=parquet)
        except Exception as error:  # one bad book must not stop the batch
            if verbose:
                print(f"  FAIL {pdf.name}: {type(error).__name__}: {error}")
            failed.append({"pdf": pdf.name, "error": f"{type(error).__name__}: {error}"})
            continue
        elapsed = time.perf_counter() - began

        stats = result["stats"]
        if verbose:
            print(f"  {pdf.name}: {result['pages']} pages, {stats['leaves']} leaves, "
                  f"median {stats['leaf_chars_median']} chars — {clock(elapsed)}")
        books.append({
            "pdf": pdf.name,
            "book": pdf.stem,
            "pages": result["pages"],
            "leaves": stats["leaves"],
            "nodes": stats["nodes"],
            "seconds": round(elapsed, 1),
        })

    total = time.perf_counter() - started
    if verbose:
        print(f"\nprocessed {len(books)}, skipped {len(skipped)}, failed {len(failed)} "
              f"in {clock(total)}")

    return {
        "data": str(data),
        "out": str(out),
        "books": books,
        "skipped": skipped,
        "failed": failed,
        "seconds": round(total, 1),
    }


# --------------------------------------------------------------------------
# Active index, loaded on demand and embedded once per process.
# --------------------------------------------------------------------------
class Index:
    """Leaves + chunks of one agastya_index folder, with lazy embeddings."""

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        config_path = self.root / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"{self.root} has no config.json — run config(folder=...) first"
            )
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.model_choice = self.config.get("model", "local")

        self.leaves: list[dict] = []
        for book in self.config.get("books", []):
            path = self.root / book["book"] / "leaves.jsonl"
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    leaf = json.loads(line)
                    leaf["book"] = book["book"]
                    self.leaves.append(leaf)

        self.leaf_texts = [
            f"{leaf['title']}\n{leaf['content'][:DOC_CHAR_LIMIT]}" for leaf in self.leaves
        ]
        self.chunk_texts: list[str] = []
        self.chunk_bounds: list[tuple[int, int]] = []
        self.leaf_vecs: np.ndarray | None = None
        self.chunk_vecs: np.ndarray | None = None
        self._embedder = None

    def _ensure_vectors(self) -> None:
        if self.leaf_vecs is not None:
            return
        if not self.leaves:
            raise RuntimeError("index contains no readable sections; check the PDF outlines and rebuild")
        from gemma import chunk_text, embed_documents, load_model

        self._embedder = load_model()
        self.chunk_texts = []
        self.chunk_bounds = []
        for leaf in self.leaves:
            chunks = chunk_text(leaf["content"])
            self.chunk_bounds.append((len(self.chunk_texts),
                                      len(self.chunk_texts) + len(chunks)))
            self.chunk_texts.extend(chunks)

        self.leaf_vecs = embed_documents(self._embedder, self.leaf_texts, batch_size=2)
        self.chunk_vecs = embed_documents(self._embedder, self.chunk_texts, batch_size=8)

    def search(self, question: str, *, leaf: bool = False,
               relevant_match: bool = False) -> dict[str, Any]:
        from gemma import embed_query, top_k

        self._ensure_vectors()
        assert self._embedder is not None
        assert self.leaf_vecs is not None
        assert self.chunk_vecs is not None
        query_vec = embed_query(self._embedder, question)
        sims = self.leaf_vecs @ query_vec
        order = np.argsort(-sims)

        leaves = []
        for i in order[:LEAF_TOP_K]:
            i = int(i)
            leaf_row = self.leaves[i]
            leaves.append({
                "book": leaf_row["book"],
                "path": leaf_row["path"] + [leaf_row["title"]],
                "page_start": leaf_row["page_start"],
                "page_end": leaf_row["page_end"],
                "similarity": round(float(sims[i]), 4),
                "text": leaf_row["content"] if leaf else leaf_row["content"][:400],
            })

        payload: dict[str, Any] = {
            "question": question,
            "index": str(self.root),
            "model": self.model_choice,
            "leaves": leaves,
        }

        if relevant_match:
            top = float(sims[order[0]]) if len(order) else 0.0
            payload["relevant"] = top >= RELEVANCE_FLOOR
            payload["relevance_floor"] = RELEVANCE_FLOOR

        if not leaf:
            hits = []
            for i in order[:LEAF_TOP_K]:
                i = int(i)
                start, end = self.chunk_bounds[i]
                for chunk_index, score in top_k(
                    query_vec, self.chunk_vecs[start:end], CHUNKS_PER_LEAF
                ):
                    leaf_row = self.leaves[i]
                    hits.append({
                        "book": leaf_row["book"],
                        "path": leaf_row["path"] + [leaf_row["title"]],
                        "page_start": leaf_row["page_start"],
                        "page_end": leaf_row["page_end"],
                        "score": round(float(score), 4),
                        "chunk": self.chunk_texts[start + chunk_index],
                    })
            hits.sort(key=lambda hit: -hit["score"])
            payload["chunks"] = hits[:5]   # the R@5 evidence set for the caller
        return payload


_ACTIVE: Index | None = None
_LOCK = threading.Lock()


def get_index(root: Path | str | None = None) -> Index:
    global _ACTIVE
    with _LOCK:
        if root is not None:
            _ACTIVE = Index(Path(root))
        if _ACTIVE is None:
            configured = os.environ.get("AGASTYA_INDEX")
            if not configured and state_file().exists():
                configured = json.loads(state_file().read_text(encoding="utf-8"))["index"]
            if not configured:
                raise RuntimeError("no index loaded — call config(folder=...) first or set AGASTYA_INDEX")
            _ACTIVE = Index(Path(configured))
        return _ACTIVE


# --------------------------------------------------------------------------
# MCP server
# --------------------------------------------------------------------------
def make_server():
    try:
        # mcp 2.x renamed FastMCP to MCPServer; keep 1.x working too.
        from mcp.server.mcpserver import MCPServer as FastMCP
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP  # type: ignore[no-redef]
        except ImportError as error:  # pragma: no cover - only when mcp is absent
            raise SystemExit(
                "the MCP server needs `mcp`; install Agastya with pip install ."
            ) from error

    server = FastMCP(
        name="agastya",
        instructions=(
            "Agastya builds a ToC-structured index from PDF books and answers "
            "questions over it. Call config(folder) once to build/select an index, "
            "then query(question). Use decide(...) for typed decisions through the "
            "cloud (Jev) or local (unsloth) decision model."
        ),
    )

    @server.tool()
    def config(
        folder: str,
        model: str = "local",
        out: str | None = None,
        parquet: bool = False,
    ) -> dict[str, Any]:
        """Build an index for every PDF in `folder` and make it the active one.

        Output goes to <folder>/agastya_index/<book>/ (or `out` when given).
        `model` records which decision backend ("local" or "cloud") the index was
        configured for.
        """
        if model not in BACKENDS:
            return {"ok": False, "error": f"model must be one of {BACKENDS}"}
        folder_path = Path(folder).expanduser().resolve()
        index_root = Path(out).expanduser().resolve() if out else folder_path / INDEX_NAME
        try:
            summary = build(folder_path, index_root, parquet=parquet, verbose=False)
        except FileNotFoundError as error:
            return {"ok": False, "error": str(error)}

        if not summary["books"]:
            return {"ok": False, "error": "No PDFs were successfully indexed", "failed": summary["failed"]}

        manifest = {
            "folder": str(folder_path),
            "model": model,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "books": summary["books"],
            "failed": summary["failed"],
        }
        index_root.mkdir(parents=True, exist_ok=True)
        (index_root / "config.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        active_file = state_file()
        active_file.parent.mkdir(parents=True, exist_ok=True)
        active_file.write_text(json.dumps({"index": str(index_root)}), encoding="utf-8")
        with _LOCK:
            global _ACTIVE
            _ACTIVE = Index(index_root)

        return {
            "ok": not summary["failed"],
            "index": str(index_root),
            "model": model,
            "books": summary["books"],
            "failed": summary["failed"],
            "seconds": summary["seconds"],
        }

    @server.tool()
    def query(question: str, leaf: bool = False, relevant_match: bool = False) -> dict[str, Any]:
        """Retrieve evidence for a question from the active index.

        leaf=False returns the strongest chunks (the R@5 evidence set),
        leaf=True returns whole leaves. relevant_match=True adds a relevance
        verdict computed from the best leaf similarity.
        """
        try:
            return get_index().search(
                question, leaf=leaf, relevant_match=relevant_match
            )
        except Exception as error:
            return {"ok": False, "error": str(error)}

    @server.tool()
    def decide(state: str, questions: dict, backend: str = "local") -> dict[str, Any]:
        """Typed decision (choice / score / noul) via cloud Jev or the local model."""
        if backend not in BACKENDS:
            return {"ok": False, "error": f"backend must be one of {BACKENDS}"}
        try:
            import decision_model
            request = decision_model.DecideRequest(
                state=state, questions=questions, backend=backend
            )
            return decision_model.decide(request)
        except Exception as error:  # HTTPException from the backend, or load failure
            status = getattr(error, "status_code", None)
            detail = getattr(error, "detail", str(error))
            return {"ok": False, "status": status, "error": detail}

    return server


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=BASE / "data",
                        help="folder to scan for PDFs (default: data/)")
    parser.add_argument("--out", type=Path, default=BASE / "data" / "out",
                        help="corpus output folder (default: data/out/)")
    parser.add_argument("--parquet", dest="parquet", action="store_true", default=True,
                        help="also write leaves.parquet (default)")
    parser.add_argument("--no-parquet", dest="parquet", action="store_false",
                        help="skip leaves.parquet")
    parser.add_argument("--skip-unchanged", action="store_true",
                        help="skip books whose book.json sha256 matches the PDF")
    parser.add_argument("--mcp", action="store_true",
                        help="run the MCP server on stdio instead of building")
    args = parser.parse_args()

    if args.mcp:
        make_server().run(transport="stdio")
        return 0

    try:
        summary = build(args.data, args.out, parquet=args.parquet,
                        skip_unchanged=args.skip_unchanged)
    except FileNotFoundError as error:
        print(error)
        return 1

    if summary["books"]:
        print(f"corpus: {summary['out']}")
        print("serve it: /usr/bin/python3.14 core.py --serve   "
              "or   /usr/bin/python3.14 main.py --mcp")
    return 1 if summary["failed"] else 0


def cli() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    raise SystemExit(main())
