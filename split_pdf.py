#!/usr/bin/env python3
"""
split_pdf.py -- turn a PDF's table of contents into a STAIR-style section tree.

The hierarchy comes from the PDF's embedded outline (bookmarks). Each heading is
located on the page as a large-font line, so a section's text is cut at the exact
heading rather than at a page boundary. Nodes with no children are the leaves --
the retrieval units a ToC retriever like STAIR predicts.

Use it as a function:

    from split_pdf import split_pdf
    result = split_pdf("data/books/slp3_ed3book.pdf")
    leaves = result["leaves"]          # list of dicts, corpus ready

    # choose where the files go, and also emit Parquet
    result = split_pdf("book.pdf", "out/mybook", parquet=True)

    # return the data without writing anything
    result = split_pdf("book.pdf", write=False)

Outputs (when write=True):
    <out_dir>/toc.json        full tree
    <out_dir>/book.json       manifest: source, sha256, pages, counts
    <out_dir>/leaves.jsonl    one JSON object per leaf (the corpus)
    <out_dir>/leaves.parquet  optional, needs pyarrow

CLI:
    python split_pdf.py book.pdf [--out DIR] [--doc-id ID] [--parquet]
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pymupdf

BODY_TOP = 100           # running heads sit above this
BODY_BOTTOM_MARGIN = 40  # page numbers sit below height - margin
HEADING_MIN_SIZE = 11.7  # body is 10pt, sub-subheadings are 11.955pt
MATCH_FLOOR = 0.5        # fuzzy score below this = heading not found
BACKMATTER = {"bibliography", "subject index", "index"}
ID_MAX = 100             # section_id slug budget before a hash suffix is added


@dataclass
class Heading:
    page: int
    y0: float
    y1: float
    text: str
    size: float
    number: str = ""


@dataclass
class Node:
    title: str
    level: int
    page: int
    number: str = ""
    y0: float = BODY_TOP
    y1: float = BODY_TOP
    matched: bool = True
    section_id: str = ""
    parent_section_id: str | None = None
    children: list["Node"] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return not self.children


def norm(text: str) -> str:
    """Compare headings ignoring numbering, case and punctuation."""
    text = re.sub(r"^\s*\d+(\.\d+)*\s+", "", text.strip())
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def unique_section_id(full_path: str, taken: dict[str, int]) -> str:
    """Readable slug, hash-suffixed only when truncation would collide."""
    base = slug(full_path)
    if len(base) > ID_MAX:
        base = base[:ID_MAX].rstrip("-") + "-" + hashlib.sha1(base.encode()).hexdigest()[:8]
    if base in taken:
        taken[base] += 1
        return f"{base}-{taken[base]}"
    taken[base] = 1
    return base


def collect_headings(doc: pymupdf.Document) -> dict[int, list[Heading]]:
    """Large-font lines in the body area, per page, plus their section numbers."""
    by_page: dict[int, list[Heading]] = {}
    for pno in range(doc.page_count):
        lines: list[Heading] = []
        for block in doc[pno].get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                if not line["spans"]:
                    continue
                size = max(span["size"] for span in line["spans"])
                text = "".join(span["text"] for span in line["spans"]).strip()
                y0, y1 = line["bbox"][1], line["bbox"][3]
                if (text and size >= HEADING_MIN_SIZE
                        and BODY_TOP <= y0 <= doc[pno].rect.height - BODY_BOTTOM_MARGIN):
                    lines.append(Heading(pno, y0, y1, text, size))
        # a heading number ("1.6.3") is its own line at the same y as the title
        numbers = [h for h in lines if re.fullmatch(r"\d+(\.\d+)*", h.text)]
        for h in lines:
            for num in numbers:
                if abs(num.y0 - h.y0) < 3 and num is not h and not re.fullmatch(r"\d+(\.\d+)*", h.text):
                    h.number = num.text
        by_page[pno] = lines
    return by_page


def match_toc(doc: pymupdf.Document, toc: list[list]) -> list[Node]:
    """Attach a page position to every outline entry, in document order."""
    headings = collect_headings(doc)
    nodes: list[Node] = []
    cursor = (-1, 0.0)

    for level, title, page in toc:
        p0 = page - 1
        best: tuple[float, Heading] | None = None
        # bookmark page numbers drift by a page or two, so search a window
        # around the declared page; the cursor keeps results in document order
        start = max(0, min(p0, cursor[0]) - 1)
        end = min(doc.page_count, max(p0, cursor[0]) + 5)
        for p in range(start, end):
            for h in headings.get(p, []):
                if (p, h.y0) < cursor:
                    continue
                score = difflib.SequenceMatcher(None, norm(title), norm(h.text)).ratio()
                if best is None or score > best[0]:
                    best = (score, h)

        if best and best[0] >= MATCH_FLOOR:
            h = best[1]
            node = Node(title=title, level=level, page=h.page, number=h.number,
                        y0=h.y0, y1=h.y1, matched=True)
            cursor = (h.page, h.y0)
        else:
            # no heading found: keep the position monotonic so page ranges stay valid
            fallback_page = max(p0, cursor[0])
            if fallback_page == cursor[0] and fallback_page + 1 < doc.page_count:
                fallback_page += 1
            node = Node(title=title, level=level, page=fallback_page, matched=False)
            cursor = (fallback_page, BODY_TOP)
        nodes.append(node)
    return nodes


def build_tree(nodes: list[Node]) -> list[Node]:
    roots: list[Node] = []
    stack: list[Node] = []
    for node in nodes:
        while stack and stack[-1].level >= node.level:
            stack.pop()
        (stack[-1].children if stack else roots).append(node)
        stack.append(node)
    return roots


def page_text(doc: pymupdf.Document, page: int, top: float, bottom: float) -> str:
    rect = doc[page].rect
    clip = pymupdf.Rect(rect.x0 + 50, top, rect.x1 - 50, bottom)
    if clip.height <= 0:
        return ""
    return doc[page].get_text("text", clip=clip, sort=True)


def clean(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)   # de-hyphenate line breaks
    return re.sub(r"\s+", " ", text).strip()


def leaf_content(doc: pymupdf.Document, node: Node, nxt: Node | None) -> str:
    bottom = doc.page_count - 1
    end_page = nxt.page if nxt else bottom
    end_y = nxt.y0 if nxt else doc[end_page].rect.height - BODY_BOTTOM_MARGIN
    parts: list[str] = []
    for p in range(node.page, end_page + 1):
        top = node.y1 + 2 if p == node.page else BODY_TOP
        bot = end_y if p == end_page else doc[p].rect.height - BODY_BOTTOM_MARGIN
        parts.append(page_text(doc, p, top, bot))
    return clean("\n".join(parts))


def split_pdf(
    pdf_path: str | Path,
    out_dir: str | Path | None = None,
    *,
    doc_id: str | None = None,
    write: bool = True,
    parquet: bool = False,
) -> dict:
    """Split one PDF into a ToC tree and section-level leaves.

    Args:
        pdf_path: path to a PDF with an embedded outline.
        out_dir: where to write outputs. Defaults to a folder next to the PDF
            named after it (book.pdf -> ./book/). Ignored when write=False.
        doc_id: override the id derived from the filename.
        write: set False to return data without touching disk.
        parquet: also write leaves.parquet (requires pyarrow).

    Returns:
        {"doc_id", "source", "pages", "sha256", "tree", "leaves", "stats", "paths"}
    """
    pdf_path = Path(pdf_path).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    doc = pymupdf.open(pdf_path)
    toc = doc.get_toc(simple=True)
    if not toc:
        raise ValueError(f"{pdf_path.name} has no embedded outline; cannot build a section tree. Agastya builds its tree from PDF bookmarks. Add bookmarks in a PDF editor (or with pymupdf set_toc) and rerun.")

    roots = build_tree(match_toc(doc, toc))
    doc_id = doc_id or slug(pdf_path.stem)

    # assign readable, unique ids to every node; leaves inherit their parent's id
    taken: dict[str, int] = {}

    def assign(node: Node, ancestors: list[str], parent_id: str | None) -> None:
        full = ancestors + [node.title]
        node.section_id = unique_section_id(" / ".join(full), taken)
        node.parent_section_id = parent_id
        for child in node.children:
            assign(child, full, node.section_id)

    for root in roots:
        assign(root, [], None)

    # flat document order; a leaf ends where the next node starts
    flat: list[tuple[Node, list[Node]]] = []

    def walk(node: Node, ancestors: list[Node]) -> None:
        flat.append((node, ancestors))
        for child in node.children:
            walk(child, ancestors + [node])

    for root in roots:
        walk(root, [])

    leaves: list[dict] = []
    skipped = 0
    for i, (node, ancestors) in enumerate(flat):
        if not node.is_leaf:
            continue
        if node.level <= 2 and node.title.strip().lower() in BACKMATTER:
            skipped += 1
            continue
        nxt = flat[i + 1][0] if i + 1 < len(flat) else None
        content = leaf_content(doc, node, nxt)
        leaves.append({
            "leaf_id": hashlib.sha1(f"{doc_id}|{node.section_id}".encode()).hexdigest()[:12],
            "doc_id": doc_id,
            "section_id": node.section_id,
            "parent_section_id": node.parent_section_id,
            "title": node.title,
            "path": [n.title for n in ancestors],
            "level": node.level,
            "number": node.number,
            "page_start": node.page + 1,
            "page_end": (nxt.page + 1) if nxt else doc.page_count,
            "heading_matched": node.matched,
            "chars": len(content),
            "words": len(content.split()),
            "approx_tokens": (len(content) + 3) // 4,
            "content_sha1": hashlib.sha1(content.encode()).hexdigest(),
            "content": content,
        })

    for i, leaf in enumerate(leaves):
        leaf["prev_leaf_id"] = leaves[i - 1]["leaf_id"] if i else None
        leaf["next_leaf_id"] = leaves[i + 1]["leaf_id"] if i + 1 < len(leaves) else None

    def tree_dict(node: Node) -> dict:
        return {
            "section_id": node.section_id,
            "title": node.title,
            "level": node.level,
            "number": node.number,
            "page": node.page + 1,
            "heading_matched": node.matched,
            "children": [tree_dict(c) for c in node.children],
        }

    levels: dict[str, int] = {}
    for node, _ in flat:
        levels[str(node.level)] = levels.get(str(node.level), 0) + 1
    lengths = sorted(leaf["chars"] for leaf in leaves) or [0]
    stats = {
        "outline_entries": len(toc),
        "nodes": len(flat),
        "levels": levels,
        "leaves": len(leaves),
        "empty_leaves": sum(1 for leaf in leaves if leaf["chars"] == 0),
        "unmatched_headings": sum(1 for n, _ in flat if not n.matched),
        "skipped_backmatter": skipped,
        "leaf_chars_min": lengths[0],
        "leaf_chars_median": lengths[len(lengths) // 2],
        "leaf_chars_max": lengths[-1],
    }

    sha = hashlib.sha256()
    with pdf_path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            sha.update(block)

    result = {
        "doc_id": doc_id,
        "source": str(pdf_path),
        "pages": doc.page_count,
        "sha256": sha.hexdigest(),
        "tree": [tree_dict(r) for r in roots],
        "leaves": leaves,
        "stats": stats,
        "paths": {},
    }
    if not write:
        return result

    out = Path(out_dir).expanduser().resolve() if out_dir else pdf_path.parent / pdf_path.stem
    out.mkdir(parents=True, exist_ok=True)
    (out / "toc.json").write_text(json.dumps(
        {"doc_id": doc_id, "source": str(pdf_path), "pages": doc.page_count,
         "tree": result["tree"]}, indent=2))
    (out / "leaves.jsonl").write_text("".join(json.dumps(leaf) + "\n" for leaf in leaves))
    (out / "book.json").write_text(json.dumps({
        "doc_id": doc_id,
        "source": str(pdf_path),
        "sha256": result["sha256"],
        "pages": doc.page_count,
        "parser": "split_pdf.py",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "counts": stats,
    }, indent=2))
    result["paths"] = {"toc": str(out / "toc.json"), "leaves": str(out / "leaves.jsonl"),
                       "book": str(out / "book.json")}

    if parquet:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("parquet=True needs pyarrow: uv pip install pyarrow") from exc
        pq.write_table(pa.Table.from_pylist(leaves), out / "leaves.parquet", compression="zstd")
        result["paths"]["parquet"] = str(out / "leaves.parquet")

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Split a PDF into ToC leaves.")
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="output folder (default: <pdf folder>/<pdf stem>)")
    ap.add_argument("--doc-id", default=None)
    ap.add_argument("--parquet", action="store_true", help="also write leaves.parquet")
    args = ap.parse_args()

    result = split_pdf(args.pdf, args.out, doc_id=args.doc_id, parquet=args.parquet)
    s = result["stats"]
    print(f"{Path(result['source']).name}: {result['pages']} pages, {s['outline_entries']} outline entries")
    print(f"levels: {s['levels']} | leaves: {s['leaves']} | unmatched: {s['unmatched_headings']} "
          f"| skipped back-matter: {s['skipped_backmatter']}")
    print(f"leaf chars: min={s['leaf_chars_min']} median={s['leaf_chars_median']} max={s['leaf_chars_max']}")
    for name, path in result["paths"].items():
        print(f"wrote {name:<8} {path}")
    print("\nfirst 6 leaves:")
    for leaf in result["leaves"][:6]:
        print(f"  p{leaf['page_start']:>3}  {' > '.join(leaf['path'] + [leaf['title']])[:110]}")


if __name__ == "__main__":
    main()
