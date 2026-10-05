"""Parse HTML and plain-text/markdown policy files into sections."""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from pathlib import Path

from app.config import ROOT, corpus_dirs

HEADING = re.compile(r"^(#{1,4})\s+(.*\S)\s*$")
WORD = re.compile(r"[A-Za-z0-9']+")


class _HTMLText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style"}:
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"}:
            self._skip = False
        if tag == "p":
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def html_to_text(raw: str) -> str:
    parser = _HTMLText()
    parser.feed(raw)
    parser.close()
    return html.unescape("".join(parser.parts))


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("401(k)", "401k")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def load_source(path: Path) -> dict:
    suffix = path.suffix.lower()
    raw = path.read_text(encoding="utf-8", errors="replace")
    if suffix in {".html", ".htm"}:
        text = clean_text(html_to_text(raw))
        fmt = "html"
    elif suffix in {".txt", ".md", ".markdown"}:
        text = clean_text(raw)
        fmt = "txt" if suffix == ".txt" else "md"
    else:
        raise ValueError(f"Unsupported policy format: {path.name}")
    match = re.match(r"(\d+)", path.name)
    document_id = match.group(1) if match else path.stem.upper()
    title = path.stem
    for line in text.splitlines():
        heading = HEADING.match(line)
        if heading:
            title = heading.group(2).strip()
            break
    return {
        "document_id": document_id,
        "title": title,
        "source": path.name,
        "format": fmt,
        "text": text,
    }


def iter_policy_files() -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for directory in corpus_dirs():
        if not directory.exists():
            continue
        # The project root contributes the HTML policy exports only.
        # Notes such as README.md and design-and-evaluation.md stay out of the index.
        if directory.resolve() == ROOT.resolve():
            paths = list(directory.glob("*.html"))
        else:
            paths = (
                list(directory.glob("*.html"))
                + list(directory.glob("*.txt"))
                + list(directory.glob("*.md"))
                + list(directory.glob("*.markdown"))
            )
        for path in paths:
            resolved = path.resolve()
            if resolved in seen or not path.is_file():
                continue
            seen.add(resolved)
            found.append(path)
    return sorted(found, key=lambda item: item.name)


def load_corpus() -> list[dict]:
    return [load_source(path) for path in iter_policy_files()]


def chunk_document(document: dict, max_words: int = 180, overlap_words: int = 40) -> list[dict]:
    """Heading-aware chunks. Long sections use a word window with overlap."""
    sections: list[tuple[str, list[str]]] = []
    current_name = "Introduction"
    current_lines: list[str] = []
    for line in document["text"].splitlines():
        heading = HEADING.match(line)
        if heading and len(heading.group(1)) >= 2:
            if any(part.strip() for part in current_lines):
                sections.append((current_name, current_lines))
            current_name = heading.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)
    if any(part.strip() for part in current_lines):
        sections.append((current_name, current_lines))

    chunks: list[dict] = []
    for index, (section, lines) in enumerate(sections):
        body = "\n".join(lines).strip()
        if len(body.split()) < 8:
            continue
        words = body.split()
        if len(words) <= max_words:
            windows = [body]
        else:
            windows = []
            step = max(max_words - overlap_words, 1)
            for start in range(0, len(words), step):
                piece = " ".join(words[start : start + max_words]).strip()
                if piece:
                    windows.append(piece)
                if start + max_words >= len(words):
                    break
        for part, window in enumerate(windows):
            prefix = f"{document['title']} — {section}"
            chunks.append(
                {
                    "id": f"{document['document_id']}-{index}-{part}",
                    "document_id": document["document_id"],
                    "title": document["title"],
                    "section": section,
                    "source": document["source"],
                    "format": document["format"],
                    "text": f"{prefix}\n{window}",
                    "snippet": window[:1200],
                }
            )
    return chunks


def word_count(text: str) -> int:
    return len(WORD.findall(text))
