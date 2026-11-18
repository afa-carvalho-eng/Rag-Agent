"""Split documents into overlapping chunks sized for embedding."""

from __future__ import annotations

import re


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    """Break a paragraph longer than max_chars at word boundaries."""
    pieces = []
    while len(paragraph) > max_chars:
        cut = paragraph.rfind(" ", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        pieces.append(paragraph[:cut].strip())
        paragraph = paragraph[cut:].strip()
    if paragraph:
        pieces.append(paragraph)
    return pieces


def _tail(text: str, overlap: int) -> str:
    """Last `overlap` characters of text, trimmed to start on a whole word."""
    if overlap <= 0 or len(text) <= overlap:
        return ""
    tail = text[-overlap:]
    space = tail.find(" ")
    return tail[space + 1 :] if space != -1 else ""


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> list[str]:
    """Pack paragraphs into chunks of at most max_chars characters.

    Paragraph boundaries are kept where possible. Each chunk after the first
    starts with the tail of the previous one, so a sentence cut by a chunk
    boundary is still retrievable in full from one of the two.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not 0 <= overlap < max_chars:
        raise ValueError("overlap must be >= 0 and smaller than max_chars")

    pieces: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if paragraph:
            pieces.extend(_split_long(paragraph, max_chars))

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if not current:
            current = piece
        elif len(current) + 2 + len(piece) <= max_chars:
            current = f"{current}\n\n{piece}"
        else:
            chunks.append(current)
            tail = _tail(current, overlap)
            with_tail = f"{tail}\n\n{piece}"
            current = with_tail if tail and len(with_tail) <= max_chars else piece
    if current:
        chunks.append(current)
    return chunks
