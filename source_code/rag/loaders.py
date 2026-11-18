"""
Document loading (Task 1: PDF support) and text cleaning.

Supported formats: .pdf, .txt, .md
Each file becomes ONE langchain Document whose text is the concatenation of
all pages. For PDFs we also keep `page_offsets` (character offset where each
page starts) so every chunk can later be mapped back to its page number.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

# pypdf prints many harmless "Ignoring wrong pointing object" warnings for
# PDFs exported from macOS Quartz; silence them.
logging.getLogger("pypdf").setLevel(logging.ERROR)


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
# Some fonts in the handbooks encode the "ti"/"tt" ligatures as odd glyphs, so
# pypdf returns e.g. "Descrip;on", "Educa8onal", "Founda'onal", "A2ributes".
_LIGATURE_FIXES = [
    (re.compile(r"(?<=[a-z])[;8](?=[a-z])"), "ti"),       # Descrip;on, Educa8onal
    (re.compile(r"(?<=[a-z])['\u2019](?=on)"), "ti"),    # Founda'onal / Founda’onal
    (re.compile(r"\bA[02]ribute"), "Attribute"),          # A2ributes, A0ributes
]


def clean_text(text: str) -> str:
    """Normalise PDF-extracted text so that it chunks and embeds well."""
    # NFKC turns typographic ligatures (ﬁ, ﬀ, ﬃ ...) into plain letters.
    text = unicodedata.normalize("NFKC", text)
    for pattern, repl in _LIGATURE_FIXES:
        text = pattern.sub(repl, text)
    # Drop table-of-contents lines ("Mission ............ 2"): they only repeat
    # headings and would match almost every query.
    text = re.sub(r"^.*\.{4,}.*$", "", text, flags=re.M)
    # Strip trailing spaces on each line so "\n \n" becomes a real paragraph break.
    text = "\n".join(line.rstrip() for line in text.splitlines())
    # Collapse 3+ blank lines and repeated spaces.
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_pdf(path: Path) -> Document:
    """Extract the selectable text of every page of a PDF file."""
    reader = PdfReader(str(path))
    parts: List[str] = []
    page_offsets: List[int] = []
    offset = 0
    for page in reader.pages:
        page_text = clean_text(page.extract_text() or "")
        page_offsets.append(offset)
        parts.append(page_text)
        offset += len(page_text) + 2          # +2 for the "\n\n" separator
    full_text = "\n\n".join(parts)
    if len(full_text.strip()) < 50:
        print(f"  WARNING: {path.name} contains almost no selectable text "
              "(scanned PDF?). OCR would be required.")
    return Document(
        page_content=full_text,
        metadata={
            "source": path.name,
            "type": "pdf",
            "num_pages": len(reader.pages),
            # stored as a string because Chroma metadata must be scalar
            "page_offsets": ",".join(map(str, page_offsets)),
        },
    )


def load_text(path: Path) -> Document:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    return Document(page_content=clean_text(content),
                    metadata={"source": path.name, "type": path.suffix.lstrip(".")})


def list_supported_files(directory: str | Path) -> List[Path]:
    directory = Path(directory)
    return sorted(p for p in directory.iterdir()
                  if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
                  and not p.name.startswith(("~$", ".")))


def load_documents(directory: str | Path, verbose: bool = True) -> List[Document]:
    """Load every supported file in `directory`."""
    documents = []
    for path in list_supported_files(directory):
        if verbose:
            print("  Loading: " + path.name)
        try:
            if path.suffix.lower() == ".pdf":
                documents.append(load_pdf(path))
            else:
                documents.append(load_text(path))
        except Exception as exc:  # a corrupt file should not stop indexing
            print(f"  ERROR loading {path.name}: {exc}")
    return documents


def page_for_offset(page_offsets: str, start_index: int) -> int:
    """Return the 1-based page number that contains character `start_index`."""
    if not page_offsets:
        return 0
    offsets = [int(x) for x in page_offsets.split(",")]
    page = 1
    for i, off in enumerate(offsets):
        if start_index >= off:
            page = i + 1
        else:
            break
    return page
