#!/usr/bin/env python3
"""
FedInsight - PDF Parser for Federal Reserve documents

Extracts clean text and metadata from FOMC Minutes PDFs (and similar documents).

Engine: PyMuPDF (fitz).
FOMC minutes are typeset in two columns. PyMuPDF's default text extraction
correctly detects the column layout and returns text in natural reading order
(full left column, then full right column), whereas a naive top-to-bottom
extraction interleaves the two columns into garbage. pdfplumber is kept only as
an optional fallback.

Design goals:
- High-quality, reading-order text suitable for RAG
- Conservative cleaning that never deletes large spans of real content
- Preserves useful metadata
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import re

from loguru import logger

try:
    import fitz  # PyMuPDF

    _HAS_PYMUPDF = True
except ImportError:  # pragma: no cover
    _HAS_PYMUPDF = False

try:
    import pdfplumber

    _HAS_PDFPLUMBER = True
except ImportError:  # pragma: no cover
    _HAS_PDFPLUMBER = False


# Running header that appears at the top of every page, e.g.
# "Page 2                Federal Open Market Committee                _"
_RUNNING_HEADER = re.compile(
    r"(?im)^\s*Page\s+\d+\s+Federal Open Market Committee.*$"
)
# Bare page-number lines, e.g. "Page 14"
_PAGE_NUMBER_LINE = re.compile(r"(?im)^\s*Page\s+\d+\s*_?\s*$")
# Stray underscore separators used as header rules
_RULE_LINE = re.compile(r"(?im)^\s*_+\s*$")

# Conservative Unicode normalization that helps lexical search without
# distorting meaning. Ligatures break word matching ("ﬁnancial" != "financial")
# and the Unicode minus sign hides negative numbers from plain-text search.
_CHAR_NORMALIZATION = {
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb00": "ff",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\u2212": "-",  # minus sign -> hyphen-minus
}
_NORMALIZE_RE = re.compile("|".join(map(re.escape, _CHAR_NORMALIZATION)))


def _normalize_chars(text: str) -> str:
    return _NORMALIZE_RE.sub(lambda m: _CHAR_NORMALIZATION[m.group(0)], text)


def clean_text(text: str) -> str:
    """
    Conservative cleaning for RAG.

    - Joins words split by hyphenation at line breaks ("Af-\\nfairs" -> "Affairs")
    - Removes running headers / page-number lines / rule lines
    - Normalizes whitespace without discarding content

    Note: deliberately avoids broad ".*" deletions that can wipe out an entire
    document (the previous implementation did exactly that).
    """
    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Normalize ligatures / minus sign for better lexical search
    text = _normalize_chars(text)

    # De-hyphenate words broken across line breaks: "Af-\nfairs" -> "Affairs"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Strip running headers / page numbers / horizontal rules (line-anchored)
    text = _RUNNING_HEADER.sub("", text)
    text = _PAGE_NUMBER_LINE.sub("", text)
    text = _RULE_LINE.sub("", text)

    # Collapse intra-line whitespace, then collapse newlines/spaces to single spaces
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


def _extract_pages_pymupdf(pdf_path: Path) -> List[str]:
    """Extract per-page text using PyMuPDF in natural reading order."""
    pages: List[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            # Default "text" mode respects the multi-column reading order.
            pages.append(page.get_text("text"))
    return pages


def _extract_pages_pdfplumber(pdf_path: Path) -> List[str]:
    """Fallback extractor (lower quality on multi-column layouts)."""
    pages: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return pages


def parse_pdf(
    pdf_path: Path | str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Parse a single PDF and return structured content + metadata.

    Args:
        pdf_path: Path to the PDF file
        metadata: Optional metadata dict from the scraper (meeting_date, etc.)

    Returns:
        dict with keys: text, metadata, page_count, char_count, source_file
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info(f"Parsing PDF: {pdf_path.name}")

    try:
        if _HAS_PYMUPDF:
            extracted_pages = _extract_pages_pymupdf(pdf_path)
            engine = "pymupdf"
        elif _HAS_PDFPLUMBER:
            logger.warning("PyMuPDF not available, falling back to pdfplumber")
            extracted_pages = _extract_pages_pdfplumber(pdf_path)
            engine = "pdfplumber"
        else:
            raise RuntimeError(
                "No PDF backend available. Install pymupdf (recommended) or pdfplumber."
            )

        page_count = len(extracted_pages)
        full_text = "\n\n".join(extracted_pages)
        cleaned_text = clean_text(full_text)

        result = {
            "text": cleaned_text,
            "metadata": dict(metadata or {}),
            "page_count": page_count,
            "char_count": len(cleaned_text),
            "source_file": str(pdf_path.resolve()),
            "parser_engine": engine,
        }

        result["metadata"]["document_type"] = result["metadata"].get(
            "document_type", "fomc_minutes"
        )

        logger.success(
            f"Parsed {pdf_path.name} | {page_count} pages | "
            f"{len(cleaned_text):,} chars | engine={engine}"
        )
        return result

    except Exception as e:
        logger.error(f"Failed to parse {pdf_path}: {e}")
        raise


def parse_pdf_to_file(
    pdf_path: Path | str,
    output_json_path: Optional[Path | str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Convenience wrapper: parse a PDF and save the result as JSON."""
    import json

    result = parse_pdf(pdf_path, metadata)

    if output_json_path is None:
        pdf_path = Path(pdf_path)
        output_json_path = pdf_path.with_suffix(".json")

    output_json_path = Path(output_json_path)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved parsed result → {output_json_path}")
    return output_json_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.parser.pdf_parser <path_to_pdf>")
        sys.exit(1)

    test_path = Path(sys.argv[1])
    result = parse_pdf(test_path)

    print("\n=== PARSE RESULT ===")
    print(f"Engine:    {result['parser_engine']}")
    print(f"Pages:     {result['page_count']}")
    print(f"Characters:{result['char_count']:,}")
    print(f"Metadata:  {result['metadata']}")
    print("\n--- First 800 chars of cleaned text ---")
    print(result["text"][:800])
    print("...")
