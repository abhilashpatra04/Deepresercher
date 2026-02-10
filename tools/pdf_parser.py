"""
PDF Parser — T1: Agent-Agnostic Tool.

Extracts text and sections from PDF files.
Works independently of any agent (plug-and-play).

Paradigm: T1 (Tool adaptation, independent metrics)
"""

import os
import re
import requests
from typing import Dict, Optional


class PDFParser:
    """
    T1: Agent-Agnostic Tool — PDF Text Extraction.

    Downloads and extracts text from academic PDFs.
    Works with any agent — no agent-specific optimization.
    """

    def __init__(self, papers_dir: str = "papers"):
        self.papers_dir = papers_dir
        os.makedirs(papers_dir, exist_ok=True)

    def download_and_parse(self, pdf_url: str, paper_id: str = None) -> Dict[str, str]:
        """
        Download PDF from URL and extract text.
        Returns dict with text, status, and sections.
        """
        if not paper_id:
            paper_id = pdf_url.split("/")[-1].replace(".pdf", "")

        local_path = os.path.join(self.papers_dir, f"{paper_id}.pdf")

        # Download
        download_ok = self._download(pdf_url, local_path)
        if not download_ok:
            return {
                "status": "failed",
                "error": "Download failed",
                "text": "",
                "paper_id": paper_id,
            }

        # Extract text
        text = self._extract_text(local_path)
        if not text:
            return {
                "status": "failed",
                "error": "Text extraction failed",
                "text": "",
                "paper_id": paper_id,
            }

        # Extract sections
        sections = self._extract_sections(text)

        return {
            "status": "success",
            "text": text,
            "paper_id": paper_id,
            "sections": sections,
            "char_count": len(text),
        }

    def parse_local(self, file_path: str) -> Dict[str, str]:
        """Parse a local PDF file."""
        if not os.path.exists(file_path):
            return {"status": "failed", "error": "File not found", "text": ""}

        text = self._extract_text(file_path)
        if not text:
            return {"status": "failed", "error": "Extraction failed", "text": ""}

        sections = self._extract_sections(text)
        return {
            "status": "success",
            "text": text,
            "sections": sections,
            "char_count": len(text),
        }

    def _download(self, url: str, local_path: str) -> bool:
        """Download PDF from URL."""
        try:
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()

            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception as e:
            print(f"[PDFParser] Download failed: {e}")
            return False

    def _extract_text(self, file_path: str) -> str:
        """Extract text from PDF. Tries PyMuPDF first, falls back to basic."""
        # Try PyMuPDF
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(file_path)
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            doc.close()
            text = "\n".join(text_parts)
            if len(text.strip()) > 100:
                return text
        except ImportError:
            pass
        except Exception as e:
            print(f"[PDFParser] PyMuPDF failed: {e}")

        # Fallback: basic text extraction
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            # Extract readable text between stream markers
            text_parts = []
            decoded = content.decode("latin-1", errors="ignore")
            # Find text between parentheses (PDF text objects)
            matches = re.findall(r"\(([^)]+)\)", decoded)
            for match in matches:
                cleaned = match.strip()
                if len(cleaned) > 2 and any(c.isalpha() for c in cleaned):
                    text_parts.append(cleaned)
            return " ".join(text_parts)
        except Exception as e:
            print(f"[PDFParser] Basic extraction failed: {e}")
            return ""

    def _extract_sections(self, text: str) -> Dict[str, str]:
        """
        Try to extract common academic paper sections.
        Returns dict mapping section name to content.
        """
        sections = {}
        section_patterns = [
            r"(?i)\b(abstract)\b",
            r"(?i)\b(introduction)\b",
            r"(?i)\b(related work)\b",
            r"(?i)\b(method(?:ology|s)?)\b",
            r"(?i)\b(experiment(?:s|al)?(?:\s+results)?)\b",
            r"(?i)\b(results?)\b",
            r"(?i)\b(discussion)\b",
            r"(?i)\b(conclusion(?:s)?)\b",
        ]

        lines = text.split("\n")
        current_section = "preamble"
        section_content = []

        for line in lines:
            matched = False
            for pattern in section_patterns:
                if re.match(pattern, line.strip()):
                    # Save previous section
                    if section_content:
                        sections[current_section] = "\n".join(section_content)
                    current_section = line.strip().lower()
                    section_content = []
                    matched = True
                    break
            if not matched:
                section_content.append(line)

        # Save last section
        if section_content:
            sections[current_section] = "\n".join(section_content)

        return sections


if __name__ == "__main__":
    parser = PDFParser()
    # Test with local paper
    result = parser.parse_local("paper.pdf")
    print(f"Status: {result['status']}")
    print(f"Characters: {result.get('char_count', 0)}")
    print(f"Sections: {list(result.get('sections', {}).keys())}")
