"""
URL Extractor — T1: Agent-Agnostic Tool.

Extracts readable text content from any URL.
Works with any agent as a plug-and-play tool.

Paradigm: T1 (Tool adaptation, independent metrics)
"""

import re
import requests
from typing import Dict


class URLExtractor:
    """
    T1: Agent-Agnostic Tool — URL Content Extraction.

    Fetches and extracts readable text from web pages.
    Handles HTML pages, PDFs, and plain text.
    """

    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36"
        }

    def extract(self, url: str) -> Dict[str, str]:
        """
        Extract text content from a URL.
        Returns dict with text, title, status.
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")

            if "pdf" in content_type:
                return self._extract_pdf(response.content, url)
            else:
                return self._extract_html(response.text, url)

        except requests.RequestException as e:
            return {
                "status": "failed",
                "error": str(e),
                "text": "",
                "url": url,
            }

    def _extract_html(self, html: str, url: str) -> Dict[str, str]:
        """Extract readable text from HTML."""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")

            # Remove script and style elements
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            # Get title
            title = soup.title.string if soup.title else ""

            # Get main content
            # Try common content containers first
            main = (
                soup.find("article") or
                soup.find("main") or
                soup.find("div", class_=re.compile(r"content|article|post|entry")) or
                soup.find("body")
            )

            text = main.get_text(separator="\n", strip=True) if main else soup.get_text(separator="\n", strip=True)

            # Clean up excessive whitespace
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            text = "\n".join(lines)

            return {
                "status": "success",
                "text": text[:10000],  # Limit to 10k chars
                "title": title.strip() if title else "",
                "url": url,
                "char_count": len(text),
            }

        except ImportError:
            # Fallback without BeautifulSoup
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text).strip()
            return {
                "status": "success",
                "text": text[:10000],
                "title": "",
                "url": url,
                "char_count": len(text),
            }

    def _extract_pdf(self, content: bytes, url: str) -> Dict[str, str]:
        """Extract text from PDF content."""
        import tempfile
        import os

        # Save to temp file and use PDFParser
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(content)
                temp_path = f.name

            from tools.pdf_parser import PDFParser
            parser = PDFParser()
            result = parser.parse_local(temp_path)
            os.unlink(temp_path)

            result["url"] = url
            return result
        except Exception as e:
            return {
                "status": "failed",
                "error": f"PDF extraction failed: {e}",
                "text": "",
                "url": url,
            }


if __name__ == "__main__":
    extractor = URLExtractor()
    result = extractor.extract("https://arxiv.org/abs/2503.00001")
    print(f"Status: {result['status']}")
    print(f"Title: {result.get('title', '')}")
    print(f"Text: {result.get('text', '')[:300]}...")
