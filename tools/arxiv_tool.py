"""
arXiv Tool — T1: Agent-Agnostic, Pre-trained Tool.

This is a T1 tool: it works independently of any specific agent.
Any agent can use it as-is (plug-and-play).
Uses the free arXiv API to search for academic papers.
"""

import requests
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Paper:
    """Represents an arXiv paper."""
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    published: str
    pdf_url: str
    categories: List[str] = field(default_factory=list)

    def short_summary(self) -> str:
        return f"[{self.arxiv_id}] {self.title} ({self.published[:10]})"


class ArxivTool:
    """
    T1: Agent-Agnostic Tool — arXiv Paper Search.

    This tool is pre-trained (the arXiv API is a fixed service)
    and works with ANY agent. It doesn't need to know which LLM
    is calling it. This is the essence of T1.

    Paradigm: T1 (Tool adaptation, independent metrics)
    """

    BASE_URL = "http://export.arxiv.org/api/query"
    NAMESPACE = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

    def search(self, query: str, max_results: int = 5) -> List[Paper]:
        """
        Search arXiv for papers matching the query.
        Returns list of Paper objects with metadata.
        """
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=15)
            response.raise_for_status()
            return self._parse_response(response.text)
        except requests.RequestException as e:
            print(f"[ArxivTool] Search failed: {e}")
            return []

    def _parse_response(self, xml_text: str) -> List[Paper]:
        """Parse arXiv Atom XML response into Paper objects."""
        papers = []
        root = ET.fromstring(xml_text)

        for entry in root.findall("atom:entry", self.NAMESPACE):
            try:
                # Extract ID
                id_text = entry.find("atom:id", self.NAMESPACE).text
                arxiv_id = id_text.split("/abs/")[-1] if "/abs/" in id_text else id_text

                # Extract title
                title = entry.find("atom:title", self.NAMESPACE).text
                title = " ".join(title.split())  # Clean whitespace

                # Extract abstract
                abstract = entry.find("atom:summary", self.NAMESPACE).text
                abstract = " ".join(abstract.strip().split())

                # Extract authors
                authors = []
                for author in entry.findall("atom:author", self.NAMESPACE):
                    name = author.find("atom:name", self.NAMESPACE).text
                    authors.append(name)

                # Extract published date
                published = entry.find("atom:published", self.NAMESPACE).text

                # Extract PDF URL
                pdf_url = ""
                for link in entry.findall("atom:link", self.NAMESPACE):
                    if link.get("title") == "pdf":
                        pdf_url = link.get("href", "")
                        break

                if not pdf_url:
                    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

                # Extract categories
                categories = []
                for cat in entry.findall("arxiv:primary_category", self.NAMESPACE):
                    categories.append(cat.get("term", ""))
                for cat in entry.findall("atom:category", self.NAMESPACE):
                    term = cat.get("term", "")
                    if term and term not in categories:
                        categories.append(term)

                papers.append(Paper(
                    arxiv_id=arxiv_id,
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    published=published,
                    pdf_url=pdf_url,
                    categories=categories,
                ))
            except (AttributeError, IndexError) as e:
                print(f"[ArxivTool] Skipping entry: {e}")
                continue

        return papers


# Quick test
if __name__ == "__main__":
    tool = ArxivTool()
    papers = tool.search("LLM tool use", max_results=3)
    for p in papers:
        print(f"\n{p.short_summary()}")
        print(f"  Authors: {', '.join(p.authors[:3])}")
        print(f"  Abstract: {p.abstract[:150]}...")
