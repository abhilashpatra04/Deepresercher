"""
Web Search Tool — T1: Agent-Agnostic, Pre-trained Tool.

Uses DuckDuckGo (free, no API key required) for web search.
Works with any agent as a plug-and-play tool.

Paradigm: T1 (Tool adaptation, independent metrics)
"""

from dataclasses import dataclass
from typing import List

try:
    from duckduckgo_search import DDGS
    HAS_DDG = True
except ImportError:
    HAS_DDG = False


@dataclass
class WebResult:
    """A single web search result."""
    title: str
    url: str
    snippet: str
    source: str = ""

    def short_summary(self) -> str:
        return f"{self.title} ({self.url[:50]}...)"


class WebSearchTool:
    """
    T1: Agent-Agnostic Tool — Web Search via DuckDuckGo.

    Free, no API key needed. Works with any agent.
    Searches the general web for blogs, docs, news, etc.
    Complements arXiv (academic) with broader web results.
    """

    def search(self, query: str, max_results: int = 5) -> List[WebResult]:
        """Search the web using DuckDuckGo."""
        if not HAS_DDG:
            print("[WebSearch] duckduckgo-search not installed. pip install duckduckgo-search")
            return []

        try:
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append(WebResult(
                        title=r.get("title", ""),
                        url=r.get("href", r.get("link", "")),
                        snippet=r.get("body", r.get("snippet", "")),
                        source="duckduckgo",
                    ))
            return results
        except Exception as e:
            print(f"[WebSearch] Search failed: {e}")
            return []

    def search_academic(self, query: str, max_results: int = 5) -> List[WebResult]:
        """Search with academic focus (adds site filters)."""
        academic_query = f"{query} site:arxiv.org OR site:scholar.google.com OR site:semanticscholar.org OR site:paperswithcode.com"
        return self.search(academic_query, max_results)


if __name__ == "__main__":
    tool = WebSearchTool()
    results = tool.search("LLM tool use 2024", max_results=3)
    for r in results:
        print(f"\n{r.title}")
        print(f"  URL: {r.url}")
        print(f"  {r.snippet[:120]}...")
