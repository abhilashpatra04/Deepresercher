"""
Search Agent — Step 2 of Deep Research Pipeline.

Performs ITERATIVE search: search → validate → refine → search again.
This is the core retrieval step using T1 (pre-trained tools) and
T2 (query rewriter optimized for the frozen LLM).

Paradigm: T1 + T2
  T1: arXiv API + DuckDuckGo web search (pre-trained, agent-agnostic)
  T2: Query rewriter is optimized to serve the frozen main LLM
"""

from typing import List, Dict
from dataclasses import dataclass, field


@dataclass
class SearchResults:
    """Results from iterative search for one sub-question."""
    sub_question: str
    papers: list = field(default_factory=list)
    web_results: list = field(default_factory=list)
    iterations: int = 0
    queries_used: List[str] = field(default_factory=list)


class SearchAgent:
    """
    Step 2: Iterative Search (T1 + T2).

    Paper context: "Reliable research requires grounded evidence...
    deep research agents must incorporate diverse tools that provide
    direct access to external knowledge" (Section 7.1).

    T1: arXiv API + Web Search are plug-and-play (works with any agent).
    T2: Query rewriter is optimized for the specific frozen LLM
        (like s3 in the paper — 70x more data-efficient than A2).
    """

    def __init__(self, arxiv_tool, query_rewriter, tool_verifier, web_search_tool=None):
        self.arxiv = arxiv_tool
        self.web_search = web_search_tool       # T1 (web)
        self.rewriter = query_rewriter           # T2
        self.verifier = tool_verifier            # A1 (for validation)

    def search_for_subquestion(self, sub_question, main_query: str, max_iterations: int = 3) -> SearchResults:
        """
        Iterative search loop for a single sub-question.
        
        Loop: search → validate (A1) → refine query (T2) → search again
        """
        question_text = sub_question.question if hasattr(sub_question, 'question') else str(sub_question)
        search_query = sub_question.search_query if hasattr(sub_question, 'search_query') else str(sub_question)

        results = SearchResults(sub_question=question_text)
        seen_ids = set()

        for iteration in range(max_iterations):
            results.iterations += 1
            results.queries_used.append(search_query)

            # T1: Use pre-trained arXiv tool (agent-agnostic)
            papers = self.arxiv.search(search_query, max_results=5)

            # T1: Also search web for broader coverage
            if self.web_search:
                try:
                    web_hits = self.web_search.search(search_query, max_results=3)
                    for wr in web_hits:
                        if wr.url not in [w.url for w in results.web_results]:
                            results.web_results.append(wr)
                except Exception:
                    pass  # Web search is supplementary, don't fail

            # A1: Verify tool execution
            verification = self.verifier.verify_search_results(papers)

            if not verification["valid"]:
                # T2: Rewrite query using agent-supervised rewriter
                if iteration < max_iterations - 1:
                    search_query = self.rewriter.rewrite_for_search(
                        question_text,
                        context=f"Previous query '{search_query}' returned no results"
                    )
                continue

            # Add new papers (avoid duplicates)
            for paper in papers:
                pid = paper.arxiv_id if hasattr(paper, 'arxiv_id') else id(paper)
                if pid not in seen_ids:
                    seen_ids.add(pid)
                    results.papers.append(paper)

            # Check if we have enough quality papers
            if len(results.papers) >= 5:
                break

            # T2: Refine query for next iteration (find complementary papers)
            if iteration < max_iterations - 1:
                search_query = self.rewriter.refine_after_results(
                    original_query=search_query,
                    found_papers=results.papers,
                )

        return results

    def search_all(self, research_plan, main_query: str) -> Dict[str, SearchResults]:
        """
        Search for all sub-questions in the research plan.
        Returns dict mapping sub-question → SearchResults.
        """
        all_results = {}

        for sub_q in research_plan.sub_questions:
            question_text = sub_q.question if hasattr(sub_q, 'question') else str(sub_q)
            results = self.search_for_subquestion(sub_q, main_query)
            all_results[question_text] = results

        return all_results

    def search_for_gap(self, gap: str, main_query: str) -> SearchResults:
        """
        Search specifically for a gap identified by the critic.
        Uses T2 rewriter to optimize the gap query.
        """
        optimized_query = self.rewriter.rewrite_for_search(gap, context=main_query)
        
        from agents.planner_agent import SubQuestion
        sub_q = SubQuestion(
            question=gap,
            search_query=optimized_query,
            evidence_type="survey"
        )
        return self.search_for_subquestion(sub_q, main_query, max_iterations=2)
