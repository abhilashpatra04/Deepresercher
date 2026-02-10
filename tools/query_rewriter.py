"""
Query Rewriter — T2: Agent-Supervised Tool.

This small tool is optimized to help a FROZEN LLM perform
better searches. It rewrites queries based on the main agent's
needs and previous search results.

In production, this would be fine-tuned (like s3 in the paper).
For our demo, we simulate T2 behavior with a smaller LLM + prompts
tailored to serve the main agent.

Paradigm: T2 (Tool adaptation, frozen agent's output signal)
"""

from typing import List, Optional


class QueryRewriter:
    """
    T2: Agent-Supervised Tool — Query Rewriting.

    Key insight from paper: s3 (T2) uses ~2.4k examples to match
    Search-R1 (A2) which needs ~170k examples. T2 is 70x more
    data-efficient because it only trains the small tool, not
    the entire agent.

    In this demo, we simulate T2 behavior using prompt engineering.
    In production, this would be a fine-tuned 7B model.
    """

    def __init__(self, llm_client):
        self.llm = llm_client

    def rewrite_for_search(self, query: str, context: str = "") -> str:
        """
        Rewrite a user query into an optimized arXiv search query.
        T2: Optimized to produce queries that the FROZEN main LLM
        can best use when processing the results.
        """
        prompt = f"""You are a search query optimizer for academic paper retrieval.
Your job is to rewrite the user's query into an optimized arXiv search query.

Rules:
- Use specific technical terms
- Include relevant synonyms separated by OR
- Remove filler words
- Focus on the most searchable concepts
- Keep it concise (under 15 words)

User query: {query}
{f"Context: {context}" if context else ""}

Respond with ONLY the optimized search query, nothing else."""

        result = self.llm.generate(prompt, temperature=0.3)
        return result.strip().strip('"').strip("'")

    def refine_after_results(self, original_query: str, found_papers: list, gap: str = "") -> str:
        """
        Refine search query based on what was already found.
        T2: Uses knowledge of what the frozen agent already has
        to find complementary papers.
        """
        found_titles = [p.title if hasattr(p, "title") else str(p) for p in found_papers[:5]]
        found_summary = "\n".join(f"- {t}" for t in found_titles)

        prompt = f"""You are refining a search query for academic papers.

Original query: {original_query}
Papers already found:
{found_summary}
{f"Gap to fill: {gap}" if gap else ""}

Write a NEW search query that finds DIFFERENT papers covering
aspects not yet covered by the papers above.

Respond with ONLY the new search query, nothing else."""

        result = self.llm.generate(prompt, temperature=0.4)
        return result.strip().strip('"').strip("'")

    def expand_sub_question(self, sub_question: str, main_query: str) -> str:
        """
        Expand a sub-question into a targeted search query.
        T2: Generates queries the frozen agent can best process.
        """
        prompt = f"""Convert this research sub-question into an arXiv search query.

Main research topic: {main_query}
Sub-question: {sub_question}

Write a focused arXiv search query (under 10 words).
Respond with ONLY the query, nothing else."""

        result = self.llm.generate(prompt, temperature=0.3)
        return result.strip().strip('"').strip("'")
