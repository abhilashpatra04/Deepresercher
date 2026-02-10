"""
Quality Scorer — A2: Agent Learns from Final Output.

Evaluates the quality of the agent's final research output.
Uses LLM-as-judge to score summaries on multiple criteria.

In production, these scores would be the RL reward signal
to fine-tune the agent (like Search-R1, DeepSeek-R1).

Paradigm: A2 (Agent adaptation, final output signal)
"""


class QualityScorer:
    """
    A2: Final Output Quality Scoring.

    In the paper, A2 means the agent is adapted based on whether
    its FINAL output is good (not just whether tools worked).

    Examples from paper:
    - Search-R1: RL reward = answer correctness
    - DeepSeek-R1: RL reward = math/code solution quality
    - Kimi-1.5: RL reward = reasoning quality

    For demo, we use LLM-as-judge + rule-based checks.
    """

    def __init__(self, llm_client):
        self.llm = llm_client
        self.scoring_history = []

    def score(self, output: str, query: str, papers_used: list = None) -> dict:
        """
        Score the quality of a research output.
        Returns dict with overall score and per-criterion breakdowns.

        A2 signal: These scores would be the reward in production RL.
        """
        scores = {}

        # Rule-based checks (fast, free)
        scores["length"] = self._score_length(output)
        scores["structure"] = self._score_structure(output)
        scores["citations"] = self._score_citations(output, papers_used)

        # LLM-as-judge (costs 1 API call)
        scores["relevance"] = self._score_relevance_llm(output, query)

        # Overall score
        weights = {"length": 0.15, "structure": 0.2, "citations": 0.3, "relevance": 0.35}
        overall = sum(scores[k] * weights[k] for k in weights)
        scores["overall"] = round(overall, 2)

        result = {
            "scores": scores,
            "overall": scores["overall"],
            "pass": scores["overall"] >= 0.6,
            "feedback": self._generate_feedback(scores),
        }

        self.scoring_history.append(result)
        return result

    def _score_length(self, output: str) -> float:
        """Is the output a reasonable length? Not too short, not too long."""
        word_count = len(output.split())
        if word_count < 50:
            return 0.2  # Too short
        elif word_count < 100:
            return 0.5
        elif word_count < 500:
            return 1.0  # Good range
        elif word_count < 1500:
            return 0.9  # Slightly long
        else:
            return 0.6  # Too long

    def _score_structure(self, output: str) -> float:
        """Does the output have good structure (headers, paragraphs, lists)?"""
        score = 0.3  # Base score

        # Check for headers/sections
        if any(line.startswith("#") or line.startswith("**") for line in output.split("\n")):
            score += 0.2

        # Check for paragraphs (multiple line breaks)
        if output.count("\n\n") >= 2:
            score += 0.2

        # Check for bullet points
        if "- " in output or "• " in output or "* " in output:
            score += 0.15

        # Check for numbered items
        if any(f"{i}." in output or f"{i})" in output for i in range(1, 6)):
            score += 0.15

        return min(score, 1.0)

    def _score_citations(self, output: str, papers_used: list = None) -> float:
        """Does the output cite actual papers found?"""
        if not papers_used:
            # Check for any citation-like patterns
            import re
            citation_patterns = re.findall(
                r"\[[^\]]+\]|\([^)]*\d{4}[^)]*\)|arXiv:\s*\d+", output
            )
            return min(len(citation_patterns) * 0.2, 1.0)

        # Check how many actual papers are referenced
        cited = 0
        for paper in papers_used:
            title = paper.title if hasattr(paper, "title") else str(paper)
            # Check if title keywords appear in output
            title_words = [w for w in title.split() if len(w) > 4]
            if any(w.lower() in output.lower() for w in title_words[:3]):
                cited += 1

        if not papers_used:
            return 0.5
        return min(cited / len(papers_used), 1.0)

    def _score_relevance_llm(self, output: str, query: str) -> float:
        """Use LLM-as-judge to score relevance."""
        prompt = f"""Rate the relevance and quality of this research summary on a scale of 0.0 to 1.0.

Query: {query}

Summary:
{output[:2000]}

Criteria:
- Does it answer the query?
- Is the information accurate and specific?
- Are claims supported?

Respond with ONLY a number between 0.0 and 1.0, nothing else."""

        try:
            response = self.llm.generate(prompt, temperature=0.1)
            score = float(response.strip())
            return max(0.0, min(1.0, score))
        except (ValueError, TypeError):
            return 0.5  # Default if LLM fails

    def _generate_feedback(self, scores: dict) -> str:
        """Generate actionable feedback based on scores."""
        feedback = []

        if scores["length"] < 0.5:
            feedback.append("Output is too short — need more detailed analysis")
        if scores["structure"] < 0.5:
            feedback.append("Improve structure — add headers, bullet points, or sections")
        if scores["citations"] < 0.5:
            feedback.append("Cite more of the papers found — ground claims in sources")
        if scores["relevance"] < 0.5:
            feedback.append("Output doesn't address the query well — refocus on the question")

        if not feedback:
            feedback.append("Quality looks good ✅")

        return "; ".join(feedback)
