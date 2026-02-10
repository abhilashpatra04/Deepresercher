"""
Critic Agent — Step 3 of Deep Research Pipeline.

Self-critique: validates findings, identifies gaps, and decides
whether to go back to search or proceed to synthesis.

Paradigm: A1 + A2
  A1: Checks that all tool outputs are valid (tool execution feedback)
  A2: Judges the overall research quality (final output signal)
"""

from typing import List, Dict
from dataclasses import dataclass, field


@dataclass
class CritiqueResult:
    """Result of the critic's evaluation."""
    tool_issues: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    gaps: List[str] = field(default_factory=list)
    needs_more_search: bool = False
    feedback: str = ""


class CriticAgent:
    """
    Step 3: Self-Critique (A1 + A2).

    Paper context: Deep research requires "multi-step self-critique"
    and "hypothesis refinement" (Section 7.1).

    A1: Verify tool outputs are valid (did search/parse actually work?)
    A2: Judge overall completeness and quality of findings so far.

    If gaps found → triggers re-search (back to Step 2).
    """

    def __init__(self, llm_client, tool_verifier):
        self.llm = llm_client
        self.verifier = tool_verifier  # A1

    def critique(self, research_plan, search_results: dict) -> CritiqueResult:
        """
        Evaluate research findings for completeness and quality.
        Returns CritiqueResult with gaps and quality score.
        """
        result = CritiqueResult()

        # A1: Verify all tool outputs
        result.tool_issues = self._check_tool_outputs(search_results)

        # A2: Judge overall research quality
        quality = self._judge_quality(research_plan, search_results)
        result.quality_score = quality["score"]
        result.gaps = quality["gaps"]
        result.feedback = quality["feedback"]

        # Decision: need more research?
        result.needs_more_search = (
            result.quality_score < 0.6
            or len(result.gaps) > 0
            or len(result.tool_issues) > 2
        )

        return result

    def _check_tool_outputs(self, search_results: dict) -> List[str]:
        """
        A1: Verify all tool execution results.
        Check that search actually returned usable data.
        """
        issues = []

        for sub_q, results in search_results.items():
            papers = results.papers if hasattr(results, 'papers') else []

            if not papers:
                issues.append(f"No papers found for: '{sub_q[:60]}'")
                continue

            # Check paper quality
            empty_abstracts = sum(
                1 for p in papers
                if not (hasattr(p, 'abstract') and p.abstract and len(p.abstract) > 20)
            )
            if empty_abstracts > len(papers) // 2:
                issues.append(f"Many empty abstracts for: '{sub_q[:60]}'")

        return issues

    def _judge_quality(self, research_plan, search_results: dict) -> dict:
        """
        A2: Judge overall research quality using LLM-as-judge.
        This is the final output signal used in A2 paradigm.
        """
        # Build summary of what we found
        findings_summary = []
        total_papers = 0

        for sub_q, results in search_results.items():
            papers = results.papers if hasattr(results, 'papers') else []
            total_papers += len(papers)
            paper_titles = [
                p.title if hasattr(p, 'title') else str(p)
                for p in papers[:3]
            ]
            findings_summary.append(
                f"Sub-question: {sub_q}\n"
                f"  Papers found: {len(papers)}\n"
                f"  Top papers: {', '.join(paper_titles)}"
            )

        sub_questions = [
            sq.question if hasattr(sq, 'question') else str(sq)
            for sq in research_plan.sub_questions
        ]

        prompt = f"""You are a research quality evaluator.

Research Goal: {research_plan.main_query}

Sub-questions planned:
{chr(10).join(f'- {sq}' for sq in sub_questions)}

Findings so far:
{chr(10).join(findings_summary)}

Total papers found: {total_papers}

Evaluate:
1. Score overall completeness from 0.0 to 1.0
2. List any gaps (sub-questions not well-covered)
3. Brief feedback

Return JSON:
{{
  "score": 0.0-1.0,
  "gaps": ["gap1", "gap2"],
  "feedback": "brief assessment"
}}"""

        result = self.llm.generate_json(prompt)

        if isinstance(result, dict) and "score" in result:
            return {
                "score": float(result.get("score", 0.5)),
                "gaps": result.get("gaps", []),
                "feedback": result.get("feedback", ""),
            }

        # Fallback: use heuristic scoring
        coverage = 0
        for sub_q in sub_questions:
            if sub_q in search_results and len(search_results[sub_q].papers) > 0:
                coverage += 1

        score = coverage / max(len(sub_questions), 1)
        gaps = [
            sq for sq in sub_questions
            if sq not in search_results or len(search_results[sq].papers) == 0
        ]

        return {
            "score": score,
            "gaps": gaps,
            "feedback": f"Covered {coverage}/{len(sub_questions)} sub-questions",
        }
