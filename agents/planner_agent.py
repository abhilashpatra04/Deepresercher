"""
Planner Agent — Step 1 of Deep Research Pipeline.

Decomposes a complex research query into structured sub-questions
and a research plan. This is the first step before iterative search.

Paradigm: A2 (Agent learns from final output quality)
— In production, the planner would be fine-tuned using RL
  with reward = quality of the final research report.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class SubQuestion:
    """A single research sub-question with search guidance."""
    question: str
    search_query: str
    evidence_type: str  # "survey", "empirical", "theoretical"


@dataclass
class ResearchPlan:
    """Structured research plan from the planner."""
    main_query: str
    sub_questions: List[SubQuestion]
    approach_notes: str = ""


class PlannerAgent:
    """
    Step 1: Research Planner (A2).

    Paper context: Deep research systems require "decomposing complex
    scientific questions into structured research plans" (Section 7.1).

    A2 paradigm: The planner's quality is judged by the FINAL output
    of the entire pipeline. In production, this would be trained via
    RL where reward = final report quality (like Search-R1).
    """

    def __init__(self, llm_client):
        self.llm = llm_client

    def create_plan(self, query: str, context: str = "") -> ResearchPlan:
        """
        Decompose a research query into 3-5 sub-questions.
        Each sub-question has a targeted search query.
        
        Args:
            query: The research question
            context: Optional context from uploaded paper/URL
        """
        context_section = ""
        if context:
            context_section = f"""
The user has also provided the following source material. Use it to create
more targeted and specific sub-questions:

---
{context[:3000]}
---
"""
        result = self.llm.generate_json(
            prompt=f"""Decompose this research query into 3-5 specific sub-questions.

Research Query: "{query}"
{context_section}
For each sub-question provide:
- "question": the specific research sub-question
- "search_query": an optimized search query (under 10 words, works for arXiv and web)
- "evidence_type": one of "survey", "empirical", "theoretical"

Return JSON format:
{{
  "sub_questions": [
    {{
      "question": "...",
      "search_query": "...",
      "evidence_type": "..."
    }}
  ],
  "approach_notes": "Brief note on research strategy"
}}""",
            system_prompt="You are a research planning assistant that breaks down complex questions into specific, searchable sub-questions."
        )

        # Parse response
        sub_questions = []
        if isinstance(result, dict) and "sub_questions" in result:
            for sq in result["sub_questions"]:
                sub_questions.append(SubQuestion(
                    question=sq.get("question", ""),
                    search_query=sq.get("search_query", ""),
                    evidence_type=sq.get("evidence_type", "survey"),
                ))

        # Fallback: if parsing failed, create basic plan
        if not sub_questions:
            sub_questions = [
                SubQuestion(query, query, "survey"),
                SubQuestion(f"Recent developments in {query}", f"{query} recent 2024 2025", "empirical"),
                SubQuestion(f"Challenges and limitations of {query}", f"{query} challenges limitations", "theoretical"),
            ]

        return ResearchPlan(
            main_query=query,
            sub_questions=sub_questions,
            approach_notes=result.get("approach_notes", "") if isinstance(result, dict) else "",
        )
