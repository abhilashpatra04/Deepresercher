"""
Deep Research Pipeline — Orchestrator.

Runs all 4 steps of the Deep Research pipeline:
  Step 1: Plan (A2) → Step 2: Search (T1+T2) →
  Step 3: Critique (A1+A2) → Step 4: Synthesize (A2+T2)

With retry loop: if Step 3 finds gaps, goes back to Step 2.

Supports multiple input modes:
  - Text query only
  - Text query + uploaded paper (PDF)
  - Text query + URL
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Any

from core.llm_client import LLMClient
from core.tool_verifier import ToolVerifier
from core.quality_scorer import QualityScorer
from core.memory import PersistentMemory
from core.report_generator import ResearchReportPDF
from tools.arxiv_tool import ArxivTool
from tools.query_rewriter import QueryRewriter
from tools.web_search import WebSearchTool
from tools.url_extractor import URLExtractor
from tools.pdf_parser import PDFParser
from agents.planner_agent import PlannerAgent
from agents.search_agent import SearchAgent
from agents.critic_agent import CriticAgent
from agents.synthesis_agent import SynthesisAgent


@dataclass
class PipelineStep:
    """Log entry for one pipeline step."""
    name: str
    paradigm: str
    status: str = "pending"
    duration: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineLog:
    """Full log of the pipeline execution."""
    steps: List[PipelineStep] = field(default_factory=list)
    total_duration: float = 0.0

    def add_step(self, name: str, paradigm: str) -> PipelineStep:
        step = PipelineStep(name=name, paradigm=paradigm, status="running")
        self.steps.append(step)
        return step


class DeepResearchPipeline:
    """
    Orchestrates the 4-step Deep Research pipeline.

    Step 1: PLAN        → A2 (agent learns research strategy)
    Step 2: SEARCH      → T1+T2 (pre-trained tools + optimized rewriter)
    Step 3: CRITIQUE    → A1+A2 (verify tools + judge quality)
    Step 4: SYNTHESIZE  → A2+T2 (quality output + persistent memory)

    Supports input modes: query, query+paper, query+URL.
    """

    def __init__(self, provider="groq", model=None):
        # Initialize LLM
        self.llm = LLMClient(provider=provider, model=model)

        # Initialize T1 tools (agent-agnostic)
        self.arxiv_tool = ArxivTool()
        self.web_search_tool = WebSearchTool()
        self.url_extractor = URLExtractor()
        self.pdf_parser = PDFParser()

        # Initialize T2 tools (optimized for this frozen LLM)
        self.query_rewriter = QueryRewriter(self.llm)

        # Initialize A1 verifier
        self.tool_verifier = ToolVerifier()

        # Initialize A2 scorer
        self.quality_scorer = QualityScorer(self.llm)

        # Initialize T2 memory
        self.memory = PersistentMemory()

        # Report generator
        self.report_gen = ResearchReportPDF()

        # Initialize agents
        self.planner = PlannerAgent(self.llm)
        self.searcher = SearchAgent(
            self.arxiv_tool, self.query_rewriter,
            self.tool_verifier, self.web_search_tool
        )
        self.critic = CriticAgent(self.llm, self.tool_verifier)
        self.synthesizer = SynthesisAgent(self.llm, self.memory, self.quality_scorer)

    def _extract_context(self, paper_path: str = None, url: str = None,
                         callback=None) -> str:
        """Extract text context from uploaded paper or URL."""
        context = ""

        if paper_path:
            if callback:
                callback("Input Processing", "T1", "Parsing uploaded paper...")
            result = self.pdf_parser.parse_local(paper_path)
            if result.get("status") == "success":
                context = result.get("text", "")[:5000]
            if callback:
                chars = len(context)
                callback("Input Processing", "T1",
                         f"Extracted {chars} chars from paper")

        elif url:
            if callback:
                callback("Input Processing", "T1", "Extracting content from URL...")
            result = self.url_extractor.extract(url)
            if result.get("status") == "success":
                context = result.get("text", "")[:5000]
            if callback:
                chars = len(context)
                callback("Input Processing", "T1",
                         f"Extracted {chars} chars from URL")

        return context

    def research(self, query: str, paper_path: str = None, url: str = None,
                 callback=None) -> dict:
        """
        Run the full Deep Research pipeline.

        Args:
            query: The research question
            paper_path: Optional path to uploaded PDF paper
            url: Optional URL to extract content from
            callback: Optional function called at each step for UI updates
                      callback(step_name, paradigm, status, details)
        
        Returns dict with report, quality, pipeline log, etc.
        """
        log = PipelineLog()
        start_time = time.time()

        def notify(step_name, paradigm, status, details=None):
            if callback:
                callback(step_name, paradigm, status, details or {})

        # ──────────────────────────────────────────────
        # Step 0a: Process paper/URL input (T1)
        # ──────────────────────────────────────────────
        user_context = self._extract_context(paper_path, url, callback)

        # ──────────────────────────────────────────────
        # Step 0b: Check Memory (T2)
        # ──────────────────────────────────────────────
        notify("Memory Recall", "T2", "Checking past research...")
        past_research = self.memory.recall(query)
        memory_context = ""
        if past_research:
            memory_context = "\n\nPrevious research on related topics:\n"
            for entry in past_research[:2]:
                memory_context += f"- {entry['query']}: {entry['findings'][:200]}...\n"
            notify("Memory Recall", "T2",
                   f"Found {len(past_research)} related past research entries")

        # ──────────────────────────────────────────────
        # Step 1: PLAN (A2)
        # ──────────────────────────────────────────────
        step1 = log.add_step("Research Planning", "A2")
        step1_start = time.time()
        notify("Research Planning", "A2", "Decomposing query into sub-questions...")

        # Combine all context for planning
        full_context = ""
        if user_context:
            full_context += f"User provided source material:\n{user_context[:3000]}\n\n"
        if memory_context:
            full_context += memory_context

        plan = self.planner.create_plan(query, context=full_context)

        step1.duration = time.time() - step1_start
        step1.status = "complete"
        step1.details = {
            "sub_questions": [sq.question for sq in plan.sub_questions],
            "count": len(plan.sub_questions),
        }
        notify("Research Planning", "A2", "Complete", step1.details)

        # ──────────────────────────────────────────────
        # Step 2: ITERATIVE SEARCH (T1 + T2)
        # ──────────────────────────────────────────────
        step2 = log.add_step("Iterative Search", "T1+T2")
        step2_start = time.time()
        notify("Iterative Search", "T1+T2",
               "Searching arXiv + web for each sub-question...")

        search_results = self.searcher.search_all(plan, query)

        step2.duration = time.time() - step2_start
        step2.status = "complete"
        total_papers = sum(
            len(r.papers) for r in search_results.values()
        )
        total_web = sum(
            len(r.web_results) for r in search_results.values()
        )
        step2.details = {
            "total_papers": total_papers,
            "total_web_results": total_web,
            "per_question": {
                q: {"papers": len(r.papers), "web": len(r.web_results)}
                for q, r in search_results.items()
            },
        }
        notify("Iterative Search", "T1+T2", "Complete", step2.details)

        # ──────────────────────────────────────────────
        # Step 3: SELF-CRITIQUE with retry (A1 + A2)
        # ──────────────────────────────────────────────
        max_critique_rounds = 2
        critique = None
        for round_num in range(max_critique_rounds):
            step3 = log.add_step(
                f"Self-Critique (Round {round_num + 1})",
                "A1+A2"
            )
            step3_start = time.time()
            notify("Self-Critique", "A1+A2",
                   f"Evaluating findings (round {round_num + 1})...")

            critique = self.critic.critique(plan, search_results)

            step3.duration = time.time() - step3_start
            step3.details = {
                "quality_score": critique.quality_score,
                "gaps": critique.gaps,
                "tool_issues": critique.tool_issues,
                "needs_more": critique.needs_more_search,
            }

            if not critique.needs_more_search or round_num == max_critique_rounds - 1:
                step3.status = "complete"
                notify("Self-Critique", "A1+A2", "Complete", step3.details)
                break

            # Go back to Step 2 for gaps
            step3.status = "gaps_found"
            notify("Self-Critique", "A1+A2",
                   f"Found {len(critique.gaps)} gaps, re-searching...",
                   step3.details)

            gap_step = log.add_step("Gap Re-Search", "T2")
            gap_start = time.time()

            for gap in critique.gaps[:3]:  # Limit re-search
                gap_results = self.searcher.search_for_gap(gap, query)
                search_results[gap] = gap_results

            gap_step.duration = time.time() - gap_start
            gap_step.status = "complete"

        # ──────────────────────────────────────────────
        # Step 4: SYNTHESIS + MEMORY (A2 + T2)
        # ──────────────────────────────────────────────
        step4 = log.add_step("Synthesis + Memory", "A2+T2")
        step4_start = time.time()
        notify("Synthesis + Memory", "A2+T2", "Generating research report...")

        # Include user context in synthesis if provided
        synthesis = self.synthesizer.synthesize(
            plan, search_results, query,
            extra_context=user_context
        )

        step4.duration = time.time() - step4_start
        step4.status = "complete"
        step4.details = {
            "quality_score": synthesis["quality"]["overall"],
            "papers_cited": synthesis["papers_cited"],
            "memory_id": synthesis["memory_id"],
        }
        notify("Synthesis + Memory", "A2+T2", "Complete", step4.details)

        # Final
        log.total_duration = time.time() - start_time

        # Collect all papers for PDF export
        all_papers = []
        for r in search_results.values():
            all_papers.extend(r.papers)

        return {
            "report": synthesis["report"],
            "quality": synthesis["quality"],
            "plan": plan,
            "search_results": search_results,
            "critique": critique,
            "pipeline_log": log,
            "memory_id": synthesis["memory_id"],
            "all_papers": all_papers,
            "user_context": user_context,
        }

    def generate_pdf(self, query: str, result: dict) -> str:
        """Generate a PDF report from research results."""
        pipeline_info = {
            "steps": [
                {"name": s.name, "paradigm": s.paradigm}
                for s in result["pipeline_log"].steps
            ]
        }
        return self.report_gen.generate(
            title=query,
            report_text=result["report"],
            papers=result.get("all_papers", []),
            quality_score=result["quality"]["overall"],
            pipeline_info=pipeline_info,
        )

    def followup(self, question: str, research_context: dict) -> str:
        """
        Answer a follow-up question using the research context.
        This keeps the conversation contextual.
        """
        # Build context from previous research
        context_parts = []
        if research_context.get("report"):
            context_parts.append(f"Research Report:\n{research_context['report'][:3000]}")
        if research_context.get("user_context"):
            context_parts.append(
                f"User's Source Material:\n{research_context['user_context'][:2000]}"
            )

        context = "\n\n---\n\n".join(context_parts)

        response = self.llm.generate(
            prompt=f"""Based on the following research context, answer the user's follow-up question.

{context}

User's Question: {question}

Provide a detailed, well-referenced answer based on the research findings above.
If the answer requires information not in the context, say so clearly.""",
            system_prompt="You are a research assistant. Answer questions based on "
                          "the research findings provided. Be specific and cite "
                          "relevant papers or sources when possible."
        )
        return response


class BaselineAgent:
    """
    Baseline: NO adaptation.
    
    Simple single-shot agent for comparison.
    No planning, no iteration, no critique, no memory.
    """

    def __init__(self, provider="groq", model=None):
        self.llm = LLMClient(provider=provider, model=model)
        self.arxiv = ArxivTool()

    def research(self, query: str, callback=None) -> dict:
        """Simple single-step research: search once → summarize."""
        start_time = time.time()

        if callback:
            callback("Baseline Search", "None", "Searching...")

        # Single search — no query optimization
        papers = self.arxiv.search(query, max_results=5)

        if callback:
            callback("Baseline Search", "None", f"Found {len(papers)} papers")

        # Single-shot summary — no critique or iteration
        paper_text = "\n".join(
            f"- {p.title}: {p.abstract[:200]}"
            for p in papers
        ) if papers else "No papers found."

        prompt = f"""Summarize the following papers for this query: {query}

Papers:
{paper_text}

Write a brief research summary."""

        report = self.llm.generate(prompt)
        duration = time.time() - start_time

        if callback:
            callback("Baseline Summary", "None", "Complete")

        return {
            "report": report,
            "quality": {"overall": 0.0, "scores": {}},
            "papers_found": len(papers),
            "duration": duration,
            "paradigms_used": "None (no adaptation)",
        }
