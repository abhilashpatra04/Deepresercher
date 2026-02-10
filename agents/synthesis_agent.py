"""
Synthesis Agent — Step 4 of Deep Research Pipeline.

Combines findings from all sub-questions into a structured
research report, then stores in persistent memory.

Paradigm: A2 + T2
  A2: Agent synthesizes high-quality output (judged by final quality)
  T2: Memory module stores findings optimized for future agent recall
"""


class SynthesisAgent:
    """
    Step 4: Synthesis + Memory (A2 + T2).

    Paper context: Deep research systems perform "iterative search,
    validation, and synthesis" (Section 7.1).

    A2: The synthesis quality is the ultimate signal. In production,
        RL reward = how good is the final report.
    T2: Memory stores findings in a format optimized for the frozen
        agent to recall in future sessions (like Mem-α, Memento).
    """

    def __init__(self, llm_client, memory, quality_scorer):
        self.llm = llm_client
        self.memory = memory               # T2: persistent memory
        self.quality_scorer = quality_scorer  # A2: output scoring

    def synthesize(self, research_plan, search_results: dict, query: str,
                   extra_context: str = "") -> dict:
        """
        Combine all findings into a structured research report.
        Score quality (A2) and store in memory (T2).
        """
        # Build context from all search results
        context = self._build_context(research_plan, search_results)

        # Add user-provided context (from paper/URL)
        if extra_context:
            context = f"### User-Provided Source Material\n{extra_context[:2000]}\n\n{context}"

        # A2: Generate synthesis
        report = self._generate_report(query, context)

        # Collect all papers used
        all_papers = []
        for results in search_results.values():
            papers = results.papers if hasattr(results, 'papers') else []
            all_papers.extend(papers)

        # A2: Score the final output
        quality = self.quality_scorer.score(report, query, all_papers)

        # If quality is low, try to improve
        if not quality["pass"]:
            report = self._improve_report(report, query, context, quality["feedback"])
            quality = self.quality_scorer.score(report, query, all_papers)

        # T2: Store in persistent memory for future queries
        memory_id = self.memory.store(
            query=query,
            findings=report,
            papers=all_papers,
            metadata={
                "quality_score": quality["overall"],
                "sub_questions": [
                    sq.question if hasattr(sq, 'question') else str(sq)
                    for sq in research_plan.sub_questions
                ],
                "total_papers": len(all_papers),
            }
        )

        return {
            "report": report,
            "quality": quality,
            "memory_id": memory_id,
            "papers_cited": len(all_papers),
        }

    def _build_context(self, research_plan, search_results: dict) -> str:
        """Build context string from all search results."""
        sections = []

        for sub_q in research_plan.sub_questions:
            question = sub_q.question if hasattr(sub_q, 'question') else str(sub_q)
            results = search_results.get(question)

            if not results:
                sections.append(f"### {question}\nNo results found.\n")
                continue

            papers = results.papers if hasattr(results, 'papers') else []
            paper_summaries = []
            for p in papers[:5]:
                title = p.title if hasattr(p, 'title') else str(p)
                abstract = p.abstract if hasattr(p, 'abstract') else ""
                arxiv_id = p.arxiv_id if hasattr(p, 'arxiv_id') else ""
                paper_summaries.append(
                    f"- **{title}** [{arxiv_id}]\n  {abstract[:300]}"
                )

            sections.append(
                f"### {question}\n"
                f"Papers found: {len(papers)}\n"
                f"{chr(10).join(paper_summaries)}\n"
            )

        return "\n".join(sections)

    def _generate_report(self, query: str, context: str) -> str:
        """Generate the final research report."""
        prompt = f"""You are a research synthesizer. Write a comprehensive research
summary based on the papers found for each sub-question.

Research Query: {query}

Findings by Sub-Question:
{context}

Write a structured research summary with:
1. **Overview** — Brief answer to the main query
2. **Key Findings** — Most important discoveries from the papers
3. **Methodology Trends** — Common approaches and techniques
4. **Open Challenges** — Gaps and future directions
5. **Key Papers** — List the most important papers with brief descriptions

Use specific paper titles and findings. Do NOT hallucinate — only
reference papers that appear in the findings above.
"""
        return self.llm.generate(prompt, temperature=0.5)

    def _improve_report(self, report: str, query: str, context: str, feedback: str) -> str:
        """
        A2: Improve report based on quality feedback.
        This is the self-improvement loop driven by A2 signal.
        """
        prompt = f"""Improve this research summary based on the feedback below.

Original Query: {query}

Current Summary:
{report}

Quality Feedback: {feedback}

Available Papers (for grounding):
{context[:3000]}

Write an improved version that addresses the feedback.
Keep citations grounded in actual papers listed above."""

        return self.llm.generate(prompt, temperature=0.5)
