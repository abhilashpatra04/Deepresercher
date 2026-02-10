"""
Tool Verifier — A1: Agent Learns from Tool Execution.

Verifies tool outputs are valid before the agent processes them.
In production, the agent would be fine-tuned using these verification
signals. For demo, we check validity programmatically.

Paradigm: A1 (Agent adaptation, tool execution feedback)
"""

from typing import List, Dict, Any


class ToolVerifier:
    """
    A1: Tool Execution Feedback.

    In the paper, A1 means the agent learns from tool execution signals
    (e.g., code runs, retrieval recalls, SQL executes correctly).

    In our demo, this module provides the verification signals that
    would be used to train/adapt the agent in production:
    - Did the arXiv search return valid papers?
    - Did the PDF parse successfully?
    - Are the extracted sections non-empty?

    Examples from paper: DeepRetrieval (retrieval metrics as reward),
    Code-R1 (test pass rate as reward), Toolformer (tool call success).
    """

    def __init__(self):
        self.verification_log = []

    def verify_search_results(self, papers: list) -> Dict[str, Any]:
        """
        Verify arXiv search returned valid results.
        A1 signal: Did the search tool execute successfully?
        """
        issues = []

        if not papers:
            issues.append("No papers returned from search")
            return self._log_result("arxiv_search", False, issues)

        if len(papers) < 2:
            issues.append(f"Very few results: only {len(papers)} paper(s)")

        for paper in papers:
            if not hasattr(paper, "abstract") or not paper.abstract:
                issues.append(f"Missing abstract for: {paper.arxiv_id if hasattr(paper, 'arxiv_id') else 'unknown'}")
            if not hasattr(paper, "title") or not paper.title:
                issues.append(f"Missing title for paper")

        valid = len(issues) == 0
        return self._log_result("arxiv_search", valid, issues)

    def verify_pdf_parse(self, result: Dict) -> Dict[str, Any]:
        """
        Verify PDF parsing was successful.
        A1 signal: Did the PDF tool produce usable text?
        """
        issues = []

        if result.get("status") == "failed":
            issues.append(f"Parse failed: {result.get('error', 'unknown')}")
            return self._log_result("pdf_parse", False, issues)

        text = result.get("text", "")
        if len(text) < 100:
            issues.append(f"Too little text extracted: {len(text)} chars")

        sections = result.get("sections", {})
        if not sections:
            issues.append("No sections detected in PDF")

        valid = len(issues) == 0
        return self._log_result("pdf_parse", valid, issues)

    def verify_query_rewrite(self, original: str, rewritten: str) -> Dict[str, Any]:
        """
        Verify query rewrite is reasonable.
        A1 signal: Did the rewriter produce a usable query?
        """
        issues = []

        if not rewritten or len(rewritten.strip()) < 3:
            issues.append("Rewritten query is empty or too short")

        if rewritten.lower() == original.lower():
            issues.append("Rewrite is identical to original (no improvement)")

        if len(rewritten) > 200:
            issues.append("Rewritten query is too long")

        if "[" in rewritten or "error" in rewritten.lower():
            issues.append("Rewrite contains error markers")

        valid = len(issues) == 0
        return self._log_result("query_rewrite", valid, issues)

    def _log_result(self, tool_name: str, valid: bool, issues: List[str]) -> Dict[str, Any]:
        """Log verification result for A1 training signal."""
        result = {
            "tool": tool_name,
            "valid": valid,
            "issues": issues,
            "signal": 1.0 if valid else 0.0,  # A1 reward signal
        }
        self.verification_log.append(result)
        return result

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all verification results."""
        total = len(self.verification_log)
        passed = sum(1 for r in self.verification_log if r["valid"])
        return {
            "total_checks": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": passed / total if total > 0 else 0,
            "all_issues": [
                issue
                for r in self.verification_log
                if not r["valid"]
                for issue in r["issues"]
            ],
        }
