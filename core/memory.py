"""
Persistent Memory — T2: Tool Optimized for Frozen Agent.

Stores research findings across sessions so the frozen agent
can recall and build on previous research.

In the paper, memory-as-T2 means:
- Treat memory as a tool
- Optimize memory to serve the frozen agent
- Examples: Mem-α, Memento, ReasoningBank, Dynamic Cheatsheet

Paradigm: T2 (Tool adaptation, frozen agent's output signal)
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Optional


class PersistentMemory:
    """
    T2: Memory as a Tool for Frozen Agent.

    Key paper insight: Memory is a T2 tool. It's trained/optimized
    to store and retrieve information in a way that helps the
    frozen agent perform better on future queries.

    For demo: JSON-based persistent storage with keyword search.
    In production: would use vector DB + fine-tuned memory retriever.
    """

    def __init__(self, memory_dir: str = "summaries"):
        self.memory_dir = memory_dir
        self.memory_file = os.path.join(memory_dir, "_memory_index.json")
        os.makedirs(memory_dir, exist_ok=True)
        self.index = self._load_index()

    def store(self, query: str, findings: str, papers: list = None, metadata: dict = None) -> str:
        """
        Store research findings for future recall.
        T2: Optimized to store in format the frozen agent can best use.
        """
        entry_id = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        entry = {
            "id": entry_id,
            "query": query,
            "findings": findings,
            "timestamp": datetime.now().isoformat(),
            "papers": [
                {
                    "title": p.title if hasattr(p, "title") else str(p),
                    "arxiv_id": p.arxiv_id if hasattr(p, "arxiv_id") else "",
                }
                for p in (papers or [])
            ],
            "metadata": metadata or {},
            # T2: Extract keywords to help frozen agent find relevant past research
            "keywords": self._extract_keywords(query, findings),
        }

        # Save full entry
        entry_path = os.path.join(self.memory_dir, f"{entry_id}.json")
        with open(entry_path, "w") as f:
            json.dump(entry, f, indent=2)

        # Update index
        self.index[entry_id] = {
            "query": query,
            "timestamp": entry["timestamp"],
            "keywords": entry["keywords"],
            "summary": findings[:200],
        }
        self._save_index()

        return entry_id

    def recall(self, query: str, max_results: int = 3) -> List[Dict]:
        """
        Recall relevant past research for a new query.
        T2: Retrieval optimized to serve the frozen agent.
        """
        query_words = set(query.lower().split())
        scored_entries = []

        for entry_id, meta in self.index.items():
            # Simple keyword overlap scoring
            entry_keywords = set(meta.get("keywords", []))
            overlap = len(query_words & entry_keywords)

            # Also check query similarity
            entry_query_words = set(meta.get("query", "").lower().split())
            query_overlap = len(query_words & entry_query_words)

            score = overlap + query_overlap * 2  # Boost query match

            if score > 0:
                scored_entries.append((score, entry_id, meta))

        # Sort by relevance
        scored_entries.sort(reverse=True)

        results = []
        for score, entry_id, meta in scored_entries[:max_results]:
            # Load full entry
            entry_path = os.path.join(self.memory_dir, f"{entry_id}.json")
            if os.path.exists(entry_path):
                with open(entry_path) as f:
                    full_entry = json.load(f)
                full_entry["relevance_score"] = score
                results.append(full_entry)

        return results

    def get_all_queries(self) -> List[str]:
        """List all past research queries."""
        return [meta["query"] for meta in self.index.values()]

    def _extract_keywords(self, query: str, findings: str) -> List[str]:
        """Extract keywords for future retrieval."""
        # Simple keyword extraction
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "shall",
            "in", "on", "at", "to", "for", "of", "with", "by", "from",
            "and", "or", "but", "not", "this", "that", "these", "those",
            "it", "its", "they", "them", "their", "we", "our", "you",
            "your", "he", "she", "his", "her", "what", "which", "who",
            "how", "when", "where", "why", "about", "into", "each",
            "such", "as", "also", "more", "than", "very", "most",
        }

        text = f"{query} {findings[:500]}".lower()
        words = [w.strip(".,;:!?()[]{}\"'") for w in text.split()]
        keywords = [w for w in words if w and len(w) > 2 and w not in stop_words]

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for w in keywords:
            if w not in seen:
                seen.add(w)
                unique.append(w)

        return unique[:30]

    def _load_index(self) -> Dict:
        if os.path.exists(self.memory_file):
            with open(self.memory_file) as f:
                return json.load(f)
        return {}

    def _save_index(self):
        with open(self.memory_file, "w") as f:
            json.dump(self.index, f, indent=2)
