"""Sovereign Brain — local-first persistent memory with semantic search.

sqlite-vec powered. No cloud. No API keys (except Ollama for embeddings).
Hybrid retrieval: keyword + semantic + graph. Temporal awareness. Anti-repetition.

Usage:
    from sovereign_brain import BrainAPI
    brain = BrainAPI()  # uses ~/.sovereign_brain/brain.db by default
    brain.write("project.notes", {"status": "active"}, description="Project notes")
    results = brain.hybrid_search("project status", k=5)
"""

from sovereign_brain.brain import BrainAPI, get_brain

__version__ = "1.0.0"
__all__ = ["BrainAPI", "get_brain"]
