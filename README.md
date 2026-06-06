# Sovereign Brain

[![CI](https://github.com/sophiacave/sovereign-brain/actions/workflows/ci.yml/badge.svg)](https://github.com/sophiacave/sovereign-brain/actions/workflows/ci.yml)


[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Tests: 30 passing](https://img.shields.io/badge/Tests-30%20passing-green.svg)](tests/)

**Local-first persistent memory with semantic search.** sqlite-vec powered. No cloud dependency. Built for AI agents that need to remember.

## Features

| Feature | Description |
|---------|-------------|
| Key-value store | Read, write, delete brain entries with categories and priorities |
| Semantic search | Vector similarity via sqlite-vec + Ollama embeddings |
| Knowledge graph | Entries linked with typed, weighted edges |
| Hybrid retrieval | 3-signal search: keyword + semantic + graph, re-ranked |
| Temporal awareness | Auto-detects event dates, tracks lifecycle (upcoming/today/past) |
| Anti-repetition | Penalizes recently-shown results per session (up to 90%) |
| Freshness decay | Exponential decay scoring with configurable half-life |
| Time-window queries | Filter by date range |
| Auto-archive | Completed events archived after N days |
| Chat history | Per-session conversation tracking |
| Zero cloud | Everything runs locally. Your data stays yours. |

## Install

```bash
pip install sovereign-brain

# For semantic search (recommended):
pip install sovereign-brain[vec]
ollama pull mxbai-embed-large
```

## Quick Start

```python
from sovereign_brain import BrainAPI

brain = BrainAPI()

# Write
brain.write("project.api", {"status": "active", "date": "2026-06-05"},
            category="projects", description="API redesign", priority=8)

# Read
data = brain.read("project.api")  # {"status": "active", ...}

# Search (keyword)
results = brain.search("api redesign", limit=5)

# Hybrid search (keyword + semantic + graph + freshness)
results = brain.hybrid_search("what's the api status", k=5, chat_id="session1")

# Knowledge graph
brain.add_edge("project.api", "project.frontend", "blocks", weight=0.9)
neighbors = brain.graph_neighbors("project.api", depth=2)

# Temporal awareness
temporal = brain.temporal_extract(data)  # {"event_date": "2026-06-05", "status": "active"}
status = brain.temporal_status(temporal)  # "today"

# Chat history
brain.save_chat("session1", "user", "what's the api status?")
brain.save_chat("session1", "assistant", "The API redesign is active.")
history = brain.chat_history("session1")

# Stats
brain.stats()  # {"entries": 42, "graph_edges": 15, "vectors": 200, "chats": 3}

brain.close()
```

## Anti-Repetition

When you pass `chat_id` to `hybrid_search`, the brain tracks what was shown and penalizes repeated results:

```python
# First call: normal results
r1 = brain.hybrid_search("deployment", k=3, chat_id="s1")

# Second call: previously-shown entries get 30% penalty per occurrence
r2 = brain.hybrid_search("deployment", k=3, chat_id="s1")
# Different results surface to avoid repetition
```

## Freshness Decay

Results are scored with exponential freshness decay (48h half-life by default):

```python
brain.freshness_score(datetime.now().isoformat())     # 1.0 (just updated)
brain.freshness_score("2026-06-01T00:00:00")           # ~0.25 (4 days old)
brain.freshness_score("2026-05-01T00:00:00")           # ~0.0 (very old)
```

## Architecture

```
sovereign_brain/
  brain.py    # BrainAPI class — all features in one module
  __init__.py # Exports BrainAPI, get_brain

Storage: SQLite (brain_context + brain_graph + brain_chat_history + brain_retrieval_log)
Vectors: sqlite-vec extension (optional, graceful fallback to keyword search)
Embeddings: Ollama local API (mxbai-embed-large, 1024 dimensions)
```

## Configuration

```python
brain = BrainAPI(
    db_path="./my_brain.db",                    # Custom DB location
    ollama_url="http://localhost:11434/api/embed",  # Ollama endpoint
    embed_model="mxbai-embed-large",            # Embedding model
    vec_extension_path="/path/to/vec0",         # sqlite-vec extension
)
```

## License

MIT. Built by [Like One Foundation](https://likeone.ai).
