#!/usr/bin/env python3
"""
sovereign_brain.brain — Unified Brain API

Local-first persistent memory with:
  - Key-value store (brain_context table)
  - Semantic search via sqlite-vec + Ollama embeddings
  - Knowledge graph (brain_graph table)
  - Hybrid 3-signal retrieval (keyword + semantic + graph)
  - Temporal awareness (event lifecycle: upcoming/today/past)
  - Anti-repetition engine (penalizes recently-shown results)
  - Freshness decay scoring (exponential, configurable half-life)
  - Time-window filtering (date range queries)
  - Auto-archive (completed events -> archive after N days)
  - Chat history tracking

Built with love. No cloud dependency.
"""

import json
import math
import sqlite3
import struct
from datetime import date, datetime, timedelta
from pathlib import Path

DEFAULT_DB = Path.home() / ".sovereign_brain" / "brain.db"
OLLAMA_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "mxbai-embed-large"
DIMENSIONS = 1024

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS brain_context (
    key TEXT PRIMARY KEY,
    category TEXT DEFAULT 'general',
    description TEXT DEFAULT '',
    value TEXT DEFAULT '',
    priority INTEGER DEFAULT 5,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS brain_graph (
    from_key TEXT NOT NULL,
    to_key TEXT NOT NULL,
    relationship TEXT DEFAULT '',
    weight REAL DEFAULT 1.0,
    PRIMARY KEY (from_key, to_key)
);

CREATE TABLE IF NOT EXISTS brain_chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_history_chat ON brain_chat_history(chat_id);

CREATE TABLE IF NOT EXISTS brain_retrieval_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL,
    chat_id TEXT DEFAULT 'default',
    timestamp TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_retrieval_chat_ts ON brain_retrieval_log(chat_id, timestamp);
"""

# Temporal date fields to auto-detect in entry values
TEMPORAL_DATE_FIELDS = ["date", "event_date", "deadline", "scheduled_at", "completed_at", "due_date"]


class BrainAPI:
    """Unified brain access — sqlite-vec powered, singleton-friendly."""

    def __init__(self, db_path=None, ollama_url=None, embed_model=None, vec_extension_path=None):
        """Initialize brain.

        Args:
            db_path: Path to SQLite database. Default: ~/.sovereign_brain/brain.db
            ollama_url: Ollama API URL. Default: http://localhost:11434/api/embed
            embed_model: Embedding model name. Default: mxbai-embed-large
            vec_extension_path: Path to sqlite-vec extension (.so/.dylib). Auto-detected if None.
        """
        self._db_path = str(db_path or DEFAULT_DB)
        self._ollama_url = ollama_url or OLLAMA_URL
        self._embed_model = embed_model or EMBED_MODEL
        self._vec_ext_path = vec_extension_path
        self._db = None
        self._vec_loaded = False

    def _conn(self):
        if self._db is None:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(self._db_path)
            self._db.row_factory = sqlite3.Row
            self._db.executescript(SCHEMA_SQL)
        return self._db

    def _load_vec(self):
        if not self._vec_loaded:
            db = self._conn()
            ext_path = self._vec_ext_path
            if not ext_path:
                ext_path = self._find_vec_extension()
            if ext_path:
                try:
                    db.enable_load_extension(True)
                    db.load_extension(ext_path)
                    self._vec_loaded = True
                except Exception:
                    pass

    @staticmethod
    def _find_vec_extension():
        """Auto-detect sqlite-vec extension path."""
        import sysconfig
        candidates = [
            Path(sysconfig.get_path("purelib")) / "sqlite_vec" / "vec0",
            Path("/opt/homebrew/lib/python3.14/site-packages/sqlite_vec/vec0"),
            Path("/opt/homebrew/lib/python3.13/site-packages/sqlite_vec/vec0"),
            Path("/opt/homebrew/lib/python3.12/site-packages/sqlite_vec/vec0"),
        ]
        for c in candidates:
            for suffix in ["", ".dylib", ".so"]:
                p = Path(str(c) + suffix)
                if p.exists():
                    return str(c)
        return None

    # ── READ / WRITE ──────────────────────────────────────

    def read(self, key):
        """Read a brain entry by key. Returns parsed JSON or raw string."""
        row = self._conn().execute(
            "SELECT value FROM brain_context WHERE key = ?", (key,)
        ).fetchone()
        if not row:
            return None
        try:
            return json.loads(row["value"])
        except (json.JSONDecodeError, TypeError):
            return row["value"]

    def read_full(self, key):
        """Read full entry with metadata."""
        row = self._conn().execute(
            "SELECT key, category, description, value, priority, updated_at FROM brain_context WHERE key = ?",
            (key,)
        ).fetchone()
        if not row:
            return None
        val = row["value"]
        try:
            val = json.loads(val)
        except (json.JSONDecodeError, TypeError):
            pass
        return {
            "key": row["key"], "category": row["category"],
            "description": row["description"], "value": val,
            "priority": row["priority"], "updated_at": row["updated_at"]
        }

    def write(self, key, value, category="general", description="", priority=5):
        """Write or update a brain entry."""
        val_str = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
        ts = datetime.now().isoformat()
        self._conn().execute("""
            INSERT OR REPLACE INTO brain_context (key, category, description, value, priority, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (key, category, description, val_str, priority, ts))
        self._conn().commit()

    def delete(self, key):
        """Delete a brain entry."""
        self._conn().execute("DELETE FROM brain_context WHERE key = ?", (key,))
        self._conn().commit()

    # ── SEARCH ────────────────────────────────────────────

    def search(self, query, limit=5):
        """Keyword search across keys, descriptions, and values."""
        pattern = f"%{query}%"
        rows = self._conn().execute("""
            SELECT key, category, description, value, priority, updated_at
            FROM brain_context
            WHERE key LIKE ? OR description LIKE ? OR value LIKE ?
            ORDER BY priority DESC LIMIT ?
        """, (pattern, pattern, pattern, limit)).fetchall()
        return [dict(r) for r in rows]

    def vec_search(self, query, k=5, collection=None):
        """Semantic vector search using sqlite-vec + Ollama embeddings."""
        self._load_vec()
        if not self._vec_loaded:
            return self.search(query, k)

        embedding = self._embed(query)
        if not embedding:
            return self.search(query, k)

        vec_bytes = struct.pack(f'{len(embedding)}f', *embedding)
        fetch_k = k * 3 if collection else k

        try:
            rows = self._conn().execute("""
                SELECT v.distance, m.doc_id, m.collection, m.document, m.metadata
                FROM vec_brain v
                JOIN vec_brain_meta m ON v.rowid = m.rowid
                WHERE v.embedding MATCH ? AND k = ?
                ORDER BY v.distance
            """, (vec_bytes, fetch_k)).fetchall()
        except Exception:
            return self.search(query, k)

        results = []
        for r in rows:
            if collection and r["collection"] != collection:
                continue
            meta = json.loads(r["metadata"]) if r["metadata"] else {}
            brain_key = meta.get("key", r["doc_id"])

            bc = self._conn().execute(
                "SELECT key, category, description, value, priority, updated_at FROM brain_context WHERE key = ?",
                (brain_key,)
            ).fetchone()

            if bc:
                results.append({
                    "key": bc["key"], "category": bc["category"] or "",
                    "description": bc["description"] or "",
                    "value": bc["value"] or "",
                    "priority": bc["priority"] or 5,
                    "updated_at": bc["updated_at"] or "",
                    "similarity": round(1 - r["distance"], 3) if r["distance"] <= 1 else 0,
                    "collection": r["collection"]
                })
            else:
                results.append({
                    "key": r["doc_id"], "category": r["collection"],
                    "description": (r["document"] or "")[:200],
                    "value": "", "priority": 5, "updated_at": "",
                    "similarity": round(1 - r["distance"], 3) if r["distance"] <= 1 else 0,
                    "collection": r["collection"]
                })
            if len(results) >= k:
                break

        return results

    def _embed(self, text):
        """Get embedding vector from Ollama."""
        try:
            import requests
            resp = requests.post(self._ollama_url, json={
                "model": self._embed_model, "input": text
            }, timeout=15)
            if resp.status_code == 200:
                vecs = resp.json().get("embeddings", [])
                return vecs[0] if vecs else None
        except Exception:
            pass
        return None

    # ── GRAPH ─────────────────────────────────────────────

    def graph_neighbors(self, key, depth=1):
        """Find connected entries in the knowledge graph."""
        results = []
        visited = {key}
        frontier = [key]

        for _ in range(depth):
            next_frontier = []
            for k in frontier:
                rows = self._conn().execute("""
                    SELECT from_key, to_key, relationship, weight
                    FROM brain_graph
                    WHERE from_key = ? OR to_key = ?
                """, (k, k)).fetchall()
                for r in rows:
                    neighbor = r["to_key"] if r["from_key"] == k else r["from_key"]
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
                        results.append({
                            "key": neighbor,
                            "relationship": r["relationship"],
                            "weight": r["weight"],
                            "from": r["from_key"]
                        })
            frontier = next_frontier

        return results

    def add_edge(self, from_key, to_key, relationship="related", weight=1.0):
        """Add a graph edge between two brain entries."""
        self._conn().execute("""
            INSERT OR REPLACE INTO brain_graph (from_key, to_key, relationship, weight)
            VALUES (?, ?, ?, ?)
        """, (from_key, to_key, relationship, weight))
        self._conn().commit()

    # ── HYBRID SEARCH ─────────────────────────────────────

    def hybrid_search(self, query, k=5, chat_id=None, since=None, until=None):
        """Three-signal retrieval: keyword + semantic + graph.
        Freshness decay + anti-repetition + time-window filtering.

        Args:
            query: Search query string
            k: Max results
            chat_id: Session ID for anti-repetition tracking
            since: Filter entries updated after this date (YYYY-MM-DD)
            until: Filter entries updated before this date (YYYY-MM-DD)
        """
        seen = {}

        # Anti-repetition: get recently shown keys
        recent_keys = self.get_recent_retrievals(chat_id, hours=4) if chat_id else {}

        # Signal 1: Keyword
        for r in self.search(query, limit=k):
            key = r["key"]
            if key not in seen:
                seen[key] = {"key": key, "score": 0, "signals": [], **r}
            seen[key]["score"] += r.get("priority", 5) * 0.1
            seen[key]["signals"].append("keyword")

        # Signal 2: Semantic
        for r in self.vec_search(query, k=k):
            key = r["key"]
            sim = r.get("similarity", 0)
            if key not in seen:
                seen[key] = {"key": key, "score": 0, "signals": [], **r}
            seen[key]["score"] += sim * 2.0
            seen[key]["signals"].append(f"semantic({sim:.2f})")

        # Signal 3: Graph expansion
        top_keys = sorted(seen.items(), key=lambda x: -x[1]["score"])[:3]
        for key, _ in top_keys:
            neighbors = self.graph_neighbors(key, depth=1)
            for n in neighbors[:3]:
                nk = n["key"]
                if nk not in seen:
                    entry = self.read_full(nk)
                    if entry:
                        seen[nk] = {"key": nk, "score": 0, "signals": [], **entry}
                seen.get(nk, {}).get("signals", []).append(f"graph({n['relationship'][:20]})")
                if nk in seen:
                    seen[nk]["score"] += n.get("weight", 0.5) * 0.5

        # Freshness decay (0.70 base + 0.30 freshness)
        for key, entry in seen.items():
            updated_at = entry.get("updated_at", "")
            if not updated_at:
                full = self.read_full(key)
                updated_at = full.get("updated_at", "") if full else ""
                entry["updated_at"] = updated_at
            freshness = self.freshness_score(updated_at)
            entry["score"] = entry["score"] * 0.70 + freshness * 2.0 * 0.30
            entry["_freshness"] = round(freshness, 3)

        # Anti-repetition penalty (up to 90%)
        for key, times in recent_keys.items():
            if key in seen:
                penalty = min(times * 0.3, 0.9)
                seen[key]["score"] *= (1.0 - penalty)
                seen[key]["signals"].append(f"repeat(-{penalty:.0%})")

        # Time-window filter
        if since or until:
            filtered = {}
            for key, entry in seen.items():
                updated_at = (entry.get("updated_at") or "")[:10]
                if since and updated_at and updated_at < since:
                    continue
                if until and updated_at and updated_at > until:
                    continue
                filtered[key] = entry
            seen = filtered

        results = sorted(seen.values(), key=lambda x: -x["score"])[:k]

        # Log retrievals for anti-repetition
        if chat_id and results:
            self.log_retrieval([r["key"] for r in results], chat_id)

        return results

    # ── TEMPORAL MIND ─────────────────────────────────────

    def temporal_extract(self, value):
        """Extract temporal metadata from a brain entry value."""
        if not isinstance(value, dict):
            try:
                value = json.loads(value) if isinstance(value, str) else {}
            except (json.JSONDecodeError, TypeError):
                return None
        if not isinstance(value, dict):
            return None

        if "_temporal" in value:
            return value["_temporal"]

        temporal = {}
        for field in TEMPORAL_DATE_FIELDS:
            if field in value:
                temporal["event_date"] = str(value[field])
                break

        if "status" in value:
            status = str(value["status"]).upper()
            if any(kw in status for kw in ["COMPLETE", "DONE", "FINISHED", "PASSED", "ARCHIVED"]):
                temporal["status"] = "completed"
            elif any(kw in status for kw in ["ACTIVE", "IN PROGRESS", "LIVE", "RUNNING"]):
                temporal["status"] = "active"
            elif any(kw in status for kw in ["PLANNED", "UPCOMING", "SCHEDULED", "PENDING"]):
                temporal["status"] = "upcoming"

        return temporal if temporal else None

    def temporal_status(self, temporal_meta):
        """Determine temporal status: past | today | upcoming | unknown."""
        if not temporal_meta:
            return "unknown"

        if temporal_meta.get("status") in ("completed", "archived"):
            return "past"

        event_date_str = temporal_meta.get("event_date", "")
        if not event_date_str:
            return temporal_meta.get("status", "unknown")

        try:
            event_date = None
            for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
                try:
                    event_date = datetime.strptime(event_date_str[:len(fmt.replace('%', 'X'))], fmt).date()
                    break
                except ValueError:
                    continue
            if not event_date:
                event_date = datetime.fromisoformat(event_date_str[:10]).date()

            today = date.today()
            if event_date < today:
                return "past"
            elif event_date == today:
                return "today"
            else:
                return "upcoming"
        except Exception:
            return temporal_meta.get("status", "unknown")

    def temporal_annotate(self, results):
        """Annotate retrieval results with temporal status tags."""
        for r in results:
            val = r.get("value", "")
            temporal = self.temporal_extract(val)
            if temporal:
                r["_temporal_status"] = self.temporal_status(temporal)
                r["_temporal"] = temporal
            else:
                r["_temporal_status"] = "static"
        return results

    # ── ANTI-REPETITION ───────────────────────────────────

    def log_retrieval(self, keys, chat_id="default"):
        """Log retrieved keys for anti-repetition tracking."""
        ts = datetime.now().isoformat()
        for key in keys:
            self._conn().execute(
                "INSERT INTO brain_retrieval_log (key, chat_id, timestamp) VALUES (?, ?, ?)",
                (key, str(chat_id), ts)
            )
        self._conn().commit()

    def get_recent_retrievals(self, chat_id="default", hours=4):
        """Get keys retrieved in the last N hours. Returns {key: times_shown}."""
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        rows = self._conn().execute("""
            SELECT DISTINCT key, COUNT(*) as times_shown
            FROM brain_retrieval_log
            WHERE chat_id = ? AND timestamp > ?
            GROUP BY key ORDER BY times_shown DESC
        """, (str(chat_id), cutoff)).fetchall()
        return {r["key"]: r["times_shown"] for r in rows}

    def cleanup_retrieval_log(self, hours=24):
        """Purge retrieval log entries older than N hours."""
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        cur = self._conn().execute(
            "DELETE FROM brain_retrieval_log WHERE timestamp < ?", (cutoff,)
        )
        self._conn().commit()
        return cur.rowcount

    # ── FRESHNESS ─────────────────────────────────────────

    def freshness_score(self, updated_at_str, half_life_hours=48):
        """Exponential decay freshness score (0.0-1.0).
        score = exp(-ln(2) * hours_ago / half_life)"""
        if not updated_at_str:
            return 0.5
        try:
            updated = datetime.fromisoformat(updated_at_str[:19])
            hours_ago = max(0, (datetime.now() - updated).total_seconds() / 3600)
            return math.exp(-math.log(2) * hours_ago / half_life_hours)
        except Exception:
            return 0.5

    # ── AUTO-ARCHIVE ──────────────────────────────────────

    def auto_archive(self, days=30):
        """Archive completed temporal events older than N days."""
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        archived = 0

        rows = self._conn().execute("""
            SELECT key, value, category FROM brain_context WHERE category != 'archive'
        """).fetchall()

        for row in rows:
            temporal = self.temporal_extract(row["value"])
            if not temporal:
                continue
            if self.temporal_status(temporal) != "past":
                continue
            event_date = temporal.get("event_date", "")
            if event_date and event_date[:10] < cutoff:
                self._conn().execute(
                    "UPDATE brain_context SET category = 'archive', updated_at = ? WHERE key = ?",
                    (datetime.now().isoformat(), row["key"])
                )
                archived += 1

        if archived:
            self._conn().commit()
        return archived

    # ── CHAT HISTORY ──────────────────────────────────────

    def save_chat(self, chat_id, role, content):
        """Save a chat message."""
        ts = datetime.now().isoformat()
        self._conn().execute(
            "INSERT INTO brain_chat_history (chat_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (str(chat_id), role, content, ts)
        )
        self._conn().commit()

    def chat_history(self, chat_id, limit=20):
        """Get recent chat messages for a session."""
        rows = self._conn().execute("""
            SELECT role, content, timestamp FROM brain_chat_history
            WHERE chat_id = ? ORDER BY id DESC LIMIT ?
        """, (str(chat_id), limit)).fetchall()
        return [dict(r) for r in reversed(rows)]

    # ── METADATA ──────────────────────────────────────────

    def categories(self):
        """List all categories."""
        rows = self._conn().execute(
            "SELECT DISTINCT category FROM brain_context WHERE category IS NOT NULL ORDER BY category"
        ).fetchall()
        return [r["category"] for r in rows]

    def keys(self, category=None):
        """List all keys, optionally filtered by category."""
        if category:
            rows = self._conn().execute(
                "SELECT key FROM brain_context WHERE category = ? ORDER BY priority DESC, key",
                (category,)
            ).fetchall()
        else:
            rows = self._conn().execute(
                "SELECT key FROM brain_context ORDER BY category, priority DESC, key"
            ).fetchall()
        return [r["key"] for r in rows]

    def stats(self):
        """Get brain statistics."""
        db = self._conn()
        self._load_vec()
        result = {
            "entries": db.execute("SELECT COUNT(*) FROM brain_context").fetchone()[0],
            "graph_edges": db.execute("SELECT COUNT(*) FROM brain_graph").fetchone()[0],
        }
        try:
            result["chats"] = db.execute("SELECT COUNT(DISTINCT chat_id) FROM brain_chat_history").fetchone()[0]
        except Exception:
            result["chats"] = 0
        if self._vec_loaded:
            try:
                result["vectors"] = db.execute("SELECT COUNT(*) FROM vec_brain").fetchone()[0]
            except Exception:
                result["vectors"] = 0
        return result

    # ── CLEANUP ───────────────────────────────────────────

    def close(self):
        """Close database connection."""
        if self._db:
            self._db.close()
            self._db = None
            self._vec_loaded = False


# Convenience singleton
_instance = None


def get_brain(db_path=None):
    """Get or create a singleton BrainAPI instance."""
    global _instance
    if _instance is None:
        _instance = BrainAPI(db_path=db_path)
    return _instance
