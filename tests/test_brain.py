"""Tests for sovereign_brain."""

import json
import os
import tempfile
from datetime import datetime, timedelta

import pytest

from sovereign_brain import BrainAPI


@pytest.fixture
def brain():
    """Create a temporary brain for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    b = BrainAPI(db_path=db_path)
    yield b
    b.close()
    os.unlink(db_path)


class TestReadWrite:
    def test_write_and_read_string(self, brain):
        brain.write("test.key", "hello world")
        assert brain.read("test.key") == "hello world"

    def test_write_and_read_dict(self, brain):
        brain.write("test.dict", {"status": "active", "count": 42})
        result = brain.read("test.dict")
        assert result["status"] == "active"
        assert result["count"] == 42

    def test_read_full(self, brain):
        brain.write("test.full", {"x": 1}, category="testing", description="A test", priority=8)
        full = brain.read_full("test.full")
        assert full["key"] == "test.full"
        assert full["category"] == "testing"
        assert full["description"] == "A test"
        assert full["priority"] == 8
        assert full["updated_at"] is not None

    def test_read_missing(self, brain):
        assert brain.read("nonexistent") is None

    def test_delete(self, brain):
        brain.write("test.delete", "bye")
        brain.delete("test.delete")
        assert brain.read("test.delete") is None

    def test_upsert(self, brain):
        brain.write("test.upsert", "v1")
        brain.write("test.upsert", "v2")
        assert brain.read("test.upsert") == "v2"


class TestSearch:
    def test_keyword_search(self, brain):
        brain.write("project.alpha", {"name": "Alpha"}, description="The alpha project")
        brain.write("project.beta", {"name": "Beta"}, description="The beta project")
        results = brain.search("alpha")
        assert len(results) >= 1
        assert results[0]["key"] == "project.alpha"

    def test_search_in_value(self, brain):
        brain.write("secret.key", {"password": "dragon"})
        results = brain.search("dragon")
        assert len(results) >= 1


class TestGraph:
    def test_add_and_find_neighbors(self, brain):
        brain.write("node.a", "A")
        brain.write("node.b", "B")
        brain.add_edge("node.a", "node.b", relationship="depends_on", weight=0.9)
        neighbors = brain.graph_neighbors("node.a")
        assert len(neighbors) == 1
        assert neighbors[0]["key"] == "node.b"
        assert neighbors[0]["relationship"] == "depends_on"

    def test_graph_depth(self, brain):
        brain.write("n.1", "1")
        brain.write("n.2", "2")
        brain.write("n.3", "3")
        brain.add_edge("n.1", "n.2", "link")
        brain.add_edge("n.2", "n.3", "link")
        depth1 = brain.graph_neighbors("n.1", depth=1)
        depth2 = brain.graph_neighbors("n.1", depth=2)
        assert len(depth1) == 1
        assert len(depth2) == 2


class TestTemporal:
    def test_temporal_extract_with_date(self, brain):
        val = {"date": "2026-06-05", "status": "UPCOMING"}
        temporal = brain.temporal_extract(val)
        assert temporal["event_date"] == "2026-06-05"
        assert temporal["status"] == "upcoming"

    def test_temporal_extract_completed(self, brain):
        val = {"date": "2026-01-01", "status": "COMPLETED"}
        temporal = brain.temporal_extract(val)
        assert temporal["status"] == "completed"

    def test_temporal_extract_non_dict(self, brain):
        assert brain.temporal_extract(42) is None
        assert brain.temporal_extract("just a string") is None

    def test_temporal_status_past(self, brain):
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        assert brain.temporal_status({"event_date": yesterday}) == "past"

    def test_temporal_status_today(self, brain):
        today = datetime.now().strftime("%Y-%m-%d")
        assert brain.temporal_status({"event_date": today}) == "today"

    def test_temporal_status_upcoming(self, brain):
        future = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        assert brain.temporal_status({"event_date": future}) == "upcoming"

    def test_temporal_annotate(self, brain):
        results = [{"key": "e.1", "value": json.dumps({"date": "2020-01-01", "status": "DONE"})}]
        annotated = brain.temporal_annotate(results)
        assert annotated[0]["_temporal_status"] == "past"


class TestAntiRepetition:
    def test_log_and_retrieve(self, brain):
        brain.log_retrieval(["k1", "k2"], "sess1")
        recent = brain.get_recent_retrievals("sess1", hours=1)
        assert "k1" in recent
        assert "k2" in recent

    def test_cleanup(self, brain):
        brain.log_retrieval(["old"], "sess1")
        cleaned = brain.cleanup_retrieval_log(hours=0)
        assert cleaned >= 1


class TestFreshness:
    def test_freshness_now(self, brain):
        score = brain.freshness_score(datetime.now().isoformat())
        assert score > 0.99

    def test_freshness_old(self, brain):
        old = (datetime.now() - timedelta(hours=96)).isoformat()
        score = brain.freshness_score(old)
        assert score < 0.30

    def test_freshness_none(self, brain):
        assert brain.freshness_score(None) == 0.5


class TestAutoArchive:
    def test_archive_old_completed(self, brain):
        old_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        brain.write("event.old", {"date": old_date, "status": "COMPLETED"}, category="session")
        archived = brain.auto_archive(days=30)
        assert archived == 1
        full = brain.read_full("event.old")
        assert full["category"] == "archive"

    def test_no_archive_recent(self, brain):
        recent = datetime.now().strftime("%Y-%m-%d")
        brain.write("event.recent", {"date": recent, "status": "ACTIVE"}, category="session")
        archived = brain.auto_archive(days=30)
        assert archived == 0


class TestChatHistory:
    def test_save_and_read(self, brain):
        brain.save_chat("chat1", "user", "hello")
        brain.save_chat("chat1", "assistant", "hi there")
        history = brain.chat_history("chat1")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"


class TestMetadata:
    def test_categories(self, brain):
        brain.write("a.1", "x", category="alpha")
        brain.write("b.1", "y", category="beta")
        cats = brain.categories()
        assert "alpha" in cats
        assert "beta" in cats

    def test_keys(self, brain):
        brain.write("t.a", "1")
        brain.write("t.b", "2")
        all_keys = brain.keys()
        assert "t.a" in all_keys
        assert "t.b" in all_keys

    def test_stats(self, brain):
        brain.write("s.1", "data")
        s = brain.stats()
        assert s["entries"] >= 1


class TestHybridSearch:
    def test_hybrid_with_keyword(self, brain):
        brain.write("topic.ai", {"desc": "artificial intelligence"}, description="AI topic", priority=8)
        results = brain.hybrid_search("artificial intelligence", k=3)
        assert len(results) >= 1
        assert results[0]["key"] == "topic.ai"
        assert "_freshness" in results[0]

    def test_hybrid_anti_repetition(self, brain):
        brain.write("rep.test", {"x": 1}, description="Repetition test", priority=9)
        r1 = brain.hybrid_search("repetition test", k=1, chat_id="sess_rep")
        score1 = r1[0]["score"] if r1 else 0
        r2 = brain.hybrid_search("repetition test", k=1, chat_id="sess_rep")
        score2 = r2[0]["score"] if r2 else 0
        assert score2 < score1  # second call should be penalized

    def test_hybrid_time_window(self, brain):
        brain.write("old.entry", "old data", category="general")
        # The entry was just written so its updated_at is today
        results = brain.hybrid_search("old data", k=3, since="2030-01-01")
        # Should be filtered out since updated_at < 2030
        assert len(results) == 0
