"""Unit tests for the service layer.

These tests target the pure-logic guard branches in HybridSearchService and
BookmarkService — paths that the integration tests don't reliably hit because
they go through the real DB. They use the dependency-injection seam built into
each service: a fake embedder for HybridSearchService, and the static
BOOKMARK_REGISTRY lookup for BookmarkService.
"""
import pytest

from services.library_search import (
    ASSET_REGISTRY,
    AssetSpec,
    HybridSearchService,
)
from services.bookmarks import BOOKMARK_REGISTRY, BookmarkSpec, BookmarkService, _spec


# ---------------------------------------------------------------------------
# HybridSearchService — guard-clause coverage
# ---------------------------------------------------------------------------


def _real_embedder(text: str):
    """Embedder that returns a fixed-length non-empty vector. The values are
    deterministic so tests can assert on them, but they're never sent to a
    real DB in these unit tests because we exercise the early-return paths."""
    return [0.1] * 384


def _empty_embedder(text: str):
    """Models the case where the embedder cache is cold and the warmup hasn't
    finished yet — embed_query returns []."""
    return []


def _exploding_embedder(text: str):
    raise RuntimeError("embedder unavailable")


class TestHybridSearchServiceGuards:
    """Cover the early-return branches that the integration tests skip past."""

    def test_empty_query_returns_empty_list(self):
        service = HybridSearchService(embedder=_real_embedder)
        result = service.search(query_text="", asset_types=["rule"])
        assert result == []

    def test_whitespace_only_query_returns_empty_list(self):
        # query_text.strip() should drop "   " to "" and trigger the empty-query guard.
        service = HybridSearchService(embedder=_real_embedder)
        assert service.search(query_text="   \t\n", asset_types=["rule"]) == []

    def test_unknown_asset_type_returns_empty_list(self):
        # If every requested asset type is missing from the registry, search
        # should bail out before doing any DB work.
        service = HybridSearchService(embedder=_real_embedder)
        assert service.search(query_text="foo", asset_types=["nope"]) == []

    def test_empty_asset_types_returns_empty_list(self):
        service = HybridSearchService(embedder=_real_embedder)
        assert service.search(query_text="foo", asset_types=[]) == []

    def test_empty_embedder_returns_empty_list(self):
        # Models the case where embed_query returns [] (no warmed embedder).
        service = HybridSearchService(embedder=_empty_embedder)
        assert service.search(query_text="foo", asset_types=["rule"]) == []

    def test_unknown_types_filtered_out_before_db_call(self):
        # Mixed valid + invalid: invalid drop, valid would hit the DB. We use
        # an embedder that returns empty so we don't actually need a DB.
        service = HybridSearchService(embedder=_empty_embedder)
        # Should NOT raise on the unknown type; should drop "nope" and return []
        # because the embedder is empty.
        assert service.search(query_text="foo", asset_types=["rule", "nope"]) == []


class TestAssetRegistry:
    """The registry is the open/closed extension point. Make sure the entries
    we ship satisfy the implicit contract that the SQL builder relies on."""

    def test_registry_has_rule_and_ce(self):
        assert "rule" in ASSET_REGISTRY
        assert "ce" in ASSET_REGISTRY

    def test_each_spec_has_required_fields(self):
        for asset_type, spec in ASSET_REGISTRY.items():
            assert isinstance(spec, AssetSpec)
            assert spec.asset_type == asset_type, f"asset_type mismatch for {asset_type}"
            assert spec.table, f"empty table for {asset_type}"
            assert spec.id_col, f"empty id_col for {asset_type}"
            assert spec.content_col, f"empty content_col for {asset_type}"

    def test_specs_are_frozen(self):
        # The dataclass is frozen=True; mutation should raise FrozenInstanceError.
        spec = ASSET_REGISTRY["rule"]
        with pytest.raises(Exception):
            spec.table = "tampered"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# BookmarkService — guard-clause coverage
# ---------------------------------------------------------------------------


class TestBookmarkServiceSpec:
    """Bookmark _spec() is the only place we validate the asset_type. The
    SQL builders downstream all trust whatever it returns, so this is a real
    correctness boundary."""

    def test_known_asset_type_resolves_to_spec(self):
        spec = _spec("rule")
        assert isinstance(spec, BookmarkSpec)
        assert spec.asset_type == "rule"
        # After the central-server migration, bookmarks no longer live in a
        # local table; the spec only carries the LOCAL asset table that the
        # bookmark's public_id is joined against on read.
        assert spec.asset_table == "rules"
        assert spec.asset_id_col == "rule_id"

    def test_unknown_asset_type_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown bookmark asset type"):
            _spec("not_a_real_type")

    def test_unknown_asset_type_includes_value_in_message(self):
        # The error message includes the bad value via repr() so a caller
        # logging the exception sees exactly what was passed in. The chosen
        # bad value contains a quote, which is the kind of typo that's easy
        # to miss in a non-repr'd error message.
        with pytest.raises(ValueError) as excinfo:
            _spec("'rule")
        # repr("'rule") produces "\"'rule\"" — a substring check on "'rule"
        # is sufficient regardless of which quote style Python picks.
        assert "'rule" in str(excinfo.value)

    def test_empty_string_asset_type_raises(self):
        with pytest.raises(ValueError):
            _spec("")


class TestBookmarkRegistry:
    """The registry is the open/closed extension point for bookmarks. Each
    entry must satisfy the implicit contract used by the SQL builders."""

    def test_registry_has_rule_and_ce(self):
        assert "rule" in BOOKMARK_REGISTRY
        assert "ce" in BOOKMARK_REGISTRY

    def test_each_spec_has_required_fields(self):
        for asset_type, spec in BOOKMARK_REGISTRY.items():
            assert isinstance(spec, BookmarkSpec)
            assert spec.asset_type == asset_type
            assert spec.asset_table, f"empty asset_table for {asset_type}"
            assert spec.asset_id_col, f"empty asset_id_col for {asset_type}"
            assert spec.list_columns, f"empty list_columns for {asset_type}"

    def test_asset_id_col_consistent_with_search_registry(self):
        # The bookmark spec's asset_id_col must match the same field in the
        # search registry's AssetSpec, otherwise SQL JOINs would fail when both
        # services are used together (e.g. searching bookmarks).
        for asset_type in BOOKMARK_REGISTRY:
            if asset_type in ASSET_REGISTRY:
                bookmark_id_col = BOOKMARK_REGISTRY[asset_type].asset_id_col
                search_id_col = ASSET_REGISTRY[asset_type].id_col
                assert bookmark_id_col == search_id_col, (
                    f"id_col mismatch for {asset_type}: "
                    f"bookmark={bookmark_id_col}, search={search_id_col}"
                )

    def test_specs_are_frozen(self):
        spec = BOOKMARK_REGISTRY["rule"]
        with pytest.raises(Exception):
            spec.bookmark_table = "tampered"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# HybridSearchService — happy-path mutation coverage
#
# The earlier guard tests only cover the "return early with []" branches. A
# mutation that turns `query_vector = self._embed(...)` into `query_vector =
# None`, or `or []` into `and []`, would still produce empty results in those
# tests. We need a test that actually goes through the full pipeline and
# asserts on real returned rows.
# ---------------------------------------------------------------------------


class _RecordingEmbedder:
    """Embedder that records every call and returns a fixed non-empty vector.
    Used to assert that the service actually invokes the embedder rather than
    accidentally bypassing it."""

    def __init__(self):
        self.calls = []

    def __call__(self, text):
        self.calls.append(text)
        return [0.1] * 384


class TestHybridSearchHappyPath:
    """End-to-end-on-the-service-level: real embedder call, mocked SQL."""

    def test_search_calls_embedder_and_returns_db_rows(self, monkeypatch):
        # Mock the two DB touchpoints of the SQLite implementation: the
        # candidate fetch (per-asset table scan) and the FTS5 keyword query.
        # The candidate carries a real float32-blob embedding so the semantic
        # signal ranks it; its name contains the query so the trigram bonus
        # fires too.
        import numpy as np
        from services import library_search as ls

        embedding_blob = np.asarray([0.1] * 384, dtype=np.float32).tobytes()
        candidate = {
            "id": 1, "name": "hello_rule", "content": "predicate",
            "type": None, "categories": [], "is_local_draft": False,
            "created_by_username": "someone", "public_id": None,
            "embedding": embedding_blob,
        }

        def fake_query(sql, params=None):
            if "MATCH" in sql:
                return [{"id": 1, "score": -1.5}]
            return [dict(candidate)]

        monkeypatch.setattr(ls, "execute_query_dict", fake_query)

        embedder = _RecordingEmbedder()
        service = ls.HybridSearchService(embedder=embedder)

        result = service.search(query_text="hello", asset_types=["rule"])

        # 1) The embedder was actually called. If the line `query_vector =
        # self._embed(...)` is replaced by `query_vector = None` (mutation
        # #60), this assertion fails.
        assert embedder.calls == ["hello"], (
            f"embedder was not called as expected: {embedder.calls}"
        )

        # 2) The candidate flows through to the caller, tagged and scored;
        # the raw embedding blob must NOT leak into the result row.
        assert len(result) == 1
        row = result[0]
        assert row["id"] == 1
        assert row["asset_type"] == "rule"
        assert row["name"] == "hello_rule"
        assert row["content"] == "predicate"
        assert row["final_score"] > 0
        assert "embedding" not in row

    def test_search_returns_empty_list_when_db_returns_none(self, monkeypatch):
        # The `or []` defends against the DB layer returning None on a no-
        # results query. Verify that's what happens.
        from services import library_search as ls

        monkeypatch.setattr(ls, "execute_query_dict", lambda sql, params: None)

        service = ls.HybridSearchService(embedder=_RecordingEmbedder())
        result = service.search(query_text="hello", asset_types=["rule"])
        assert result == []
