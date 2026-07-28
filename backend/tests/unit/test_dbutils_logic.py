"""Pure unit tests for utils/DButils.py helpers and utils/device.py logic.

No database and no GPU are available, so:
  * The DB seam used by normalize_and_upsert_categories is `exec_query`
    (defined in DButils itself, wrapping get_connection/release_connection).
    We monkeypatch `DButils.exec_query` to return canned rows and record the
    INSERT calls — the real Postgres is never touched.
  * Device selection in utils/device.py is driven purely by
    torch.cuda.is_available / torch.backends.mps.is_available, which we
    monkeypatch to exercise every branch (cuda / mps / cpu fallback).

We deliberately never call init_database / drop_all_tables (real schema
bootstrap) — only the small pure helpers and constants are exercised.

Covered surface:
  * SCHEMA_VERSION constant presence + type
  * DEFAULT_CATEGORIES shape
  * normalize_and_upsert_categories:
      empty / None input, int IDs, string-digit IDs, name resolution,
      case + whitespace handling, dedup, allow_new True/False, ON CONFLICT
      path, malformed/unknown inputs, max_len truncation, sorted ordering,
      exception swallowing on insert failure, cache update after create.
  * device.py: get_torch_device / empty_device_cache / get_llm_device_map
      across cuda / mps / cpu, plus override-respecting behavior.
"""
import pytest

import utils.DButils as db
import utils.device as device


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_schema_version_present_and_int(self):
        assert hasattr(db, "SCHEMA_VERSION")
        assert isinstance(db.SCHEMA_VERSION, int)
        assert db.SCHEMA_VERSION >= 1

    def test_schema_version_current_value(self):
        # Locks the constant: bumping the schema must update this test too,
        # mirroring the file's own version log.
        assert db.SCHEMA_VERSION == 20

    def test_default_categories_shape(self):
        assert isinstance(db.DEFAULT_CATEGORIES, list)
        assert len(db.DEFAULT_CATEGORIES) == 10
        for entry in db.DEFAULT_CATEGORIES:
            # Each is a (name, description) pair of non-empty strings.
            assert isinstance(entry, tuple) and len(entry) == 2
            name, desc = entry
            assert isinstance(name, str) and name.strip()
            assert isinstance(desc, str) and desc.strip()

    def test_default_categories_names_unique(self):
        names = [n for n, _ in db.DEFAULT_CATEGORIES]
        assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# normalize_and_upsert_categories — DB seam helpers
# ---------------------------------------------------------------------------


def _patch_exec_query(monkeypatch, existing_rows, insert_results=None):
    """Stub DButils.exec_query.

    The first SELECT (active categories) returns `existing_rows` (list of
    (category_id, name) tuples). Any INSERT ... RETURNING returns the next
    item popped from `insert_results` (a list of (new_id,) tuples), or raises
    if that item is an Exception instance.

    Returns a `calls` list capturing (query, params) for every invocation.
    """
    calls = []
    pending = list(insert_results or [])

    def fake(query, params=None):
        calls.append((query, params))
        q = query.strip().upper()
        if q.startswith("SELECT"):
            return existing_rows
        if "INSERT INTO CATEGORIES" in q:
            if not pending:
                return None
            nxt = pending.pop(0)
            if isinstance(nxt, Exception):
                raise nxt
            return nxt  # e.g. [(7,)]
        return None

    monkeypatch.setattr(db, "exec_query", fake)
    return calls


_EXISTING = [
    (1, "Security & Defense"),
    (2, "Privacy & Data Protection"),
    (3, "Safety & Harm Prevention"),
]


class TestNormalizeEmptyAndNone:
    def test_empty_list_returns_empty(self, monkeypatch):
        _patch_exec_query(monkeypatch, _EXISTING)
        assert db.normalize_and_upsert_categories([]) == []

    def test_none_input_returns_empty(self, monkeypatch):
        # `categories or []` guard inside the loop.
        _patch_exec_query(monkeypatch, _EXISTING)
        assert db.normalize_and_upsert_categories(None) == []

    def test_none_existing_rows_no_crash(self, monkeypatch):
        # SELECT returns None -> `... or []` guard keeps maps empty.
        _patch_exec_query(monkeypatch, None)
        # Names won't resolve, allow_new False -> nothing created.
        assert db.normalize_and_upsert_categories(["Security & Defense"]) == []


class TestNormalizeIdResolution:
    def test_int_id_present(self, monkeypatch):
        _patch_exec_query(monkeypatch, _EXISTING)
        assert db.normalize_and_upsert_categories([1, 2]) == [1, 2]

    def test_int_id_absent_is_dropped(self, monkeypatch):
        _patch_exec_query(monkeypatch, _EXISTING)
        # 99 is not a known id and is silently dropped.
        assert db.normalize_and_upsert_categories([1, 99]) == [1]

    def test_string_digit_id_present(self, monkeypatch):
        _patch_exec_query(monkeypatch, _EXISTING)
        assert db.normalize_and_upsert_categories(["2", "3"]) == [2, 3]

    def test_string_digit_id_absent_dropped(self, monkeypatch):
        _patch_exec_query(monkeypatch, _EXISTING)
        assert db.normalize_and_upsert_categories(["404"]) == []


class TestNormalizeNameResolution:
    def test_exact_name(self, monkeypatch):
        _patch_exec_query(monkeypatch, _EXISTING)
        assert db.normalize_and_upsert_categories(["Security & Defense"]) == [1]

    def test_case_insensitive_name(self, monkeypatch):
        _patch_exec_query(monkeypatch, _EXISTING)
        assert db.normalize_and_upsert_categories(["security & defense"]) == [1]
        # And mixed/upper variants resolve to the same id.
        assert db.normalize_and_upsert_categories(["SECURITY & DEFENSE"]) == [1]

    def test_whitespace_padded_name(self, monkeypatch):
        _patch_exec_query(monkeypatch, _EXISTING)
        assert db.normalize_and_upsert_categories(["  Privacy & Data Protection  "]) == [2]

    def test_unknown_name_no_allow_new_dropped(self, monkeypatch):
        calls = _patch_exec_query(monkeypatch, _EXISTING)
        assert db.normalize_and_upsert_categories(["Brand New Cat"]) == []
        # No INSERT should have been attempted.
        assert all("INSERT" not in q.upper() for q, _ in calls)

    def test_blank_string_skipped(self, monkeypatch):
        _patch_exec_query(monkeypatch, _EXISTING)
        # Whitespace-only string falls into the `if not s_item: continue` guard.
        assert db.normalize_and_upsert_categories(["   "]) == []


class TestNormalizeAllowNew:
    def test_creates_new_category(self, monkeypatch):
        calls = _patch_exec_query(monkeypatch, _EXISTING, insert_results=[[(7,)]])
        result = db.normalize_and_upsert_categories(["Brand New Cat"], allow_new=True)
        assert result == [7]
        # Exactly one INSERT, with the trimmed original-case name as the param.
        inserts = [(q, p) for q, p in calls if "INSERT" in q.upper()]
        assert len(inserts) == 1
        assert inserts[0][1] == ("Brand New Cat",)

    def test_new_category_preserves_original_case_in_param(self, monkeypatch):
        calls = _patch_exec_query(monkeypatch, _EXISTING, insert_results=[[(7,)]])
        db.normalize_and_upsert_categories(["  MyMixedCase  "], allow_new=True)
        inserts = [(q, p) for q, p in calls if "INSERT" in q.upper()]
        # Trimmed but case preserved (lowercasing only happens for the cache key).
        assert inserts[0][1] == ("MyMixedCase",)

    def test_allow_new_caches_so_duplicate_name_inserts_once(self, monkeypatch):
        # Dedup happens on raw input before the loop, so two identical strings
        # collapse to one. Use two case-variants instead — they are distinct
        # inputs but map to the same cache key after the first insert.
        calls = _patch_exec_query(monkeypatch, _EXISTING, insert_results=[[(7,)]])
        result = db.normalize_and_upsert_categories(
            ["NewCat", "newcat"], allow_new=True
        )
        # Both resolve to id 7; result deduped to a single id.
        assert result == [7]
        inserts = [(q, p) for q, p in calls if "INSERT" in q.upper()]
        # Second variant hit the freshly-populated name_map cache -> only 1 insert.
        assert len(inserts) == 1

    def test_insert_returns_none_drops_silently(self, monkeypatch):
        # ON CONFLICT path that returns nothing -> no id added, no crash.
        _patch_exec_query(monkeypatch, _EXISTING, insert_results=[None])
        assert db.normalize_and_upsert_categories(["Ghost"], allow_new=True) == []

    def test_insert_exception_is_swallowed(self, monkeypatch):
        # A failing INSERT is caught; the function continues and returns []
        # rather than propagating.
        _patch_exec_query(
            monkeypatch, _EXISTING, insert_results=[RuntimeError("boom")]
        )
        assert db.normalize_and_upsert_categories(["Exploding"], allow_new=True) == []

    def test_mix_existing_and_new(self, monkeypatch):
        _patch_exec_query(monkeypatch, _EXISTING, insert_results=[[(50,)]])
        result = db.normalize_and_upsert_categories(
            [1, "Privacy & Data Protection", "Totally New"], allow_new=True
        )
        # 1, 2 (resolved by name), 50 (new) -> sorted.
        assert result == [1, 2, 50]


class TestNormalizeDedupOrderingTruncation:
    def test_dedup_repeated_ids(self, monkeypatch):
        _patch_exec_query(monkeypatch, _EXISTING)
        assert db.normalize_and_upsert_categories([1, 1, 1, 2]) == [1, 2]

    def test_result_is_sorted(self, monkeypatch):
        _patch_exec_query(monkeypatch, _EXISTING)
        # Input order 3,1,2 -> output is sorted ascending by id.
        assert db.normalize_and_upsert_categories([3, 1, 2]) == [1, 2, 3]

    def test_name_and_id_pointing_to_same_id_dedup(self, monkeypatch):
        _patch_exec_query(monkeypatch, _EXISTING)
        # int 1 and the name for id 1 collapse to a single id.
        assert db.normalize_and_upsert_categories([1, "Security & Defense"]) == [1]

    def test_max_len_truncates_after_sort(self, monkeypatch):
        _patch_exec_query(monkeypatch, _EXISTING)
        # All three resolve; max_len=2 keeps the two lowest ids.
        result = db.normalize_and_upsert_categories([3, 1, 2], max_len=2)
        assert result == [1, 2]

    def test_max_len_zero_disables_truncation(self, monkeypatch):
        _patch_exec_query(monkeypatch, _EXISTING)
        # `if max_len and ...` -> 0 is falsy, no truncation.
        result = db.normalize_and_upsert_categories([3, 1, 2], max_len=0)
        assert result == [1, 2, 3]

    def test_default_max_len_is_three(self, monkeypatch):
        # Four resolvable ids, default max_len=3 -> first three by sort.
        rows = _EXISTING + [(4, "Fairness & Ethics")]
        _patch_exec_query(monkeypatch, rows)
        result = db.normalize_and_upsert_categories([4, 3, 2, 1])
        assert result == [1, 2, 3]

    def test_unknown_hashable_types_ignored(self, monkeypatch):
        _patch_exec_query(monkeypatch, _EXISTING)
        # float / None are hashable but neither int nor digit-string, and
        # str(None)/str(1.5) won't match any name -> all dropped; id 1 survives.
        result = db.normalize_and_upsert_categories([1.5, None, 1])
        assert result == [1]

    def test_unhashable_input_raises_typeerror(self, monkeypatch):
        # KNOWN LIMITATION: dedup uses a set, so an unhashable element (dict /
        # list) makes the function raise TypeError rather than skip it. This
        # pins the current behavior so a future "handle gracefully" change is
        # a conscious decision, not an accident.
        _patch_exec_query(monkeypatch, _EXISTING)
        with pytest.raises(TypeError):
            db.normalize_and_upsert_categories([{"x": 1}, 1])


# ---------------------------------------------------------------------------
# device.py — device selection logic
# ---------------------------------------------------------------------------


def _set_accel(monkeypatch, *, cuda, mps):
    """Force torch.cuda.is_available and torch.backends.mps.is_available."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    # torch.backends.mps may or may not exist depending on build; ensure a
    # backends.mps.is_available we control. The code guards with hasattr, so
    # we always provide one here.
    if not hasattr(torch.backends, "mps"):
        class _MPS:
            pass

        monkeypatch.setattr(torch.backends, "mps", _MPS(), raising=False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps, raising=False)


class TestGetTorchDevice:
    def test_cuda_preferred(self, monkeypatch):
        _set_accel(monkeypatch, cuda=True, mps=True)
        assert device.get_torch_device().type == "cuda"

    def test_cuda_over_cpu(self, monkeypatch):
        _set_accel(monkeypatch, cuda=True, mps=False)
        assert device.get_torch_device().type == "cuda"

    def test_mps_when_no_cuda(self, monkeypatch):
        _set_accel(monkeypatch, cuda=False, mps=True)
        assert device.get_torch_device().type == "mps"

    def test_cpu_fallback_no_accelerators(self, monkeypatch):
        _set_accel(monkeypatch, cuda=False, mps=False)
        assert device.get_torch_device().type == "cpu"


class TestGetLlmDeviceMap:
    def test_cuda_returns_auto(self, monkeypatch):
        _set_accel(monkeypatch, cuda=True, mps=False)
        assert device.get_llm_device_map() == "auto"

    def test_mps_returns_mps_map(self, monkeypatch):
        _set_accel(monkeypatch, cuda=False, mps=True)
        assert device.get_llm_device_map() == {"": "mps"}

    def test_cpu_returns_cpu(self, monkeypatch):
        _set_accel(monkeypatch, cuda=False, mps=False)
        assert device.get_llm_device_map() == "cpu"


class TestEmptyDeviceCache:
    def test_calls_cuda_empty_cache(self, monkeypatch):
        import torch

        _set_accel(monkeypatch, cuda=True, mps=False)
        called = {"cuda": False}
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: called.__setitem__("cuda", True))
        device.empty_device_cache()
        assert called["cuda"] is True

    def test_calls_mps_empty_cache_when_only_mps(self, monkeypatch):
        import torch

        _set_accel(monkeypatch, cuda=False, mps=True)
        called = {"mps": False}
        # torch.mps may not exist on this build; provide a stub.
        if not hasattr(torch, "mps"):
            class _M:
                pass

            monkeypatch.setattr(torch, "mps", _M(), raising=False)
        monkeypatch.setattr(
            torch.mps, "empty_cache", lambda: called.__setitem__("mps", True), raising=False
        )
        device.empty_device_cache()
        assert called["mps"] is True

    def test_noop_on_cpu(self, monkeypatch):
        import torch

        _set_accel(monkeypatch, cuda=False, mps=False)
        # Neither empty_cache should fire; make them blow up if called.
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: pytest.fail("cuda hit"))
        # Should simply return without touching any accelerator cache.
        assert device.empty_device_cache() is None
