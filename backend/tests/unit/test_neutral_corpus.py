"""Unit tests for the neutral evaluation corpus.

The corpus is DB-only (no bundled seed fallback): the loader reads the
`neutral_corpus` table and returns EMPTY groups when it's empty/unavailable —
the evaluation layer turns that into a hard "cannot evaluate". Tests cover the
conversation identity hash and the DB loader. DB access is monkeypatched; no
real DB or network.
"""
import evaluation.neutral_corpus as nc


# --- content_hash / canonicalization ----------------------------------------

class TestContentHash:
    def test_is_stable_and_hex(self):
        conv = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
        assert nc.content_hash(conv) == nc.content_hash(conv)
        h = nc.content_hash(conv)
        assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)

    def test_invariant_to_whitespace_and_role_case(self):
        a = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
        b = [{"role": "USER", "content": "  hi  "}, {"role": "Assistant", "content": "yo"}]
        assert nc.content_hash(a) == nc.content_hash(b)

    def test_distinguishes_different_content(self):
        a = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
        b = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "yo"}]
        assert nc.content_hash(a) != nc.content_hash(b)


# --- DB loader (monkeypatched) — DB-only, no seed fallback -------------------

class TestDbLoader:
    def test_groups_db_rows_by_category(self, monkeypatch):
        rows = [
            {"category": "instructive", "conversation": [{"role": "user", "content": "how to x"}, {"role": "assistant", "content": "a"}]},
            {"category": "conversational", "conversation": [{"role": "user", "content": "hey"}, {"role": "assistant", "content": "b"}]},
            {"category": "weird", "conversation": [{"role": "user", "content": "z"}, {"role": "assistant", "content": "c"}]},
        ]
        import utils.sqlite_db as pg
        monkeypatch.setattr(pg, "execute_query_dict", lambda *a, **k: rows)
        grouped = nc.load_neutral_corpus_by_category()
        assert len(grouped["instructive"]) == 1
        # unknown 'weird' category folds into conversational
        assert len(grouped["conversational"]) == 2

    def test_empty_db_returns_empty_groups_no_fallback(self, monkeypatch):
        import utils.sqlite_db as pg
        monkeypatch.setattr(pg, "execute_query_dict", lambda *a, **k: [])
        grouped = nc.load_neutral_corpus_by_category()
        assert grouped == {"conversational": [], "instructive": []}
        assert nc.load_neutral_corpus() == []

    def test_db_error_returns_empty_groups(self, monkeypatch):
        import utils.sqlite_db as pg
        def boom(*a, **k):
            raise RuntimeError("no db")
        monkeypatch.setattr(pg, "execute_query_dict", boom)
        grouped = nc.load_neutral_corpus_by_category()
        assert grouped == {"conversational": [], "instructive": []}

    def test_only_two_categories(self):
        assert set(nc.CATEGORIES) == {"conversational", "instructive"}
