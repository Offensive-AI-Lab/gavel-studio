"""Contribution counts come from the LOCAL synced library, not the drift-prone
central counter — so a profile/community count stays accurate even after items
are removed from HF. Covers routes.user._local_published_counts.
"""
import time

from utils.sqlite_db import execute_query
from routes.user import _local_published_counts


def _uniq() -> str:
    return f"countuser_{int(time.time() * 1000) % 1_000_000_000}"


class TestLocalPublishedCounts:
    def test_counts_published_only_excluding_drafts(self, client):
        uname = _uniq()
        # 2 published rules + 1 draft rule
        for i, draft in enumerate([False, False, True]):
            execute_query(
                "INSERT INTO rules (name, predicate, created_by_username, is_local_draft) "
                "VALUES (%s, 'CE', %s, %s)",
                (f"{uname}_r{i}", uname, draft),
            )
        # 1 published CE + 2 draft CEs
        for i, draft in enumerate([False, True, True]):
            execute_query(
                "INSERT INTO cognitive_elements (name, definition, created_by_username, is_local_draft) "
                "VALUES (%s, 'd', %s, %s)",
                (f"{uname}_ce{i}", uname, draft),
            )

        counts = _local_published_counts([uname])
        assert counts[uname.lower()] == {"rules": 2, "ces": 1}

    def test_matching_is_case_insensitive(self, client):
        uname = _uniq()
        execute_query(
            "INSERT INTO rules (name, predicate, created_by_username, is_local_draft) "
            "VALUES (%s, 'CE', %s, FALSE)",
            (f"{uname}_r", uname.upper()),  # stored uppercase
        )
        counts = _local_published_counts([uname])  # queried lowercase
        assert counts[uname.lower()]["rules"] == 1

    def test_unknown_user_is_zero(self, client):
        counts = _local_published_counts([_uniq()])
        # Present in the map but zeroed.
        assert all(v == {"rules": 0, "ces": 0} for v in counts.values())

    def test_empty_input(self, client):
        assert _local_published_counts([]) == {}
        assert _local_published_counts([None, "", "  "]) == {}
