"""Unit tests for guardrail-folder membership (manual grouping, DB-free).

Folders are purely manual now (no auto-arrange). We patch the thin DB helpers
and assert the membership writes: ownership checks, assign into a folder, and
ungroup.
"""
import pytest

import services.guardrail_folders as gf


def test_assign_sets_folder_membership(monkeypatch):
    calls = []
    monkeypatch.setattr(gf, "_owns_guardrail", lambda u, c: True)
    monkeypatch.setattr(gf, "_owns_folder", lambda u, f: True)
    monkeypatch.setattr(gf, "execute_query", lambda q, p=None: calls.append((q, p)))

    gf.assign(1, 10, 5)

    assert any("UPDATE classifiers SET folder_id" in q and p == (5, 10, 1) for q, p in calls)


def test_assign_ungroup_sets_null(monkeypatch):
    calls = []
    monkeypatch.setattr(gf, "_owns_guardrail", lambda u, c: True)
    monkeypatch.setattr(gf, "execute_query", lambda q, p=None: calls.append((q, p)))

    gf.assign(1, 10, None)

    assert any("UPDATE classifiers SET folder_id" in q and p == (None, 10, 1) for q, p in calls)


def test_assign_rejects_foreign_guardrail(monkeypatch):
    monkeypatch.setattr(gf, "_owns_guardrail", lambda u, c: False)
    monkeypatch.setattr(gf, "execute_query", lambda q, p=None: (_ for _ in ()).throw(AssertionError("must not write")))
    with pytest.raises(PermissionError):
        gf.assign(1, 99, 5)


def test_assign_rejects_foreign_folder(monkeypatch):
    monkeypatch.setattr(gf, "_owns_guardrail", lambda u, c: True)
    monkeypatch.setattr(gf, "_owns_folder", lambda u, f: False)
    monkeypatch.setattr(gf, "execute_query", lambda q, p=None: (_ for _ in ()).throw(AssertionError("must not write")))
    with pytest.raises(PermissionError):
        gf.assign(1, 10, 999)


def test_create_folder_defaults_blank_name(monkeypatch):
    captured = {}
    def fake_dict(q, p=None):
        captured["params"] = p
        return [{"folder_id": 1, "name": p[1], "created_at": None}]
    monkeypatch.setattr(gf, "execute_query_dict", fake_dict)
    row = gf.create_folder(7, "   ")
    assert captured["params"] == (7, "New folder")
    assert row["name"] == "New folder"


def test_rename_folder_rejects_empty(monkeypatch):
    monkeypatch.setattr(gf, "execute_query_dict", lambda q, p=None: (_ for _ in ()).throw(AssertionError("must not write")))
    with pytest.raises(ValueError):
        gf.rename_folder(1, 5, "  ")
