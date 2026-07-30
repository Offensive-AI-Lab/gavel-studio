"""Pure unit tests for the guardrail (classifier) clone + create helpers added
for the model-last flow, in `sql_scripts.model_scripts`.

NO database is available, so the DB seams (`execute_query` / `execute_query_dict`,
imported INTO `model_scripts` at load time) are monkeypatched on the
`model_scripts` namespace — same convention as test_model_scripts_logic.py.

Covers:
  * create_classifier        — owner + nullable model_id INSERT shape
  * clone_classifier_policy  — name dedupe, per-setup link remap, untrained copy
"""
import pytest

from sql_scripts import model_scripts


class _ScriptedDictDB:
    """execute_query_dict fake. `router(sql, params)` returns the rows for each
    call so a test can branch on the SQL text; records every (sql, params)."""

    def __init__(self, router):
        self._router = router
        self.calls = []

    def __call__(self, sql, params=None):
        self.calls.append((sql, params))
        return self._router(sql, params)


class _Writer:
    def __init__(self):
        self.calls = []

    def __call__(self, sql, params=None):
        self.calls.append((sql, params))
        return None


# ===========================================================================
# create_classifier — owner + optional model_id
# ===========================================================================


class TestCreateClassifier:
    def test_inserts_user_and_nullable_model(self, monkeypatch):
        captured = {}

        def router(sql, params):
            captured["sql"] = sql
            captured["params"] = params
            return [{"classifier_id": 7, "model_id": None, "name": "G", "status": "untrained"}]

        monkeypatch.setattr(model_scripts, "execute_query_dict", _ScriptedDictDB(router))
        out = model_scripts.create_classifier(42, "G")  # model_id defaults to None

        assert out["classifier_id"] == 7
        assert out["model_id"] is None
        # user_id, model_id (NULL), name — in that order.
        assert captured["params"] == (42, None, "G")
        assert "user_id" in captured["sql"] and "INSERT INTO classifiers" in captured["sql"]

    def test_attached_create_passes_model_id(self, monkeypatch):
        captured = {}

        def router(sql, params):
            captured["params"] = params
            return [{"classifier_id": 8, "model_id": 3, "name": "G2", "status": "untrained"}]

        monkeypatch.setattr(model_scripts, "execute_query_dict", _ScriptedDictDB(router))
        out = model_scripts.create_classifier(42, "G2", 3)
        assert out["model_id"] == 3
        assert captured["params"] == (42, 3, "G2")


# ===========================================================================
# clone_classifier_policy — deep-copy of the policy layer
# ===========================================================================


class TestCloneClassifierPolicy:
    def _setup(self, monkeypatch, *, existing_names=(), source_name="Finance Guard",
               setups=None):
        """Wire a scripted DB. `existing_names` are the names already present in
        the target model (drives dedupe); `setups` are the source rule_setup rows."""
        if setups is None:
            setups = [
                {"setup_id": 11, "rule_id": 100, "custom_name": "r1", "predicate": "CE_1",
                 "ce_groups": {"required": ["CE_1"]}, "condition": "all of required",
                 "is_active": True},
                {"setup_id": 12, "rule_id": None, "custom_name": "manual", "predicate": "CE_2 AND CE_3",
                 "ce_groups": {"required": ["CE_2", "CE_3"]}, "condition": "all of required",
                 "is_active": True},
            ]
        state = {"next_setup_id": 500, "inserted_classifier": None}

        def router(sql, params):
            s = " ".join(sql.split())
            if s.startswith("SELECT name FROM classifiers WHERE classifier_id"):
                return [{"name": source_name}]
            if s.startswith("SELECT 1 FROM classifiers WHERE model_id = %s AND LOWER(name)"):
                # params: (target_model_id, candidate_name)
                candidate = params[1]
                return [{"1": 1}] if candidate in existing_names else []
            if s.startswith("INSERT INTO classifiers"):
                state["inserted_classifier"] = params  # (user_id, model_id, name)
                return [{"classifier_id": 999, "model_id": params[1], "name": params[2], "status": "untrained"}]
            if s.startswith("SELECT setup_id, rule_id, custom_name, predicate, ce_groups, condition, is_active FROM rule_setup WHERE classifier_id"):
                return list(setups)
            if s.startswith("INSERT INTO rule_setup"):
                state["next_setup_id"] += 1
                return [{"setup_id": state["next_setup_id"]}]
            return []

        dict_db = _ScriptedDictDB(router)
        writer = _Writer()
        monkeypatch.setattr(model_scripts, "execute_query_dict", dict_db)
        monkeypatch.setattr(model_scripts, "execute_query", writer)
        return dict_db, writer, state

    def test_copies_setups_and_links_to_new_untrained_classifier(self, monkeypatch):
        dict_db, writer, state = self._setup(monkeypatch)
        out = model_scripts.clone_classifier_policy(11, target_model_id=3, user_id=42)

        # New classifier created under the target model, owned by the user.
        assert state["inserted_classifier"][0] == 42        # user_id
        assert state["inserted_classifier"][1] == 3         # target_model_id
        assert out["classifier_id"] == 999
        assert out["status"] == "untrained"

        # One link-copy write per source setup, each remapping to the freshly
        # minted setup_id and pulling rows from the OLD setup_id.
        link_writes = [c for c in writer.calls if "INSERT INTO setup_ce_link" in c[0]]
        assert len(link_writes) == 2
        # First copied setup -> new setup_id 501 from old 11; second -> 502 from 12.
        assert link_writes[0][1] == (501, 11)
        assert link_writes[1][1] == (502, 12)

    def test_name_deduped_against_target_model(self, monkeypatch):
        # The source name already exists on the target model -> append "(copy)".
        dict_db, writer, state = self._setup(
            monkeypatch, existing_names={"Finance Guard"}, source_name="Finance Guard")
        model_scripts.clone_classifier_policy(11, target_model_id=3, user_id=42)
        inserted_name = state["inserted_classifier"][2]
        assert inserted_name == "Finance Guard (copy)"

    def test_name_dedupe_increments_when_copy_also_taken(self, monkeypatch):
        dict_db, writer, state = self._setup(
            monkeypatch,
            existing_names={"Finance Guard", "Finance Guard (copy)"},
            source_name="Finance Guard")
        model_scripts.clone_classifier_policy(11, target_model_id=3, user_id=42)
        assert state["inserted_classifier"][2] == "Finance Guard (copy 2)"

    def test_explicit_name_overrides_source(self, monkeypatch):
        dict_db, writer, state = self._setup(monkeypatch)
        model_scripts.clone_classifier_policy(11, target_model_id=3, user_id=42, name="Custom")
        assert state["inserted_classifier"][2] == "Custom"

    def test_missing_source_raises(self, monkeypatch):
        def router(sql, params):
            return []  # source lookup returns nothing

        monkeypatch.setattr(model_scripts, "execute_query_dict", _ScriptedDictDB(router))
        monkeypatch.setattr(model_scripts, "execute_query", _Writer())
        with pytest.raises(ValueError):
            model_scripts.clone_classifier_policy(404, target_model_id=3, user_id=42)

    def test_no_setups_still_creates_empty_copy(self, monkeypatch):
        dict_db, writer, state = self._setup(monkeypatch, setups=[])
        out = model_scripts.clone_classifier_policy(11, target_model_id=3, user_id=42)
        assert out["classifier_id"] == 999
        assert [c for c in writer.calls if "INSERT INTO setup_ce_link" in c[0]] == []
