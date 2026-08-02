"""End-to-end export of CEs, rules and rule sets as gavel-rules artifacts.

Everything here builds its own dummy artifacts (a CE with excitation and
calibration data, a rule with all three test buckets, a rule set holding that
rule) so the tests do not depend on whatever the synced library happens to
contain. Per-test cleanup in conftest tracks every table touched below.
"""
import json

import pytest
import yaml

from utils.sqlite_db import execute_query_dict, execute_update


def _convo(text):
    return [{"role": "user", "content": text},
            {"role": "assistant", "content": f"re: {text}"}]


@pytest.fixture
def dummy_ce(client, auth_headers, test_user):
    """A CE with both of its datasets present, marked as already in the library
    (is_local_draft = FALSE) so rule exports over it are warning-free by
    default; the draft-CE warning has its own test that flips the flag back."""
    name = f"export_ce_{id(test_user)}"
    res = client.post("/cognitive/create", json={
        "user_id": test_user["user_id"],
        "name": name,
        "definition": "Detects an export-test concept.",
    }, headers=auth_headers)
    if res.status_code not in (200, 201):
        pytest.skip(f"could not create CE: {res.status_code}")
    ce_id = res.json().get("ce_id")

    execute_update(
        "UPDATE cognitive_elements SET role = %s, title = %s, examples = %s, "
        "is_local_draft = FALSE WHERE ce_id = %s",
        ("llm_behavior", "Export CE",
         json.dumps([{"input": "yes case", "output": "YES"},
                     {"input": "no case", "output": "NO"}]), ce_id),
    )
    execute_update(
        "INSERT INTO excitation_datasets (ce_id, dataset) VALUES (%s, %s)",
        (ce_id, json.dumps({"samples": [_convo("excite me")]})),
    )
    # Local calibration stores `conversations`; the registry keys it on `samples`.
    execute_update(
        "INSERT INTO calibration_datasets (ce_id, dataset) VALUES (%s, %s)",
        (ce_id, json.dumps({"conversations": [_convo("calibrate me")]})),
    )
    return {"ce_id": ce_id, "name": name}


@pytest.fixture
def dummy_rule(client, auth_headers, test_user, dummy_ce):
    """A rule over the dummy CE, with all three test buckets populated."""
    name = f"export_rule_{id(test_user)}"
    res = client.post("/rules/public/create", json={
        "name": name,
        "groups": {"trigger": [dummy_ce["name"]]},
        "condition": "all of trigger",
        "user_id": test_user["user_id"],
        "description": "Fires when the export-test concept appears.",
        "categories": [],
    }, headers=auth_headers)
    if res.status_code not in (200, 201):
        pytest.skip(f"could not create rule: {res.status_code}")
    rule_id = res.json().get("rule_id")

    # 9 = Security & Defense, 4 = Other (deliberately not a registry category).
    execute_update("UPDATE rules SET categories = %s WHERE rule_id = %s",
                   (json.dumps([9]), rule_id))
    for bucket in ("positive", "negative", "positive_calibration"):
        execute_update(
            "INSERT INTO test_datasets (rule_id, dataset_type, conversations, config, "
            "status, is_default) VALUES (%s, %s, %s, %s, 'ready', TRUE)",
            (rule_id, bucket, json.dumps([_convo(f"{bucket} case")]),
             json.dumps({"scenario_instructions": "an export-test scenario"})),
        )
    return {"rule_id": rule_id, "name": name}


@pytest.fixture
def dummy_ruleset(client, auth_headers, test_classifier, dummy_rule):
    """The test classifier with the dummy rule attached — the UI's 'rule set'."""
    cid = test_classifier["classifier_id"]
    res = client.post(f"/classifiers/{cid}/rules/add",
                      json={"rule_id": dummy_rule["rule_id"]}, headers=auth_headers)
    if res.status_code not in (200, 201):
        pytest.skip(f"could not attach rule: {res.status_code}")
    return {"classifier_id": cid, "rule_name": dummy_rule["name"]}


class TestCeExport:
    def test_preflight_lists_all_three_files(self, client, auth_headers, dummy_ce):
        res = client.get(f"/export/ce/{dummy_ce['ce_id']}/preflight",
                         params={"author": "octocat"}, headers=auth_headers)
        assert res.status_code == 200
        body = res.json()
        assert body["kind"] == "ce"
        assert body["directory"] == f"ces/{dummy_ce['name']}"
        assert {f["filename"] for f in body["files"]} == {
            "ce.yaml", "excitation.json", "calibration.json"}
        assert body["errors"] == [] and body["warnings"] == []
        assert yaml.safe_load(body["preview"])["name"] == dummy_ce["name"]

    def test_yaml_download_is_valid_and_attached(self, client, auth_headers, dummy_ce):
        res = client.get(f"/export/ce/{dummy_ce['ce_id']}/ce.yaml",
                         params={"author": "octocat"}, headers=auth_headers)
        assert res.status_code == 200
        assert "attachment" in res.headers["content-disposition"]
        assert 'filename="ce.yaml"' in res.headers["content-disposition"]
        doc = yaml.safe_load(res.text)
        assert doc["name"] == dummy_ce["name"]
        assert doc["schema_version"] == 2
        assert doc["role"] == "llm_behavior"
        assert doc["provenance"]["created_by"] == "octocat"
        assert doc["examples"][0]["output"] in ("YES", "NO")

    def test_excitation_json_uses_the_registry_envelope(self, client, auth_headers, dummy_ce):
        res = client.get(f"/export/ce/{dummy_ce['ce_id']}/excitation.json",
                         params={"author": "octocat"}, headers=auth_headers)
        assert res.status_code == 200
        doc = res.json()
        assert doc["type"] == "excitation"
        assert doc["ce"] == dummy_ce["name"]
        assert doc["sample_count"] == len(doc["samples"]) == 1

    def test_calibration_conversations_are_remapped_to_samples(
            self, client, auth_headers, dummy_ce):
        res = client.get(f"/export/ce/{dummy_ce['ce_id']}/calibration.json",
                         params={"author": "octocat"}, headers=auth_headers)
        assert res.status_code == 200
        doc = res.json()
        assert doc["type"] == "calibration"
        assert "conversations" not in doc
        assert doc["sample_count"] == 1

    def test_public_id_is_shared_between_yaml_and_datasets(
            self, client, auth_headers, dummy_ce):
        """validate.py rejects a dataset whose ce_public_id != the artifact's."""
        y = yaml.safe_load(client.get(f"/export/ce/{dummy_ce['ce_id']}/ce.yaml",
                                      params={"author": "octocat"},
                                      headers=auth_headers).text)
        d = client.get(f"/export/ce/{dummy_ce['ce_id']}/excitation.json",
                       params={"author": "octocat"}, headers=auth_headers).json()
        assert d["ce_public_id"] == y["public_id"]

    def test_missing_role_is_reported_not_guessed(self, client, auth_headers, dummy_ce):
        execute_update("UPDATE cognitive_elements SET role = '' WHERE ce_id = %s",
                       (dummy_ce["ce_id"],))
        body = client.get(f"/export/ce/{dummy_ce['ce_id']}/preflight",
                          params={"author": "octocat"},
                          headers=auth_headers).json()
        assert any("role" in e for e in body["errors"])
        assert body["roles"] == ["directive_to_user", "llm_task", "llm_behavior", "topic"]

    def test_blocked_download_states_what_is_wrong(self, client, auth_headers, dummy_ce):
        execute_update("UPDATE cognitive_elements SET role = '' WHERE ce_id = %s",
                       (dummy_ce["ce_id"],))
        res = client.get(f"/export/ce/{dummy_ce['ce_id']}/ce.yaml",
                         params={"author": "octocat"}, headers=auth_headers)
        assert res.status_code == 422
        assert "role" in res.json()["detail"]

    def test_role_override_unblocks_the_download(self, client, auth_headers, dummy_ce):
        execute_update("UPDATE cognitive_elements SET role = '' WHERE ce_id = %s",
                       (dummy_ce["ce_id"],))
        res = client.get(f"/export/ce/{dummy_ce['ce_id']}/excitation.json",
                         params={"author": "octocat", "role": "topic"},
                         headers=auth_headers)
        assert res.status_code == 200

    def test_role_override_is_applied(self, client, auth_headers, dummy_ce):
        res = client.get(f"/export/ce/{dummy_ce['ce_id']}/ce.yaml",
                         params={"author": "octocat", "role": "topic"},
                         headers=auth_headers)
        assert yaml.safe_load(res.text)["role"] == "topic"

    def test_missing_dataset_blocks_in_preflight(self, client, auth_headers, dummy_ce):
        execute_update("DELETE FROM excitation_datasets WHERE ce_id = %s",
                       (dummy_ce["ce_id"],))
        body = client.get(f"/export/ce/{dummy_ce['ce_id']}/preflight",
                          params={"author": "octocat"}, headers=auth_headers).json()
        assert any("excitation" in e for e in body["errors"])
        assert "excitation.json" not in {f["filename"] for f in body["files"]}

    def test_unknown_ce_is_404(self, client, auth_headers):
        assert client.get("/export/ce/99999999/preflight",
                          headers=auth_headers).status_code == 404

    def test_unknown_dataset_kind_is_404(self, client, auth_headers, dummy_ce):
        assert client.get(f"/export/ce/{dummy_ce['ce_id']}/banana.json",
                          headers=auth_headers).status_code == 404


class TestRuleExport:
    def test_preflight_lists_yaml_plus_three_test_files(
            self, client, auth_headers, dummy_rule):
        body = client.get(f"/export/rule/{dummy_rule['rule_id']}/preflight",
                          params={"author": "octocat"}, headers=auth_headers).json()
        assert body["kind"] == "rule"
        assert body["directory"] == f"rules/{dummy_rule['name']}"
        assert {f["filename"] for f in body["files"]} == {
            "rule.yaml", "positive.json", "negative.json", "positive_calibration.json"}
        assert body["errors"] == [] and body["warnings"] == []

    def test_yaml_carries_groups_condition_and_named_categories(
            self, client, auth_headers, dummy_rule):
        res = client.get(f"/export/rule/{dummy_rule['rule_id']}/rule.yaml",
                         params={"author": "octocat"}, headers=auth_headers)
        assert res.status_code == 200
        doc = yaml.safe_load(res.text)
        assert doc["schema_version"] == 2
        assert doc["condition"] == "all of trigger"
        assert list(doc["groups"]) == ["trigger"]
        # Stored locally as integer ids; the registry wants names.
        assert doc["categories"] == ["Security & Defense"]

    def test_non_registry_category_blocks(self, client, auth_headers, dummy_rule):
        execute_update("UPDATE rules SET categories = %s WHERE rule_id = %s",
                       (json.dumps([4]), dummy_rule["rule_id"]))  # 4 = 'Other'
        body = client.get(f"/export/rule/{dummy_rule['rule_id']}/preflight",
                          params={"author": "octocat"}, headers=auth_headers).json()
        assert any("Other" in e for e in body["errors"])

    def test_test_bucket_uses_the_registry_envelope(self, client, auth_headers, dummy_rule):
        res = client.get(f"/export/rule/{dummy_rule['rule_id']}/tests/positive.json",
                         params={"author": "octocat"}, headers=auth_headers)
        assert res.status_code == 200
        doc = res.json()
        assert doc["type"] == "positive"
        assert doc["rule"] == dummy_rule["name"]
        assert doc["conversation_count"] == len(doc["conversations"]) == 1
        assert set(doc["config"]) == {"scenario_instructions"}

    def test_every_bucket_downloads(self, client, auth_headers, dummy_rule):
        for bucket in ("positive", "negative", "positive_calibration"):
            res = client.get(f"/export/rule/{dummy_rule['rule_id']}/tests/{bucket}.json",
                             params={"author": "octocat"}, headers=auth_headers)
            assert res.status_code == 200, bucket
            assert res.json()["type"] == bucket

    def test_public_id_is_shared_between_yaml_and_tests(
            self, client, auth_headers, dummy_rule):
        y = yaml.safe_load(client.get(f"/export/rule/{dummy_rule['rule_id']}/rule.yaml",
                                      params={"author": "octocat"},
                                      headers=auth_headers).text)
        t = client.get(f"/export/rule/{dummy_rule['rule_id']}/tests/positive.json",
                       params={"author": "octocat"}, headers=auth_headers).json()
        assert t["rule_public_id"] == y["public_id"]

    def test_draft_ce_member_warns(self, client, auth_headers, dummy_ce, dummy_rule):
        execute_update("UPDATE cognitive_elements SET is_local_draft = TRUE "
                       "WHERE ce_id = %s", (dummy_ce["ce_id"],))
        body = client.get(f"/export/rule/{dummy_rule['rule_id']}/preflight",
                          params={"author": "octocat"}, headers=auth_headers).json()
        assert any(dummy_ce["name"] in w for w in body["warnings"])

    def test_deleted_ce_member_warns(self, client, auth_headers, dummy_ce, dummy_rule):
        # ce_groups keeps the name even after the CE row is gone.
        execute_update("DELETE FROM cognitive_elements WHERE ce_id = %s",
                       (dummy_ce["ce_id"],))
        body = client.get(f"/export/rule/{dummy_rule['rule_id']}/preflight",
                          params={"author": "octocat"}, headers=auth_headers).json()
        assert any(dummy_ce["name"] in w for w in body["warnings"])

    def test_missing_test_set_blocks(self, client, auth_headers, dummy_rule):
        execute_update("DELETE FROM test_datasets WHERE rule_id = %s AND dataset_type = %s",
                       (dummy_rule["rule_id"], "negative"))
        body = client.get(f"/export/rule/{dummy_rule['rule_id']}/preflight",
                          params={"author": "octocat"}, headers=auth_headers).json()
        assert any("negative" in e for e in body["errors"])

    def test_blocked_download_states_what_is_wrong(self, client, auth_headers, dummy_rule):
        # One missing test set blocks every file of the rule, not just that one.
        execute_update("DELETE FROM test_datasets WHERE rule_id = %s AND dataset_type = %s",
                       (dummy_rule["rule_id"], "negative"))
        res = client.get(f"/export/rule/{dummy_rule['rule_id']}/rule.yaml",
                         params={"author": "octocat"}, headers=auth_headers)
        assert res.status_code == 422
        assert "negative" in res.json()["detail"]

    def test_unknown_bucket_is_404(self, client, auth_headers, dummy_rule):
        assert client.get(f"/export/rule/{dummy_rule['rule_id']}/tests/sideways.json",
                          headers=auth_headers).status_code == 404

    def test_unknown_rule_is_404(self, client, auth_headers):
        assert client.get("/export/rule/99999999/preflight",
                          headers=auth_headers).status_code == 404


class TestRuleSetExport:
    def test_preflight_lists_members_and_one_file(
            self, client, auth_headers, dummy_ruleset):
        body = client.get(f"/export/ruleset/{dummy_ruleset['classifier_id']}/preflight",
                          params={"author": "octocat", "description": "A test bundle."},
                          headers=auth_headers).json()
        assert body["kind"] == "ruleset"
        assert body["directory"] == "rulesets"
        assert len(body["files"]) == 1
        assert dummy_ruleset["rule_name"] in body["members"]

    def test_yaml_lists_member_names_only(self, client, auth_headers, dummy_ruleset):
        res = client.get(f"/export/ruleset/{dummy_ruleset['classifier_id']}/ruleset.yaml",
                         params={"author": "octocat", "description": "A test bundle."},
                         headers=auth_headers)
        assert res.status_code == 200
        doc = yaml.safe_load(res.text)
        assert doc["schema_version"] == 1
        assert doc["description"] == "A test bundle."
        assert dummy_ruleset["rule_name"] in doc["rules"]
        # A rule set carries no logic — only names.
        assert "groups" not in doc and "condition" not in doc

    def test_local_only_member_is_flagged_by_name(
            self, client, auth_headers, dummy_ruleset, dummy_rule):
        execute_update("UPDATE rules SET is_local_draft = TRUE WHERE rule_id = %s",
                       (dummy_rule["rule_id"],))
        body = client.get(f"/export/ruleset/{dummy_ruleset['classifier_id']}/preflight",
                          params={"author": "octocat", "description": "d"},
                          headers=auth_headers).json()
        assert dummy_rule["name"] in body["local_only"]
        assert any(dummy_rule["name"] in w for w in body["warnings"])

    def test_empty_description_blocks_the_download(
            self, client, auth_headers, dummy_ruleset):
        res = client.get(f"/export/ruleset/{dummy_ruleset['classifier_id']}/ruleset.yaml",
                         params={"author": "octocat"}, headers=auth_headers)
        assert res.status_code == 422
        assert "description" in res.json()["detail"]

    def test_local_only_member_does_not_block_the_download(
            self, client, auth_headers, dummy_ruleset, dummy_rule):
        execute_update("UPDATE rules SET is_local_draft = TRUE WHERE rule_id = %s",
                       (dummy_rule["rule_id"],))
        res = client.get(f"/export/ruleset/{dummy_ruleset['classifier_id']}/ruleset.yaml",
                         params={"author": "octocat", "description": "d"},
                         headers=auth_headers)
        assert res.status_code == 200

    def test_empty_rule_set_is_rejected(self, client, auth_headers, test_classifier):
        cid = test_classifier["classifier_id"]
        execute_update("DELETE FROM rule_setup WHERE classifier_id = %s", (cid,))
        res = client.get(f"/export/ruleset/{cid}/preflight",
                         params={"author": "octocat"}, headers=auth_headers)
        assert res.status_code == 400

    def test_name_is_slugged_for_the_registry(self, client, auth_headers, dummy_ruleset):
        cid = dummy_ruleset["classifier_id"]
        execute_update("UPDATE classifiers SET name = %s WHERE classifier_id = %s",
                       ("My Scam Detector!", cid))
        body = client.get(f"/export/ruleset/{cid}/preflight",
                          params={"author": "octocat", "description": "d"},
                          headers=auth_headers).json()
        assert body["name"] == "my_scam_detector"
        assert body["files"][0]["path"] == "rulesets/my_scam_detector.yaml"

    def test_unknown_rule_set_is_404_or_403(self, client, auth_headers):
        assert client.get("/export/ruleset/99999999/preflight",
                          headers=auth_headers).status_code in (403, 404)


class TestExportSettings:
    @pytest.fixture(autouse=True)
    def _restore_setting(self):
        """_app_meta is not tracked by per-test cleanup, so restore it here."""
        rows = execute_query_dict(
            "SELECT value FROM _app_meta WHERE key = 'export.github_username'")
        before = rows[0]["value"] if rows else None
        yield
        if before is None:
            execute_update("DELETE FROM _app_meta WHERE key = 'export.github_username'")
        else:
            execute_update(
                "INSERT INTO _app_meta (key, value) VALUES (%s, %s) "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                ("export.github_username", before))

    def test_username_round_trips(self, client, auth_headers):
        assert client.put("/export/settings", json={"github_username": "octocat"},
                          headers=auth_headers).status_code == 200
        assert client.get("/export/settings",
                          headers=auth_headers).json()["github_username"] == "octocat"

    def test_remembered_username_is_used_when_none_is_passed(
            self, client, auth_headers, dummy_ce):
        client.put("/export/settings", json={"github_username": "remembered"},
                   headers=auth_headers)
        res = client.get(f"/export/ce/{dummy_ce['ce_id']}/ce.yaml", headers=auth_headers)
        assert yaml.safe_load(res.text)["provenance"]["created_by"] == "remembered"

    def test_missing_username_blocks_rather_than_emitting_a_placeholder(
            self, client, auth_headers, dummy_ce):
        client.put("/export/settings", json={"github_username": ""}, headers=auth_headers)
        body = client.get(f"/export/ce/{dummy_ce['ce_id']}/preflight",
                          headers=auth_headers).json()
        assert any("GitHub username" in e for e in body["errors"])
        assert client.get(f"/export/ce/{dummy_ce['ce_id']}/ce.yaml",
                          headers=auth_headers).status_code == 422
