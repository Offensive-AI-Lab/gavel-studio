"""Tests for rule management: CRUD, CE linking (groups+condition), bookmarks."""
import pytest


class TestPublicRules:
    """Public rule library."""

    def test_get_public_rules(self, client, auth_headers):
        res = client.get("/rules/public/library", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        rules = data.get("rules", data) if isinstance(data, dict) else data
        assert isinstance(rules, list)

    def test_create_public_rule_with_groups_condition(self, client, auth_headers, test_user):
        # Create two CEs first (the rule references them BY NAME).
        ce_names = []
        for i in range(2):
            name = f"rule_test_ce_{i}_{id(test_user)}"
            ce_res = client.post("/cognitive/create", json={
                "user_id": test_user["user_id"],
                "name": name,
                "definition": f"Test CE {i}",
            }, headers=auth_headers)
            if ce_res.status_code == 200:
                ce_names.append(name)

        if len(ce_names) < 2:
            pytest.skip("Could not create enough CEs")

        res = client.post("/rules/public/create", json={
            "name": f"test_rule_{id(test_user)}",
            "groups": {"required": ce_names},
            "condition": "all of required",
            "user_id": test_user["user_id"],
            "description": "created by the v2 shape test",
            "categories": [],
        }, headers=auth_headers)
        assert res.status_code in (200, 201)

    def test_create_public_rule_legacy_ce_names_maps_to_required_group(
            self, client, auth_headers, test_user):
        # The legacy flat ce_names payload maps to a single 'required' group
        # with 'all of required'.
        name = f"legacy_shape_ce_{id(test_user)}"
        ce_res = client.post("/cognitive/create", json={
            "user_id": test_user["user_id"],
            "name": name,
            "definition": "legacy",
        }, headers=auth_headers)
        if ce_res.status_code != 200:
            pytest.skip("Could not create CE")

        res = client.post("/rules/public/create", json={
            "name": f"legacy_rule_{id(test_user)}",
            "ce_names": [name],
            "user_id": test_user["user_id"],
            "description": "legacy payload",
            "categories": [],
        }, headers=auth_headers)
        assert res.status_code in (200, 201)

    def test_create_public_rule_invalid_condition_rejected(self, client, auth_headers, test_user):
        name = f"badcond_ce_{id(test_user)}"
        ce_res = client.post("/cognitive/create", json={
            "user_id": test_user["user_id"],
            "name": name,
            "definition": "x",
        }, headers=auth_headers)
        if ce_res.status_code != 200:
            pytest.skip("Could not create CE")
        res = client.post("/rules/public/create", json={
            "name": f"badcond_rule_{id(test_user)}",
            "groups": {"required": [name]},
            "condition": "all of some_undefined_group",
            "user_id": test_user["user_id"],
            "categories": [],
        }, headers=auth_headers)
        assert res.status_code in (400, 409, 422)


class TestRuleSetup:
    """Rule setup operations (classifier-specific)."""

    def _make_setup(self, client, cid, auth_headers):
        """Seed a manual (empty-logic) rule setup so the link tests have a
        real target instead of skipping on an empty classifier."""
        res = client.post(f"/classifiers/{cid}/rules/manual", json={
            "name": f"linkce_target_{id(client)}_{id(auth_headers)}",
        }, headers=auth_headers)
        if res.status_code != 200:
            pytest.skip(f"Could not create manual rule setup: {res.status_code}")
        return res.json()["setup_id"]

    def test_link_ce_to_setup(self, client, test_classifier, auth_headers, test_user):
        cid = test_classifier["classifier_id"]
        setup_id = self._make_setup(client, cid, auth_headers)

        # Create a CE to link
        ce_res = client.post("/cognitive/create", json={
            "user_id": test_user["user_id"],
            "name": f"link_ce_{id(test_user)}",
            "definition": "linkable",
        }, headers=auth_headers)
        if ce_res.status_code != 200:
            pytest.skip("Could not create CE")

        ce_id = ce_res.json()["ce_id"]
        # v2 contract: the CE joins a logic GROUP (default 'additional'), not
        # a role.
        res = client.post(f"/rules/setup/{setup_id}/link-ce", json={
            "ce_id": ce_id,
            "group": "additional",
        }, headers=auth_headers)
        assert res.status_code == 200, res.text

        # The CE landed in the setup's membership AND its logic group.
        detail = client.get(f"/classifiers/{cid}/rules", headers=auth_headers).json()["rules"]
        seeded = next(r for r in detail if r["setup_id"] == setup_id)
        assert any(ce["ce_id"] == ce_id for ce in seeded["active_ces"])
        groups = seeded["logic"]["groups"]
        assert any(
            any(m["ce_id"] == ce_id for m in members)
            for members in groups.values()
        )

    def test_link_ce_rejects_invalid_group_name(self, client, test_classifier,
                                                auth_headers, test_user):
        cid = test_classifier["classifier_id"]
        setup_id = self._make_setup(client, cid, auth_headers)

        ce_res = client.post("/cognitive/create", json={
            "user_id": test_user["user_id"],
            "name": f"badgroup_ce_{id(test_user)}",
            "definition": "x",
        }, headers=auth_headers)
        if ce_res.status_code != 200:
            pytest.skip("Could not create CE")
        res = client.post(f"/rules/setup/{setup_id}/link-ce", json={
            "ce_id": ce_res.json()["ce_id"],
            "group": "Not A Valid Group!",   # must match [a-z][a-z0-9_]*
        }, headers=auth_headers)
        assert res.status_code in (400, 422)


# Rule-bookmark CRUD is exercised via /rules/public/bookmark* in
# test_models_user_rules_edges.py.
