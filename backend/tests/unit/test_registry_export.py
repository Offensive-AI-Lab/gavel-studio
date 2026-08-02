"""Format contract for gavel-rules contribution files.

The assertions here mirror gavel-rules `tools/validate.py`, because that script
is what decides whether a contributed file is accepted. Anything it rejects
(public_id shape, schema_version, required fields, dataset envelope) is
asserted directly rather than through a golden file, so the test fails loudly
if the emitted shape drifts.
"""
import json
import re

import pytest
import yaml

from services import registry_export as rx

# --- mirrored from gavel-rules tools/validate.py ---------------------------
PUBLIC_ID_RE = {
    "ce": re.compile(r"ce_[0-9a-f]{32}$"),
    "rule": re.compile(r"rule_[0-9a-f]{32}$"),
    "ruleset": re.compile(r"ruleset_[0-9a-f]{32}$"),
}
NAME_RE = re.compile(r"[A-Za-z0-9_]+$")
MESSAGE_ROLES = {"system", "user", "assistant"}


def _ce_row(**over):
    row = {
        "ce_id": 1,
        "name": "trust_seeding",
        "title": "Trust Seeding",
        "definition": "Attempts to build rapport as groundwork for influence.",
        "role": "llm_behavior",
        "examples": json.dumps([
            {"input": "You can trust me, I've done this 15 years.", "output": "YES"},
            {"input": "I understand, that must be frustrating.", "output": "NO"},
        ]),
        "tags": None,
        "public_id": "ce_f4ddb5bca4264f4c8fc0d6233e1bccf9",
        "is_local_draft": False,
    }
    row.update(over)
    return row


def _rule_row(**over):
    row = {
        "rule_id": 1,
        "name": "phishing_content_creation",
        "title": "Phishing Content Creation",
        "description": "Detects the assistant drafting credential-harvesting lures.",
        "categories": json.dumps([9, 8]),
        "tags": None,
        "ce_groups": json.dumps({
            "lure_creation": ["content_creation", "click_or_enter"],
            "info_solicitation": ["personal_information", "provide_or_give"],
        }),
        "condition": "all of lure_creation and 1 of info_solicitation",
        "public_id": "rule_366dae57d7e1422aa8335c3782ead96a",
        "is_local_draft": False,
    }
    row.update(over)
    return row


class TestPublicIds:
    def test_minted_ids_match_the_registry_pattern(self):
        for kind in ("ce", "rule", "ruleset"):
            assert PUBLIC_ID_RE[kind].match(rx.mint_public_id(kind, "some_name"))

    def test_minting_is_stable_across_calls(self):
        # Re-exporting must not fork the artifact's identity.
        assert rx.mint_public_id("ce", "abc") == rx.mint_public_id("ce", "abc")

    def test_different_names_get_different_ids(self):
        assert rx.mint_public_id("ce", "abc") != rx.mint_public_id("ce", "abd")

    def test_kind_is_validated(self):
        with pytest.raises(rx.ExportError):
            rx.mint_public_id("banana", "abc")


class TestCognitiveElement:
    def test_emits_every_field_the_validator_requires(self):
        doc, errors, warnings = rx.build_ce_doc(_ce_row(), author="octocat")
        assert doc["name"] == "trust_seeding"
        assert doc["title"] == "Trust Seeding"
        assert doc["schema_version"] == 2
        assert doc["role"] in rx.CE_ROLES
        assert PUBLIC_ID_RE["ce"].match(doc["public_id"])
        assert doc["definition"]
        assert doc["provenance"]["created_by"] == "octocat"
        assert doc["provenance"]["published_at"].endswith("Z")
        assert errors == [] and warnings == []

    def test_retired_v1_fields_are_never_emitted(self):
        # validate.py rejects `type` and `categories` on a v2 CE.
        doc, _, _ = rx.build_ce_doc(_ce_row(), author="octocat")
        assert "type" not in doc and "categories" not in doc

    def test_examples_keep_only_input_output_pairs(self):
        doc, _, _ = rx.build_ce_doc(_ce_row(examples=json.dumps([
            {"input": "keep me", "output": "YES"},
            {"input": "bad verdict", "output": "MAYBE"},
            {"input": "", "output": "NO"},
            "not a dict",
        ])), author="octocat")
        assert [dict(e) for e in doc["examples"]] == [{"input": "keep me", "output": "YES"}]

    def test_examples_accept_already_parsed_lists(self):
        doc, _, _ = rx.build_ce_doc(
            _ce_row(examples=[{"input": "x", "output": "NO"}]), author="octocat")
        assert [dict(e) for e in doc["examples"]] == [{"input": "x", "output": "NO"}]

    def test_missing_public_id_is_minted(self):
        doc, _, _ = rx.build_ce_doc(_ce_row(public_id=None), author="octocat")
        assert PUBLIC_ID_RE["ce"].match(doc["public_id"])

    def test_title_falls_back_to_a_readable_name(self):
        doc, _, _ = rx.build_ce_doc(_ce_row(title=None), author="octocat")
        assert doc["title"] == "Trust Seeding"

    def test_override_wins_over_the_stored_value(self):
        doc, _, _ = rx.build_ce_doc(_ce_row(), author="octocat",
                                    title="Custom", role="topic")
        assert doc["title"] == "Custom" and doc["role"] == "topic"

    def test_bad_role_blocks_instead_of_silently_emitting(self):
        _, errors, _ = rx.build_ce_doc(_ce_row(role="nonsense"), author="octocat")
        assert any("role" in e for e in errors)

    def test_empty_examples_block(self):
        _, errors, _ = rx.build_ce_doc(_ce_row(examples="[]"), author="octocat")
        assert any("examples" in e for e in errors)

    def test_nameless_ce_is_an_error(self):
        with pytest.raises(rx.ExportError):
            rx.build_ce_doc(_ce_row(name=""), author="octocat")

    def test_tags_are_emitted_only_when_present(self):
        assert "tags" not in rx.build_ce_doc(_ce_row(), author="o")[0]
        doc, _, _ = rx.build_ce_doc(_ce_row(tags=json.dumps(["a.b"])), author="o")
        assert doc["tags"] == ["a.b"]


class TestRule:
    def test_emits_every_field_the_validator_requires(self):
        doc, errors, warnings = rx.build_rule_doc(
            _rule_row(), author="octocat",
            category_names=["Security & Defense", "Safety & Harm Prevention"])
        assert doc["schema_version"] == 2
        assert PUBLIC_ID_RE["rule"].match(doc["public_id"])
        assert doc["categories"] == ["Security & Defense", "Safety & Harm Prevention"]
        assert doc["condition"]
        assert set(doc["groups"]) == {"lure_creation", "info_solicitation"}
        assert doc["provenance"]["created_by"] == "octocat"
        assert errors == [] and warnings == []

    def test_retired_v1_fields_are_never_emitted(self):
        doc, _, _ = rx.build_rule_doc(_rule_row(), author="o",
                                      category_names=["Tone & Style"])
        for retired in ("required", "any_of", "support"):
            assert retired not in doc

    def test_category_outside_the_registry_taxonomy_blocks(self):
        _, errors, _ = rx.build_rule_doc(_rule_row(), author="o",
                                         category_names=["Other"])
        assert any("Other" in e for e in errors)

    def test_no_categories_blocks(self):
        _, errors, _ = rx.build_rule_doc(_rule_row(), author="o", category_names=[])
        assert any("categories" in e for e in errors)

    def test_dead_group_blocks(self):
        # validate.py rejects groups defined but never referenced by the condition.
        _, errors, _ = rx.build_rule_doc(
            _rule_row(ce_groups=json.dumps({"used": ["a"], "orphan": ["b"]}),
                      condition="all of used"),
            author="o", category_names=["Tone & Style"])
        assert any("orphan" in e for e in errors)

    def test_missing_condition_blocks(self):
        _, errors, _ = rx.build_rule_doc(_rule_row(condition=""), author="o",
                                         category_names=["Tone & Style"])
        assert any("condition" in e for e in errors)

    def test_local_only_ce_warns_by_name_without_blocking(self):
        # Advisory only: contributing the rule together with its draft CEs in
        # one pull request is legitimate, so this must never land in errors.
        _, errors, warnings = rx.build_rule_doc(_rule_row(), author="o",
                                                category_names=["Tone & Style"],
                                                local_only_ces=["click_or_enter"])
        assert any("click_or_enter" in w for w in warnings)
        assert errors == []


class TestRuleSet:
    def test_emits_every_field_the_validator_requires(self):
        doc, errors, warnings = rx.build_ruleset_doc(
            name="scam_detector", members=["phishing_content_creation", "tax_scam"],
            author="octocat", title="Scam Detector", description="Scam-adjacent rules.")
        assert doc["schema_version"] == 1
        assert PUBLIC_ID_RE["ruleset"].match(doc["public_id"])
        assert doc["rules"] == ["phishing_content_creation", "tax_scam"]
        assert doc["description"]
        assert errors == [] and warnings == []

    def test_duplicate_members_are_collapsed(self):
        # validate.py rejects duplicate member rules outright.
        doc, _, _ = rx.build_ruleset_doc(name="s", members=["a", "b", "a"],
                                         author="o", description="d")
        assert doc["rules"] == ["a", "b"]

    def test_local_only_members_warn_by_name_without_blocking(self):
        _, errors, warnings = rx.build_ruleset_doc(
            name="s", members=["published", "mine"], author="o",
            description="d", local_only=["mine"])
        assert any("mine" in w for w in warnings)
        assert errors == []

    def test_empty_description_blocks(self):
        _, errors, _ = rx.build_ruleset_doc(name="s", members=["a"], author="o")
        assert any("description" in e for e in errors)

    def test_a_set_with_no_rules_is_an_error(self):
        with pytest.raises(rx.ExportError):
            rx.build_ruleset_doc(name="s", members=[], author="o", description="d")


class TestDatasets:
    def test_ce_dataset_envelope(self):
        samples = [[{"role": "user", "content": "hi"}]]
        doc = rx.build_ce_dataset("excitation", ce_name="trust_seeding",
                                  ce_public_id="ce_" + "a" * 32, samples=samples)
        assert doc["schema_version"] == 1
        assert doc["ce"] == "trust_seeding"
        assert doc["type"] == "excitation"
        assert doc["sample_count"] == len(doc["samples"]) == 1
        for msg in doc["samples"][0]:
            assert set(msg) == {"role", "content"} and msg["role"] in MESSAGE_ROLES

    def test_calibration_uses_the_samples_key_too(self):
        doc = rx.build_ce_dataset("calibration", ce_name="x",
                                  ce_public_id="ce_" + "a" * 32, samples=[])
        assert "samples" in doc and "conversations" not in doc

    def test_unknown_ce_dataset_kind_is_an_error(self):
        with pytest.raises(rx.ExportError):
            rx.build_ce_dataset("nonsense", ce_name="x", ce_public_id="y", samples=[])

    def test_rule_test_envelope(self):
        convos = [[{"role": "user", "content": "hi"}]]
        doc = rx.build_rule_test("positive", rule_name="r",
                                 rule_public_id="rule_" + "a" * 32,
                                 conversations=convos, scenario_instructions="do a thing")
        assert doc["schema_version"] == 1
        assert doc["type"] == "positive"
        assert doc["conversation_count"] == 1
        # validate.py: config must be EXACTLY {scenario_instructions}
        assert set(doc["config"]) == {"scenario_instructions"}

    def test_unknown_test_bucket_is_an_error(self):
        with pytest.raises(rx.ExportError):
            rx.build_rule_test("sideways", rule_name="r", rule_public_id="p",
                               conversations=[], scenario_instructions="x")


class TestYamlEmission:
    def test_output_parses_back_to_the_same_document(self):
        doc, _, _ = rx.build_ce_doc(_ce_row(), author="octocat")
        assert yaml.safe_load(rx.dump_yaml(doc)) == json.loads(json.dumps(doc))

    def test_key_order_matches_the_registry_files(self):
        doc, _, _ = rx.build_ce_doc(_ce_row(), author="octocat")
        text = rx.dump_yaml(doc)
        assert list(yaml.safe_load(text)) == [
            "name", "title", "public_id", "schema_version",
            "role", "definition", "examples", "provenance",
        ]
        assert text.index("name:") < text.index("public_id:") < text.index("provenance:")

    def test_examples_render_inline_like_the_registry(self):
        doc, _, _ = rx.build_ce_doc(_ce_row(), author="octocat")
        assert "- {input:" in rx.dump_yaml(doc)

    def test_group_members_render_inline_like_the_registry(self):
        doc, _, _ = rx.build_rule_doc(_rule_row(), author="o",
                                      category_names=["Tone & Style"])
        assert "lure_creation: [content_creation, click_or_enter]" in rx.dump_yaml(doc)

    def test_non_ascii_survives_unescaped(self):
        doc, _, _ = rx.build_ce_doc(
            _ce_row(definition="uses an em dash — and no escaping"), author="o")
        assert "—" in rx.dump_yaml(doc)


class TestRepoPlacement:
    def test_paths_match_the_registry_layout(self):
        assert rx.repo_path("ce", "trust_seeding") == "ces/trust_seeding/ce.yaml"
        assert rx.repo_path("ce", "x", "excitation.json") == "ces/x/excitation.json"
        assert rx.repo_path("rule", "phishing") == "rules/phishing/rule.yaml"
        assert rx.repo_path("rule", "x", "tests/positive.json") == "rules/x/tests/positive.json"
        assert rx.repo_path("ruleset", "default") == "rulesets/default.yaml"

    def test_unknown_kind_is_an_error(self):
        with pytest.raises(rx.ExportError):
            rx.repo_path("banana", "x")

    @pytest.mark.parametrize("raw,expected", [
        ("Scam Detector", "scam_detector"),
        ("my rule-set!", "my_rule_set"),
        ("  spaced  out  ", "spaced_out"),
        ("!!!", "unnamed"),
        ("", "unnamed"),
    ])
    def test_names_are_slugged_to_the_registry_grammar(self, raw, expected):
        slug = rx.slugify_name(raw)
        assert slug == expected
        assert NAME_RE.match(slug)
