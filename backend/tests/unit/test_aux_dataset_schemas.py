"""Unit tests for the lazily-fetched dataset record schemas
(services.library_schemas).

The catalog side of the gavel-rules registry (index.json) is consumed as
plain dicts by services/library_sync.py; only the per-record dataset files
are validated through Pydantic before they touch the local DB. These tests
lock the contract for those files:

  * ces/<name>/excitation.json      -> ExcitationRecord
  * ces/<name>/calibration.json     -> CECalibrationRecord
  * rules/<name>/tests/<type>.json  -> RuleDatasetRecord (upstream carries
                                       the bucket under `type`; locally the
                                       field is `dataset_type` — both keys
                                       must validate)

The v1 Manifest/CERecord/RuleRecord catalog models are gone with the HF
registry — index.json needs no Pydantic layer.
"""
import pytest
from pydantic import ValidationError

from services.library_schemas import (
    CECalibrationRecord,
    ExcitationRecord,
    RuleDatasetRecord,
)


# ---------------------------------------------------------------------------
# ExcitationRecord
# ---------------------------------------------------------------------------


class TestExcitationRecord:
    def test_happy_path_round_trips_samples(self):
        convo = [{"role": "user", "content": "hi"}]
        rec = ExcitationRecord.model_validate({
            "schema_version": 2,
            "ce_public_id": "ce_abc",
            "samples": [convo, convo],
            "sample_count": 2,
            "published_at": "2025-01-01T00:00:00Z",
        })
        assert rec.ce_public_id == "ce_abc"
        assert rec.samples == [convo, convo]
        assert rec.sample_count == 2

    def test_samples_default_to_empty(self):
        rec = ExcitationRecord.model_validate({
            "schema_version": 2,
            "ce_public_id": "ce_x",
        })
        assert rec.samples == []
        assert rec.sample_count == 0

    def test_upstream_extra_fields_ignored(self):
        # Real gavel-rules files carry extras (ce name, type, ...) — a v2
        # record with new fields must not break this client.
        rec = ExcitationRecord.model_validate({
            "schema_version": 2,
            "ce_public_id": "ce_x",
            "ce": "some_ce_name",
            "type": "excitation",
            "future_field": {"some": "thing"},
        })
        assert not hasattr(rec, "future_field")
        assert not hasattr(rec, "ce")


# ---------------------------------------------------------------------------
# CECalibrationRecord
# ---------------------------------------------------------------------------


class TestCECalibrationRecord:
    def test_happy_path(self):
        sample_convo = [{"role": "user", "content": "hi"}]
        rec = CECalibrationRecord.model_validate({
            "schema_version": 2,
            "ce_public_id": "ce_abc",
            "samples": [sample_convo, sample_convo],
            "sample_count": 2,
            "published_at": "2025-01-01T00:00:00Z",
        })
        assert rec.ce_public_id == "ce_abc"
        assert len(rec.samples) == 2
        assert rec.sample_count == 2

    def test_samples_default_to_empty_list(self):
        # Defensive: a malformed record without samples still validates,
        # just empty. The downstream upsert will write an empty dataset
        # rather than raising.
        rec = CECalibrationRecord.model_validate({
            "schema_version": 2,
            "ce_public_id": "ce_x",
        })
        assert rec.samples == []
        assert rec.sample_count == 0

    def test_missing_public_id_rejected(self):
        with pytest.raises(ValidationError):
            CECalibrationRecord.model_validate({"schema_version": 2})


# ---------------------------------------------------------------------------
# RuleDatasetRecord — the `type` alias is load-bearing
# ---------------------------------------------------------------------------


class TestRuleDatasetRecord:
    def _payload(self, **over):
        base = {
            "schema_version": 2,
            "rule_public_id": "rule_abc",
            "type": "positive",
            "config": {"scenario_instructions": "do the thing"},
            "conversations": [[{"role": "user", "content": "hi"}]],
            "conversation_count": 1,
            "published_at": "2025-01-01T00:00:00Z",
        }
        base.update(over)
        return base

    def test_upstream_type_key_populates_dataset_type(self):
        rec = RuleDatasetRecord.model_validate(self._payload())
        assert rec.dataset_type == "positive"
        assert rec.rule_public_id == "rule_abc"
        assert rec.conversation_count == 1
        assert rec.config == {"scenario_instructions": "do the thing"}

    def test_local_dataset_type_key_also_validates(self):
        # populate_by_name=True: the local column name works too.
        payload = self._payload()
        payload.pop("type")
        payload["dataset_type"] = "negative"
        rec = RuleDatasetRecord.model_validate(payload)
        assert rec.dataset_type == "negative"

    def test_missing_bucket_key_rejected(self):
        payload = self._payload()
        payload.pop("type")
        with pytest.raises(ValidationError):
            RuleDatasetRecord.model_validate(payload)

    def test_conversations_round_trip_unchanged(self):
        convos = [[{"role": "user", "content": "a"}], [{"role": "assistant", "content": "b"}]]
        rec = RuleDatasetRecord.model_validate(self._payload(conversations=convos))
        assert rec.conversations == convos

    def test_extra_fields_ignored(self):
        rec = RuleDatasetRecord.model_validate(self._payload(rule="some_rule", extra=1))
        assert not hasattr(rec, "rule")
        assert not hasattr(rec, "extra")

    def test_config_and_conversations_default_empty(self):
        rec = RuleDatasetRecord.model_validate({
            "schema_version": 2,
            "rule_public_id": "r",
            "type": "positive_calibration",
        })
        assert rec.config == {}
        assert rec.conversations == []
        assert rec.conversation_count == 0


# ---------------------------------------------------------------------------
# Deleted v1 catalog models must stay deleted
# ---------------------------------------------------------------------------


def test_v1_catalog_models_are_gone():
    import services.library_schemas as schemas

    for name in ("Manifest", "CERecord", "RuleRecord", "RuleSetRecord",
                 "CategoriesFile", "NeutralCorpusFile"):
        assert not hasattr(schemas, name), f"{name} should have died with the HF registry"
