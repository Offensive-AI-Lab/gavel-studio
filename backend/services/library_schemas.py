"""Pydantic models for the dataset JSON files shipped by the gavel-rules
GitHub registry.

The catalog side of the registry (index.json — rules, CEs, rule sets,
categories) is consumed as plain dicts by services/library_sync.py; only the
LAZILY fetched per-record dataset files are validated through these models
before they touch the local DB. Malformed payloads (schema drift upstream,
partial downloads) surface here as a ValidationError instead of corrupting
local state.

Upstream paths (all JSON, name-addressed):
    ces/<name>/excitation.json     -> ExcitationRecord
    ces/<name>/calibration.json    -> CECalibrationRecord
    rules/<name>/tests/<type>.json -> RuleDatasetRecord
                                      (type: positive | negative |
                                       positive_calibration)

Forward-compatibility policy: every model uses `extra="ignore"`. Future
fields land in newer records, get silently dropped by older clients, and the
older client keeps working. New required fields would force a schema_version
bump and a coordinated client release.
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ExcitationRecord(BaseModel):
    """The training data attached to a library CE
    (ces/<name>/excitation.json). samples is preserved as the raw
    conversation list — the guardrail engine consumes this shape directly."""

    model_config = ConfigDict(extra="ignore")

    schema_version: int
    ce_public_id: str
    samples: List[Any] = []
    sample_count: int = 0
    published_at: Optional[str] = None


class CECalibrationRecord(BaseModel):
    """CE-level calibration conversations attached to a library CE
    (ces/<name>/calibration.json). Same shape as ExcitationRecord but used
    by the calibration pipeline (Youden-J threshold sweep) rather than
    training. The library ships these so a fresh client can run `evaluate`
    end-to-end without generating calibration data from scratch."""

    model_config = ConfigDict(extra="ignore")

    schema_version: int
    ce_public_id: str
    samples: List[Any] = []
    sample_count: int = 0
    published_at: Optional[str] = None


class RuleDatasetRecord(BaseModel):
    """A rule's DEFAULT test/calibration dialogues
    (rules/<name>/tests/<type>.json). Three exist per library rule —
    positive, negative, and positive_calibration.

    The upstream files carry the bucket under the key `type`; locally the
    field is `dataset_type` (matching the test_datasets column), so `type`
    is accepted as an alias and either key validates.

    Carries DIALOGUES + the generation config only (v2 configs are
    {scenario_instructions}). Never thresholds or metrics: those are
    computed per-guardrail against the adopter's own trained model, so they
    can't be shared."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    schema_version: int
    rule_public_id: str
    # 'positive' | 'negative' | 'positive_calibration'
    dataset_type: str = Field(alias="type")
    config: Dict[str, Any] = {}
    conversations: List[Any] = []
    conversation_count: int = 0
    published_at: Optional[str] = None
