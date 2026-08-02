"""Unit cover for routes.evaluation._jsonb — the NaN/Infinity-safe serializer.

Regression for: evaluation crashing with
  psycopg2.errors.InvalidTextRepresentation: invalid input syntax for type json
  DETAIL: Token "NaN" is invalid
when a use-case's ROC_AUC (etc.) comes back NaN and json.dumps writes the bare
`NaN` literal, which Postgres jsonb rejects.
"""
import json
import math

from routes.evaluation import _jsonb


def test_nan_and_inf_become_null():
    out = _jsonb({
        "metrics": [
            {"Usecase": "tax_scam", "ROC_AUC": float("nan"), "F1": 0.96},
            {"Usecase": "phish", "ROC_AUC": float("inf"), "PR_AUC": float("-inf")},
        ],
        "ok": 1.0,
    })
    # No bare NaN/Infinity tokens (those are what Postgres rejects).
    assert "NaN" not in out
    assert "Infinity" not in out
    # Valid JSON that round-trips, with the non-finite floats turned to null.
    parsed = json.loads(out)
    assert parsed["metrics"][0]["ROC_AUC"] is None
    assert parsed["metrics"][0]["F1"] == 0.96
    assert parsed["metrics"][1]["ROC_AUC"] is None
    assert parsed["metrics"][1]["PR_AUC"] is None
    assert parsed["ok"] == 1.0


def test_finite_values_and_nesting_preserved():
    payload = {"a": [1, 2.5, {"b": 0.0, "c": [True, "x", None]}]}
    assert json.loads(_jsonb(payload)) == payload


def test_plain_values_pass_through():
    assert _jsonb({"error": "boom"}) == json.dumps({"error": "boom"})
    assert json.loads(_jsonb([1, 2, 3])) == [1, 2, 3]
