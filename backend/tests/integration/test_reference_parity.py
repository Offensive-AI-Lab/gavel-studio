"""Parity verification for the reference evaluation pipeline under the v2
(groups + condition) rule model.

This test does NOT exercise an LLM. The reference functions
(`gavel.evaluation.calibrate`, `evaluate`, `compute_triggers`,
`build_ruleset_logic`, `detect_usecase`, `update_label_level_stats`) take
already-computed logit matrices as input, so a synthetic dialogue cache with
known properties is enough to verify our adapter:

  1. Passes the right shapes IN (labels dict, unified_ruleset dict with
     {groups, condition}, dialogue_data list with {logits, metadata}).
  2. Reads the right shapes OUT (per-topic optimal thresholds, per-usecase
     metric tables, AUC scores).
  3. Produces the DOCUMENTED numbers: the ±5-logit fixtures separate
     cleanly, so the calibration optimum and the per-usecase TPR/FPR are
     computed by hand below and asserted exactly.

Label-stats semantics under the v2 model (see
calibration.update_label_level_stats):
  * member of an `all of` selector -> fired TP, unfired FN
  * member of a `k of` selector    -> fired TP; unfired TN when the selector
    was satisfied anyway, FN when it wasn't
  * member of a defined-but-unreferenced group ('supporting') -> fired TP,
    unfired TN
  * outside the rule (or only under `not`) -> fired FP, unfired TN
"""
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Fixture: synthetic ruleset + labels + dialogue cache
# ---------------------------------------------------------------------------

@pytest.fixture
def labels():
    # 4 CEs, indexed in label-name order
    return {"ce_alpha": 0, "ce_beta": 1, "ce_gamma": 2, "ce_delta": 3}


@pytest.fixture
def unified_ruleset():
    """Two rules with different logic shapes — exercises `all of` and the
    `all of + 1 of` combination (the migrated necessary+fallback pattern)."""
    return {
        "rule_strict": {
            "enabled": True,
            "groups": {"required": ["ce_alpha", "ce_beta"]},   # both must fire
            "condition": "all of required",
        },
        "rule_anyof": {
            "enabled": True,
            "groups": {"req": ["ce_alpha"], "opt": ["ce_gamma", "ce_delta"]},
            "condition": "all of req and 1 of opt",            # alpha AND (gamma OR delta)
        },
    }


def _make_dialogue(logits_per_window, split, usecase_path, dialogue_id):
    """Build one entry in the {logits, metadata} shape calibrate/evaluate expect."""
    return {
        "logits": np.array(logits_per_window, dtype=np.float32),
        "metadata": {
            "split": split,
            "usecase_path": usecase_path,
            "dialogue_id": dialogue_id,
        },
    }


HIGH, LOW = 5.0, -5.0     # sigmoid ≈ 0.9933 / 0.0067 — clean separation


@pytest.fixture
def calibration_dialogue_data(labels):
    """Usecase-level calibration data.

    IMPORTANT: the reference `run_threshold_sweep` only consumes dialogues
    whose split is "usecase_level" — that's the upstream contract; our
    fixture has to honour it.

    Per-label expectation at any threshold inside (0.007, 0.993):
      * alpha: `all of` member of both rules, fires in all 7 dialogues
               -> TP=7, FN=0, no negative exposure -> TPR 1.0, FPR 0.0
      * beta:  `all of` member of rule_strict (fires in its 3 dialogues,
               TP=3); OUTSIDE rule_anyof and silent there -> TN=4
      * gamma: `1 of opt` member. Fires in 2 anyof dialogues (TP); in the
               2 delta-dialogues the selector is satisfied by delta -> TN;
               silent + irrelevant in the 3 strict dialogues -> TN
      * delta: symmetric to gamma
    So every CE separates perfectly: youden_j == 1.0 at the optimum.
    """
    cache = []
    n = len(labels)

    def make_logits(fire_indices):
        rows = [[LOW] * n, [LOW] * n]
        for idx in fire_indices:
            rows[0][idx] = HIGH
            rows[1][idx] = HIGH
        return rows

    idx_alpha, idx_beta, idx_gamma, idx_delta = 0, 1, 2, 3

    # rule_strict positives: alpha + beta fire together
    for i in range(3):
        cache.append(_make_dialogue(
            make_logits([idx_alpha, idx_beta]),
            split="usecase_level", usecase_path="rule_strict",
            dialogue_id=f"calib_strict_pos_{i}",
        ))

    # rule_anyof positives: alpha + gamma (or alpha + delta) fire
    for i in range(2):
        cache.append(_make_dialogue(
            make_logits([idx_alpha, idx_gamma]),
            split="usecase_level", usecase_path="rule_anyof",
            dialogue_id=f"calib_anyof_pos_gamma_{i}",
        ))
        cache.append(_make_dialogue(
            make_logits([idx_alpha, idx_delta]),
            split="usecase_level", usecase_path="rule_anyof",
            dialogue_id=f"calib_anyof_pos_delta_{i}",
        ))

    return cache


@pytest.fixture
def evaluation_dialogue_data(labels):
    """Hand-crafted eval set with predictable rule outcomes.

    rule_strict (all of {alpha, beta}):
        3 positives firing both        -> TP
        2 positives firing only one CE -> FN        => TPR = 3/5 = 0.6
        2 negatives firing nothing     -> TN        => FPR = 0.0
    rule_anyof (alpha and (gamma or delta)):
        2 positives firing alpha+gamma -> TP
        1 positive firing gamma only   -> FN (req unmet) => TPR = 2/3
        1 negative firing delta only   -> no fire   => FPR = 0.0
    """
    cache = []

    def fire(ce_indices, n_windows=2):
        logits = [[LOW] * len(labels) for _ in range(n_windows)]
        for ce_idx in ce_indices:
            for w in range(n_windows):
                logits[w][ce_idx] = HIGH
        return logits

    idx_alpha, idx_beta, idx_gamma, idx_delta = 0, 1, 2, 3

    for i in range(3):
        cache.append(_make_dialogue(
            fire([idx_alpha, idx_beta]),
            split="positive", usecase_path="rule_strict",
            dialogue_id=f"eval_strict_tp_{i}",
        ))
    cache.append(_make_dialogue(
        fire([idx_alpha]),
        split="positive", usecase_path="rule_strict",
        dialogue_id="eval_strict_fn_alpha_only",
    ))
    cache.append(_make_dialogue(
        fire([idx_beta]),
        split="positive", usecase_path="rule_strict",
        dialogue_id="eval_strict_fn_beta_only",
    ))
    for i in range(2):
        cache.append(_make_dialogue(
            fire([]),
            split="negative", usecase_path="rule_strict",
            dialogue_id=f"eval_strict_tn_{i}",
        ))

    for i in range(2):
        cache.append(_make_dialogue(
            fire([idx_alpha, idx_gamma]),
            split="positive", usecase_path="rule_anyof",
            dialogue_id=f"eval_anyof_tp_{i}",
        ))
    cache.append(_make_dialogue(
        fire([idx_gamma]),
        split="positive", usecase_path="rule_anyof",
        dialogue_id="eval_anyof_fn_no_alpha",
    ))
    cache.append(_make_dialogue(
        fire([idx_delta]),
        split="negative", usecase_path="rule_anyof",
        dialogue_id="eval_anyof_tn_delta_only",
    ))
    return cache


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReferenceImport:
    """The sys.modules aliasing trick must work — every reference symbol
    we depend on has to be importable through both `gavel.*` and
    `classifier_engine.reference.*` paths."""

    def test_calibrate_importable(self):
        import classifier_engine.reference  # noqa: F401
        from gavel.evaluation.calibration import calibrate
        assert callable(calibrate)

    def test_evaluate_importable(self):
        import classifier_engine.reference  # noqa: F401
        from gavel.evaluation.metrics import evaluate
        assert callable(evaluate)

    def test_compute_triggers_importable(self):
        import classifier_engine.reference  # noqa: F401
        from gavel.evaluation.metrics import compute_triggers
        assert callable(compute_triggers)

    def test_ruleset_logic_builders_importable(self):
        # The groups+condition replacements for the retired
        # convert_labels_to_tensors / load_any_of_conditions builders.
        import classifier_engine.reference  # noqa: F401
        from gavel.evaluation.metrics import build_ruleset_logic, detect_usecase
        assert callable(build_ruleset_logic) and callable(detect_usecase)

    def test_outcomerecorder_importable(self):
        import classifier_engine.reference  # noqa: F401
        from gavel.evaluation.debug import OutcomeRecorder
        assert callable(OutcomeRecorder)


class TestComputeTriggers:
    """Sanity-check the core trigger math at the reference level. If this
    breaks, our calibration + evaluation are also broken."""

    def test_single_high_window_triggers_with_patience_1(self):
        import classifier_engine.reference  # noqa: F401
        from gavel.evaluation.metrics import compute_triggers

        logits = torch.tensor([[5.0], [-5.0]])  # window 0 fires, window 1 doesn't
        out = compute_triggers(logits, thresholds=0.5, patience_rate=1)
        assert out.shape == (1,)
        assert bool(out[0].item()) is True

    def test_patience_2_requires_two_windows(self):
        import classifier_engine.reference  # noqa: F401
        from gavel.evaluation.metrics import compute_triggers

        logits = torch.tensor([[5.0], [-5.0]])  # only one window above
        out = compute_triggers(logits, thresholds=0.5, patience_rate=2)
        assert bool(out[0].item()) is False

    def test_per_label_thresholds(self):
        import classifier_engine.reference  # noqa: F401
        from gavel.evaluation.metrics import compute_triggers

        logits = torch.tensor([[3.0, 3.0]])  # sigmoid ≈ 0.953 for both
        thresholds = torch.tensor([0.5, 0.99])
        out = compute_triggers(logits, thresholds=thresholds, patience_rate=1)
        assert bool(out[0].item()) is True   # 0.953 > 0.5
        assert bool(out[1].item()) is False  # 0.953 < 0.99


class TestRulesetLogicSemantics:
    """build_ruleset_logic + detect_usecase — the groups/condition firing
    semantics, checked directly against hand-picked trigger sets."""

    def _logic(self, unified_ruleset, labels):
        import classifier_engine.reference  # noqa: F401
        from gavel.evaluation.metrics import build_ruleset_logic
        return build_ruleset_logic(unified_ruleset, labels)

    def _fires(self, logic, uc, triggered_idxs):
        import classifier_engine.reference  # noqa: F401
        from gavel.evaluation.metrics import detect_usecase
        return detect_usecase(logic, uc, torch.tensor(triggered_idxs, dtype=torch.long))

    def test_all_of_requires_every_member(self, unified_ruleset, labels):
        logic = self._logic(unified_ruleset, labels)
        assert self._fires(logic, "rule_strict", [0, 1]) is True    # alpha+beta
        assert self._fires(logic, "rule_strict", [0]) is False      # alpha only
        assert self._fires(logic, "rule_strict", [1]) is False      # beta only
        assert self._fires(logic, "rule_strict", []) is False

    def test_one_of_group_needs_any_single_member(self, unified_ruleset, labels):
        logic = self._logic(unified_ruleset, labels)
        assert self._fires(logic, "rule_anyof", [0, 2]) is True     # alpha+gamma
        assert self._fires(logic, "rule_anyof", [0, 3]) is True     # alpha+delta
        assert self._fires(logic, "rule_anyof", [0]) is False       # opt unmet
        assert self._fires(logic, "rule_anyof", [2, 3]) is False    # req unmet

    def test_unknown_usecase_never_fires(self, unified_ruleset, labels):
        logic = self._logic(unified_ruleset, labels)
        assert self._fires(logic, "no_such_rule", [0, 1, 2, 3]) is False

    def test_conditionless_rule_never_fires(self, labels):
        logic = self._logic(
            {"supporting_only": {"groups": {"s": ["ce_alpha"]}, "condition": ""}},
            labels,
        )
        assert self._fires(logic, "supporting_only", [0]) is False

    def test_k_of_quantifier(self, labels):
        logic = self._logic(
            {"two_of": {"groups": {"g": ["ce_alpha", "ce_beta", "ce_gamma"]},
                        "condition": "2 of g"}},
            labels,
        )
        assert self._fires(logic, "two_of", [0]) is False
        assert self._fires(logic, "two_of", [0, 2]) is True
        assert self._fires(logic, "two_of", [0, 1, 2]) is True

    def test_not_selector(self, labels):
        logic = self._logic(
            {"guarded": {"groups": {"g": ["ce_alpha"], "blocked": ["ce_beta"]},
                         "condition": "all of g and not all of blocked"}},
            labels,
        )
        assert self._fires(logic, "guarded", [0]) is True
        assert self._fires(logic, "guarded", [0, 1]) is False


class TestLabelLevelStats:
    """update_label_level_stats — the calibration bucketing documented in its
    docstring, exercised bucket by bucket."""

    def _stats_for(self, unified_ruleset, labels, use_case, triggered_idxs):
        import classifier_engine.reference  # noqa: F401
        from gavel.evaluation.calibration import update_label_level_stats
        from gavel.evaluation.metrics import build_ruleset_logic

        logic = build_ruleset_logic(unified_ruleset, labels)
        triggers = torch.zeros(len(labels), dtype=torch.bool)
        for i in triggered_idxs:
            triggers[i] = True
        stats = [
            {"true_positive": 0, "true_negative": 0,
             "false_positive": 0, "false_negative": 0}
            for _ in labels
        ]
        update_label_level_stats(
            triggers=triggers, ruleset_logic=logic,
            use_case=use_case, labels_statistics=stats,
        )
        return stats

    def _bucket(self, stats, i):
        return next(k for k, v in stats[i].items() if v == 1)

    def test_required_member_fired_tp_unfired_fn(self, unified_ruleset, labels):
        stats = self._stats_for(unified_ruleset, labels, "rule_strict", [0])
        assert self._bucket(stats, 0) == "true_positive"    # alpha fired
        assert self._bucket(stats, 1) == "false_negative"   # beta required, silent

    def test_k_of_member_tn_when_selector_satisfied(self, unified_ruleset, labels):
        # rule_anyof, alpha+gamma fired: delta (unfired 1-of member) is TN
        # because the selector was satisfied by gamma.
        stats = self._stats_for(unified_ruleset, labels, "rule_anyof", [0, 2])
        assert self._bucket(stats, 2) == "true_positive"    # gamma fired
        assert self._bucket(stats, 3) == "true_negative"    # delta excused

    def test_k_of_member_fn_when_selector_unsatisfied(self, unified_ruleset, labels):
        # rule_anyof with only alpha fired: gamma AND delta both missed.
        stats = self._stats_for(unified_ruleset, labels, "rule_anyof", [0])
        assert self._bucket(stats, 2) == "false_negative"
        assert self._bucket(stats, 3) == "false_negative"

    def test_label_outside_rule_fired_is_fp(self, unified_ruleset, labels):
        # beta is not part of rule_anyof — firing there is a false positive.
        stats = self._stats_for(unified_ruleset, labels, "rule_anyof", [0, 1, 2])
        assert self._bucket(stats, 1) == "false_positive"

    def test_label_outside_rule_silent_is_tn(self, unified_ruleset, labels):
        stats = self._stats_for(unified_ruleset, labels, "rule_anyof", [0, 2])
        assert self._bucket(stats, 1) == "true_negative"

    def test_supporting_group_fired_tp_unfired_tn(self, labels):
        ruleset = {"with_support": {
            "groups": {"req": ["ce_alpha"], "supporting": ["ce_beta"]},
            "condition": "all of req",     # 'supporting' never referenced
        }}
        fired = self._stats_for(ruleset, labels, "with_support", [0, 1])
        assert self._bucket(fired, 1) == "true_positive"
        silent = self._stats_for(ruleset, labels, "with_support", [0])
        assert self._bucket(silent, 1) == "true_negative"


class TestCalibratePipeline:
    """End-to-end test of the reference calibrate() with our adapter shapes."""

    def test_calibrate_produces_perfect_separation_thresholds(
        self, labels, unified_ruleset, calibration_dialogue_data,
    ):
        import classifier_engine.reference  # noqa: F401
        from gavel.evaluation.calibration import calibrate

        with tempfile.TemporaryDirectory() as tmpdir:
            calibrate(
                output_dir=tmpdir,
                labels=labels,
                unified_ruleset=unified_ruleset,
                dialogue_data=calibration_dialogue_data,
                show_progress=False,
                generate_plots=False,
            )
            thresholds_path = Path(tmpdir) / "thresholds.json"
            assert thresholds_path.is_file(), "calibrate() did not produce thresholds.json"

            with open(thresholds_path, "r") as f:
                thresholds = json.load(f)

        for ce_name in labels.keys():
            assert ce_name in thresholds, f"missing threshold entry for {ce_name}"
            entry = thresholds[ce_name]
            for required_key in ("threshold", "patience", "youden_j",
                                 "tpr_at_optimal", "fpr_at_optimal"):
                assert required_key in entry, f"{ce_name} missing {required_key}"
            assert isinstance(entry["threshold"], (int, float))
            assert isinstance(entry["patience"], int)
            assert 0.0 <= entry["threshold"] <= 1.0
            # The ±5-logit fixtures separate perfectly under the v2 label
            # bucketing (see the calibration fixture docstring) — the sweep
            # must find a clean optimum for every CE.
            assert entry["tpr_at_optimal"] == pytest.approx(1.0), ce_name
            assert entry["fpr_at_optimal"] == pytest.approx(0.0), ce_name
            assert entry["youden_j"] == pytest.approx(1.0), ce_name


class TestEvaluatePipeline:
    """End-to-end test of reference evaluate() — read back CSV artifacts and
    check the hand-computed per-usecase numbers."""

    def test_evaluate_with_calibrated_thresholds(
        self, labels, unified_ruleset,
        calibration_dialogue_data, evaluation_dialogue_data,
    ):
        import csv

        import classifier_engine.reference  # noqa: F401
        from gavel.evaluation.calibration import calibrate
        from gavel.evaluation.metrics import evaluate

        with tempfile.TemporaryDirectory() as tmpdir:
            calibrate(
                output_dir=tmpdir,
                labels=labels,
                unified_ruleset=unified_ruleset,
                dialogue_data=calibration_dialogue_data,
                show_progress=False,
                generate_plots=False,
            )
            thresholds_path = os.path.join(tmpdir, "thresholds.json")

            ruleset_path = os.path.join(tmpdir, "unified_ruleset.json")
            with open(ruleset_path, "w") as f:
                json.dump(unified_ruleset, f)

            eval_out_dir = os.path.join(tmpdir, "eval")
            os.makedirs(eval_out_dir, exist_ok=True)
            evaluate(
                output_dir=eval_out_dir,
                labels=labels,
                thresholds_path=thresholds_path,
                unified_ruleset_path=ruleset_path,
                dialogue_data=evaluation_dialogue_data,
                compute_auc=True,
                show_progress=False,
            )

            for fname in ("usecase_metrics_fprtpr.csv",
                          "usecase_weighted_averages.csv"):
                assert (Path(eval_out_dir) / fname).is_file(), f"{fname} missing"

            with open(Path(eval_out_dir) / "usecase_metrics_fprtpr.csv", "r") as f:
                rows = list(csv.DictReader(f))

        usecase_col = next(c for c in rows[0].keys() if c.lower().startswith("use"))
        by_rule = {r[usecase_col]: r for r in rows}
        assert "rule_strict" in by_rule, "rule_strict not in metrics CSV"
        assert "rule_anyof" in by_rule, "rule_anyof not in metrics CSV"

        # Hand-computed (see the evaluation fixture docstring):
        # rule_strict: 3 TP / 2 FN -> TPR 0.6; negatives silent -> FPR 0.
        assert float(by_rule["rule_strict"]["TPR"]) == pytest.approx(0.6)
        assert float(by_rule["rule_strict"]["FPR"]) == pytest.approx(0.0)
        # rule_anyof: 2 TP / 1 FN -> TPR 2/3; delta-only negative can't
        # fire the rule (req group unmet) -> FPR 0.
        assert float(by_rule["rule_anyof"]["TPR"]) == pytest.approx(2 / 3)
        assert float(by_rule["rule_anyof"]["FPR"]) == pytest.approx(0.0)


class TestAdapterParity:
    """Our adapter is supposed to be a thin orchestrator. These tests
    verify it produces the same answers as the raw reference calls would.

    The DB-backed build_unified_ruleset is stubbed with the same-shaped
    fixture (labels normally come from on-disk classifier_meta.json)."""

    def test_run_calibration_threshold_keys_match_labels(
        self, labels, unified_ruleset, calibration_dialogue_data, monkeypatch,
    ):
        import classifier_engine.reference  # noqa: F401
        from evaluation import adapter

        monkeypatch.setattr(adapter, "build_unified_ruleset",
                            lambda classifier_id: unified_ruleset)

        result = adapter.run_calibration(
            classifier_id=999,
            labels=labels,
            dialogue_data=calibration_dialogue_data,
        )

        assert set(result.keys()) == set(labels.keys())
        for entry in result.values():
            assert "threshold" in entry
            assert "patience" in entry
            assert "youden_j" in entry
            # Same clean-separation optimum as the raw reference call.
            assert entry["youden_j"] == pytest.approx(1.0)

    def test_run_evaluation_returns_serialisable_metrics(
        self, labels, unified_ruleset,
        calibration_dialogue_data, evaluation_dialogue_data, monkeypatch,
    ):
        import classifier_engine.reference  # noqa: F401
        from evaluation import adapter

        monkeypatch.setattr(adapter, "build_unified_ruleset",
                            lambda classifier_id: unified_ruleset)

        thresholds = adapter.run_calibration(
            classifier_id=999, labels=labels,
            dialogue_data=calibration_dialogue_data,
        )

        result = adapter.run_evaluation(
            classifier_id=999, labels=labels,
            dialogue_data=evaluation_dialogue_data,
            thresholds=thresholds, compute_auc=True,
        )

        # The adapter promises JSONB-safe (no DataFrames, no tensors) output
        json.dumps(result)
        # Adapter unwraps the CSVs into a dict keyed by stem
        assert "csvs" in result
