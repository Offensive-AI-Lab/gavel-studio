"""TRUE end-to-end ML-pipeline integration tests (the real heart of GAVEL).

Everything else in the suite tests the API/contract layer around the ML, or the
calibration/evaluation *math* on synthetic logits (test_reference_parity.py). These
tests close the one remaining gap: the genuine process with a REAL model loaded —

    seed CEs + rule  ->  run_training (LLM forward + per-sequence readout +
    RNN probe training + artifact save + status flip + policy fingerprint)
    ->  realtime classify a stored dialogue (load_or_get the trained model,
        inference forward, windowed/per-token logits, calibrated compute_triggers,
        rule-predicate evaluation)
    ->  real-activation calibration (run_inference_on_dialogues + the reference
        Youden-J threshold sweep -> thresholds.json persisted)

They are marked `@pytest.mark.slow`, so the fast suite (`-m "not slow"`) skips
them. Run explicitly with:  python -m pytest tests/integration/test_ml_pipeline_e2e.py

The model (HuggingFaceTB/SmolLM2-360M-Instruct) loads on CPU. If it can't be
loaded (no weights cached AND no network) the test SKIPS rather than fails — so
CI without model access stays green while a developer machine exercises the
real path.

CLEANUP: every DB row is created in a conftest-tracked table, so the per-test
snapshot/restore removes it automatically. The only out-of-DB artifact is the
on-disk trained_classifiers/<user>/classifier_<id>/ directory, which each test
removes explicitly in a finally block.
"""
import json
import os
import time

import pytest

from utils.sqlite_db import execute_query, execute_query_dict


pytestmark = pytest.mark.slow


# A training-config tuned for SPEED, not quality — we only assert the pipeline
# runs end-to-end and produces well-formed artifacts, not that the probe is
# accurate. Tiny RNN, 2 epochs, last 2 LLM layers, short sequences.
FAST_TRAINING_CONFIG = {
    "batch_size_text": 2,
    # Must be large enough that the system+user prompt does NOT truncate the
    # ASSISTANT turn to zero tokens — the per-sequence readout only keeps
    # assistant-token reps, so a truncated assistant => 0 windows => the
    # stratified loader caps every class to 0 (num_samples=0).
    "max_length": 192,
    "rnn_sequence_length": 2,
    "batch_size": 8,
    "epochs": 2,
    "num_layers_to_use": 2,
    "hidden_dim": 32,
    "num_rnn_layers": 1,
    "learning_rate": 1e-3,
}

# Network/availability failure signatures — if run_training dies on one of
# these, the environment can't load the model, so we skip instead of fail.
_UNAVAILABLE_MARKERS = (
    "connection", "could not download", "couldn't connect", "timed out",
    "max retries", "offline", "repository not found", "can't load",
    "cannot find the requested files", "no such file", "proxyerror",
    "failed to establish", "name or service not known", "getaddrinfo",
)


def _unique(prefix: str) -> str:
    return f"{prefix}_{int(time.time() * 1000) % 100_000_000}"


def _assistant_paragraph(seed: str) -> str:
    """A long assistant turn so the per-sequence readout yields plenty of
    assistant-token reps to window (short replies => 0 windows for that class
    => the stratified loader caps every class to 0)."""
    return (
        f"Here is a thorough answer about {seed}. First, it helps to consider "
        f"the overall context and the goal we are trying to reach. Second, we "
        f"break the problem into smaller steps and examine each one in turn. "
        f"Third, we weigh the trade-offs, note the common pitfalls, and explain "
        f"why each decision is reasonable. Finally, we summarize the takeaways "
        f"so the explanation stays clear, complete, and easy to follow along."
    )


def _conversation(seed: str) -> list:
    # Keep the system+user turns short so the long assistant turn is never
    # truncated away by max_length.
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Explain {seed}."},
        {"role": "assistant", "content": _assistant_paragraph(seed)},
    ]


def _make_dataset(n: int, theme: str) -> dict:
    return {"samples": [_conversation(f"{theme} topic number {i}") for i in range(n)],
            "sample_count": n}


def _make_calibration_dataset(n: int, theme: str) -> dict:
    return {"conversations": [_conversation(f"{theme} calibration case {i}") for i in range(n)],
            "sample_count": n}


def _seed_trainable_classifier(client, auth_headers, test_user, test_model):
    """Build a fully-wired, trainable classifier:

      2 CEs (each with an excitation dataset) -> 1 rule (both CEs all_required)
      -> rule_setup attached to a fresh classifier -> setup_ce_link x2.

    Returns (classifier_id, user_id, [ce_id_a, ce_id_b], [ce_name_a, ce_name_b]).
    All rows land in conftest-tracked tables.
    """
    model_id = test_model["model_id"]
    user_id = test_user["user_id"]

    # Fresh classifier via the real API.
    cres = client.post(
        "/classifiers/create",
        json={"model_id": model_id, "name": _unique("e2e_cls")},
        headers=auth_headers,
    )
    assert cres.status_code == 200, cres.text
    classifier_id = cres.json().get("classifier", cres.json())["classifier_id"]

    # Two CEs with excitation datasets (16 conversations each).
    # NOTE: the CE name MUST NOT end in `_<digits>`. The text dataloader groups
    # topics by the regex `(.+?)_\d+$`, which would strip a trailing `_12345`
    # off the topic-dir name while the labels dict keeps the full sanitized
    # name — the resulting mismatch leaves zero sequence dirs and the RNN loader
    # dies with num_samples=0. Real CE names don't end in digits; we append a
    # non-digit suffix so the unique stamp sits in the MIDDLE of the name.
    ce_ids, ce_names = [], []
    for theme in ("alpha", "beta"):
        name = f"{_unique(f'e2e_ce_{theme}')}_probe"
        row = execute_query_dict(
            "INSERT INTO cognitive_elements (name, definition) VALUES (%s, %s) RETURNING ce_id",
            (name, f"end-to-end test CE for {theme}"),
        )[0]
        ce_id = row["ce_id"]
        execute_query(
            "INSERT INTO excitation_datasets (ce_id, dataset) VALUES (%s, %s)",
            (ce_id, json.dumps(_make_dataset(16, theme))),
        )
        ce_ids.append(ce_id)
        ce_names.append(name)

    # One rule requiring both CEs, attached to the classifier. v2 logic:
    # a single 'required' group with 'all of required'.
    ce_groups = json.dumps({"required": list(ce_names)})
    condition = "all of required"
    rule_id = execute_query_dict(
        "INSERT INTO rules (name, predicate, ce_groups, condition) "
        "VALUES (%s, %s, %s, %s) RETURNING rule_id",
        (_unique("e2e_rule"), " AND ".join(ce_names), ce_groups, condition),
    )[0]["rule_id"]
    setup_id = execute_query_dict(
        "INSERT INTO rule_setup (classifier_id, rule_id, custom_name, predicate, "
        "ce_groups, condition, is_active) "
        "VALUES (%s, %s, %s, %s, %s, %s, TRUE) RETURNING setup_id",
        (classifier_id, rule_id, _unique("e2e_setup"), " AND ".join(ce_names),
         ce_groups, condition),
    )[0]["setup_id"]
    for ce_id in ce_ids:
        execute_query(
            "INSERT INTO setup_ce_link (setup_id, ce_id) VALUES (%s, %s)",
            (setup_id, ce_id),
        )

    # Speed-tuned training config.
    execute_query(
        "UPDATE classifiers SET training_config = %s WHERE classifier_id = %s",
        (json.dumps(FAST_TRAINING_CONFIG), classifier_id),
    )
    return classifier_id, user_id, ce_ids, ce_names


def _cleanup_workdir(classifier_id, user_id):
    try:
        from classifier_engine.trainer import delete_classifier_workdir
        delete_classifier_workdir(classifier_id, user_id)
    except Exception:
        pass


def _skip_if_unavailable(exc: Exception):
    msg = str(exc).lower()
    if any(m in msg for m in _UNAVAILABLE_MARKERS):
        pytest.skip(f"Model could not be loaded in this environment: {exc}")


# ---------------------------------------------------------------------------
# Training -> realtime classification (the core forward path)
# ---------------------------------------------------------------------------


class TestTrainThenClassify:
    def test_train_to_active_then_realtime_classify(
        self, client, auth_headers, test_user, test_model
    ):
        """The whole forward path on a real model, in one run (train once)."""
        classifier_id, user_id, ce_ids, ce_names = _seed_trainable_classifier(
            client, auth_headers, test_user, test_model
        )
        # main.py warms torch/transformers on a background thread; importing
        # transformers HERE while that thread is mid-import can observe a
        # partially-initialized package ("cannot import name 'AutoTokenizer'").
        # Gate on the warmup flag /health exposes for exactly this purpose.
        import time as _time
        for _ in range(180):
            health = client.get("/health").json()
            if health.get("components", {}).get("transformers"):
                break
            _time.sleep(1)
        try:
            from classifier_engine.trainer import run_training, classifier_workdir

            # ---- TRAIN (real LLM forward + readout + RNN probe) ----
            try:
                run_training(classifier_id)
            except Exception as exc:  # noqa: BLE001
                _skip_if_unavailable(exc)
                raise

            # Status flipped to active and persisted.
            row = execute_query_dict(
                "SELECT status, trained_policy_fingerprint, model_path "
                "FROM classifiers WHERE classifier_id = %s",
                (classifier_id,),
            )[0]
            assert row["status"] == "active", f"status={row['status']}"
            # Policy fingerprint snapshotted at train time (drift detection input).
            assert row["trained_policy_fingerprint"], "no trained_policy_fingerprint stored"

            # On-disk artifacts exist and are well-formed.
            work_dir = classifier_workdir(classifier_id, user_id)
            rnn_path = os.path.join(work_dir, "trained_rnn.pth")
            meta_path = os.path.join(work_dir, "classifier_meta.json")
            assert os.path.isfile(rnn_path), "trained_rnn.pth not written"
            assert os.path.isfile(meta_path), "classifier_meta.json not written"
            with open(meta_path) as f:
                meta = json.load(f)
            assert set(meta["labels"].keys()) == {  # both CEs became labels
                _san(n) for n in ce_names
            }, f"labels={meta['labels']}"
            assert meta["num_classes"] == 2
            assert meta["readout_dim"] > 0
            assert len(meta["selected_layers"]) == 2

            # ---- MONITORING IS BLOCKED UNTIL CALIBRATED ----
            # Scoring against a default 0.5 threshold per CE would look
            # functional and mean nothing, so realtime refuses (409) rather
            # than serving meaningless verdicts. In the app, training chains
            # into calibration automatically; the suite disables that chain
            # (see conftest), so here the rule set is genuinely uncalibrated.
            convo = _conversation("alpha topic number 0")
            res = client.post(
                f"/realtime/{classifier_id}/analyze-stored",
                json={"messages": convo},
                headers=auth_headers,
            )
            assert res.status_code == 409, res.text
            assert "calibrat" in res.json()["detail"].lower()

            # Stand in for a finished calibration. The real sweep is covered by
            # TestCalibrationOnRealActivations; here we only need thresholds to
            # exist so the forward path is reachable.
            execute_query(
                "INSERT INTO evaluation_results (classifier_id, eval_type, thresholds) "
                "VALUES (%s, 'calibration', %s::jsonb)",
                (
                    classifier_id,
                    json.dumps({
                        name: {"threshold": 0.5, "patience": 1}
                        for name in meta["labels"]
                    }),
                ),
            )

            # ---- REALTIME CLASSIFY (inference forward + triggers + rule eval) ----
            res = client.post(
                f"/realtime/{classifier_id}/analyze-stored",
                json={"messages": convo},
                headers=auth_headers,
            )
            assert res.status_code == 200, res.text
            data = res.json()
            # Shape contract the frontend renders. analyze-stored is TURNS-
            # based: each assistant turn carries its own windows/tokens span;
            # windows/tokens are not top-level keys.
            assert set(data["labels"].keys()) == set(meta["labels"].keys())
            assistant_turns = [
                t for t in data["turns"]
                if t.get("role") == "assistant" and "windows" in t
            ]
            assert assistant_turns, f"no scored assistant turn in {data['turns']}"
            turn = assistant_turns[0]
            assert isinstance(turn["windows"], list) and len(turn["windows"]) >= 1
            assert isinstance(turn["tokens"], list) and len(turn["tokens"]) >= 1
            assert turn["num_windows"] == len(turn["windows"])
            # One rule was wired -> exactly one rule verdict, with a boolean fired.
            assert len(data["rule_triggers"]) == 1
            assert isinstance(data["rule_triggers"][0]["fired"], bool)
            # Each window carries real per-CE probabilities in [0,1] for both CEs.
            probs = turn["windows"][0]["probabilities"]
            assert set(probs.keys()) == set(meta["labels"].keys())
            for v in probs.values():
                assert 0.0 <= float(v) <= 1.0
            # Tokens drive the per-token chart -> raw logits are stripped
            # (NaN-safe payloads); each token carries per-CE probabilities.
            tok_probs = turn["tokens"][0]["probabilities"]
            assert set(tok_probs.keys()) == set(meta["labels"].keys())
            # No calibration row yet -> thresholds default to 0.5 for every CE.
            for spec in data["thresholds_used"].values():
                assert spec["threshold"] == 0.5
                assert spec["patience"] >= 1
        finally:
            _cleanup_workdir(classifier_id, user_id)

    def test_realtime_rejects_untrained_classifier(
        self, client, auth_headers, test_classifier
    ):
        """analyze-stored must 400 a classifier that was never trained — the
        forward path needs a model on disk. (Fast guard; no model load.)"""
        cid = test_classifier["classifier_id"]
        # Ensure it's not active for this assertion.
        execute_query(
            "UPDATE classifiers SET status = 'untrained' WHERE classifier_id = %s "
            "AND status NOT IN ('active','needs_retraining')",
            (cid,),
        )
        res = client.post(
            f"/realtime/{cid}/analyze-stored",
            json={"messages": _conversation("x")},
            headers=auth_headers,
        )
        assert res.status_code in (400, 404)


# ---------------------------------------------------------------------------
# Training -> real-activation calibration (Youden-J sweep on real logits)
# ---------------------------------------------------------------------------


class TestCalibrationOnRealActivations:
    def test_calibration_produces_thresholds_from_real_inference(
        self, client, auth_headers, test_user, test_model
    ):
        """Train, then run the REAL calibration: extract logits from stored
        dialogues via the trained model and run the reference threshold sweep.
        The sweep only consumes usecase_level (rule-level) dialogues, so we seed
        a rule-level positive_calibration test_dataset as well as per-CE sets."""
        classifier_id, user_id, ce_ids, ce_names = _seed_trainable_classifier(
            client, auth_headers, test_user, test_model
        )
        try:
            from classifier_engine.trainer import run_training

            try:
                run_training(classifier_id)
            except Exception as exc:  # noqa: BLE001
                _skip_if_unavailable(exc)
                raise

            # Per-CE calibration sets (CE_level) — loaded but, per the upstream
            # contract, not swept; seed them so the loader path is exercised.
            for ce_id, theme in zip(ce_ids, ("alpha", "beta")):
                execute_query(
                    "INSERT INTO calibration_datasets (ce_id, dataset) VALUES (%s, %s)",
                    (ce_id, json.dumps(_make_calibration_dataset(6, theme))),
                )
            # Rule-level (usecase_level) calibration — the set the sweep actually
            # consumes. Attach to the rule wired into this classifier.
            rule_id = execute_query_dict(
                "SELECT rule_id FROM rule_setup WHERE classifier_id = %s LIMIT 1",
                (classifier_id,),
            )[0]["rule_id"]
            execute_query(
                "INSERT INTO test_datasets (rule_id, dataset_type, status, is_default, conversations) "
                "VALUES (%s, 'positive_calibration', 'ready', TRUE, %s::jsonb)",
                (rule_id, json.dumps([_conversation(f"rule calib {i}") for i in range(6)])),
            )

            from routes.evaluation import _run_calibration

            try:
                _run_calibration(classifier_id, patience_values=[1])
            except Exception as exc:  # noqa: BLE001
                _skip_if_unavailable(exc)
                raise

            # A calibration row with real thresholds was persisted.
            rows = execute_query_dict(
                "SELECT thresholds FROM evaluation_results "
                "WHERE classifier_id = %s AND eval_type = 'calibration' "
                "AND thresholds IS NOT NULL ORDER BY created_at DESC LIMIT 1",
                (classifier_id,),
            )
            assert rows, "no calibration thresholds were saved"
            thresholds = rows[0]["thresholds"]
            if isinstance(thresholds, str):
                thresholds = json.loads(thresholds)
            assert thresholds, "thresholds dict is empty"
            # Every threshold entry has the documented schema with sane ranges.
            for ce_name, spec in thresholds.items():
                assert "threshold" in spec and "patience" in spec
                assert 0.0 <= float(spec["threshold"]) <= 1.0
                assert int(spec["patience"]) >= 1
        finally:
            _cleanup_workdir(classifier_id, user_id)


def _san(name: str) -> str:
    """Mirror trainer._sanitize_label so the test predicts the label keys."""
    import re
    return re.sub(r"[^\w\-]", "_", name).strip("_") or "label"
