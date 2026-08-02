"""Debug and diagnostic utilities for GAVEL evaluation.

This module contains utilities for detailed per-dialogue logging,
outcome recording, and sanity checking during evaluation runs.
These are primarily useful for debugging and detailed analysis.
"""
from __future__ import annotations
import csv
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class OutcomeRecorder:
    """Records per-dialogue evaluation outcomes for detailed analysis.

    Used for debugging and generating detailed CSV reports of
    the evaluation process.
    """

    rows: List[Dict[str, Any]] = field(default_factory=list)

    def add_row(
        self,
        *,
        dialogue: str,
        ground_truth: str,
        usecase: str,
        outcome: str,
        predicted_labels: List[str],
        missing_labels_names_for_gt: List[str],
        any_of_failed_for_gt: bool,
        gt_any_of_groups_failed: Optional[List[int]] = None,
        gt_any_of_groups_total: Optional[int] = None,
        superset_guard_result: str = "",
        superset_guard_outcome: str = "",
        fp_reason: str = "",
        is_gt: bool = False,
    ):
        predicted_labels_field = "|".join(predicted_labels)

        # Build missing_labels field for GT row (req names + optional tag)
        gt_missing_parts: List[str] = []
        if is_gt:
            if missing_labels_names_for_gt:
                gt_missing_parts.extend(missing_labels_names_for_gt)
            if any_of_failed_for_gt:
                gt_missing_parts.append("any_of_not_met")

        row = dict(
            dialogue=dialogue,
            ground_truth=ground_truth,
            usecase=usecase,
            outcome=outcome,
            predicted_labels=predicted_labels_field,
            missing_labels=("".join([]) if not is_gt else "|".join(gt_missing_parts)),
            superset_guard_result=superset_guard_result,
            superset_guard_outcome=superset_guard_outcome,
            fp_reason=fp_reason,
        )

        # Only populate the any_of group diagnostics on the GT row
        if is_gt:
            row["any_of_groups_failed"] = (
                ""
                if not gt_any_of_groups_failed
                else "|".join(f"G{g}" for g in gt_any_of_groups_failed)
            )
            row["any_of_groups_total"] = (
                "" if gt_any_of_groups_total is None else str(gt_any_of_groups_total)
            )

        self.rows.append(row)


def write_rows_to_csv(rows: List[Dict[str, Any]], path: str) -> None:
    """Write outcome recorder rows to a CSV file.

    Args:
        rows: List of row dictionaries from OutcomeRecorder.
        path: Output file path.
    """
    fieldnames = [
        "dialogue",
        "ground_truth",
        "usecase",
        "outcome",
        "predicted_labels",
        "missing_labels",
        "superset_guard_result",
        "superset_guard_outcome",
        "fp_reason",
        "any_of_groups_failed",
        "any_of_groups_total",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def logger_usecase_stats(
    triggers: torch.Tensor,
    all_required_labels: torch.Tensor,
    supporting_labels: torch.Tensor,
    any_of_conditions: dict[str, list[torch.Tensor]],
    dialogue_use_case: str,
    use_cases_stats: dict[str, dict[str, int]],
    all_usecase_gt_labels: dict[str, dict[str, torch.Tensor]],
    dialogue_name: str,
    idxs_to_labels: dict[int, str],
    score_neutral=False,
    recorder: Optional[OutcomeRecorder] = None,
) -> None:
    """Log detailed per-dialogue use case statistics.

    This function provides verbose logging of detection outcomes for
    each dialogue, including superset guard logic for false positive
    analysis.

    Args:
        triggers: Tensor of triggered labels.
        all_required_labels: Required labels for the ground truth use case.
        supporting_labels: Supporting labels for the ground truth use case.
        any_of_conditions: Any-of conditions for all use cases.
        dialogue_use_case: Ground truth use case name.
        use_cases_stats: Dictionary to update with TP/FP/TN/FN counts.
        all_usecase_gt_labels: Ground truth labels for all use cases.
        dialogue_name: Dialogue identifier for logging.
        idxs_to_labels: Mapping from indices to label names.
        score_neutral: Whether to score neutral use cases.
        recorder: Optional OutcomeRecorder for CSV output.
    """
    # Predictions
    predicted_idxs = triggers.bool().nonzero(as_tuple=True)[0]
    fired_any = predicted_idxs.numel() > 0

    # Ensure all masks/indices live on the same device
    device = all_required_labels.device
    all_required_labels = all_required_labels.to(device)
    supporting_labels = supporting_labels.to(device)
    predicted_idxs = predicted_idxs.to(device)

    gt_usecase_any_of_groups = any_of_conditions.get(dialogue_use_case, []) or []
    if len(gt_usecase_any_of_groups) > 0:
        gt_usecase_any_of_union = torch.cat(gt_usecase_any_of_groups).to(device)
    else:
        gt_usecase_any_of_union = torch.tensor([], dtype=torch.long, device=device)

    # allowed set = all_required | supporting | any_of_union   (boolean mask)
    num_labels = int(all_required_labels.numel())
    any_of_mask = torch.zeros(num_labels, dtype=torch.bool, device=device)
    if gt_usecase_any_of_union.numel() > 0:
        any_of_mask[gt_usecase_any_of_union] = True

    allowed_mask = all_required_labels.bool() | supporting_labels.bool() | any_of_mask
    gt_allowed_idxs = allowed_mask.nonzero(as_tuple=True)[0]

    # Type predicates
    def is_bu(uc: str) -> bool:
        return uc == "neutral"

    def is_mal(uc: str) -> bool:
        return not is_bu(uc)

    # Detection primitives
    def has_all_required(uc: str) -> bool:
        uc_req = (all_usecase_gt_labels[uc]["all_required_labels"] == 1).nonzero(as_tuple=True)[0]
        return (uc_req.numel() == 0) or torch.all(torch.isin(uc_req, predicted_idxs))

    def passes_any_of(uc: str) -> bool:
        if uc in any_of_conditions.keys():
            if len(any_of_conditions[uc]) > 1:
                for group in any_of_conditions[uc]:
                    if not torch.any(torch.isin(predicted_idxs, group)):
                        return False
            else:
                return torch.any(torch.isin(predicted_idxs, any_of_conditions[uc][0]))
        return True

    def detect_uc(uc: str) -> bool:
        return has_all_required(uc) and passes_any_of(uc)

    # Prepare display info {LOGGING}
    predicted_labels = [idxs_to_labels[i.item()] for i in predicted_idxs]
    # gt_allowed_labels = [idxs_to_labels[i.item()] for i in gt_allowed_idxs]

    # Find missing labels (all_required labels + any_of labels that didn't fire)
    gt_required_idxs = (all_required_labels == 1).nonzero(as_tuple=True)[0]
    missing_required = gt_required_idxs[~torch.isin(gt_required_idxs, predicted_idxs)]
    missing_req_names_for_gt = [idxs_to_labels[i.item()] for i in missing_required]

    gt_any_of_groups_total = 0
    gt_any_of_groups_failed: List[int] = []
    if dialogue_use_case in any_of_conditions and len(any_of_conditions[dialogue_use_case]) > 0:
        groups = any_of_conditions[dialogue_use_case]
        gt_any_of_groups_total = len(groups)
        for gi, group in enumerate(groups, start=1):
            hit = torch.any(torch.isin(predicted_idxs, group.to(predicted_idxs.device)))
            if not hit:
                gt_any_of_groups_failed.append(gi)

    any_of_failed_for_gt = len(gt_any_of_groups_failed) > 0

    def emit(
        uc: str,
        outcome: str,
        *,
        is_gt: bool = False,
        superset_guard_result: str = "",
        superset_guard_outcome: str = "",
        fp_reason: str = "",
    ):
        if recorder is None:
            return
        recorder.add_row(
            dialogue=str(dialogue_name),
            ground_truth=str(dialogue_use_case),
            usecase=uc,
            outcome=outcome,
            predicted_labels=predicted_labels,
            missing_labels_names_for_gt=missing_req_names_for_gt,
            any_of_failed_for_gt=any_of_failed_for_gt if is_gt else False,
            gt_any_of_groups_failed=(gt_any_of_groups_failed if is_gt else None),
            gt_any_of_groups_total=(gt_any_of_groups_total if is_gt else None),
            superset_guard_result=superset_guard_result,
            superset_guard_outcome=superset_guard_outcome,
            fp_reason=fp_reason,
            is_gt=is_gt,
        )

    # Partitions
    all_ucs = list(use_cases_stats.keys())
    mal_ucs = [uc for uc in all_ucs if is_mal(uc)]

    if not fired_any:
        use_cases_stats[dialogue_use_case]["false_negative"] += 1
        emit(dialogue_use_case, "FN", is_gt=True)
        for uc in all_ucs:
            if uc not in ("neutral", dialogue_use_case):
                use_cases_stats[uc]["true_negative"] += 1
                emit(uc, "TN")

        return

    if is_mal(dialogue_use_case):
        for uc in mal_ucs:
            detected = detect_uc(uc)

            if uc == dialogue_use_case:
                if detected:
                    use_cases_stats[uc]["true_positive"] += 1
                    emit(uc, "TP", is_gt=True)
                else:
                    use_cases_stats[uc]["false_negative"] += 1
                    emit(uc, "FN", is_gt=True)
            else:
                if detected:
                    other_uc_req_idxs = (
                        all_usecase_gt_labels[uc]["all_required_labels"] == 1
                    ).nonzero(as_tuple=True)[0]

                    triggering_any_of_idxs = torch.tensor([], dtype=torch.long)
                    if uc in any_of_conditions:
                        if len(any_of_conditions[uc]) > 1:
                            triggering_altogether = []
                            for group in any_of_conditions[uc]:
                                triggering_altogether.append(
                                    predicted_idxs[torch.isin(predicted_idxs, group)]
                                )
                            triggering_any_of_idxs = torch.cat(triggering_altogether)

                        else:
                            any_of_options = any_of_conditions[uc][0]
                            triggering_any_of_idxs = predicted_idxs[
                                torch.isin(predicted_idxs, any_of_options)
                            ]

                    req_ok = torch.all(torch.isin(other_uc_req_idxs, gt_allowed_idxs))
                    has_any_of_constraint = uc in any_of_conditions

                    if has_any_of_constraint:
                        if len(any_of_conditions[uc]) > 1:
                            for group_triggered in triggering_altogether:
                                any_of_ok = torch.any(torch.isin(group_triggered, gt_allowed_idxs))
                                if not any_of_ok:
                                    break
                        else:
                            any_of_ok = torch.any(
                                torch.isin(triggering_any_of_idxs, gt_allowed_idxs)
                            )

                        allowed_guard = req_ok and any_of_ok

                    else:
                        allowed_guard = req_ok

                    if allowed_guard:
                        emit(
                            uc,
                            "superset TP",
                            superset_guard_result="true",
                            superset_guard_outcome="",
                        )

                    else:
                        use_cases_stats[uc]["false_positive"] += 1
                        emit(
                            uc,
                            "FP",
                            superset_guard_result="false",
                            superset_guard_outcome="FP",
                            fp_reason="Labels outside allowed set",
                        )
                else:
                    use_cases_stats[uc]["true_negative"] += 1
                    emit(uc, "TN")


def sanity_check_gt(
    gt: torch.Tensor,
    any_of_conditions: dict[str, list[torch.Tensor]],
    all_usecase_gt_labels: dict[str, dict[str, torch.Tensor]],
    gt_uc_name: str,
    dialogue_name,
    split: str,
    use_cases_stats: dict[str, dict[str, int]],
) -> None:
    """Validate ground truth labels meet detection requirements.

    This sanity check verifies that explicit ground truth labels
    would actually trigger detection for the specified use case.
    Useful for debugging annotation quality.

    Args:
        gt: Ground truth one-hot tensor.
        any_of_conditions: Any-of conditions for all use cases.
        all_usecase_gt_labels: Ground truth labels for all use cases.
        gt_uc_name: Ground truth use case name.
        dialogue_name: Dialogue identifier for logging.
        split: Data split ("positive" or "negative").
        use_cases_stats: Dictionary to update with TP/FN counts.
    """
    logger.debug(
        f"Sanity checking GT for dialogue: {dialogue_name}, use case: {gt_uc_name}, split {split}"
    )
    predicted_idxs = gt.bool().nonzero(as_tuple=True)[0]

    def has_all_required(uc: str) -> bool:
        req_idxs = (all_usecase_gt_labels[uc]["all_required_labels"] == 1).nonzero(as_tuple=True)[0]
        return (req_idxs.numel() == 0) or torch.all(torch.isin(req_idxs, predicted_idxs))

    def passes_any_of(uc: str) -> bool:
        if uc not in any_of_conditions:
            return True
        groups = any_of_conditions[uc]
        if len(groups) > 1:
            return all(torch.any(torch.isin(predicted_idxs, group)) for group in groups)
        else:
            return torch.any(torch.isin(predicted_idxs, groups[0]))

    def detect_uc(uc: str) -> bool:
        return has_all_required(uc) and passes_any_of(uc)

    detected = detect_uc(gt_uc_name)
    if split == "positive":
        if detected:
            use_cases_stats[gt_uc_name]["true_positive"] += 1
            logger.debug(f"SANITY CHECK: {gt_uc_name} (pos): DETECTED → TP")
        else:
            use_cases_stats[gt_uc_name]["false_negative"] += 1
            logger.debug(f"SANITY CHECK: {gt_uc_name} (pos): NOT DETECTED → FN")
