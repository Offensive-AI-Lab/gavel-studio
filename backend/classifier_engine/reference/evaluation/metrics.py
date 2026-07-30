from __future__ import annotations
import json
import logging
import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

def build_ruleset_logic(data: dict, labels_dict: dict) -> dict:
    """Parse each rule's (groups, condition) pair into an evaluatable
    structure — the groups+condition replacement for the retired
    all_required/any_of/supporting tensor builders.

    Args:
        data: unified ruleset — {use_case: {"groups": {gname: [label names]},
            "condition": "all of gname and 1 of other", ...}}.
        labels_dict: label name -> index mapping of the trained model.

    Returns:
        {use_case: {
            "ast":           parsed condition AST (None when absent/invalid),
            "groups_idx":    {gname: LongTensor of member label indices},
            "member_labels": one-hot FloatTensor over every group member,
        }}

    Member names unknown to labels_dict are dropped — same tolerance the old
    tensor builders had; a referenced group left EMPTY by that filtering
    evaluates vacuously True (the legacy builders dropped empty groups)."""
    from utils.rule_condition import ConditionError, parse

    out = {}
    num_labels = len(labels_dict)
    for uc, spec in data.items():
        groups_idx = {}
        member_idxs: set = set()
        for gname, members in (spec.get("groups") or {}).items():
            idxs = [labels_dict[m] for m in (members or []) if m in labels_dict]
            groups_idx[gname] = torch.tensor(idxs, dtype=torch.long)
            member_idxs.update(idxs)
        ast = None
        condition = (spec.get("condition") or "").strip()
        if condition:
            try:
                ast = parse(condition)
            except ConditionError as e:
                logger.warning("ruleset logic: condition for '%s' does not parse (%s)", uc, e)
        one_hot = torch.zeros(num_labels, dtype=torch.float32)
        if member_idxs:
            one_hot[sorted(member_idxs)] = 1.0
        out[uc] = {"ast": ast, "groups_idx": groups_idx, "member_labels": one_hot}
    return out


def _eval_condition_idx(node, groups_idx: dict, predicted_idxs: torch.Tensor) -> bool:
    """Evaluate a condition AST against the set of triggered label indices.
    Selector semantics: at least k distinct group members triggered ('all' =
    every member). A group missing/empty after label filtering is vacuously
    True (mirrors the legacy dropped-empty-group behavior)."""
    kind = node[0]
    if kind == "or":
        return any(_eval_condition_idx(c, groups_idx, predicted_idxs) for c in node[1])
    if kind == "and":
        return all(_eval_condition_idx(c, groups_idx, predicted_idxs) for c in node[1])
    if kind == "not":
        return not _eval_condition_idx(node[1], groups_idx, predicted_idxs)
    _, quant, name = node
    members = groups_idx.get(name)
    if members is None or members.numel() == 0:
        return True
    hits = int(torch.isin(members, predicted_idxs).sum().item())
    k = members.numel() if quant == "all" else quant
    return hits >= k


def detect_usecase(ruleset_logic: dict, uc: str, predicted_idxs: torch.Tensor) -> bool:
    """Does the triggered-label set fire this use case's condition? Unknown
    use cases and rules without a parsed condition never fire."""
    entry = ruleset_logic.get(uc)
    if entry is None or entry["ast"] is None:
        return False
    return _eval_condition_idx(entry["ast"], entry["groups_idx"], predicted_idxs)


def usecase_soft_score(entry: dict, topic_scores: torch.Tensor) -> float:
    """Fuzzy [0,1] confidence that a use case fires, for AUC curves.
    Generalizes the legacy min(required)/min-over-groups(max) score:
    and=min, or=max, not=1-x, 'all of g'=min over members, 'k of g'=k-th
    largest member score. Vacuous (missing/empty) selectors score 1.0."""
    def rec(node) -> float:
        kind = node[0]
        if kind == "or":
            return max(rec(c) for c in node[1])
        if kind == "and":
            return min(rec(c) for c in node[1])
        if kind == "not":
            return 1.0 - rec(node[1])
        _, quant, name = node
        members = entry["groups_idx"].get(name)
        if members is None or members.numel() == 0:
            return 1.0
        scores = topic_scores[members]
        if quant == "all":
            return float(scores.min().item())
        k = min(int(quant), scores.numel())
        if k <= 1:
            return float(scores.max().item())
        return float(scores.topk(k).values[-1].item())

    if entry["ast"] is None:
        return 0.0
    return rec(entry["ast"])


def compute_triggers(
    logits_list: torch.Tensor | np.ndarray,
    thresholds: float | torch.Tensor | np.ndarray = 0.8,
    patience_rate: int = 1,
) -> torch.Tensor:
    """Compute trigger vector from window-level logits.

    Applies sigmoid to convert logits to probabilities, then determines
    which labels triggered based on thresholds and patience requirements.

    Args:
        logits_list: Logits tensor of shape (num_windows, num_labels).
            Can be numpy array or torch tensor.
        thresholds: Decision threshold(s). Can be a scalar (applied to all labels)
            or a tensor/array of per-label thresholds. Defaults to 0.8.
        patience_rate: Minimum number of windows that must exceed threshold
            for a label to trigger. Defaults to 1.

    Returns:
        1D tensor of shape (num_labels,) with 1.0 for triggered labels, 0.0 otherwise.

    Raises:
        ValueError: If logits_list is not a numpy array or torch tensor.
    """
    if isinstance(logits_list, np.ndarray):
        logits_tensor = torch.from_numpy(logits_list).float()
    elif torch.is_tensor(logits_list):
        logits_tensor = logits_list.float()
    else:
        raise ValueError(
            "logits_list should be a numpy.ndarray or torch.Tensor of shape (num_windows, num_labels)."
        )

    probs = torch.sigmoid(logits_tensor)
    if isinstance(thresholds, float) or isinstance(thresholds, int):
        hits = probs >= thresholds  # broadcast scalar over all labels
    else:
        if isinstance(thresholds, np.ndarray):
            thresholds = torch.from_numpy(thresholds).float()
        hits = probs >= thresholds.view(1, -1)  # broadcast thresholds to all windows
    above_thresh = hits.sum(dim=0)  # (#hits per label)
    triggers = (above_thresh >= patience_rate).float()
    return triggers


def _labels_from_idxs(idxs: torch.Tensor, idxs_to_labels: Dict[int, str]) -> List[str]:
    return [idxs_to_labels[i.item()] for i in idxs]


def save_to_csv(labels_statistics: List[Dict[str, Any]], output_path: str) -> None:
    """Save label-level statistics to a CSV file.

    Args:
        labels_statistics: List of dictionaries containing label statistics.
        output_path: Path to the output CSV file.
    """
    df = pd.DataFrame(labels_statistics)
    df.to_csv(output_path, index=False)
    logger.debug(f"Saved label-level statistics to {output_path}")


def eval_usecase_detection(
    triggers: torch.Tensor,
    ruleset_logic: dict,
    gt_uc_name: str,
    dialogue_name: str,
    split: str,
    use_cases_stats: dict[str, dict[str, int]],
) -> None:
    """Evaluate use case detection and update TPR/FPR statistics.

    Determines if the ground truth use case was correctly detected based on
    triggered labels, and updates the confusion matrix statistics accordingly.

    Args:
        triggers: 1D tensor of triggered labels (1.0 for triggered, 0.0 otherwise).
        ruleset_logic: parsed rule logic from build_ruleset_logic.
        gt_uc_name: Name of the ground truth use case for this dialogue.
        dialogue_name: Identifier for the dialogue (for logging).
        split: Data split - "positive", "negative", or "neutral".
        use_cases_stats: Dictionary to update with TP/FP/TN/FN counts.
    """
    logger.debug(
        f"Evaluating use case stats for dialogue: {dialogue_name}, use case: {gt_uc_name}, split {split}"
    )
    predicted_idxs = triggers.bool().nonzero(as_tuple=True)[0]
    fired_any = predicted_idxs.numel() > 0
    logger.debug(f"Predicted idx: {predicted_idxs}")

    def detect_uc(uc: str) -> bool:
        return detect_usecase(ruleset_logic, uc, predicted_idxs)

    mal_detected_in_neutral = False
    if split == "neutral":
        if not fired_any:
            use_cases_stats[gt_uc_name]["true_negative"] += 1
            logger.debug(f"{gt_uc_name}: NOT DETECTED → TN in neutral {gt_uc_name}")
        else:
            for uc in use_cases_stats.keys():
                if uc in ["conversational", "instructive"]:
                    continue
                if detect_uc(uc):
                    use_cases_stats[gt_uc_name]["false_positive"] += 1
                    logger.debug(f"{uc}: DETECTED → FP in neutral {gt_uc_name}")
                    mal_detected_in_neutral = True
                    break

        if not mal_detected_in_neutral and fired_any:
            use_cases_stats[gt_uc_name]["true_negative"] += 1
            logger.debug(f"{gt_uc_name}: NOT DETECTED → TN in neutral {gt_uc_name}")

    else:
        detected = detect_uc(gt_uc_name)
        if split == "positive":
            # For positive malicious samples, we want detection (true positive = detection)
            if detected:
                use_cases_stats[gt_uc_name]["true_positive"] += 1
                logger.debug(f"{gt_uc_name} (pos): DETECTED → TP")
            else:
                use_cases_stats[gt_uc_name]["false_negative"] += 1
                logger.debug(f"{gt_uc_name} (pos): NOT DETECTED → FN")
        else:  # split == "neg"
            # For negative malicious samples, we want NO detection (true negative = no detection)
            if detected:
                use_cases_stats[gt_uc_name]["false_positive"] += 1
                logger.debug(f"{gt_uc_name} (neg): INCORRECTLY DETECTED → FP")
            else:
                use_cases_stats[gt_uc_name]["true_negative"] += 1
                logger.debug(f"{gt_uc_name} (neg): NOT DETECTED → TN")


def update_usecase_confusion_matrix(
    triggers: torch.Tensor,
    ruleset_logic: dict,
    gt_uc_name: str,
    split: str,
    use_cases_stats: dict[str, dict[str, int]],
) -> None:
    """Update confusion matrix statistics for use case detection.

    Similar to eval_usecase_detection but used for accuracy-style metrics
    where we evaluate detection against all malicious use cases.

    Args:
        triggers: 1D tensor of triggered labels.
        ruleset_logic: parsed rule logic from build_ruleset_logic.
        gt_uc_name: Name of the ground truth use case for this dialogue.
        split: Data split - "positive", "negative", or "neutral".
        use_cases_stats: Dictionary to update with TP/FP/TN/FN counts.
    """
    predicted_idxs = triggers.bool().nonzero(as_tuple=True)[0]

    def detect_uc(uc: str) -> bool:
        return detect_usecase(ruleset_logic, uc, predicted_idxs)

    if split == "neutral":
        for uc in use_cases_stats.keys():
            if uc in ["conversational", "instructive"]:
                continue
            if detect_uc(uc):
                use_cases_stats[uc]["false_positive"] += 1
            else:
                use_cases_stats[uc]["true_negative"] += 1

    else:
        if split == "positive":
            detected = detect_uc(gt_uc_name)
            if detected:
                use_cases_stats[gt_uc_name]["true_positive"] += 1
            else:
                use_cases_stats[gt_uc_name]["false_negative"] += 1


def compute_usecase_detection_metrics(use_cases_stats: Dict[str, Dict[str, int]]) -> pd.DataFrame:
    """Compute detection metrics from use case confusion matrix statistics.

    Calculates True Positive Rate (TPR), False Positive Rate (FPR),
    Accuracy, and F1 score for each use case.

    Args:
        use_cases_stats: Dictionary mapping use case names to their
            confusion matrix counts (true_positive, false_positive,
            true_negative, false_negative).

    Returns:
        DataFrame with columns: Usecase, TPR, FPR, Accuracy, F1,
        TP, FP, TN, FN for each use case.
    """
    rows = []
    for uc, stats in use_cases_stats.items():
        tp = stats["true_positive"]
        fp = stats["false_positive"]
        tn = stats["true_negative"]
        fn = stats["false_negative"]

        # TPR = TP / (TP + FN) - from positive samples
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # FPR = FP / (FP + TN) - from negative samples
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # Recall (same as tpr)
        recall = tpr

        # Accuracy
        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0

        # F1 score
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        support_pos = tp + fn  # Total positive samples
        support_neg = fp + tn  # Total negative samples

        rows.append(
            {
                "Usecase": uc,
                "TPR": tpr,
                "FPR": fpr,
                "Accuracy": accuracy,
                "F1": f1,
                "Support_Pos": support_pos,
                "Support_Neg": support_neg,
            }
        )

    return pd.DataFrame(rows).sort_values("Usecase")


def compute_weighted_metrics(
    stats: Dict[str, Dict[str, int]],
) -> tuple[Dict[str, float], pd.DataFrame]:
    """Compute support-weighted average metrics across use cases.

    Calculates weighted averages of TPR, FPR, precision, and accuracy
    where weights are the support (number of positive samples) for each use case.

    Args:
        stats: Dictionary mapping use case names to their confusion matrix counts.

    Returns:
        Tuple of (weighted_averages_dict, detailed_dataframe) where:
            - weighted_averages_dict contains weighted TPR, FPR, precision, accuracy
            - detailed_dataframe contains per-use-case metrics
    """
    # Exclude 'benign_untargeted' if it exists in the stats dict keys
    malicious_stats = {k: v for k, v in stats.items() if not k.startswith("neutral_")}
    if not malicious_stats:
        logger.warning("No malicious use case stats found to average.")
        return {}
    else:
        logger.debug(f"Malicious use cases found: {list(malicious_stats.keys())}")

    df = pd.DataFrame.from_dict(malicious_stats, orient="index")

    # Ensure all columns exist, fill with 0 if not
    for col in ["true_positive", "false_negative", "false_positive", "true_negative"]:
        if col not in df.columns:
            df[col] = 0
    df = df.fillna(0)

    # --- Calculate metrics for each use case ---
    # Support (total positive samples for each class)
    df["support"] = df["true_positive"] + df["false_negative"]

    # True Positive Rate (Recall)
    df["tpr"] = df["true_positive"] / df["support"]

    # Precision
    df["precision"] = df["true_positive"] / (df["true_positive"] + df["false_positive"])

    # False Positive Rate
    df["fpr"] = df["false_positive"] / (df["false_positive"] + df["true_negative"])

    # Accuracy
    df["accuracy"] = (df["true_positive"] + df["true_negative"]) / (
        df["true_positive"] + df["true_negative"] + df["false_positive"] + df["false_negative"]
    )

    df = df.fillna(0)  # Handle division-by-zero cases (e.g., no positives)

    # --- Calculate weighted averages ---
    total_support = df["support"].sum()
    if total_support == 0:
        logger.warning("Total support is zero, cannot calculate weighted averages.")
        return {}

    weighted_avg = {
        "Weighted Avg TPR (Recall)": (df["tpr"] * df["support"]).sum() / total_support,
        "Weighted Avg Precision": (df["precision"] * df["support"]).sum() / total_support,
        "Weighted Avg FPR": (df["fpr"] * df["support"]).sum() / total_support,
        "Weighted Avg Accuracy": (df["accuracy"] * df["support"]).sum() / total_support,
    }

    logger.debug("--- Per-Use-Case Metrics ---")
    logger.debug(df[["tpr", "precision", "fpr", "accuracy", "support"]].round(3).to_string())
    logger.debug("--- Summary ---")

    return weighted_avg, df


def plot_usecase_metrics_table(df_metrics: pd.DataFrame, output_path: str):
    """
    Plots a single table of use case metrics (TPR, FPR, Supports).
    For usecases containing "conversational" or "instructive", prepends 'NEUTRAL_' and moves their row to the bottom,
    and ensures the usecase column is wider for potentially long names.
    Saves as a PNG.

    Args:
        df_metrics (pd.DataFrame): DataFrame containing the metrics.
            Expected columns: "Usecase", "TPR", "FPR", "Accuracy", "F1", "Support_Pos", "Support_Neg".
        output_path (str): Path to save the output PNG file.
    """
    # Define the columns to be displayed in the specified order
    cols = ["Usecase", "TPR", "FPR", "Accuracy", "F1", "Support_Pos", "Support_Neg"]
    df_plot = df_metrics[cols].copy()

    # --- Identify and handle neutral usecases ---
    # Build mask for usecases containing "conversational" or "instructive" (case insensitive)
    mask_neutral = (
        df_plot["Usecase"].str.contains("conversational", case=False)
        | df_plot["Usecase"].str.contains("instructive", case=False)
        | df_plot["Usecase"].str.contains("instructive", case=False)
    )
    df_neutral = df_plot[mask_neutral].copy()
    df_other = df_plot[~mask_neutral].copy()

    if not df_neutral.empty:
        # Prepend 'NEUTRAL_' (unless already present)
        df_neutral["Usecase"] = df_neutral["Usecase"].apply(
            lambda x: x if x.startswith("NEUTRAL_") else "NEUTRAL_" + x
        )
    # Combine such that neutral rows go at the end
    df_plot_ordered = pd.concat([df_other, df_neutral], ignore_index=True)

    # --- Prepare data for plotting ---
    # Format numbers as strings for the table cells
    table_rows = [
        [
            r["Usecase"],
            f"{r['TPR']:.3f}",
            f"{r['FPR']:.3f}",
            f"{r['Accuracy']:.3f}",
            f"{r['F1']:.3f}",
            str(int(r["Support_Pos"])),
            str(int(r["Support_Neg"])),
        ]
        for _, r in df_plot_ordered.iterrows()
    ]

    # --- Figure Setup ---
    # Dynamically calculate figure height based on the number of rows
    fig_h = max(4, len(table_rows) * 0.4 + 1.5)
    # Widen figure to accommodate possible long NEUTRAL_xxx names
    fig_w = 13
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")  # Hide the axes, we only want the table

    # --- Helper function for styling ---
    def style_table_cells(table, df):
        """
        Styles the table header and colors the data cells based on performance.
        - Header: Blue background with bold white text.
        - TPR: Green (≥0.8), Yellow (≥0.6), Pink (<0.6). Grayed out if Support_Pos is 0.
        - FPR: Green (≤0.10), Yellow (≤0.20), Pink (>0.20).
        """
        num_cols = len(df.columns)

        # Style header row
        for j in range(num_cols):
            table[(0, j)].set_facecolor("#4472C4")
            table[(0, j)].set_text_props(weight="bold", color="white")

        # Style data rows (start from row 1, as row 0 is the header)
        for i, (_, r) in enumerate(df.iterrows(), start=1):
            # TPR cell (column index 1)
            tpr = r["TPR"]
            accuracy = r["Accuracy"]
            f1 = r["F1"]
            support_pos = r["Support_Pos"]
            if support_pos == 0 or np.isnan(tpr):
                table[(i, 1)].set_facecolor("#DDDDDD")  # Gray
            elif tpr >= 0.8:
                table[(i, 1)].set_facecolor("#90EE90")  # Green
            elif tpr >= 0.6:
                table[(i, 1)].set_facecolor("#FFFFE0")  # Yellow
            else:
                table[(i, 1)].set_facecolor("#FFB6C1")  # Pink

            # FPR cell (column index 2)
            fpr = r["FPR"]
            if np.isnan(fpr):
                table[(i, 2)].set_facecolor("#DDDDDD")  # Gray
            elif fpr <= 0.10:
                table[(i, 2)].set_facecolor("#90EE90")  # Green
            elif fpr <= 0.20:
                table[(i, 2)].set_facecolor("#FFFFE0")  # Yellow
            else:
                table[(i, 2)].set_facecolor("#FFB6C1")  # Pink

            # Accuracy cell (column index 3)
            accuracy = r["Accuracy"]
            if np.isnan(accuracy):
                table[(i, 3)].set_facecolor("#DDDDDD")  # Gray
            elif accuracy >= 0.8:
                table[(i, 3)].set_facecolor("#90EE90")  # Green
            elif accuracy >= 0.6:
                table[(i, 3)].set_facecolor("#FFFFE0")  # Yellow
            else:
                table[(i, 3)].set_facecolor("#FFB6C1")  # Pink

            # F1 cell (column index 4)
            f1 = r["F1"]
            if support_pos == 0 or np.isnan(f1):
                table[(i, 4)].set_facecolor("#DDDDDD")  # Gray
            elif f1 >= 0.8:
                table[(i, 4)].set_facecolor("#90EE90")  # Green
            elif f1 >= 0.6:
                table[(i, 4)].set_facecolor("#FFFFE0")  # Yellow
            else:
                table[(i, 4)].set_facecolor("#FFB6C1")  # Pink

    # --- Create and style the table ---
    tbl = ax.table(
        cellText=([cols] + table_rows),
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    # Make the first (Usecase) column wider to fit NEUTRAL_... names
    tbl.auto_set_column_width(col=list(range(len(cols))))
    for key, cell in tbl.get_celld().items():
        if key[1] == 0:  # first column
            cell.set_width(0.33)  # make usecase column wider
        else:
            cell.set_width(0.12)

    tbl.scale(1.2, 1.9)  # Adjust column width and row height

    # Apply the custom styling
    style_table_cells(tbl, df_plot_ordered)

    ax.set_title("Use Case Performance Metrics", fontsize=16, fontweight="bold", pad=20)

    # --- Save and close the figure ---
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.debug(f"Metrics table saved to: {output_path}")


def evaluate_ce_detection_rules(
    triggers: torch.Tensor,  # bool/int tensor [L]
    all_required_labels: torch.Tensor,  # bool/int tensor [L]
    supporting_labels: torch.Tensor,  # bool/int tensor [L]
    any_of_conditions: dict[str, list[torch.Tensor]],  # {use_case: [LongTensor idx-group, ...]}
    use_case: str,
    labels_statistics: list[dict],
    idxs_to_labels: dict[int, str],
) -> None:
    """
    Policy:
      - All Required: triggered -> TP; missed -> FN
      - Supporting: triggered -> TP; missed -> TN
      - any_of (grouped):
          * per group:
              if any fired: fired -> TP; missed -> TN
              else: all in group -> FN
      - Irrelevant: triggered -> FP; missed -> TN
    """
    pred = triggers.bool()
    req = all_required_labels.bool()
    supp = supporting_labels.bool()

    L = pred.numel()

    def idxs_to_mask(idxs: torch.Tensor) -> torch.Tensor:
        m = torch.zeros(L, dtype=torch.bool, device=pred.device)
        if idxs.numel() > 0:
            m[idxs] = True
        return m

    # Build any_of groups as boolean masks (may be empty list)
    any_of_groups_idx = any_of_conditions.get(use_case, []) or []
    any_of_groups = [idxs_to_mask(g) for g in any_of_groups_idx]

    # Union of all any_of groups
    any_of_union = torch.zeros_like(pred, dtype=torch.bool)
    for g in any_of_groups:
        any_of_union |= g

    # --------- Disjointness checks (fail fast to prevent double counts) ----------
    # 1) No overlap between all_required/supporting and any any_of
    if (req & any_of_union).any() or (supp & any_of_union).any():
        raise ValueError(
            "A label appears in both any_of and all_required/supporting. Make sets disjoint or add precedence."
        )
    # 2) No overlap between all_required and supporting
    if (req & supp).any():
        raise ValueError(
            "A label is marked both all_required and supporting. Make sets disjoint or add precedence."
        )
    # 3) No overlap between any_of groups themselves
    if len(any_of_groups) > 1:
        stacked = torch.stack(any_of_groups, dim=0)  # [G, L]
        overlaps = stacked.sum(dim=0) > 1
        if overlaps.any():
            raise ValueError(
                "A label appears in multiple any_of groups. Make groups disjoint or add precedence."
            )

    allowed = req | supp | any_of_union
    irrelevant = ~allowed

    # ----- All Required -----
    tp_req = pred & req
    fn_req = (~pred) & req

    # ----- Supporting -----
    tp_supp = pred & supp
    tn_supp = (~pred) & supp

    # ----- any_of (per-group) -----
    tp_any = torch.zeros_like(pred)
    tn_any = torch.zeros_like(pred)
    fn_any = torch.zeros_like(pred)
    for g in any_of_groups:
        any_fired = bool((pred & g).any())
        if any_fired:
            tp_any |= pred & g
            tn_any |= (~pred) & g
        else:
            fn_any |= g  # whole group becomes FN

    # ----- Irrelevant -----
    fp_irr = pred & irrelevant
    tn_irr = (~pred) & irrelevant

    # Totals
    tp = tp_req | tp_supp | tp_any
    fp = fp_irr
    tn = tn_supp | tn_any | tn_irr
    fn = fn_req | fn_any

    # Partition sanity (each label in exactly one bucket)
    assert (tp | fp | tn | fn).all(), "Every label must land in a bucket."
    assert not ((tp & fp) | (tp & tn) | (tp & fn) | (fp & tn) | (fp & fn) | (tn & fn)).any(), (
        "Label landed in multiple buckets; check disjointness."
    )

    def _name(idx: int) -> str:
        return idxs_to_labels.get(idx, f"idx_{idx}")

    def _bump(mask: torch.Tensor, field: str) -> None:
        for i in mask.nonzero(as_tuple=True)[0].tolist():
            # labels_statistics[i][field] += 1
            label_name = _name(i)
            labels_statistics[label_name][field] += 1

    _bump(tp, "true_positive")
    _bump(fp, "false_positive")
    _bump(tn, "true_negative")
    _bump(fn, "false_negative")


def evaluate_ce_detection_gt(
    triggers: torch.Tensor,
    gt_one_hot: torch.Tensor,
    use_case: str,
    dialogue_name: str,
    label_logger: list[dict],
    label_stats: list[dict],
    idxs_to_labels: dict[int, str],
) -> None:
    """
    Policy (per label i):
      pred[i]=1, gt[i]=1 -> TP
      pred[i]=1, gt[i]=0 -> FP
      pred[i]=0, gt[i]=0 -> TN
      pred[i]=0, gt[i]=1 -> FN
    """

    if triggers.ndim != 1 or gt_one_hot.ndim != 1:
        raise ValueError("triggers and gt_one_hot must be 1D tensors.")
    if triggers.numel() != gt_one_hot.numel():
        raise ValueError(f"Length mismatch: pred={triggers.numel()} vs gt={gt_one_hot.numel()}")

    pred = triggers.to(dtype=torch.bool)
    gt = gt_one_hot.to(dtype=torch.bool)

    # Precompute helpful name lists for context columns
    def _name(idx: int) -> str:
        return idxs_to_labels.get(idx, f"idx_{idx}")

    predicted_label_names = "|".join(_name(i) for i in pred.nonzero(as_tuple=True)[0].tolist())
    gt_label_names = "|".join(_name(i) for i in gt.nonzero(as_tuple=True)[0].tolist())

    # Masks for the four buckets
    tp = pred & gt
    fp = pred & ~gt
    tn = ~pred & ~gt
    fn = ~pred & gt

    def _bump(mask: torch.Tensor, outcome: str) -> None:
        for i in mask.nonzero(as_tuple=True)[0].tolist():
            label_name = _name(i)
            label_logger.append(
                {
                    "dialogue": use_case,
                    "dialogue_number": dialogue_name,
                    "predicted_label_names": predicted_label_names,
                    "gt_label_names": gt_label_names,
                    "label": label_name,
                    "outcome": outcome,
                }
            )
            label_stats[label_name][outcome] += 1

    _bump(tp, "true_positive")
    _bump(fp, "false_positive")
    _bump(tn, "true_negative")

    _bump(fn, "false_negative")


def initialize_usecase_stats(mal_ruleset: dict, include_neutral: bool = True) -> Dict[str, dict]:
    """Initialize statistics dictionary for all use cases.

    Args:
        mal_ruleset: Malicious ruleset dictionary
        include_neutral: Whether to include neutral use cases (conversational, instructive)

    Returns:
        Dictionary of use case stats
    """
    use_cases_stats = {}

    if include_neutral:
        for uc in ["conversational", "instructive"]:
            use_cases_stats[uc] = {
                "true_positive": 0,
                "false_positive": 0,
                "true_negative": 0,
                "false_negative": 0,
            }

    for uc in mal_ruleset.keys():
        use_cases_stats[uc] = {
            "true_positive": 0,
            "false_positive": 0,
            "true_negative": 0,
            "false_negative": 0,
        }

    return use_cases_stats


def load_evaluation_setup(
    thresholds_path: str,
    unified_ruleset_path: str,
    labels: Dict[str, int],
) -> dict:
    """Load all evaluation setup data.

    Args:
        thresholds_path: Path to optimal thresholds JSON
        unified_ruleset_path: Path to unified ruleset JSON with enabled field
        labels: Label name to index mapping

    Returns:
        Dictionary with thresholds, rulesets, conditions, and stats initialized
    """
    num_topics = len(labels)
    idx_to_label = {v: k for k, v in labels.items()}

    # Load thresholds
    with open(thresholds_path, "r") as f:
        best_thr = json.load(f)

    topic_thresholds = torch.tensor(
        [best_thr[t]["threshold"] for t in idx_to_label.values()],
        dtype=torch.float32,
    )

    # Load unified ruleset and filter by enabled field
    with open(unified_ruleset_path, "r") as f:
        full_ruleset = json.load(f)

    # Filter to only enabled rulesets
    enabled_ruleset = {k: v for k, v in full_ruleset.items() if v.get("enabled", True)}

    ruleset_logic = build_ruleset_logic(enabled_ruleset, labels)

    return {
        "thresholds": topic_thresholds,
        "unified_ruleset": enabled_ruleset,
        "ruleset_logic": ruleset_logic,
        "idx_to_label": idx_to_label,
    }


def evaluate_usecase_detection_from_logits(
    logits_root: str,
    setup: dict,
    labels: Dict[str, int],
    show_progress: bool = True,
    logger=None,
) -> Dict[str, pd.DataFrame]:
    """Evaluate use case detection from saved logits.

    Args:
        logits_root: Root directory containing logits
        setup: Setup dict from load_evaluation_setup
        labels: Label name to index mapping
        show_progress: Show progress bar
        logger: Optional logger

    Returns:
        Dictionary with 'metrics' DataFrame and 'stats' dict
    """
    from tqdm import tqdm

    from gavel.utils.io import iter_dialogue_files

    num_topics = len(labels)

    # Initialize stats
    fpr_tpr_stats = initialize_usecase_stats(setup["unified_ruleset"], include_neutral=True)
    acc_stats = initialize_usecase_stats(setup["unified_ruleset"], include_neutral=False)

    # Process dialogues
    processed = 0
    iterator = iter_dialogue_files(logits_root)
    if show_progress:
        iterator = tqdm(iterator, desc="Evaluating", unit="dial")

    for meta_path, npy_path, meta in iterator:
        if meta.get("split", "") in ["usecase_level", "CE_level"]:
            continue

        logits_np = np.load(npy_path)
        dialogue_logits = torch.from_numpy(logits_np).float()

        split = meta.get("split", "")
        gt_uc_name = meta.get("usecase_path", "")

        # Compute triggers
        triggers = compute_triggers(
            dialogue_logits, thresholds=setup["thresholds"], patience_rate=1
        )

        # Update stats
        eval_usecase_detection(
            triggers=triggers,
            ruleset_logic=setup["ruleset_logic"],
            gt_uc_name=gt_uc_name,
            dialogue_name=meta.get("dialogue_id", meta_path.stem),
            split=split,
            use_cases_stats=fpr_tpr_stats,
        )
        update_usecase_confusion_matrix(
            triggers=triggers,
            ruleset_logic=setup["ruleset_logic"],
            gt_uc_name=gt_uc_name,
            split=split,
            use_cases_stats=acc_stats,
        )

        processed += 1

    if logger:
        logger.info(f"Processed {processed} dialogues")

    # Compute metrics
    metrics = compute_usecase_detection_metrics(fpr_tpr_stats)
    weighted_averages, df_weighted_averages = compute_weighted_metrics(acc_stats)

    return {
        "metrics": metrics,
        "fpr_tpr_stats": fpr_tpr_stats,
        "acc_stats": acc_stats,
        "weighted_averages": weighted_averages,
        "df_weighted_averages": df_weighted_averages,
        "processed": processed,
    }


def compute_usecase_auc(
    logits_root: str,
    setup: dict,
    labels: Dict[str, int],
    show_progress: bool = True,
) -> pd.DataFrame:
    """Compute AUC for each use case from saved logits.

    Args:
        logits_root: Root directory containing logits
        setup: Setup dict from load_evaluation_setup
        labels: Label name to index mapping
        show_progress: Show progress bar

    Returns:
        DataFrame with use_case, roc_auc, pr_auc columns
    """
    from sklearn.metrics import average_precision_score, roc_auc_score
    from tqdm import tqdm

    from gavel.utils.io import iter_dialogue_files

    usecases = list(setup["unified_ruleset"].keys())
    uc_true = {uc: [] for uc in usecases}
    uc_score = {uc: [] for uc in usecases}

    def max_over_windows_probs(logits_np):
        logits = torch.from_numpy(logits_np).float()
        probs = torch.sigmoid(logits)
        scores, _ = probs.max(dim=0)
        return scores

    def build_usecase_score(topic_scores, all_required_mask, any_of_groups):
        if int(all_required_mask.sum()) > 0:
            s_req = float(topic_scores[all_required_mask.bool()].min().item())
        else:
            s_req = 1.0

        if any_of_groups and len(any_of_groups) > 0:
            group_scores = []
            for g in any_of_groups:
                g = g.to(topic_scores.device)
                group_scores.append(0.0 if g.numel() == 0 else float(topic_scores[g].max().item()))
            s_any = min(group_scores)
        else:
            s_any = 1.0

        return min(s_req, s_any)

    iterator = iter_dialogue_files(logits_root)
    if show_progress:
        iterator = tqdm(iterator, desc="Computing AUC", unit="dial")

    for meta_path, npy_path, meta in iterator:
        split = meta.get("split", "")
        if split in ["usecase_level", "CE_level"]:
            continue
        uc_in_meta = meta.get("usecase_path", "")

        topic_scores = max_over_windows_probs(np.load(npy_path))

        for uc in usecases:
            s_uc = usecase_soft_score(setup["ruleset_logic"][uc], topic_scores)

            if split == "positive" and uc_in_meta == uc:
                uc_true[uc].append(1)
                uc_score[uc].append(s_uc)
            elif split == "negative" and uc_in_meta == uc:
                uc_true[uc].append(0)
                uc_score[uc].append(s_uc)

    # Compute AUC for each use case
    rows = []
    for uc in usecases:
        y_true = np.array(uc_true[uc], dtype=int)
        y_score = np.array(uc_score[uc], dtype=float)

        if len(y_true) == 0 or len(np.unique(y_true)) < 2:
            rows.append({"Usecase": uc, "ROC_AUC": None, "PR_AUC": None})
            continue

        try:
            roc_auc = roc_auc_score(y_true, y_score)
        except Exception:
            roc_auc = None
        try:
            pr_auc = average_precision_score(y_true, y_score)
        except Exception:
            pr_auc = None

        rows.append({"Usecase": uc, "ROC_AUC": roc_auc, "PR_AUC": pr_auc})

    return pd.DataFrame(rows)


def evaluate(
    output_dir: str,
    labels: Dict[str, int],
    thresholds_path: str,
    unified_ruleset_path: str,
    dialogue_data: Optional[List[Dict]] = None,
    logits_root: Optional[str] = None,
    compute_auc: bool = True,
    show_progress: bool = True,
    logger=None,
) -> dict:
    """Run full evaluation pipeline.

    Accepts either in-memory dialogue data OR a path to saved logits on disk.
    Exactly one of `dialogue_data` or `logits_root` must be provided.

    Args:
        output_dir: Output directory for results
        labels: Label name to index mapping
        thresholds_path: Path to optimal thresholds JSON
        unified_ruleset_path: Path to unified ruleset JSON
        dialogue_data: List of dicts from extract_dialogues_in_memory (in-memory mode)
        logits_root: Root directory containing saved logits (disk mode)
        compute_auc: Whether to compute AUC metrics
        show_progress: Show progress bar
        logger: Optional logger

    Returns:
        Dictionary with evaluation results

    Raises:
        ValueError: If neither or both of dialogue_data/logits_root are provided
    """
    from pathlib import Path

    # Validate inputs
    if dialogue_data is None and logits_root is None:
        raise ValueError("Must provide either dialogue_data or logits_root")
    if dialogue_data is not None and logits_root is not None:
        raise ValueError("Provide only one of dialogue_data or logits_root, not both")

    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir)

    num_topics = len(labels)

    # Load setup
    if logger:
        logger.info("Loading evaluation setup...")
    setup = load_evaluation_setup(thresholds_path, unified_ruleset_path, labels)

    if logits_root is not None:
        # Disk-based mode: use existing logits loader function
        if logger:
            logger.info("Evaluating dialogues from disk...")
        results = evaluate_usecase_detection_from_logits(
            logits_root, setup, labels, show_progress, logger
        )

        # Compute AUC for disk mode
        if compute_auc:
            if logger:
                logger.info("Computing AUC metrics...")
            auc_df = compute_usecase_auc(logits_root, setup, labels, show_progress)
            auc_df.to_csv(output_path / "usecase_auc.csv", index=False)
            results["auc"] = auc_df
    else:
        # In-memory mode
        fpr_tpr_stats = initialize_usecase_stats(setup["unified_ruleset"], include_neutral=True)
        acc_stats = initialize_usecase_stats(setup["unified_ruleset"], include_neutral=False)

        processed = 0

        if logger:
            logger.info(f"Evaluating {len(dialogue_data)} dialogues...")

        for dialogue in dialogue_data:
            meta = dialogue["metadata"]
            split = meta.get("split", "")

            # Skip calibration splits
            if split in ["usecase_level", "CE_level"]:
                continue

            # Handle logits conversion
            logits = dialogue["logits"]
            if isinstance(logits, np.ndarray):
                dialogue_logits = torch.from_numpy(logits).float()
            else:
                dialogue_logits = logits.float()

            gt_uc_name = meta.get("usecase_path", "")
            dialogue_id = meta.get("dialogue_id", "")

            # Unknown (non-neutral) use cases can't be scored — skip.
            if (
                gt_uc_name not in ["conversational", "instructive"]
                and gt_uc_name not in setup["ruleset_logic"]
            ):
                if logger:
                    logger.warning(f"Unknown use case '{gt_uc_name}' - skipping")
                continue

            # Compute triggers
            triggers = compute_triggers(
                dialogue_logits, thresholds=setup["thresholds"], patience_rate=1
            )

            # Update stats
            eval_usecase_detection(
                triggers=triggers,
                ruleset_logic=setup["ruleset_logic"],
                gt_uc_name=gt_uc_name,
                dialogue_name=dialogue_id,
                split=split,
                use_cases_stats=fpr_tpr_stats,
            )
            update_usecase_confusion_matrix(
                triggers=triggers,
                ruleset_logic=setup["ruleset_logic"],
                gt_uc_name=gt_uc_name,
                split=split,
                use_cases_stats=acc_stats,
            )

            processed += 1

        if logger:
            logger.info(f"Processed {processed} dialogues")

        # Compute metrics
        metrics = compute_usecase_detection_metrics(fpr_tpr_stats)
        weighted_averages, df_weighted_averages = compute_weighted_metrics(acc_stats)

        results = {
            "metrics": metrics,
            "fpr_tpr_stats": fpr_tpr_stats,
            "acc_stats": acc_stats,
            "weighted_averages": weighted_averages,
            "df_weighted_averages": df_weighted_averages,
            "processed": processed,
        }

        # Compute AUC for in-memory mode
        if compute_auc:
            if logger:
                logger.info("Computing AUC metrics...")
            auc_df = compute_auc_from_dialogues(dialogue_data, setup, labels, logger is not None)
            auc_df.to_csv(output_path / "usecase_auc.csv", index=False)
            results["auc"] = auc_df

    # Save metrics (common to both modes)
    results["metrics"].to_csv(output_path / "usecase_metrics_fprtpr.csv", index=False)
    results["df_weighted_averages"].to_csv(
        output_path / "usecase_weighted_averages.csv", index=True
    )
    plot_usecase_metrics_table(results["metrics"], output_path / "usecase_fprtpr.png")

    if logger:
        logger.info(f"Results saved to: {output_dir}")

    return results


# Legacy alias for backward compatibility
def evaluate_from_logits(
    logits_root: str,
    output_dir: str,
    labels: Dict[str, int],
    thresholds_path: str,
    unified_ruleset_path: str = "rulesets/unified_ruleset.json",
    compute_auc: bool = True,
    show_progress: bool = True,
    logger=None,
) -> dict:
    """Legacy wrapper for evaluate(). Use evaluate() instead."""
    return evaluate(
        output_dir=output_dir,
        labels=labels,
        thresholds_path=thresholds_path,
        unified_ruleset_path=unified_ruleset_path,
        logits_root=logits_root,
        compute_auc=compute_auc,
        show_progress=show_progress,
        logger=logger,
    )


def compute_auc_from_dialogues(
    dialogue_data: List[Dict],
    setup: dict,
    labels: Dict[str, int],
    show_progress: bool = True,
) -> pd.DataFrame:
    """Compute AUC for each use case from in-memory dialogues.

    Args:
        dialogue_data: List of dicts from extract_dialogues_in_memory
        setup: Setup dict from load_evaluation_setup
        labels: Label name to index mapping
        show_progress: Show progress bar

    Returns:
        DataFrame with use_case, roc_auc, pr_auc columns
    """
    from sklearn.metrics import average_precision_score, roc_auc_score
    from tqdm import tqdm

    usecases = list(setup["unified_ruleset"].keys())
    uc_true = {uc: [] for uc in usecases}
    uc_score = {uc: [] for uc in usecases}

    def max_over_windows_probs(logits_np):
        logits = torch.from_numpy(logits_np).float()
        probs = torch.sigmoid(logits)
        scores, _ = probs.max(dim=0)
        return scores

    def build_usecase_score(topic_scores, all_required_mask, any_of_groups):
        if int(all_required_mask.sum()) > 0:
            s_req = float(topic_scores[all_required_mask.bool()].min().item())
        else:
            s_req = 1.0

        if any_of_groups and len(any_of_groups) > 0:
            group_scores = []
            for g in any_of_groups:
                g = g.to(topic_scores.device)
                group_scores.append(0.0 if g.numel() == 0 else float(topic_scores[g].max().item()))
            s_any = min(group_scores)
        else:
            s_any = 1.0

        return min(s_req, s_any)

    iterator = dialogue_data
    if show_progress:
        iterator = tqdm(dialogue_data, desc="Computing AUC", unit="dial")

    for dialogue in iterator:
        meta = dialogue["metadata"]
        split = meta.get("split", "")

        # Skip calibration splits
        if split in ["usecase_level", "CE_level"]:
            continue

        uc_in_meta = meta.get("usecase_path", "")
        topic_scores = max_over_windows_probs(dialogue["logits"])

        for uc in usecases:
            s_uc = usecase_soft_score(setup["ruleset_logic"][uc], topic_scores)

            if split == "positive" and uc_in_meta == uc:
                uc_true[uc].append(1)
                uc_score[uc].append(s_uc)
            elif split == "negative" and uc_in_meta == uc:
                uc_true[uc].append(0)
                uc_score[uc].append(s_uc)

    # Compute AUC for each use case
    rows = []
    for uc in usecases:
        y_true = np.array(uc_true[uc], dtype=int)
        y_score = np.array(uc_score[uc], dtype=float)

        if len(y_true) == 0 or len(np.unique(y_true)) < 2:
            rows.append({"Usecase": uc, "ROC_AUC": None, "PR_AUC": None})
            continue

        try:
            roc_auc = roc_auc_score(y_true, y_score)
        except Exception:
            roc_auc = None
        try:
            pr_auc = average_precision_score(y_true, y_score)
        except Exception:
            pr_auc = None

        rows.append({"Usecase": uc, "ROC_AUC": roc_auc, "PR_AUC": pr_auc})

    return pd.DataFrame(rows)
