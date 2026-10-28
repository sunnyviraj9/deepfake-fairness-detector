"""
Fairness and Performance Metrics Module for Deepfake Detection.
Implements CVPR 2025 AI-Face Fairness Benchmark principles, Monk Skin Tone (MST 1-10) evaluation,
and Equalized Odds Disparity (F_EO) calculations.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    target_fpr: float = 0.05,
) -> Dict[str, float]:
    """
    Finds the optimal decision threshold that constrains the False Positive Rate (FPR)
    to at most ``target_fpr`` while maximising True Positive Rate (TPR).

    Strategy
    --------
    1. Walk the ROC curve and collect every threshold whose FPR ≤ target_fpr.
    2. Among those candidates, pick the one with the highest TPR  — i.e. the
       operating point that is *most permissive for fakes* while still respecting
       the real-image false-alarm budget.
    3. If **no** threshold achieves FPR ≤ target_fpr (can happen when the model
       is poorly calibrated), fall back to the Equal Error Rate threshold so the
       result is always well-defined.

    Parameters
    ----------
    y_true : array-like of int
        Ground-truth binary labels  (0 = Real, 1 = Fake).
    y_probs : array-like of float
        Predicted probabilities for the positive (Fake) class.
    target_fpr : float, optional
        Maximum tolerated FPR on real images.  Defaults to 0.05 (5 %).

    Returns
    -------
    dict with keys:
        ``threshold``   – chosen decision threshold (float)
        ``fpr``         – actual FPR achieved at that threshold
        ``tpr``         – TPR / recall achieved at that threshold
        ``method``      – ``"fpr_constrained"`` or ``"eer_fallback"``
        ``eer_threshold`` – EER threshold for reference
        ``eer``         – Equal Error Rate value
    """
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)

    if len(np.unique(y_true)) < 2:
        # Edge-case: only one class present — return neutral defaults
        return {
            "threshold": 0.5,
            "fpr": 0.0,
            "tpr": 1.0,
            "method": "single_class_fallback",
            "eer_threshold": 0.5,
            "eer": 0.0,
        }

    fpr_curve, tpr_curve, thresholds = roc_curve(y_true, y_probs, pos_label=1)
    eer, eer_threshold = compute_equal_error_rate(y_true, y_probs)

    # Candidates: all ROC points where FPR is within budget
    # roc_curve returns arrays in *ascending* threshold order (sklearn convention),
    # so we simply mask and select the best TPR among eligible points.
    eligible_mask = fpr_curve <= target_fpr
    if np.any(eligible_mask):
        eligible_tprs = tpr_curve[eligible_mask]
        eligible_fprs = fpr_curve[eligible_mask]
        eligible_thresholds = thresholds[eligible_mask]

        # Best = highest TPR among eligible (ties broken by highest threshold → fewer FPs)
        best_idx = np.argmax(eligible_tprs)
        chosen_threshold = float(eligible_thresholds[best_idx])
        chosen_fpr = float(eligible_fprs[best_idx])
        chosen_tpr = float(eligible_tprs[best_idx])
        method = "fpr_constrained"
    else:
        # Fallback: use EER threshold — balanced FPR/FNR, no budget respected
        chosen_threshold = float(eer_threshold)
        # Find the ROC point closest to the EER threshold
        eer_idx = np.nanargmin(np.abs(thresholds - eer_threshold))
        chosen_fpr = float(fpr_curve[eer_idx])
        chosen_tpr = float(tpr_curve[eer_idx])
        method = "eer_fallback"

    return {
        "threshold": round(chosen_threshold, 6),
        "fpr": round(chosen_fpr, 4),
        "tpr": round(chosen_tpr, 4),
        "method": method,
        "eer_threshold": round(float(eer_threshold), 6),
        "eer": round(float(eer), 4),
    }


def compute_equal_error_rate(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """
    Computes the Equal Error Rate (EER) where FPR == FNR, and corresponding threshold.
    """
    if len(np.unique(y_true)) < 2:
        return 0.0, 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr
    eer_index = np.nanargmin(np.absolute(fnr - fpr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2.0
    eer_threshold = thresholds[eer_index]
    return float(eer), float(eer_threshold)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Computes comprehensive binary classification metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    n_samples = len(y_true)
    if n_samples == 0:
        return {
            "accuracy": 0.0,
            "auc": 0.0,
            "precision": 0.0,
            "recall_tpr": 0.0,
            "fpr": 0.0,
            "fnr": 0.0,
            "tnr": 0.0,
            "f1": 0.0,
            "eer": 0.0,
            "support": 0,
            "pos_support": 0,
            "neg_support": 0,
        }

    pos_support = int(np.sum(y_true == 1))
    neg_support = int(np.sum(y_true == 0))

    # Confusion matrix: tn, fp, fn, tp
    if len(np.unique(y_true)) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = 0.5
        eer, _ = compute_equal_error_rate(y_true, y_prob)
    else:
        # Only one class present in subgroup
        if pos_support > 0:
            tp = int(np.sum(y_pred == 1))
            fn = int(np.sum(y_pred == 0))
            tn, fp = 0, 0
        else:
            tn = int(np.sum(y_pred == 0))
            fp = int(np.sum(y_pred == 1))
            tp, fn = 0, 0
        auc = 0.5
        eer = 0.5

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    tnr = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        "accuracy": round(accuracy, 4),
        "auc": round(auc, 4),
        "precision": round(precision, 4),
        "recall_tpr": round(recall, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "tnr": round(tnr, 4),
        "f1": round(f1, 4),
        "eer": round(eer, 4),
        "support": int(n_samples),
        "pos_support": pos_support,
        "neg_support": neg_support,
    }


def evaluate_fairness_disparities(subgroup_metrics_df: pd.DataFrame) -> Dict[str, float]:
    """
    Computes algorithmic fairness disparity metrics across demographic subgroups:
    - Delta FPR: max(FPR) - min(FPR)
    - Delta TPR: max(TPR) - min(TPR)
    - Equalized Odds Disparity (F_EO): 0.5 * (Delta FPR + Delta TPR)
    - Max Equalized Odds Gap: max(Delta FPR, Delta TPR)
    - Delta Accuracy: max(Acc) - min(Acc)
    - Delta AUC: max(AUC) - min(AUC)
    - Demographic Parity Disparity: max(Positive Selection Rate) - min(Positive Selection Rate)
    """
    if subgroup_metrics_df.empty or len(subgroup_metrics_df) < 2:
        return {
            "delta_fpr": 0.0,
            "delta_tpr": 0.0,
            "F_EO": 0.0,
            "max_eo_gap": 0.0,
            "delta_accuracy": 0.0,
            "delta_auc": 0.0,
        }

    fpr_vals = subgroup_metrics_df["fpr"].values
    tpr_vals = subgroup_metrics_df["recall_tpr"].values
    acc_vals = subgroup_metrics_df["accuracy"].values
    auc_vals = subgroup_metrics_df["auc"].values

    delta_fpr = float(np.max(fpr_vals) - np.min(fpr_vals))
    delta_tpr = float(np.max(tpr_vals) - np.min(tpr_vals))
    f_eo = float(0.5 * (delta_fpr + delta_tpr))
    max_eo_gap = float(max(delta_fpr, delta_tpr))
    delta_acc = float(np.max(acc_vals) - np.min(acc_vals))
    delta_auc = float(np.max(auc_vals) - np.min(auc_vals))

    return {
        "delta_fpr": round(delta_fpr, 4),
        "delta_tpr": round(delta_tpr, 4),
        "F_EO": round(f_eo, 4),
        "max_eo_gap": round(max_eo_gap, 4),
        "delta_accuracy": round(delta_acc, 4),
        "delta_auc": round(delta_auc, 4),
    }


class FairnessAuditor:
    """
    Audits deepfake detector predictions across demographic axes (Monk Scale 1-10, Gender, Intersectional).
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def audit(
        self,
        y_true: Union[List, np.ndarray],
        y_prob: Union[List, np.ndarray],
        demographics_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Executes a comprehensive fairness audit.

        Args:
            y_true: 1D array of ground truth labels (0=Real, 1=Fake).
            y_prob: 1D array of predicted probabilities for class 1 (Fake).
            demographics_df: DataFrame containing metadata columns ('skin_tone', 'gender', etc.).

        Returns:
            Dictionary containing overall metrics, subgroup breakdown tables, and fairness disparity scores.
        """
        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)

        df = demographics_df.copy().reset_index(drop=True)
        df["y_true"] = y_true
        df["y_prob"] = y_prob
        df["y_pred"] = (y_prob >= self.threshold).astype(int)

        # 1. Overall Performance Metrics
        overall_metrics = compute_binary_metrics(y_true, y_prob, threshold=self.threshold)

        # Helper to compute subgroup breakdown table
        def _get_breakdown(group_col: str) -> pd.DataFrame:
            records = []
            for grp_val, grp_data in df.groupby(group_col):
                m = compute_binary_metrics(
                    grp_data["y_true"].values,
                    grp_data["y_prob"].values,
                    threshold=self.threshold,
                )
                m[group_col] = grp_val
                records.append(m)
            out_df = pd.DataFrame(records)
            if not out_df.empty:
                cols = [group_col] + [c for c in out_df.columns if c != group_col]
                out_df = out_df[cols]
            return out_df

        # 2. Monk Skin Tone (MST 1-10) Breakdown
        mst_breakdown = _get_breakdown("skin_tone") if "skin_tone" in df.columns else pd.DataFrame()
        mst_fairness = evaluate_fairness_disparities(mst_breakdown) if not mst_breakdown.empty else {}

        # 3. Monk Skin Tone Group (Light 1-3, Medium 4-7, Dark 8-10) Breakdown
        if "skin_tone_group" not in df.columns and "skin_tone" in df.columns:
            from utils.dataset_loader import MST_GROUPS
            df["skin_tone_group"] = df["skin_tone"].map(lambda st: MST_GROUPS.get(st, "Unknown"))
        mst_group_breakdown = (
            _get_breakdown("skin_tone_group") if "skin_tone_group" in df.columns else pd.DataFrame()
        )
        mst_group_fairness = (
            evaluate_fairness_disparities(mst_group_breakdown) if not mst_group_breakdown.empty else {}
        )

        # 4. Gender Breakdown
        gender_breakdown = _get_breakdown("gender") if "gender" in df.columns else pd.DataFrame()
        gender_fairness = (
            evaluate_fairness_disparities(gender_breakdown) if not gender_breakdown.empty else {}
        )

        # 5. Intersectional Subgroup Breakdown (Gender + MST)
        if "subgroup" not in df.columns and "gender" in df.columns and "skin_tone" in df.columns:
            df["subgroup"] = df["gender"].astype(str) + "_MST_" + df["skin_tone"].astype(str)
        intersectional_breakdown = (
            _get_breakdown("subgroup") if "subgroup" in df.columns else pd.DataFrame()
        )
        intersectional_fairness = (
            evaluate_fairness_disparities(intersectional_breakdown)
            if not intersectional_breakdown.empty
            else {}
        )

        # Consolidated Summary
        audit_results = {
            "overall_metrics": overall_metrics,
            "mst_breakdown": mst_breakdown,
            "mst_fairness_disparities": mst_fairness,
            "mst_group_breakdown": mst_group_breakdown,
            "mst_group_fairness_disparities": mst_group_fairness,
            "gender_breakdown": gender_breakdown,
            "gender_fairness_disparities": gender_fairness,
            "intersectional_breakdown": intersectional_breakdown,
            "intersectional_fairness_disparities": intersectional_fairness,
        }

        return audit_results

    def generate_markdown_report(self, audit_results: Dict[str, Any]) -> str:
        """
        Formats the audit results into a readable Markdown report.
        """
        lines = []
        lines.append(f"# Deepfake Detection & Algorithmic Fairness Audit Report\n")
        lines.append("## 1. Overall Performance")
        lines.append("| Metric | Value |")
        lines.append("| :--- | :--- |")
        for k, v in audit_results["overall_metrics"].items():
            lines.append(f"| **{k.upper()}** | {v} |")
        lines.append("")

        lines.append("## 2. Monk Skin Tone (MST 1-10) Fairness Evaluation")
        f_mst = audit_results.get("mst_fairness_disparities", {})
        lines.append(f"- **Equalized Odds Disparity ($F_{{EO}}$)**: `{f_mst.get('F_EO', 'N/A')}`")
        lines.append(fr"- **Max EO Gap ($\Delta EO_{{max}}$)**: `{f_mst.get('max_eo_gap', 'N/A')}`")
        lines.append(fr"- **FPR Disparity ($\Delta FPR$)**: `{f_mst.get('delta_fpr', 'N/A')}`")
        lines.append(fr"- **TPR Disparity ($\Delta TPR$)**: `{f_mst.get('delta_tpr', 'N/A')}`")
        lines.append(fr"- **AUC Disparity ($\Delta AUC$)**: `{f_mst.get('delta_auc', 'N/A')}`")
        lines.append("")

        mst_df = audit_results.get("mst_breakdown", pd.DataFrame())
        if not mst_df.empty:
            lines.append("### Subgroup Table (Monk Scale 1-10)")
            lines.append(mst_df.to_markdown(index=False))
            lines.append("")

        lines.append("## 3. Gender Fairness Evaluation")
        f_gen = audit_results.get("gender_fairness_disparities", {})
        lines.append(f"- **Equalized Odds Disparity ($F_{{EO}}$)**: `{f_gen.get('F_EO', 'N/A')}`")
        lines.append(fr"- **FPR Disparity ($\Delta FPR$)**: `{f_gen.get('delta_fpr', 'N/A')}`")
        lines.append(fr"- **TPR Disparity ($\Delta TPR$)**: `{f_gen.get('delta_tpr', 'N/A')}`")
        lines.append("")

        gen_df = audit_results.get("gender_breakdown", pd.DataFrame())
        if not gen_df.empty:
            lines.append("### Subgroup Table (Gender)")
            lines.append(gen_df.to_markdown(index=False))
            lines.append("")

        return "\n".join(lines)
