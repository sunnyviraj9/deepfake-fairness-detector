import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from utils.metrics import (
    FairnessAuditor,
    compute_binary_metrics,
    compute_equal_error_rate,
    evaluate_fairness_disparities,
)


def test_compute_binary_metrics_perfect_case():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_binary_metrics(y_true, y_prob, threshold=0.5)

    assert metrics["accuracy"] == 1.0
    assert metrics["auc"] == 1.0
    assert metrics["fpr"] == 0.0
    assert metrics["fnr"] == 0.0
    assert metrics["recall_tpr"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["f1"] == 1.0


def test_fairness_disparity_calculation():
    # Construct mock subgroup dataframe
    # Group A: FPR = 0.1, TPR = 0.9
    # Group B: FPR = 0.3, TPR = 0.7
    subgroups = pd.DataFrame([
        {"skin_tone": 1, "fpr": 0.1, "recall_tpr": 0.9, "accuracy": 0.9, "auc": 0.95},
        {"skin_tone": 2, "fpr": 0.3, "recall_tpr": 0.7, "accuracy": 0.7, "auc": 0.75},
    ])

    disparities = evaluate_fairness_disparities(subgroups)

    # Delta FPR = 0.3 - 0.1 = 0.2
    # Delta TPR = 0.9 - 0.7 = 0.2
    # F_EO = 0.5 * (0.2 + 0.2) = 0.2
    assert disparities["delta_fpr"] == 0.2
    assert disparities["delta_tpr"] == 0.2
    assert disparities["F_EO"] == 0.2
    assert disparities["max_eo_gap"] == 0.2


def test_fairness_auditor_full_run():
    # 20 samples: 10 light skin tones (MST 1-3), 10 dark skin tones (MST 8-10)
    y_true = np.array([0, 1] * 10)
    # Give higher false positives to dark skin tones intentionally to test detection of bias
    y_prob = np.array(
        [0.1, 0.9] * 5 +  # Group 1: perfect predictions
        [0.7, 0.9] * 5    # Group 2: high false positives on real (y_true=0)
    )
    skin_tones = [2] * 10 + [9] * 10
    genders = ["Female"] * 10 + ["Male"] * 10

    demographics_df = pd.DataFrame({
        "skin_tone": skin_tones,
        "gender": genders,
    })

    auditor = FairnessAuditor(threshold=0.5)
    audit = auditor.audit(y_true, y_prob, demographics_df)

    assert "overall_metrics" in audit
    assert "mst_breakdown" in audit
    assert "mst_fairness_disparities" in audit

    # Dark skin tone (MST 9) should have FPR = 1.0, Light skin tone (MST 2) should have FPR = 0.0
    mst_df = audit["mst_breakdown"]
    assert len(mst_df) == 2
    assert audit["mst_fairness_disparities"]["delta_fpr"] == 1.0
    assert audit["mst_fairness_disparities"]["F_EO"] == 0.5


if __name__ == "__main__":
    test_compute_binary_metrics_perfect_case()
    test_fairness_disparity_calculation()
    test_fairness_auditor_full_run()
    print("All metrics unit tests passed successfully!")
