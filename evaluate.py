"""
Evaluation and Algorithmic Fairness Audit Pipeline for Deepfake Detection.
Produces subgroup breakdowns across Monk Skin Tone (1-10) and Gender, computes Equalized Odds (F_EO),
and exports results to evaluation_summary.csv and fairness_audit_report.md.
"""

import argparse
import os
from typing import Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import build_model
from utils.dataset_loader import DeepfakeFairnessDataset, get_default_transforms
from utils.metrics import FairnessAuditor, compute_binary_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Deepfake Detector with Algorithmic Fairness Audit.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--annotations", type=str, default="./data/mock_dataset/annotations.csv", help="Path to evaluation annotations CSV")
    parser.add_argument("--data_root", type=str, default=None, help="Base directory for image files")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--image_size", type=int, default=224, help="Image resolution")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for binary classification")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save evaluation results and plots")
    parser.add_argument("--plot", action="store_true", default=True, help="Generate and save fairness diagnostic plots")
    return parser.parse_args()


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict:
    """Runs inference across the entire dataset and collects predictions & demographic metadata."""
    model.eval()
    all_probs = []
    all_targets = []
    all_meta = []

    for images, targets, metadata in tqdm(loader, desc="Evaluating Dataset"):
        images = images.to(device)
        logits = model(images).view(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
        targets_np = targets.numpy()

        all_probs.extend(probs)
        all_targets.extend(targets_np)

        batch_len = len(targets_np)
        for i in range(batch_len):
            all_meta.append({
                "image_path": metadata["image_path"][i],
                "skin_tone": int(metadata["skin_tone"][i]),
                "gender": str(metadata["gender"][i]),
                "skin_tone_group": str(metadata["skin_tone_group"][i]),
                "subgroup": str(metadata["subgroup"][i]),
            })

    meta_df = pd.DataFrame(all_meta)
    auditor = FairnessAuditor(threshold=threshold)
    audit_results = auditor.audit(all_targets, all_probs, meta_df)
    audit_results["predictions_df"] = meta_df.assign(y_true=all_targets, y_prob=all_probs)
    return audit_results


def generate_evaluation_plots(audit_results: Dict, output_path: str):
    """Generates visual breakdown plots of subgroup FPR and TPR across Monk Skin Tone buckets."""
    mst_df = audit_results.get("mst_breakdown", pd.DataFrame())
    if mst_df.empty or "skin_tone" not in mst_df.columns:
        return

    plt.figure(figsize=(12, 5), dpi=150)

    # Subplot 1: Subgroup FPR across Monk Skin Tones
    plt.subplot(1, 2, 1)
    tones = mst_df["skin_tone"].values
    fprs = mst_df["fpr"].values
    bars = plt.bar(tones, fprs, color="#e74c3c", alpha=0.85, edgecolor="black")
    plt.axhline(np.mean(fprs), color="black", linestyle="--", label=f"Mean FPR ({np.mean(fprs):.3f})")
    plt.xlabel("Monk Skin Tone Scale (1 to 10)", fontsize=11, fontweight="bold")
    plt.ylabel("False Positive Rate (FPR)", fontsize=11, fontweight="bold")
    plt.title("Subgroup False Positive Rate (FPR) Disparity", fontsize=12, fontweight="bold")
    plt.xticks(tones)
    plt.ylim(0, max(max(fprs) * 1.25, 0.1))
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.legend()

    # Subplot 2: Subgroup TPR (Recall) across Monk Skin Tones
    plt.subplot(1, 2, 2)
    tprs = mst_df["recall_tpr"].values
    plt.bar(tones, tprs, color="#2ecc71", alpha=0.85, edgecolor="black")
    plt.axhline(np.mean(tprs), color="black", linestyle="--", label=f"Mean TPR ({np.mean(tprs):.3f})")
    plt.xlabel("Monk Skin Tone Scale (1 to 10)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (TPR / Recall)", fontsize=11, fontweight="bold")
    plt.title("Subgroup True Positive Rate (TPR) Disparity", fontsize=12, fontweight="bold")
    plt.xticks(tones)
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved fairness diagnostic plot to: {output_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"==================================================")
    print(f"Deepfake Detector Fairness & Performance Evaluation")
    print(f"Checkpoint: {args.model_path}")
    print(f"Annotations: {args.annotations}")
    print(f"==================================================")

    # 1. Load Checkpoint & Instantiate Model
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Checkpoint file not found: {args.model_path}")

    checkpoint = torch.load(args.model_path, map_location=device)
    model_type = checkpoint.get("model_type", "spatial")
    backbone = checkpoint.get("backbone", "efficientnet_b0")

    model = build_model(
        model_type=model_type,
        backbone=backbone,
        pretrained=False,
        num_classes=1,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Successfully loaded {model_type} model ({backbone}) from epoch {checkpoint.get('epoch', 'N/A')}.")

    # 2. Build Evaluation DataLoader
    transform = get_default_transforms(split="val", image_size=args.image_size)
    dataset = DeepfakeFairnessDataset(
        annotations=args.annotations,
        root_dir=args.data_root,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # 3. Perform Full Fairness Audit
    audit_results = evaluate_model(model, loader, device, threshold=args.threshold)
    auditor = FairnessAuditor(threshold=args.threshold)

    # 4. Print Summary to Console
    print("\n--- Overall Performance Metrics ---")
    for k, v in audit_results["overall_metrics"].items():
        print(f"  {k:15s}: {v}")

    print("\n--- Monk Skin Tone (MST 1-10) Disparities ---")
    for k, v in audit_results["mst_fairness_disparities"].items():
        print(f"  {k:15s}: {v}")

    print("\n--- Gender Disparities ---")
    for k, v in audit_results["gender_fairness_disparities"].items():
        print(f"  {k:15s}: {v}")

    # 5. Export Summary CSV
    summary_rows = []
    # Overall row
    overall_dict = audit_results["overall_metrics"].copy()
    overall_dict["category"] = "Overall"
    overall_dict["group_name"] = "All"
    summary_rows.append(overall_dict)

    # Monk tone rows
    for _, row in audit_results["mst_breakdown"].iterrows():
        r = row.to_dict()
        r["category"] = "Monk_Scale"
        r["group_name"] = f"MST_{r.get('skin_tone', '')}"
        summary_rows.append(r)

    # Gender rows
    for _, row in audit_results["gender_breakdown"].iterrows():
        r = row.to_dict()
        r["category"] = "Gender"
        r["group_name"] = str(r.get("gender", ""))
        summary_rows.append(r)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(args.output_dir, "evaluation_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSaved structured evaluation summary table to: {summary_csv_path}")

    # 6. Export Markdown Audit Report
    report_md = auditor.generate_markdown_report(audit_results)
    report_path = os.path.join(args.output_dir, "fairness_audit_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Saved detailed fairness markdown report to: {report_path}")

    # 7. Generate diagnostic plots if requested
    if args.plot:
        plot_path = os.path.join(args.output_dir, "fairness_disparity_plots.png")
        generate_evaluation_plots(audit_results, plot_path)


if __name__ == "__main__":
    main()
