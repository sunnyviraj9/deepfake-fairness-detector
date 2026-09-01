"""
Multi-Model Benchmark Script for Deepfake Detection.
Trains and evaluates Spatial, Frequency, and Dual-Stream models and produces a comparative report.

Usage:
    python benchmark.py --annotations ./data/mock_dataset/annotations.csv --epochs 5 --output_dir ./benchmark_results
"""

import argparse
import os
import time
import json
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from models import build_model
from utils.dataset_loader import create_dataloaders
from utils.metrics import FairnessAuditor, compute_binary_metrics


BENCHMARK_CONFIGS = [
    {"model_type": "spatial",     "backbone": "efficientnet_b0", "label": "EfficientNet-B0 (Spatial)"},
    {"model_type": "spatial",     "backbone": "resnet50",        "label": "ResNet-50 (Spatial)"},
    {"model_type": "frequency",   "backbone": "resnet18",        "label": "ResNet-18 (Frequency)"},
    {"model_type": "dual_stream", "backbone": "efficientnet_b0", "label": "EfficientNet-B0 (Dual-Stream)"},
]


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Model Deepfake Benchmark")
    parser.add_argument("--annotations", type=str, default="./data/mock_dataset/annotations.csv")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--output_dir", type=str, default="./benchmark_results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated list of models to benchmark: spatial,frequency,dual_stream")
    return parser.parse_args()


def train_and_evaluate(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> Dict:
    """Trains a model and returns validation metrics."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_auc = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        # ----- Train -----
        model.train()
        train_loss = 0.0
        n = 0
        for images, targets, _ in tqdm(train_loader, desc=f"  Epoch {epoch}/{epochs}", leave=False):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(images).view(-1)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            n += images.size(0)
        scheduler.step()
        train_loss /= max(n, 1)

        # ----- Validate -----
        model.eval()
        all_probs, all_targets = [], []
        all_meta = []
        with torch.no_grad():
            for images, targets, meta in val_loader:
                images = images.to(device)
                probs = torch.sigmoid(model(images).view(-1)).cpu().numpy()
                all_probs.extend(probs)
                all_targets.extend(targets.numpy())
                for i in range(len(targets)):
                    all_meta.append({
                        "skin_tone": int(meta["skin_tone"][i]),
                        "gender": str(meta["gender"][i]),
                        "skin_tone_group": str(meta["skin_tone_group"][i]),
                        "subgroup": str(meta["subgroup"][i]),
                    })

        meta_df = pd.DataFrame(all_meta)
        auditor = FairnessAuditor(threshold=0.5)
        audit = auditor.audit(all_targets, all_probs, meta_df)
        metrics = audit["overall_metrics"]
        f_eo = audit["mst_fairness_disparities"].get("F_EO", 0.0)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_auc": metrics["auc"],
            "val_acc": metrics["accuracy"],
            "F_EO": f_eo,
        })

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]

    return {
        "best_auc": best_auc,
        "final_metrics": metrics,
        "final_f_eo": f_eo,
        "history": history,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"{'=' * 60}")
    print(f"  Deepfake Detection Multi-Model Benchmark")
    print(f"  Device: {device}")
    print(f"{'=' * 60}")

    # Load data once
    train_loader, val_loader, test_loader, full_df = create_dataloaders(
        annotations_path=args.annotations,
        root_dir=args.data_root,
        batch_size=args.batch_size,
        num_workers=0,
        image_size=args.image_size,
        seed=args.seed,
    )
    print(f"Data: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")

    # Filter models if requested
    configs = BENCHMARK_CONFIGS
    if args.models:
        selected = set(args.models.split(","))
        configs = [c for c in BENCHMARK_CONFIGS if c["model_type"] in selected]

    all_results = []

    for cfg in configs:
        print(f"\n{'─' * 60}")
        print(f"  Training: {cfg['label']}")
        print(f"{'─' * 60}")

        model = build_model(
            model_type=cfg["model_type"],
            backbone=cfg["backbone"],
            pretrained=True,
            dropout=0.3,
            num_classes=1,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable parameters: {n_params:,}")

        t0 = time.time()
        results = train_and_evaluate(model, train_loader, val_loader, args.epochs, args.lr, device)
        elapsed = time.time() - t0

        # Save checkpoint
        ckpt_path = os.path.join(args.output_dir, f"{cfg['model_type']}_{cfg['backbone']}.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_type": cfg["model_type"],
            "backbone": cfg["backbone"],
            "results": results,
        }, ckpt_path)

        row = {
            "Model": cfg["label"],
            "Architecture": cfg["model_type"],
            "Backbone": cfg["backbone"],
            "Params (M)": round(n_params / 1e6, 2),
            "Best Val AUC": round(results["best_auc"], 4),
            "Val Accuracy": round(results["final_metrics"]["accuracy"], 4),
            "Val F1": round(results["final_metrics"].get("f1", 0.0), 4),
            "Monk F_EO↓": round(results["final_f_eo"], 4),
            "Train Time (s)": round(elapsed, 1),
        }
        all_results.append(row)
        print(f"  Done: AUC={row['Best Val AUC']:.4f} | F_EO={row['Monk F_EO↓']:.4f} | Time={elapsed:.1f}s")

    # === Final Benchmark Table ===
    print(f"\n{'=' * 60}")
    print("  BENCHMARK RESULTS")
    print(f"{'=' * 60}")
    results_df = pd.DataFrame(all_results)
    print(results_df.to_string(index=False))

    # Save CSV
    csv_path = os.path.join(args.output_dir, "benchmark_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nBenchmark table saved to: {csv_path}")

    # Save Markdown report
    md_path = os.path.join(args.output_dir, "benchmark_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Deepfake Detection Benchmark Report\n\n")
        f.write("## Model Comparison Table\n\n")
        f.write(results_df.to_markdown(index=False))
        f.write("\n\n## Fairness Note\n\n")
        f.write("**F_EO** (Equalized Odds Disparity) measures demographic bias across Monk Skin Tone groups.\n")
        f.write("Lower is better. F_EO = 0 means perfectly equalized error rates across skin tones.\n\n")
        f.write(f"_Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}_\n")

    print(f"Benchmark markdown report saved to: {md_path}")

    # Best model summary
    best = results_df.loc[results_df["Best Val AUC"].idxmax()]
    fairest = results_df.loc[results_df["Monk F_EO↓"].idxmin()]
    print(f"\n  🏆 Best AUC     : {best['Model']} ({best['Best Val AUC']:.4f})")
    print(f"  ⚖️  Fairest Model : {fairest['Model']} (F_EO={fairest['Monk F_EO↓']:.4f})")


if __name__ == "__main__":
    main()
