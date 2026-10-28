"""
Training Script for Deepfake Detection with Algorithmic Fairness Mitigation.
Supports Spatial (timm), Frequency (FFT/HPF), and Dual-Stream architectures with Equalized Odds regularization.
"""

import argparse
import os
import time
from typing import Dict, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import yaml

from models import build_model
from utils.dataset_loader import create_dataloaders
from utils.loss import build_criterion
from utils.metrics import FairnessAuditor, compute_binary_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train Deepfake Detector with Algorithmic Fairness.")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    # --annotations and --train_csv are aliases for the training CSV path
    parser.add_argument("--annotations", "--train_csv", dest="annotations",
                        type=str, default="./dataset/train_25k.csv",
                        help="Path to training annotations CSV")
    # --val_annotations and --val_csv are aliases for the validation CSV path
    parser.add_argument("--val_annotations", "--val_csv", dest="val_annotations",
                        type=str, default=None,
                        help="Path to validation/test annotations CSV (optional)")
    parser.add_argument("--data_root", type=str, default=".", help="Base directory for image paths")
    parser.add_argument("--model_type", type=str, default="spatial", choices=["spatial", "frequency", "dual_stream"], help="Model architecture")
    parser.add_argument("--backbone", type=str, default="efficientnet_b0", help="Backbone name for timm")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--image_size", type=int, default=224, help="Input image resolution")
    parser.add_argument("--loss_type", type=str, default="bce", choices=["bce", "subgroup_reweighted", "equalized_odds"], help="Loss function")
    parser.add_argument("--lambda_fair", type=float, default=0.5, help="Fairness regularization penalty weight")
    parser.add_argument("--mitigate_bias", action="store_true", help="Enable demographic balanced batch sampling and fair loss")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--results_dir", type=str, default="./results", help="Directory to save logs and plots")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (0 for Windows compatibility)")
    parser.add_argument("--torch_threads", type=int, default=0,
                        help="Number of intra-op CPU threads for PyTorch (0 = use torch default).")
    parser.add_argument("--freeze_backbone", type=int, default=0,
                        help="Freeze first N stages of backbone for fast CPU head-only training (0 = no freezing).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_type: str,
    resume_path: str = "",
    save_every: int = 50,
    current_epoch: int = 1,
) -> float:
    """
    Runs one training epoch with mid-epoch checkpointing.

    If ``resume_path`` points to an existing resume checkpoint whose
    ``epoch`` matches ``current_epoch``, batches up to and including the
    saved ``batch_idx`` are skipped (data is still iterated to keep the
    RNG state reproducible, but no gradient step is taken).

    A resume checkpoint is written every ``save_every`` batches so that
    a killed process can continue from the last saved position rather than
    restarting the full epoch.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    # ---- resume state -------------------------------------------------------
    resume_batch = -1   # last completed batch index stored in resume file
    if resume_path and os.path.exists(resume_path):
        try:
            res = torch.load(resume_path, map_location=device)
            if res.get("epoch") == current_epoch:
                resume_batch = int(res.get("batch_idx", -1))
                total_loss   = float(res.get("total_loss", 0.0))
                total_samples = int(res.get("total_samples", 0))
                model.load_state_dict(res["model_state_dict"])
                optimizer.load_state_dict(res["optimizer_state_dict"])
                # Restore Python / NumPy / Torch RNG states for reproducibility
                if "rng_state" in res:
                    torch.set_rng_state(res["rng_state"])
                if "np_rng_state" in res:
                    np.random.set_state(res["np_rng_state"])
                print(f"\n  [Resume] Epoch {current_epoch}: skipping batches 0-{resume_batch} "
                      f"(accumulated loss={total_loss:.4f}, samples={total_samples})")
        except Exception as exc:
            print(f"  [Resume] Could not load resume checkpoint: {exc}. Starting epoch from scratch.")
            resume_batch = -1
            total_loss = 0.0
            total_samples = 0
    # -------------------------------------------------------------------------

    resume_dir = os.path.dirname(resume_path) if resume_path else ""

    pbar = tqdm(enumerate(loader), desc=f"Training E{current_epoch}", leave=False,
                total=len(loader))
    for batch_idx, (images, targets, metadata) in pbar:

        # Skip already-processed batches (iterate loader to keep RNG in sync)
        if batch_idx <= resume_batch:
            pbar.set_postfix({"skip": batch_idx})
            continue

        images = images.to(device)
        targets = targets.to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()
        logits = model(images).view(-1)

        if loss_type == "equalized_odds":
            skin_tones = (
                metadata["skin_tone"].to(device)
                if isinstance(metadata["skin_tone"], torch.Tensor)
                else torch.as_tensor(metadata["skin_tone"], device=device)
            )
            loss = criterion(logits, targets, skin_tones=skin_tones)
        elif loss_type == "subgroup_reweighted":
            subgroups = metadata["subgroup"]
            loss = criterion(logits, targets, subgroups=subgroups)
        else:
            loss = criterion(logits, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss    += loss.item() * batch_size
        total_samples += batch_size
        pbar.set_postfix({"loss": f"{loss.item():.4f}",
                          "avg":  f"{total_loss/max(total_samples,1):.4f}"})

        # ---- mid-epoch checkpoint -------------------------------------------
        if resume_path and resume_dir and (batch_idx + 1) % save_every == 0:
            torch.save({
                "epoch":          current_epoch,
                "batch_idx":      batch_idx,
                "total_loss":     total_loss,
                "total_samples":  total_samples,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "rng_state":      torch.get_rng_state(),
                "np_rng_state":   np.random.get_state(),
            }, resume_path)
        # ---------------------------------------------------------------------

    # Epoch finished — remove stale resume file so next epoch starts clean
    if resume_path and os.path.exists(resume_path):
        os.remove(resume_path)

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict:
    """Evaluates model on validation loader and returns loss, performance, and fairness disparities."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    all_targets = []
    all_probs = []
    all_meta_records = []

    for images, targets, metadata in tqdm(loader, desc="Validation", leave=False):
        images = images.to(device)
        targets = targets.to(device)
        batch_size = images.size(0)

        logits = model(images).view(-1)
        loss = criterion(logits, targets)

        probs = torch.sigmoid(logits).cpu().numpy()
        targets_np = targets.cpu().numpy()

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_probs.extend(probs)
        all_targets.extend(targets_np)

        # Unpack metadata batch
        for i in range(batch_size):
            all_meta_records.append({
                "image_path": metadata["image_path"][i],
                "skin_tone": int(metadata["skin_tone"][i]),
                "gender": str(metadata["gender"][i]),
                "skin_tone_group": str(metadata["skin_tone_group"][i]),
                "subgroup": str(metadata["subgroup"][i]),
            })

    meta_df = pd.DataFrame(all_meta_records)
    auditor = FairnessAuditor(threshold=0.5)
    audit_results = auditor.audit(all_targets, all_probs, meta_df)

    val_loss = total_loss / max(total_samples, 1)
    return {
        "val_loss": val_loss,
        "metrics": audit_results["overall_metrics"],
        "fairness_mst": audit_results["mst_fairness_disparities"],
        "audit_results": audit_results,
    }


def main():
    args = parse_args()

    # Load YAML config if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
            for k, v in cfg.items():
                if hasattr(args, k) and v is not None:
                    setattr(args, k, v)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # CPU thread tuning — use all physical cores when no GPU is available
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(max(1, args.torch_threads // 2))
    elif not torch.cuda.is_available():
        import os as _os
        cpu_count = _os.cpu_count() or 4
        torch.set_num_threads(cpu_count)
        torch.set_num_interop_threads(max(1, cpu_count // 2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==================================================")
    print(f"Deepfake Detector Training & Algorithmic Fairness")
    print(f"Device: {device} | Model: {args.model_type} ({args.backbone})")
    print(f"Loss: {args.loss_type} (Mitigate Bias: {args.mitigate_bias})")
    print(f"==================================================")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # 1. Create DataLoaders
    train_loader, val_loader, test_loader, full_df = create_dataloaders(
        annotations_path=args.annotations,
        val_annotations_path=args.val_annotations,
        root_dir=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        balance_demographics=args.mitigate_bias,
        seed=args.seed,
    )
    print(f"Data splits -> Train (Entire): {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # 2. Build Model
    model = build_model(
        model_type=args.model_type,
        backbone=args.backbone,
        pretrained=True,
        dropout=0.3,
        num_classes=1,
        freeze_backbone_stages=args.freeze_backbone,
    ).to(device)

    # 3. Setup Loss Criterion
    group_weights = {}
    if args.loss_type == "subgroup_reweighted" or args.mitigate_bias:
        sg_counts = full_df["subgroup"].value_counts()
        max_c = sg_counts.max()
        group_weights = {sg: float(max_c / max(c, 1)) for sg, c in sg_counts.items()}

    loss_choice = "equalized_odds" if (args.mitigate_bias and args.loss_type == "bce") else args.loss_type
    criterion = build_criterion(
        loss_type=loss_choice,
        lambda_fair=args.lambda_fair,
        group_weights=group_weights,
    )
    val_criterion = nn.BCEWithLogitsLoss()

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_auc = -1.0
    best_f_eo = 999.0
    history = []

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, loss_choice)
        val_res = validate(model, val_loader, val_criterion, device)
        scheduler.step()

        val_loss = val_res["val_loss"]
        val_acc = val_res["metrics"]["accuracy"]
        val_auc = val_res["metrics"]["auc"]
        f_eo = val_res["fairness_mst"].get("F_EO", 0.0)

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_time:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f} | "
            f"Monk F_EO: {f_eo:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": val_acc,
            "val_auc": val_auc,
            "monk_F_EO": f_eo,
            "monk_delta_fpr": val_res["fairness_mst"].get("delta_fpr", 0.0),
            "monk_delta_tpr": val_res["fairness_mst"].get("delta_tpr", 0.0),
        })

        # Save Best Checkpoint (Primary: High AUC, Secondary: Low Equalized Odds Disparity)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_f_eo = f_eo
            best_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": val_auc,
                "monk_F_EO": f_eo,
                "model_type": args.model_type,
                "backbone": args.backbone,
                "args": vars(args),
            }, best_path)
            print(f"  --> Saved new best checkpoint to: {best_path}")

    # Save last model checkpoint
    last_path = os.path.join(args.output_dir, "last_model.pt")
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "model_type": args.model_type,
        "backbone": args.backbone,
        "args": vars(args),
    }, last_path)

    # Export training history
    history_df = pd.DataFrame(history)
    history_csv = os.path.join(args.results_dir, "training_history.csv")
    history_df.to_csv(history_csv, index=False)
    print(f"\nTraining complete in {time.time() - start_time:.1f}s.")
    print(f"Best Validation AUC: {best_val_auc:.4f} | Monk F_EO: {best_f_eo:.4f}")
    print(f"Saved training history to: {history_csv}")


if __name__ == "__main__":
    main()
