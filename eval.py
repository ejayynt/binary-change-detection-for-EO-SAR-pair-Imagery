"""
eval.py — Evaluation script for GalaxEye EO-SAR Binary Change Detection.

Evaluates on the specified split (or both val and test), prints all required
metrics, saves the confusion matrix, and saves ≥5 qualitative visualisations.

Usage:
    # Evaluate test split
    python eval.py --data_path /path/to/dataset --weights checkpoints/best_model.pth

    # Evaluate both val and test
    python eval.py --data_path /path/to/dataset --weights checkpoints/best_model.pth --split both

    # Override sigmoid threshold
    python eval.py --data_path /path/to/dataset --weights checkpoints/best_model.pth --threshold 0.5
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import GalaxEyeLocalDataset
from model import PseudoSiameseUNet


# ── Visualisation Helpers ─────────────────────────────────────────────────────

def save_visualization(eo_t, sar_t, gt_t, pred_t, save_idx: int, save_dir: str = "visualizations"):
    """
    Saves a 4-panel figure: EO | SAR | Ground Truth | Prediction.
    Fulfils the ≥5 qualitative visualisation requirement in the report.
    """
    os.makedirs(save_dir, exist_ok=True)

    eo_img   = (eo_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    sar_np   = sar_t.squeeze().cpu().numpy()
    sar_img  = (sar_np * 255).astype(np.uint8)
    gt_mask  = gt_t.squeeze().cpu().numpy()
    pred_msk = pred_t.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(eo_img)
    axes[0].set_title("Post-Event (EO RGB)")
    axes[0].axis("off")

    axes[1].imshow(sar_img, cmap='gray')
    axes[1].set_title("Pre-Event (SAR)")
    axes[1].axis("off")

    axes[2].imshow(gt_mask, cmap='Reds', vmin=0, vmax=1)
    axes[2].set_title("Ground Truth (Change=Red)")
    axes[2].axis("off")

    axes[3].imshow(pred_msk, cmap='Reds', vmin=0, vmax=1)
    axes[3].set_title("Prediction (Change=Red)")
    axes[3].axis("off")

    plt.suptitle(f"Sample {save_idx}", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(save_dir, f"sample_{save_idx:03d}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_confusion_matrix(tn: int, fp: int, fn: int, tp: int,
                           split_name: str, save_dir: str = "visualizations"):
    """Saves an annotated confusion matrix heatmap."""
    os.makedirs(save_dir, exist_ok=True)
    cm = np.array([[tn, fp], [fn, tp]])

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues",
                xticklabels=["No-Change (Pred)", "Change (Pred)"],
                yticklabels=["No-Change (GT)",   "Change (GT)"])
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title(f"Pixel-Level Confusion Matrix — {split_name}")
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"confusion_matrix_{split_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path


# ── Core Evaluation Function ──────────────────────────────────────────────────

def evaluate_split(model, data_root: str, split: str, device,
                   threshold: float = 0.5,
                   vis_dir: str = "visualizations",
                   num_vis: int = 5) -> dict:
    """
    Runs evaluation on a single split. Accumulates pixel-level TP/FP/FN/TN
    across the whole split (global aggregation — not per-image average).

    Returns a dict of metrics.
    """
    dataset = GalaxEyeLocalDataset(data_root, split=split, crop_size=None, augment=False)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    total_tp = total_fp = total_fn = total_tn = 0
    saved_vis = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating [{split}]"):
            eo   = batch['eo'].to(device)
            sar  = batch['sar'].to(device)
            mask = batch['mask'].to(device)

            logits = model(eo, sar)
            preds  = (torch.sigmoid(logits) > threshold).float()

            p_flat = preds.view(-1).cpu().numpy()
            t_flat = mask.view(-1).cpu().numpy()

            tp = int(np.sum((p_flat == 1) & (t_flat == 1)))
            fp = int(np.sum((p_flat == 1) & (t_flat == 0)))
            fn = int(np.sum((p_flat == 0) & (t_flat == 1)))
            tn = int(np.sum((p_flat == 0) & (t_flat == 0)))

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

            # Save qualitative examples that contain real change pixels
            if saved_vis < num_vis and np.sum(t_flat == 1) > 50:
                save_visualization(eo, sar, mask, preds, saved_vis,
                                   save_dir=os.path.join(vis_dir, split))
                saved_vis += 1

    # Global pixel-level metrics (change class)
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall    = total_tp / (total_tp + total_fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    iou       = total_tp / (total_tp + total_fp + total_fn + 1e-8)

    cm_path = save_confusion_matrix(total_tn, total_fp, total_fn, total_tp,
                                    split_name=split, save_dir=vis_dir)

    return {
        'split':     split,
        'iou':       iou,
        'precision': precision,
        'recall':    recall,
        'f1':        f1,
        'tp':        total_tp,
        'fp':        total_fp,
        'fn':        total_fn,
        'tn':        total_tn,
        'cm_path':   cm_path,
    }


# ── Pretty Print ──────────────────────────────────────────────────────────────

def print_metrics(m: dict):
    s = m['split'].upper()
    print(f"\n{'='*46}")
    print(f"  METRICS — {s} SPLIT  (Change class = 1)")
    print(f"{'='*46}")
    print(f"  Intersection over Union (IoU) : {m['iou']:.4f}")
    print(f"  Precision                     : {m['precision']:.4f}")
    print(f"  Recall                        : {m['recall']:.4f}")
    print(f"  F1 Score                      : {m['f1']:.4f}")
    print(f"{'─'*46}")
    print(f"  TP={m['tp']:,}  FP={m['fp']:,}  FN={m['fn']:,}  TN={m['tn']:,}")
    print(f"  Confusion matrix saved → {m['cm_path']}")
    print(f"{'='*46}")


# ── Entry Point ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate GalaxEye EO-SAR Change Detection Model"
    )
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to dataset root directory")
    parser.add_argument("--weights",   type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--split",     type=str, default="test",
                        choices=["val", "test", "both"],
                        help="Which split(s) to evaluate (default: test)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Sigmoid threshold for binary predictions (default: 0.5)")
    parser.add_argument("--vis_dir",   type=str, default="visualizations",
                        help="Directory to save visualisations (default: visualizations/)")
    parser.add_argument("--num_vis",   type=int, default=5,
                        help="Number of qualitative samples to save per split (default: 5)")
    return parser.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"Weights : {args.weights}")

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Checkpoint not found: {args.weights}")

    model = PseudoSiameseUNet().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    print("Model loaded successfully.\n")

    splits = ["val", "test"] if args.split == "both" else [args.split]

    all_results = []
    for sp in splits:
        result = evaluate_split(
            model, args.data_path, sp, device,
            threshold=args.threshold,
            vis_dir=args.vis_dir,
            num_vis=args.num_vis,
        )
        print_metrics(result)
        all_results.append(result)

    print(f"\nVisualizations saved to: {args.vis_dir}/")
