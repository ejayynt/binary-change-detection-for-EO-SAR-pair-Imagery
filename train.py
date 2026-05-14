"""
train.py — Training script for GalaxEye EO-SAR Binary Change Detection.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --data_root /path/to/dataset

Fixes applied vs original:
    1. focal_alpha flipped to 0.75  — original 0.25 penalised the minority
                                       Change class; 0.75 up-weights it.
    2. Mask normalisation guard      — masks loaded as 0/255 are divided by
                                       255 before use so targets are in [0,1].
    3. Threshold lowered to 0.3      — sigmoid outputs are near-zero in early
                                       epochs; 0.5 produced zero TPs.
    4. Multi-threshold val sweep     — best F1 across [0.2,0.3,0.4,0.5] is
                                       reported and used for checkpointing.
    5. Gradient clipping added       — stabilises early training.
    6. Sanity-check before training  — 50-step single-batch overfit test
                                       confirms mask values, model output
                                       shape and loss signal are correct.
    7. Sigmoid stats logged per val  — helps diagnose future collapse.
    8. Mixed-precision (AMP) support — optional; enabled via config flag.
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import GalaxEyeLocalDataset
from model import PseudoSiameseUNet


# ── Loss Functions ────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """Optimises for spatial overlap (directly aligned with IoU metric)."""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs        = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        union        = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice         = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """
    Down-weights easy background pixels; concentrates gradient on hard
    minority Change pixels to combat severe class imbalance.

    FIX 1 — alpha semantics:
        alpha_t is applied to the POSITIVE (change) class.
        Original alpha=0.25 penalised change pixels.
        Corrected default alpha=0.75 up-weights the minority change class.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce     = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs   = torch.sigmoid(logits)
        p_t     = probs * targets + (1 - probs) * (1 - targets)
        # alpha applied to positives; (1-alpha) applied to negatives
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_w = alpha_t * (1 - p_t) ** self.gamma
        return (focal_w * bce).mean()


class HybridLoss(nn.Module):
    """Linear sum of Focal Loss and Dice Loss."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, smooth: float = 1e-6):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice  = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.focal(logits, targets) + self.dice(logits, targets)


# ── Metric Helpers ────────────────────────────────────────────────────────────

def _tp_fp_fn(preds: torch.Tensor, targets: torch.Tensor):
    p = preds.view(-1)
    t = targets.view(-1)
    tp = ((p == 1) & (t == 1)).sum().item()
    fp = ((p == 1) & (t == 0)).sum().item()
    fn = ((p == 0) & (t == 1)).sum().item()
    return tp, fp, fn


def metrics_from_counts(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    iou       = tp / (tp + fp + fn + 1e-8)
    return iou, precision, recall, f1


# ── Mask Normalisation Guard ──────────────────────────────────────────────────

def normalise_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    FIX 2 — ensure mask values are in {0, 1}.
    Datasets that store masks as uint8 (0 / 255) will have their values
    clipped and divided automatically.
    """
    if mask.max() > 1.0:
        mask = mask / 255.0
    return mask.clamp(0.0, 1.0)


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Sanity Check (overfit single batch) ──────────────────────────────────────

def sanity_check(model, criterion, train_loader, device, steps: int = 50):
    """
    FIX 6 — Overfit on one batch for `steps` iterations.
    Confirms: mask values, loss signal, model output shape, TP > 0.
    Raises RuntimeError if the model cannot overfit a single batch.
    """
    print("\n── Sanity check: overfitting single batch ──")
    batch = next(iter(train_loader))
    eo   = batch['eo'].to(device)
    sar  = batch['sar'].to(device)
    mask = normalise_mask(batch['mask'].to(device))

    print(f"  Mask unique values : {mask.unique().tolist()}")
    print(f"  Positive pixel ratio: {mask.mean().item():.4f}")
    print(f"  EO  shape : {eo.shape}   SAR shape: {sar.shape}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()

    for i in range(steps):
        opt.zero_grad()
        logits = model(eo, sar)
        loss   = criterion(logits, mask)
        loss.backward()
        opt.step()

        if i % 10 == 0:
            preds = (torch.sigmoid(logits) > 0.3).float()
            tp, fp, fn = _tp_fp_fn(preds, mask)
            sig_mean = torch.sigmoid(logits).mean().item()
            print(f"  step {i:3d} | loss {loss.item():.4f} | "
                  f"sigmoid_mean {sig_mean:.4f} | TP {tp}")

    preds = (torch.sigmoid(logits) > 0.3).float()
    tp, _, _ = _tp_fp_fn(preds, mask)
    if tp == 0:
        raise RuntimeError(
            "Sanity check failed: model produced zero TPs after overfitting "
            "one batch. Check mask normalisation and model output dimensions."
        )
    print("  ✓ Sanity check passed\n")

    # Reset model weights for real training
    for layer in model.modules():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


# ── Validation at Multiple Thresholds ─────────────────────────────────────────

def validate(model, val_loader, device, thresholds=(0.2, 0.3, 0.4, 0.5)):
    """
    FIX 3 & 4 — Sweep multiple thresholds; return best F1 and its metrics.
    Also logs sigmoid output statistics to catch future model collapse.
    """
    model.eval()
    all_logits = []
    all_masks  = []

    with torch.no_grad():
        for batch in val_loader:
            eo   = batch['eo'].to(device)
            sar  = batch['sar'].to(device)
            mask = normalise_mask(batch['mask'].to(device))
            logits = model(eo, sar)
            all_logits.append(logits.cpu())
            all_masks.append(mask.cpu())

    all_logits = torch.cat(all_logits)
    all_masks  = torch.cat(all_masks)
    all_probs  = torch.sigmoid(all_logits)

    # FIX 7 — log sigmoid statistics
    print(f"  Sigmoid | min {all_probs.min():.4f} | "
          f"mean {all_probs.mean():.4f} | "
          f"max {all_probs.max():.4f}")

    best = {'f1': 0.0, 'iou': 0.0, 'precision': 0.0,
            'recall': 0.0, 'threshold': thresholds[0]}

    for thr in thresholds:
        preds = (all_probs > thr).float()
        tp = ((preds == 1) & (all_masks == 1)).sum().item()
        fp = ((preds == 1) & (all_masks == 0)).sum().item()
        fn = ((preds == 0) & (all_masks == 1)).sum().item()
        iou, prec, rec, f1 = metrics_from_counts(tp, fp, fn)
        print(f"  thr={thr:.2f} | IoU {iou:.4f} | "
              f"Prec {prec:.4f} | Rec {rec:.4f} | F1 {f1:.4f}")
        if f1 > best['f1']:
            best = {'f1': f1, 'iou': iou, 'precision': prec,
                    'recall': rec, 'threshold': thr}

    return best


# ── Main Training Function ────────────────────────────────────────────────────

def train(cfg: dict, data_root_override: str = None):
    # ── Setup ──────────────────────────────────────────────────────────
    seed = cfg['training'].get('random_seed', 42)
    set_seed(seed)

    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS     = cfg['training']['epochs']
    BATCH_SIZE = cfg['training']['batch_size']
    CROP_SIZE  = cfg['training']['crop_size']
    N_WORKERS  = cfg['training'].get('num_workers', 0)
    PIN_MEM    = cfg['training'].get('pin_memory', True) and torch.cuda.is_available()
    ROOT       = data_root_override or cfg['data']['root_dir']
    CKPT_PATH  = cfg['checkpoint']['save_path']
    AUGMENT    = cfg['augmentation'].get('enabled', True)
    USE_AMP    = cfg['training'].get('use_amp', False) and torch.cuda.is_available()
    GRAD_CLIP  = cfg['training'].get('grad_clip', 1.0)   # FIX 5

    # FIX 3 — use lower default threshold; sweep handled in validate()
    VAL_THRESHOLDS = cfg['evaluation'].get(
        'val_thresholds', [0.2, 0.3, 0.4, 0.5]
    )

    os.makedirs(
        os.path.dirname(CKPT_PATH) if os.path.dirname(CKPT_PATH) else '.',
        exist_ok=True
    )

    print(f"Device      : {DEVICE}")
    print(f"Dataset root: {ROOT}")
    print(f"Epochs      : {EPOCHS}  |  Batch: {BATCH_SIZE}  |  Crop: {CROP_SIZE}")
    print(f"AMP         : {USE_AMP}  |  Grad clip: {GRAD_CLIP}")

    # ── Datasets & Loaders ─────────────────────────────────────────────
    train_ds = GalaxEyeLocalDataset(ROOT, split='train', crop_size=CROP_SIZE, augment=AUGMENT)
    val_ds   = GalaxEyeLocalDataset(ROOT, split='val',   crop_size=CROP_SIZE, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=N_WORKERS, pin_memory=PIN_MEM)
    val_loader   = DataLoader(val_ds,   batch_size=1,          shuffle=False,
                              num_workers=N_WORKERS, pin_memory=PIN_MEM)

    # ── Model ──────────────────────────────────────────────────────────
    model = PseudoSiameseUNet().to(DEVICE)

    # ── Loss ───────────────────────────────────────────────────────────
    loss_cfg  = cfg['loss']
    criterion = HybridLoss(
        # FIX 1 — default changed to 0.75; config can still override
        alpha  = loss_cfg.get('focal_alpha', 0.75),
        gamma  = loss_cfg.get('focal_gamma', 2.0),
        smooth = loss_cfg.get('dice_smooth', 1e-6),
    )

    # ── Optimiser — differential learning rates ────────────────────────
    eo_params = (
        list(model.eo_init.parameters())   +
        list(model.eo_pool.parameters())   +
        list(model.eo_layer1.parameters()) +
        list(model.eo_layer2.parameters()) +
        list(model.eo_layer3.parameters()) +
        list(model.eo_layer4.parameters())
    )
    eo_ids       = {id(p) for p in eo_params}
    other_params = [p for p in model.parameters() if id(p) not in eo_ids]

    opt_cfg   = cfg['optimizer']
    optimizer = torch.optim.AdamW(
        [
            {'params': eo_params,    'lr': opt_cfg['lr_eo_encoder']},
            {'params': other_params, 'lr': opt_cfg['lr_other']},
        ],
        weight_decay=opt_cfg.get('weight_decay', 1e-4),
    )

    sched_cfg = cfg['scheduler']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=sched_cfg.get('T_max', EPOCHS)
    )

    # FIX 8 — AMP scaler (no-op on CPU)
    scaler = GradScaler(enabled=USE_AMP)

    # ── Sanity check ───────────────────────────────────────────────────
    # FIX 6 — runs before the real training loop
    sanity_check(model, criterion, train_loader, DEVICE)

    # ── Training Loop ──────────────────────────────────────────────────
    best_val_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        # ── Train ───────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{EPOCHS} [Train]", leave=False)
        for batch in loop:
            eo   = batch['eo'].to(DEVICE)
            sar  = batch['sar'].to(DEVICE)
            # FIX 2 — normalise mask before loss computation
            mask = normalise_mask(batch['mask'].to(DEVICE))

            optimizer.zero_grad()

            # FIX 8 — mixed precision forward pass
            with autocast(enabled=USE_AMP):
                logits = model(eo, sar)
                loss   = criterion(logits, mask)

            scaler.scale(loss).backward()

            # FIX 5 — gradient clipping prevents exploding gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        scheduler.step()

        # ── Validate ─────────────────────────────────────────────────
        # FIX 3 & 4 — multi-threshold sweep; best F1 used for checkpoint
        print(f"\nEpoch {epoch:3d}/{EPOCHS} | Train Loss: {avg_train_loss:.4f}")
        best = validate(model, val_loader, DEVICE, thresholds=VAL_THRESHOLDS)

        print(
            f"  Best → thr={best['threshold']:.2f} | "
            f"IoU {best['iou']:.4f} | "
            f"Prec {best['precision']:.4f} | "
            f"Rec {best['recall']:.4f} | "
            f"F1 {best['f1']:.4f}"
        )

        if best['f1'] > best_val_f1:
            best_val_f1 = best['f1']
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_f1': best_val_f1,
                    'best_threshold': best['threshold'],
                },
                CKPT_PATH,
            )
            print(f"  ✓ New best F1 {best_val_f1:.4f} — checkpoint saved to {CKPT_PATH}")

    print(f"\nTraining complete. Best Val F1: {best_val_f1:.4f}")
    print(f"Best checkpoint: {CKPT_PATH}")


# ── Entry Point ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train GalaxEye EO-SAR Change Detection Model")
    parser.add_argument("--config",    type=str, default="config.yaml",
                        help="Path to YAML config file (default: config.yaml)")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Override data root path from config (optional)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    train(cfg, data_root_override=args.data_root)