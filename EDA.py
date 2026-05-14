"""
EDA.py — Exploratory Data Analysis for the GalaxEye EO-SAR dataset.

Analyses a single image triplet (EO, SAR, mask) to understand:
  • Array shapes and dtypes
  • EO radiometric statistics
  • SAR radiometric statistics (especially speckle / outlier range)
  • Class distribution in the label mask
  • No-data (black-pixel) alignment between modalities

HOW TO RUN:
    Just update the three paths at the bottom of this file, then run:
        python EDA.py
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
import random

def analyse_sample(eo_path: str, sar_path: str, mask_path: str):
    # ── Validate paths ────────────────────────────────────────────
    for label, path in [("EO", eo_path), ("SAR", sar_path), ("Mask", mask_path)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} file not found: {path}")

    print("=" * 60)
    print("LOADING RAW TIFF ARRAYS")
    print("=" * 60)

    eo_arr   = tiff.imread(eo_path)
    sar_arr  = tiff.imread(sar_path)
    mask_arr = tiff.imread(mask_path)

    print(f"EO   shape: {eo_arr.shape}   dtype: {eo_arr.dtype}")
    print(f"SAR  shape: {sar_arr.shape}  dtype: {sar_arr.dtype}")
    print(f"Mask shape: {mask_arr.shape} dtype: {mask_arr.dtype}")

    # ── EO Statistics ─────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("EO OPTICAL STATISTICS")
    print("─" * 60)
    print(f"  Min: {np.min(eo_arr)}   Max: {np.max(eo_arr)}")
    print(f"  Mean: {np.mean(eo_arr):.2f}   Std: {np.std(eo_arr):.2f}")
    if eo_arr.ndim == 3:
        for c in range(eo_arr.shape[2]):
            ch = eo_arr[:, :, c]
            print(f"  Channel {c} — mean: {ch.mean():.2f}  std: {ch.std():.2f}")

    # ── SAR Statistics ────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("SAR RADAR STATISTICS")
    print("─" * 60)
    sar_flat = sar_arr.flatten()
    print(f"  Min: {np.min(sar_flat)}   Max: {np.max(sar_flat)}")
    print(f"  Mean: {np.mean(sar_flat):.2f}   Std: {np.std(sar_flat):.2f}")
    for p in [50, 90, 95, 98, 99]:
        print(f"  {p}th percentile: {np.percentile(sar_flat, p):.2f}")

    # ── Label Distribution ────────────────────────────────────────
    print("\n" + "─" * 60)
    print("LABEL DISTRIBUTION (original 4-class)")
    print("─" * 60)
    total_px = mask_arr.size
    for val, name in zip([0, 1, 2, 3], ["Background", "Intact", "Damaged", "Destroyed"]):
        count = np.sum(mask_arr == val)
        print(f"  Class {val} ({name:<12}): {count:>10,}  ({100*count/total_px:.2f}%)")

    # Remapped binary distribution
    change_px    = np.sum((mask_arr == 2) | (mask_arr == 3))
    no_change_px = total_px - change_px
    print(f"\n  → Binary: No-Change={no_change_px:,} ({100*no_change_px/total_px:.2f}%)  "
          f"Change={change_px:,} ({100*change_px/total_px:.2f}%)")
    if change_px > 0:
        print(f"  → Imbalance ratio (no-change / change): {no_change_px/change_px:.1f}x")
    
    # Add after the binary distribution print block
    print("\n" + "─" * 60)
    print("CROP FEASIBILITY CHECK")
    print("─" * 60)
    crop_size = 256
    change_coords = np.argwhere((mask_arr == 2) | (mask_arr == 3))
    print(f"  Change pixel coords found: {len(change_coords)}")
    if len(change_coords) > 0:
        # Simulate 1000 random crops and check how many hit a change pixel
        h, w   = mask_arr.shape[:2]
        hits   = 0
        trials = 1000
        for _ in range(trials):
            top  = random.randint(0, max(0, h - crop_size))
            left = random.randint(0, max(0, w - crop_size))
            crop = mask_arr[top:top+crop_size, left:left+crop_size]
            if np.any((crop == 2) | (crop == 3)):
                hits += 1
        print(f"  Random 256×256 crop hits change: {hits}/{trials} ({hits/10:.1f}%)")
        print(f"  → With guided crop (70% anchored): ~{int(0.7*100 + 0.3*hits/10)}% of crops contain change")
    else:
        print("  ⚠ This scene has NO change pixels at all — pure no-change scene")

    # ── No-Data (void) Alignment ──────────────────────────────────
    print("\n" + "─" * 60)
    print("NO-DATA VOID ALIGNMENT")
    print("─" * 60)
    eo_nodata  = np.all(eo_arr == 0, axis=-1) if eo_arr.ndim == 3 else (eo_arr == 0)
    sar_nodata = (sar_arr == 0) if sar_arr.ndim == 2 else np.all(sar_arr == 0, axis=-1)

    eo_cnt  = int(np.sum(eo_nodata))
    sar_cnt = int(np.sum(sar_nodata))
    overlap = int(np.sum(eo_nodata & sar_nodata))

    print(f"  EO no-data pixels : {eo_cnt:,}")
    print(f"  SAR no-data pixels: {sar_cnt:,}")
    if eo_cnt > 0 and sar_cnt > 0:
        pct = 100 * overlap / max(eo_cnt, sar_cnt)
        print(f"  Alignment match   : {pct:.1f}%")

    # ── Visualisation ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(eo_arr if eo_arr.ndim == 3 else eo_arr, cmap='gray' if eo_arr.ndim == 2 else None)
    axes[0].set_title("EO (Post-Event)")
    axes[0].axis("off")

    axes[1].imshow(sar_arr if sar_arr.ndim == 2 else sar_arr[:, :, 0], cmap='gray')
    axes[1].set_title("SAR (Pre-Event)")
    axes[1].axis("off")

    axes[2].imshow(mask_arr, cmap='tab10', vmin=0, vmax=3)
    axes[2].set_title("Label Mask (0=BG, 1=Intact, 2=Damaged, 3=Destroyed)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("eda_sample.png", dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: eda_sample.png")
    plt.show()


if __name__ == "__main__":
    # ── UPDATE THESE THREE PATHS BEFORE RUNNING ───────────────────
    # Replace the filename below with any .tif file from your target/ folder.
    # All three files must share the same filename across their folders.

    FILENAME = "scene_01_000001_building_damage.tif"  # ← change this

    EO_PATH   = r"C:\Draft\Draft\train\train\post-event\scene_01_000001_building_damage.tif"
    SAR_PATH  = r"C:\Draft\Draft\train\train\pre-event\scene_01_000001_building_damage.tif"
    MASK_PATH = r"C:\Draft\Draft\train\train\target\scene_01_000001_building_damage.tif"

    analyse_sample(EO_PATH, SAR_PATH, MASK_PATH)