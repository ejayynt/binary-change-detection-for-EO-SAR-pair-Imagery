import os
import glob
import torch
import tifffile as tiff
import numpy as np
import random
from torch.utils.data import Dataset


class GalaxEyeLocalDataset(Dataset):
    """
    Dataset loader for the GalaxEye EO-SAR change detection task.

    Directory layout expected (nested split folders as provided):
        root_dir/
            train/train/post-event/  pre-event/  target/
            val/val/post-event/      pre-event/  target/
            test/test/post-event/    pre-event/  target/

    Label remapping (mandatory per assignment):
        0 (Background)  -> 0  No-Change
        1 (Intact)      -> 0  No-Change   ← BUG FIX: was incorrectly set to 1
        2 (Damaged)     -> 1  Change
        3 (Destroyed)   -> 1  Change
    """

    def __init__(self, root_dir: str, split: str = 'train', crop_size: int = 256,
                 augment: bool = False):
        """
        Args:
            root_dir  : Path to the dataset root folder.
            split     : One of 'train', 'val', or 'test'.
            crop_size : Square crop size used during training to limit VRAM usage.
            augment   : If True, apply random horizontal/vertical flips (training only).
        """
        self.split_dir = os.path.join(root_dir, split, split)
        self.crop_size = crop_size
        self.is_train = (split == 'train')
        self.augment = augment and self.is_train

        self.samples = []

        eo_dir   = os.path.join(self.split_dir, 'post-event')   # RGB optical (post)
        sar_dir  = os.path.join(self.split_dir, 'pre-event')    # SAR (pre)
        mask_dir = os.path.join(self.split_dir, 'target')

        if not os.path.exists(self.split_dir):
            raise FileNotFoundError(
                f"Split directory not found: {self.split_dir}\n"
                f"Make sure root_dir points to the dataset root and the split "
                f"folder structure is root_dir/<split>/<split>/post-event|pre-event|target/."
            )

        mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.tif")))
        if not mask_files:
            raise RuntimeError(f"No .tif mask files found in {mask_dir}")

        missing = 0
        for mask_path in mask_files:
            base_name = os.path.basename(mask_path)
            eo_path  = os.path.join(eo_dir,  base_name)
            sar_path = os.path.join(sar_dir, base_name)

            if os.path.exists(eo_path) and os.path.exists(sar_path):
                self.samples.append({'eo': eo_path, 'sar': sar_path, 'mask': mask_path})
            else:
                print(f"[WARNING] Missing EO or SAR pair for {base_name} in '{split}' split")
                missing += 1

        if missing:
            print(f"[WARNING] {missing} incomplete samples skipped in '{split}' split.")

        print(f"[Dataset] '{split}' split: {len(self.samples)} valid samples loaded.")

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        eo_arr   = tiff.imread(sample['eo'])    # uint8, HxWx3 (RGB)
        sar_arr  = tiff.imread(sample['sar'])   # uint8, HxW or HxWxC
        mask_arr = tiff.imread(sample['mask'])  # uint8, HxW  values in {0,1,2,3}

        # ── Mandatory GalaxEye Label Remapping ────────────────────────
        # 0 (Background) → 0  |  1 (Intact)   → 0
        # 2 (Damaged)    → 1  |  3 (Destroyed) → 1
        remapped_mask = np.zeros_like(mask_arr, dtype=np.float32)
        remapped_mask[mask_arr == 2] = 1.0   # Damaged  → Change
        remapped_mask[mask_arr == 3] = 1.0   # Destroyed → Change
        # Classes 0 and 1 remain 0.0 (No-Change) — correct per spec.

        # ── Spatial Cropping (training) ────────────────────────────────
        # ── Spatial Cropping (training) ────────────────────────────────
        h, w = mask_arr.shape[:2]
        if self.is_train and self.crop_size < h:
            # Find where change pixels are
            change_coords = np.argwhere(remapped_mask == 1.0)  # shape [N, 2] (row, col)

            # 70% of the time: anchor crop on a change pixel (guaranteed TP signal)
            # 30% of the time: random crop (hard negatives, prevents bias)
            if len(change_coords) > 0 and random.random() < 0.7:
                cy, cx  = change_coords[random.randint(0, len(change_coords) - 1)]
                # Centre crop around the chosen change pixel, clamped to image bounds
                top  = int(np.clip(cy - self.crop_size // 2, 0, h - self.crop_size))
                left = int(np.clip(cx - self.crop_size // 2, 0, w - self.crop_size))
            else:
                top  = random.randint(0, h - self.crop_size)
                left = random.randint(0, w - self.crop_size)

            eo_arr        = _crop(eo_arr,  top, left, self.crop_size)
            sar_arr       = _crop(sar_arr, top, left, self.crop_size)
            remapped_mask = remapped_mask[top:top+self.crop_size, left:left+self.crop_size]

        # ── Augmentation (training only) ──────────────────────────────
        if self.augment:
            if random.random() > 0.5:   # Random horizontal flip
                eo_arr        = np.fliplr(eo_arr).copy()
                sar_arr       = np.fliplr(sar_arr).copy()
                remapped_mask = np.fliplr(remapped_mask).copy()
            if random.random() > 0.5:   # Random vertical flip
                eo_arr        = np.flipud(eo_arr).copy()
                sar_arr       = np.flipud(sar_arr).copy()
                remapped_mask = np.flipud(remapped_mask).copy()
            if random.random() > 0.5:   # Random 90° rotation
                k = random.randint(1, 3)
                eo_arr        = np.rot90(eo_arr,  k).copy()
                sar_arr       = np.rot90(sar_arr, k).copy()
                remapped_mask = np.rot90(remapped_mask, k).copy()

        # ── Normalisation uint8 → [0, 1] ──────────────────────────────
        eo_arr  = eo_arr.astype(np.float32)  / 255.0
        sar_arr = sar_arr.astype(np.float32) / 255.0

        # ── Convert to Tensors ────────────────────────────────────────
        # EO: ensure 3-channel CxHxW
        if eo_arr.ndim == 3:
            eo_tensor = torch.from_numpy(eo_arr).permute(2, 0, 1)   # HWC → CHW
        else:
            eo_3ch    = np.stack([eo_arr, eo_arr, eo_arr], axis=0)  # grayscale → 3ch
            eo_tensor = torch.from_numpy(eo_3ch)

        # SAR: always 1-channel CxHxW
        if sar_arr.ndim == 2:
            sar_tensor = torch.from_numpy(sar_arr).unsqueeze(0)
        else:
            sar_tensor = torch.from_numpy(sar_arr[:, :, 0]).unsqueeze(0)  # first channel

        mask_tensor = torch.from_numpy(remapped_mask).unsqueeze(0)  # 1xHxW

        return {'eo': eo_tensor, 'sar': sar_tensor, 'mask': mask_tensor,
                'filename': os.path.basename(sample['mask'])}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _crop(arr: np.ndarray, top: int, left: int, size: int) -> np.ndarray:
    if arr.ndim == 3:
        return arr[top:top+size, left:left+size, :]
    return arr[top:top+size, left:left+size]
