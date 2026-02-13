#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets import load_dataset, concatenate_datasets
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
    TrainingArguments,
    Trainer,
)

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import evaluate
from torch import nn
from PIL import Image

IGNORE = 255

import cv2
import numpy as np
import random

import cv2
import numpy as np
import random
import albumentations as A

import cv2
import numpy as np
import random
import albumentations as A

import albumentations as A
import cv2
import numpy as np
import random
import albumentations as A
import cv2
import numpy as np
import random

BASE_DIR = "data/2016-5 and 2016-6/Batch 3/2016-5/2016-5 P2"
PATCH = 512
N_AUG = 8
MIN_MASK_PIXELS = 10
SEED = None  # set 0 for reproducible
IGNORE = 255


# --- Probability of choosing EACH component (CC) ---
P_CHOOSE_CC = 0.35     # <-- change this (0.1 subtle, 0.3-0.4 moderate, 0.6 heavy)

# --- CC filtering / preference ---
MIN_CC_AREA = 25       # ignore tiny specks
PREFER_ELONGATED = True  # bias sampling toward long fibers

# --- ROI warp params (heavy warping) ---
ROI_MARGIN = 8
PAD = 96
ELASTIC_ALPHA = 600
ELASTIC_SIGMA = 8
ELASTIC_ALPHA_AFFINE = 0  # keep 0 for micro-local realism

# --- Seam handling ---
FEATHER_BLEND = True
FEATHER_WIDTH = 18

if SEED is not None:
    random.seed(SEED)
    np.random.seed(SEED)


    
import re, os, shutil
from transformers import TrainerCallback

def max_checkpoint_step(out_dir: str) -> int:
    """Return the largest N from checkpoint-N in out_dir, or 0 if none."""
    if not os.path.isdir(out_dir):
        return 0
    mx = 0
    for name in os.listdir(out_dir):
        m = re.match(r"checkpoint-(\d+)$", name)
        if m:
            mx = max(mx, int(m.group(1)))
    return mx

import numpy as np
import cv2
import albumentations as A

import numpy as np
import cv2
import albumentations as A

def component_bbox(labels, lab, margin, H, W):
    ys, xs = np.where(labels == lab)
    if len(ys) == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    y0 = max(0, y0 - margin); x0 = max(0, x0 - margin)
    y1 = min(H-1, y1 + margin); x1 = min(W-1, x1 + margin)
    return (y0, y1, x0, x1)

def feather_alpha(h, w, feather=18):
    y = np.minimum(np.arange(h), np.arange(h)[::-1]).astype(np.float32)
    x = np.minimum(np.arange(w), np.arange(w)[::-1]).astype(np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    d = np.minimum(yy, xx)
    a = np.clip(d / max(1, feather), 0, 1)
    return a[..., None]  # (H,W,1)

class ComponentScalePaste(A.DualTransform):
    """
    Extract one connected component (CC), optionally isolate it inside bbox,
    then RESIZE the patch (bigger/smaller) and paste it at a random location.

    - image: paste resized CC patch
    - mask : paste resized CC mask (union)
    """
    def __init__(
        self,
        p_choose=0.7,
        min_cc_area=25,
        margin=8,
        scale_range=(0.8, 1.8),   # >1 makes bigger
        max_scale=None,           # optional hard cap based on image size
        feather_blend=True,
        feather_width=18,
        allow_overlap=True,       # allow pasting on top of other CCs
        require_background=False, # if True, only paste where mask is mostly empty
        bg_max_pixels=10,         # threshold for "mostly empty"
        max_tries=80,
        p=0.5
    ):
        super().__init__(p=p)
        self.p_choose = float(p_choose)
        self.min_cc_area = int(min_cc_area)
        self.margin = int(margin)
        self.scale_range = tuple(map(float, scale_range))
        self.max_scale = float(max_scale) if max_scale is not None else None
        self.feather_blend = bool(feather_blend)
        self.feather_width = int(feather_width)
        self.allow_overlap = bool(allow_overlap)
        self.require_background = bool(require_background)
        self.bg_max_pixels = int(bg_max_pixels)
        self.max_tries = int(max_tries)

    @property
    def targets_as_params(self):
        return ["image", "mask"]

    def __call__(self, force_apply=False, **data):
        if not (force_apply or np.random.rand() < self.p):
            return data

        img = data["image"]
        msk = (data["mask"] > 0).astype(np.uint8)
        H, W = msk.shape

        nlabels, labels = cv2.connectedComponents(msk, connectivity=8)
        if nlabels <= 1 or np.random.rand() > self.p_choose:
            data["_scale_paste_ok"] = 0
            return data

        # eligible CCs (bias to larger a bit)
        labs, areas = [], []
        for lab in range(1, nlabels):
            area = int((labels == lab).sum())
            if area >= self.min_cc_area:
                labs.append(lab)
                areas.append(area)
        if not labs:
            data["_scale_paste_ok"] = 0
            return data

        probs = np.asarray(areas, np.float32)
        probs = probs / probs.sum()
        lab = int(np.random.choice(labs, p=probs))

        bbox = component_bbox(labels, lab, self.margin, H, W)
        if bbox is None:
            data["_scale_paste_ok"] = 0
            return data
        y0, y1, x0, x1 = bbox
        h0, w0 = (y1 - y0 + 1), (x1 - x0 + 1)

        # isolate the CC inside bbox
        cc = (labels[y0:y1+1, x0:x1+1] == lab).astype(np.uint8)
        roi_img = img[y0:y1+1, x0:x1+1].copy()
        roi_msk = cc

        # choose scale (bigger and bigger)
        s = float(np.random.uniform(self.scale_range[0], self.scale_range[1]))
        if self.max_scale is not None:
            s = min(s, self.max_scale)

        h1 = max(2, int(round(h0 * s)))
        w1 = max(2, int(round(w0 * s)))

        # cap to image size
        h1 = min(h1, H)
        w1 = min(w1, W)

        # resize image + mask
        roi_img_r = cv2.resize(roi_img, (w1, h1), interpolation=cv2.INTER_LINEAR)
        roi_msk_r = cv2.resize(roi_msk, (w1, h1), interpolation=cv2.INTER_NEAREST)
        roi_msk_r = (roi_msk_r > 0).astype(np.uint8)

        # build "component-only" patch: keep only CC pixels; rest transparent
        # we'll paste by alpha mask so background shows through
        alpha = roi_msk_r[..., None].astype(np.float32)

        # pick destination anywhere it fits (incl corners)
        dy0 = dx0 = None
        for _ in range(self.max_tries):
            dy0_try = np.random.randint(0, H - h1 + 1)
            dx0_try = np.random.randint(0, W - w1 + 1)

            if self.require_background:
                if int(msk[dy0_try:dy0_try+h1, dx0_try:dx0_try+w1].sum()) > self.bg_max_pixels:
                    continue

            if not self.allow_overlap:
                # disallow overlap with any mask pixels at destination
                if int(msk[dy0_try:dy0_try+h1, dx0_try:dx0_try+w1].sum()) > 0:
                    continue

            dy0, dx0 = dy0_try, dx0_try
            break

        if dy0 is None:
            data["_scale_paste_ok"] = 0
            return data

        dy1, dx1 = dy0 + h1 - 1, dx0 + w1 - 1

        img2 = img.copy()
        msk2 = msk.copy()

        # optional feather on the alpha edge
        if self.feather_blend:
            f = feather_alpha(h1, w1, feather=self.feather_width).astype(np.float32)
            alpha = alpha * f  # soften rectangle boundary (still respects CC mask)

        base = img2[dy0:dy1+1, dx0:dx1+1].astype(np.float32)
        paste = roi_img_r.astype(np.float32)

        img2[dy0:dy1+1, dx0:dx1+1] = np.clip(alpha * paste + (1 - alpha) * base, 0, 255).astype(np.uint8)

        # mask union (pasted CC)
        msk2[dy0:dy1+1, dx0:dx1+1] = np.maximum(msk2[dy0:dy1+1, dx0:dx1+1], roi_msk_r)

        data["image"] = img2
        data["mask"] = msk2
        data["_scale_paste_ok"] = 1
        data["_scale_factor"] = s
        data["_paste_box"] = (dy0, dy1, dx0, dx1)
        return data


class ComponentSwapPaste(A.DualTransform):
    """
    Pick ONE connected component, extract its bbox ROI, then SWAP that ROI with a random
    background patch (same size) anywhere in the image (including corners).

    - Image: swaps pixels (optionally feather blends the pasted component)
    - Mask : moves the component mask to the new location; source becomes 0
    """
    def __init__(
        self,
        p_choose=0.35,
        min_cc_area=25,
        margin=8,
        feather_blend=True,
        feather_width=18,
        allow_overlap=False,     # if False, avoid destination overlapping source bbox
        max_tries=50,
        p=0.5
    ):
        super().__init__(p=p)
        self.p_choose = float(p_choose)
        self.min_cc_area = int(min_cc_area)
        self.margin = int(margin)
        self.feather_blend = bool(feather_blend)
        self.feather_width = int(feather_width)
        self.allow_overlap = bool(allow_overlap)
        self.max_tries = int(max_tries)

    @property
    def targets_as_params(self):
        return ["image", "mask"]

    def __call__(self, force_apply=False, **data):
        if not (force_apply or np.random.rand() < self.p):
            return data

        img = data["image"]
        msk = (data["mask"] > 0).astype(np.uint8)
        H, W = msk.shape

        nlabels, labels = cv2.connectedComponents(msk, connectivity=8)
        if nlabels <= 1:
            data["_swap_ok"] = 0
            return data

        # collect eligible CCs
        labs = []
        areas = []
        for lab in range(1, nlabels):
            area = int((labels == lab).sum())
            if area >= self.min_cc_area:
                labs.append(lab)
                areas.append(area)

        if not labs or (np.random.rand() > self.p_choose):
            data["_swap_ok"] = 0
            return data

        # choose one CC (bias to larger a bit)
        areas = np.asarray(areas, np.float32)
        probs = areas / areas.sum()
        lab = int(np.random.choice(labs, p=probs))

        bbox = component_bbox(labels, lab, self.margin, H, W)
        if bbox is None:
            data["_swap_ok"] = 0
            return data

        y0, y1, x0, x1 = bbox
        h = y1 - y0 + 1
        w = x1 - x0 + 1

        # if ROI bigger than image (shouldn't happen, but guard)
        if h <= 1 or w <= 1 or h > H or w > W:
            data["_swap_ok"] = 0
            return data

        # pick destination (anywhere incl corners) where ROI fits
        def overlaps(src, dst):
            (sy0, sy1, sx0, sx1) = src
            (dy0, dy1, dx0, dx1) = dst
            return not (dx1 < sx0 or dx0 > sx1 or dy1 < sy0 or dy0 > sy1)

        src_box = (y0, y1, x0, x1)
        dy0 = dx0 = None

        for _ in range(self.max_tries):
            dy0_try = np.random.randint(0, H - h + 1)
            dx0_try = np.random.randint(0, W - w + 1)
            dst_box = (dy0_try, dy0_try + h - 1, dx0_try, dx0_try + w - 1)

            if self.allow_overlap or (not overlaps(src_box, dst_box)):
                dy0, dx0 = dy0_try, dx0_try
                break

        if dy0 is None:
            data["_swap_ok"] = 0
            return data

        dy1, dx1 = dy0 + h - 1, dx0 + w - 1

        # --- extract patches ---
        src_img = img[y0:y1+1, x0:x1+1].copy()
        src_msk = msk[y0:y1+1, x0:x1+1].copy()

        dst_img = img[dy0:dy1+1, dx0:dx1+1].copy()
        dst_msk = msk[dy0:dy1+1, dx0:dx1+1].copy()

        # we only want to move THIS component's mask (not other stuff in bbox)
        # so isolate that component inside the ROI
        src_cc = (labels[y0:y1+1, x0:x1+1] == lab).astype(np.uint8)

        # build a "component-only" image patch (keep background elsewhere)
        # (this helps if bbox contains other fibers)
        comp_img = src_img.copy()
        comp_img[src_cc == 0] = dst_img[src_cc == 0]  # fill non-CC pixels with destination background

        # --- do swap on IMAGE ---
        img2 = img.copy()

        # put destination background into the source rectangle (pure swap)
        img2[y0:y1+1, x0:x1+1] = dst_img

        # put component patch into destination
        if self.feather_blend:
            a = feather_alpha(h, w, feather=self.feather_width).astype(np.float32)
            base = img2[dy0:dy1+1, dx0:dx1+1].astype(np.float32)
            paste = comp_img.astype(np.float32)
            img2[dy0:dy1+1, dx0:dx1+1] = np.clip(a * paste + (1 - a) * base, 0, 255).astype(np.uint8)
        else:
            img2[dy0:dy1+1, dx0:dx1+1] = comp_img

        # --- do swap on MASK ---
        msk2 = msk.copy()

        # source becomes whatever was at destination (usually background = 0)
        # BUT: we don't want to drag destination's other CCs into the source,
        # so safer: zero out the source box
        msk2[y0:y1+1, x0:x1+1] = 0

        # destination: add the moved CC mask
        # (if you want to overwrite, use '='; if you want union, use max)
        msk2[dy0:dy1+1, dx0:dx1+1] = np.maximum(msk2[dy0:dy1+1, dx0:dx1+1], src_cc)

        data["image"] = img2
        data["mask"] = msk2
        data["_swap_ok"] = 1
        data["_swap_lab"] = lab
        data["_swap_src_box"] = (y0, y1, x0, x1)
        data["_swap_dst_box"] = (dy0, dy1, dx0, dx1)
        return data

    
class OffsetCheckpointNamer(TrainerCallback):
    """
    Renames newly-saved checkpoints checkpoint-S to checkpoint-(OFFSET+S),
    so fresh training doesn't overwrite earlier runs.
    """
    def __init__(self, output_dir: str, offset: int):
        self.output_dir = output_dir
        self.offset = int(offset)
        self._done = set()  # avoid renaming twice

    def on_save(self, args, state, control, **kwargs):
        # Trainer just saved checkpoint-{state.global_step}
        step = int(state.global_step)
        src = os.path.join(self.output_dir, f"checkpoint-{step}")
        if not os.path.isdir(src):
            return control  # nothing to do

        # Don't reprocess if already renamed
        if src in self._done:
            return control

        dst_step = self.offset + step
        dst = os.path.join(self.output_dir, f"checkpoint-{dst_step}")

        # If destination exists, bump until free
        while os.path.exists(dst):
            dst_step += 1
            dst = os.path.join(self.output_dir, f"checkpoint-{dst_step}")

        shutil.move(src, dst)
        self._done.add(dst)

        print(f"[checkpoint rename] {os.path.basename(src)} -> {os.path.basename(dst)} (offset={self.offset})")
        return control

    
class PatchGaussianNoise(A.DualTransform):
    """
    Adds Gaussian noise to random rectangular patches.
    Image only (mask unchanged).
    """

    def __init__(
        self,
        num_patches_range=(1, 4),
        patch_size_range=(32, 128),
        noise_std_range=(5, 25),
        p=0.3,
    ):
        super().__init__(p=p)
        self.num_patches_range = num_patches_range
        self.patch_size_range = patch_size_range
        self.noise_std_range = noise_std_range

    def apply(self, image, **params):
        img = image.astype(np.float32)

        H, W = img.shape[:2]
        n_patches = np.random.randint(
            self.num_patches_range[0],
            self.num_patches_range[1] + 1
        )

        for _ in range(n_patches):

            ph = np.random.randint(*self.patch_size_range)
            pw = np.random.randint(*self.patch_size_range)

            y = np.random.randint(0, max(1, H - ph))
            x = np.random.randint(0, max(1, W - pw))

            std = np.random.uniform(*self.noise_std_range)

            noise = np.random.normal(
                0, std, size=(ph, pw, 3)
            ).astype(np.float32)

            img[y:y+ph, x:x+pw] += noise

        return np.clip(img, 0, 255).astype(np.uint8)

    def apply_to_mask(self, mask, **params):
        return mask

def set_dropout(model, hidden=0.2, attn=0.2, classifier=0.3):
    if hasattr(model.config, "hidden_dropout_prob"):
        model.config.hidden_dropout_prob = hidden
    if hasattr(model.config, "attention_probs_dropout_prob"):
        model.config.attention_probs_dropout_prob = attn
    if hasattr(model.config, "classifier_dropout_prob"):
        model.config.classifier_dropout_prob = classifier

    print("Dropout settings:")
    print("hidden:", getattr(model.config, "hidden_dropout_prob", None))
    print("attention:", getattr(model.config, "attention_probs_dropout_prob", None))
    print("classifier:", getattr(model.config, "classifier_dropout_prob", None))

# ---------------- IO HELPERS ----------------
def list_pairs(folder):
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith((".tif", ".tiff"))])
    unann, ann = {}, {}
    for f in files:
        m1 = re.match(r"Image\s*(\d+)\s*unannotated\.tif(f)?$", f, flags=re.IGNORECASE)
        m2 = re.match(r"Image\s*(\d+)_FLAT\.tif(f)?$", f, flags=re.IGNORECASE)
        if m1: unann[int(m1.group(1))] = os.path.join(folder, f)
        if m2: ann[int(m2.group(1))] = os.path.join(folder, f)
    keys = sorted(set(unann.keys()) & set(ann.keys()))
    return [(k, unann[k], ann[k]) for k in keys]

def read_tif_rgb(path):
    img = tiff.imread(path)
    if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        imin, imax = float(img.min()), float(img.max())
        if imax > imin:
            img = (255.0 * (img - imin) / (imax - imin)).astype(np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)
    return img

def green_mask_from_annotated_rgb(ann_rgb):
    hsv = cv2.cvtColor(ann_rgb, cv2.COLOR_RGB2HSV)
    lower = np.array([35, 120, 60], dtype=np.uint8)
    upper = np.array([85, 255, 255], dtype=np.uint8)
    m = cv2.inRange(hsv, lower, upper)
    m = cv2.medianBlur(m, 5)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=1)
    return (m > 0).astype(np.uint8)

def draw_border(rgb, msk, color=(255, 0, 0), thickness=2):
    out = rgb.copy()
    cnts, _ = cv2.findContours((msk > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0:
        cv2.drawContours(out, cnts, -1, color, thickness)
    return out

def cc_count(binary_mask):
    nlabels, _ = cv2.connectedComponents((binary_mask > 0).astype(np.uint8), connectivity=8)
    return nlabels - 1

def feather_alpha(h, w, feather=18):
    y = np.minimum(np.arange(h), np.arange(h)[::-1]).astype(np.float32)
    x = np.minimum(np.arange(w), np.arange(w)[::-1]).astype(np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    d = np.minimum(yy, xx)
    a = np.clip(d / max(1, feather), 0, 1)
    return a[..., None]  # (H,W,1)

def component_bbox(labels, lab, margin, H, W):
    ys, xs = np.where(labels == lab)
    if len(ys) == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    y0 = max(0, y0 - margin); x0 = max(0, x0 - margin)
    y1 = min(H-1, y1 + margin); x1 = min(W-1, x1 + margin)
    return (y0, y1, x0, x1)

def sample_components(labels, min_area=25, p_choose=0.35, prefer_elongated=True, max_pick=None):
    """
    Returns list of CC labels chosen.
    If prefer_elongated=True, we do weighted Bernoulli (elongated CCs more likely).
    """
    nlabels = int(labels.max()) + 1
    if nlabels <= 1:
        return []

    labs = []
    weights = []
    for lab in range(1, nlabels):
        ys, xs = np.where(labels == lab)
        area = len(ys)
        if area < min_area:
            continue
        labs.append(lab)

        if prefer_elongated:
            h = (ys.max() - ys.min() + 1)
            w = (xs.max() - xs.min() + 1)
            elong = max(h, w) / max(1, min(h, w))  # aspect ratio
            weights.append(float(elong))
        else:
            weights.append(1.0)

    if not labs:
        return []

    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.sum()

    chosen = []
    for lab, w in zip(labs, weights):
        # weighted probability: base p_choose scaled by normalized elongation weight
        p = float(np.clip(p_choose * (w / weights.mean()), 0.0, 1.0)) if prefer_elongated else float(p_choose)
        if np.random.rand() < p:
            chosen.append(lab)

    if max_pick is not None and len(chosen) > max_pick:
        chosen = list(np.random.choice(chosen, size=max_pick, replace=False))

    return chosen

# ---------------- Per-Component ROI Warp Transform ----------------
class ProbabilisticComponentWarp(A.DualTransform):
    """
    For each CC chosen with probability p_choose, warp its fitted bbox locally.
    """
    def __init__(
        self,
        p_choose=0.35,
        min_cc_area=25,
        prefer_elongated=True,
        margin=8,
        pad=96,
        elastic_alpha=900,
        elastic_sigma=4,
        alpha_affine=0,
        feather_blend=True,
        feather_width=18,
        max_pick=8,     # safety: cap number of CCs warped per patch
        p=1.0
    ):
        super().__init__(p=p)
        self.p_choose = float(p_choose)
        self.min_cc_area = int(min_cc_area)
        self.prefer_elongated = bool(prefer_elongated)
        self.margin = int(margin)
        self.pad = int(pad)
        self.feather_blend = bool(feather_blend)
        self.feather_width = int(feather_width)
        self.max_pick = int(max_pick) if max_pick is not None else None

        # Randomize elastic strength PER CALL (so every augmentation is different)
        self.elastic_alpha_range = (200.0, float(elastic_alpha))
        self.elastic_sigma_range = (3.0, float(elastic_sigma))
        self.alpha_affine = float(alpha_affine)

    @property
    def targets_as_params(self):
        return ["image", "mask"]

    def __call__(self, force_apply=False, **data):
        if not (force_apply or np.random.rand() < self.p):
            return data

        # Sample elastic params per augmentation call
        alpha = float(np.random.uniform(*self.elastic_alpha_range))
        sigma = float(np.random.uniform(*self.elastic_sigma_range))

        img = data["image"]
        mask = (data["mask"] > 0).astype(np.uint8)
        H, W = mask.shape

        nlabels, labels = cv2.connectedComponents(mask, connectivity=8)
        if nlabels <= 1:
            return data

        chosen_labs = sample_components(
            labels,
            min_area=self.min_cc_area,
            p_choose=self.p_choose,
            prefer_elongated=self.prefer_elongated,
            max_pick=self.max_pick
        )

        if len(chosen_labs) == 0:
            data["_chosen_cc_count"] = 0
            return data

        img2 = img.copy()
        mask2 = mask.copy()

        # Make ROI augmenter with sampled params (fresh each call)
        roi_aug = A.Compose([
            A.ElasticTransform(
                alpha=alpha,
                sigma=sigma,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_REFLECT_101,
                p=1.0,
            )
        ])

        # warp each chosen CC in its OWN ROI (no union bbox)
        for lab in chosen_labs:
            bbox = component_bbox(labels, lab, self.margin, H, W)
            if bbox is None:
                continue
            y0, y1, x0, x1 = bbox

            img_roi = img2[y0:y1+1, x0:x1+1].copy()
            msk_roi = mask2[y0:y1+1, x0:x1+1].copy()

            H0, W0 = msk_roi.shape[:2]
            PAD = self.pad

            img_pad = cv2.copyMakeBorder(img_roi, PAD, PAD, PAD, PAD, borderType=cv2.BORDER_REFLECT_101)
            msk_pad = cv2.copyMakeBorder(msk_roi, PAD, PAD, PAD, PAD, borderType=cv2.BORDER_CONSTANT, value=0)

            out = roi_aug(image=img_pad, mask=msk_pad)
            img_w = out["image"][PAD:PAD+H0, PAD:PAD+W0]
            msk_w = (out["mask"][PAD:PAD+H0, PAD:PAD+W0] > 0).astype(np.uint8)

            if self.feather_blend:
                a = feather_alpha(H0, W0, feather=self.feather_width).astype(np.float32)
                base = img2[y0:y1+1, x0:x1+1].astype(np.float32)
                img2[y0:y1+1, x0:x1+1] = np.clip(a*img_w.astype(np.float32) + (1-a)*base, 0, 255).astype(np.uint8)
            else:
                img2[y0:y1+1, x0:x1+1] = img_w

            mask2[y0:y1+1, x0:x1+1] = msk_w

        data["image"] = img2
        data["mask"]  = mask2
        data["_chosen_cc_count"] = len(chosen_labs)
        data["_elastic_alpha"] = alpha
        data["_elastic_sigma"] = sigma
        return data



# ---------- Argument Parsing ----------
def parse_args():
    p = argparse.ArgumentParser(description="Train/eval SegFormer on 1+ HF datasets (concat).")

    p.add_argument("--model_id", type=str, required=True)
    p.add_argument("--dataset_ids", type=str, nargs="+", required=True)

    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--learning_rate", type=float, default=1e-5)

    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--max_steps", type=int, default=-1)  # if >0 overrides epochs

    p.add_argument("--train_batch_size", type=int, default=8)
    p.add_argument("--eval_batch_size", type=int, default=8)

    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--logging_steps", type=int, default=50)

    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()

# ---------- Utility ----------
def remap_labels(labels: np.ndarray) -> np.ndarray:
    labels = labels.copy()
    mask1 = (labels >= 0) & (labels <= 227)
    labels[mask1] = 0
    mask2 = (labels >= 228) & (labels <= 255)
    labels[mask2] = 1
    return labels

def compute_metrics(eval_pred):
    processor = SegformerImageProcessor()
    metric = evaluate.load("mean_iou")

    logits, labels = eval_pred
    with torch.no_grad():
        logits_t = torch.from_numpy(logits)
        logits_t = nn.functional.interpolate(
            logits_t,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        preds = logits_t.cpu().numpy()

        metrics = metric._compute(
            predictions=preds,
            references=labels,
            num_labels=2,
            ignore_index=IGNORE,
            reduce_labels=processor.do_reduce_labels,
        )

        acc = metrics.pop("per_category_accuracy", None)
        iou = metrics.pop("per_category_iou", None)
        if acc is not None and iou is not None:
            for i, (a, j) in enumerate(zip(acc.tolist(), iou.tolist())):
                metrics[f"accuracy_class_{i}"] = a
                metrics[f"iou_class_{i}"] = j
        return metrics

# ---------- Main ----------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Model     :", args.model_id)
    print("Datasets  :", " ".join(args.dataset_ids))
    print("Out       :", args.output_dir)
    print("LR        :", args.learning_rate)
    print("Epochs    :", args.num_epochs)
    print("MaxSteps  :", args.max_steps, "(overrides epochs if > 0)")
    print("Train BS  :", args.train_batch_size)
    print("Eval  BS  :", args.eval_batch_size)

    # ---- Load + concat datasets ----
    ds_list = [load_dataset(d) for d in args.dataset_ids]
    train_ds = concatenate_datasets([d["train"] for d in ds_list])


    eval_ds  = concatenate_datasets([d["test"]  for d in ds_list])
    # ---- Build a fixed, balanced eval subset so class-1 isn't missing ----
    def _example_has_pos(ex) -> bool:
        lbl = np.array(Image.fromarray(np.uint8(ex["label"])).convert("L"))
        lbl = remap_labels(lbl)  # must output 0/1
        return bool((lbl > 0).any())
    
    pos_idx, neg_idx = [], []
    MAX_SCAN = min(len(eval_ds), 20000)  # scan up to 20k for speed
    
    for i in range(MAX_SCAN):
        if _example_has_pos(eval_ds[i]):
            pos_idx.append(i)
        else:
            neg_idx.append(i)
    
    print(f"[eval scan] scanned={MAX_SCAN} pos={len(pos_idx)} neg={len(neg_idx)}")
    
    # If we found no positives, it's either the split or remap_labels is wrong
    if len(pos_idx) == 0:
        print("WARNING: No positive pixels found in eval_ds (after remap_labels). "
              "Class-1 metrics will be NaN. Check your labels/remap or eval split.")
    else:
        # Choose a stable fixed subset
        n_pos = min(2000, len(pos_idx))
        n_neg = min(2000, len(neg_idx))
        keep = pos_idx[:n_pos] + neg_idx[:n_neg]
        eval_ds = eval_ds.select(keep)
        print(f"[eval subset] using {len(eval_ds)} examples: pos={n_pos} neg={n_neg}")
    
        

    # ---- Processor ----
    processor = SegformerImageProcessor(do_resize=True, do_normalize=True)

    # ---- Augmentations ----
    rgb_shift_aug = A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5)


    train_aug = A.Compose([
        # Geometry (safe)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.RGBShift(r_shift_limit=18, g_shift_limit=12, b_shift_limit=18, p=0.6),
        A.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            scale=(0.50, 1.50),
            rotate=(-40, 40),
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            mode=cv2.BORDER_REFLECT_101,
            p=0.5
        ),


        # Acquisition realism (one-of)
        A.OneOf([
            A.RandomBrightnessContrast(0.10, 0.10),
            A.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.06, hue=0.02),
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MotionBlur(blur_limit=3),
            A.ISONoise(color_shift=(0.01, 0.2), intensity=(0.1, 0.35)),
            A.ImageCompression(quality_lower=60, quality_upper=95),
        ], p=0.6),

        ComponentSwapPaste(
            p_choose=0.4,        # chance to attempt choosing a CC
            min_cc_area=MIN_CC_AREA,
            margin=ROI_MARGIN,
            feather_blend=True,
            feather_width=18,
            allow_overlap=False,
            p=0.9
        ),

        ComponentScalePaste(
            p_choose=0.8,
            min_cc_area=MIN_CC_AREA,
            margin=ROI_MARGIN,
            scale_range=(1.0, 2.5),     # bigger and bigger
            feather_blend=True,
            feather_width=18,
            require_background=False,   # set True if you want only empty areas
            p=0.6
        ),

        PatchGaussianNoise(
            num_patches_range=(1, 3),
            patch_size_range=(48, 160),
            noise_std_range=(8, 30),
            p=0.4
        ),

        # Your local per-CC warp (keep, but not insane)
        ProbabilisticComponentWarp(
            p_choose=0.25,          # consider lowering from 0.35
            min_cc_area=MIN_CC_AREA,
            prefer_elongated=True,
            margin=ROI_MARGIN,
            pad=PAD,
            elastic_alpha=600,
            elastic_sigma=8,
            alpha_affine=0,
            feather_blend=True,
            feather_width=18,
            max_pick=6,
            p=0.8,                  # not always
        ),
    ])


    
    val_aug = A.Compose([])  # NO randomness, NO tensorization



    def transforms(example_batch, augmentations):
        images, labels = [], []
    
        for img, lbl in zip(example_batch["pixel_values"], example_batch["label"]):
            img = np.array(Image.fromarray(np.uint8(img)).convert("RGB"))  # HWC
            lbl = np.array(Image.fromarray(np.uint8(lbl)).convert("L"))    # HW
    
            lbl = remap_labels(lbl).astype(np.uint8)  # binary before aug
    
            aug = augmentations(image=img, mask=lbl)
        # Ensure dropout holes don't become training supervision
            aug_img = aug["image"]
            aug_msk = aug["mask"]

            holes = (aug_img[...,0] < 8) & (aug_img[...,1] < 8) & (aug_img[...,2] < 8)
            aug_msk = aug_msk.copy()
            aug_msk[holes] = IGNORE

            aug["mask"] = aug_msk


    
            images.append(aug["image"])                  # HWC numpy
            labels.append(aug["mask"].astype(np.uint8))  # HW numpy
    
        return processor(images, labels, return_tensors="pt")
    



    train_ds.set_transform(lambda ex: transforms(ex, train_aug))
    eval_ds.set_transform(lambda ex: transforms(ex, val_aug))

    # ---- Model ----
    id2label = {0: "normal", 1: "abnormality"}
    label2id = {v: k for k, v in id2label.items()}

    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model_id,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    # ---- ADD DROPOUT HERE ----
    if hasattr(model.config, "hidden_dropout_prob"):
        model.config.hidden_dropout_prob = 0.1
    
    if hasattr(model.config, "attention_probs_dropout_prob"):
        model.config.attention_probs_dropout_prob = 0.1
    
    if hasattr(model.config, "classifier_dropout_prob"):
        model.config.classifier_dropout_prob = 0.1
    
    print("Dropout settings:")
    print("hidden:", getattr(model.config, "hidden_dropout_prob", None))
    print("attention:", getattr(model.config, "attention_probs_dropout_prob", None))
    print("classifier:", getattr(model.config, "classifier_dropout_prob", None))


    # ---- Training args ----
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,

        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,  # if >0 overrides epochs

        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,

        load_best_model_at_end=True,
        push_to_hub=args.push_to_hub,
        report_to=["none"],
        seed=args.seed,
        lr_scheduler_type="cosine",        # or "linear"
        warmup_ratio=0.0                 # or warmup_steps=...
            
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )

    import re
    
    def latest_checkpoint(out_dir: str):
        if not os.path.isdir(out_dir):
            return None

        best_path = None
        best_step = -1

        for name in os.listdir(out_dir):
            m = re.match(r"checkpoint-(\d+)$", name)
            if not m:
                continue

            step = int(m.group(1))
            path = os.path.join(out_dir, name)

            if os.path.isdir(path) and step > best_step:
                best_step = step
                best_path = path

        return best_path

    
    ckpt = latest_checkpoint(args.output_dir)

    offset = max_checkpoint_step(args.output_dir)
    print(f"[checkpoint offset] max existing checkpoint step in {args.output_dir} = {offset}")
    trainer.add_callback(OffsetCheckpointNamer(args.output_dir, offset))

    if ckpt:
        print("Loading WEIGHTS ONLY from:", ckpt)
        model = SegformerForSemanticSegmentation.from_pretrained(
            ckpt,
            num_labels=2,
            id2label=id2label,
            label2id=label2id,
        )
        set_dropout(model, hidden=0.3, attn=0.3, classifier=0.3)
        trainer.model = model
    else:
        print("No checkpoint found. Starting from base model:", args.model_id)

    trainer.train(resume_from_checkpoint=False)  # fresh LR/optimizer



if __name__ == "__main__":
    main()
