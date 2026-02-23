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
ELASTIC_ALPHA = 250
ELASTIC_SIGMA = 4
ELASTIC_ALPHA_AFFINE = 0  # keep 0 for micro-local realism

# --- Seam handling ---
FEATHER_BLEND = True
FEATHER_WIDTH = 18

if SEED is not None:
    random.seed(SEED)
    np.random.seed(SEED)

import torch.nn.functional as F

# ---------------- Diffusion-like forward noise (for Option A) ----------------
# Simple DDPM-style forward process: x_t = sqrt(a_bar)*x0 + sqrt(1-a_bar)*eps
_DIFF_T = 1000
_BETA_START = 1e-4
_BETA_END = 2e-2

_betas = np.linspace(_BETA_START, _BETA_END, _DIFF_T, dtype=np.float64)
_alphas = 1.0 - _betas
_alpha_bars = np.cumprod(_alphas, axis=0).astype(np.float64)  # shape (T,)

def diffusion_forward_u8(img_u8: np.ndarray, t: int = None) -> np.ndarray:
    """
    img_u8: HxWx3 uint8 RGB
    returns: HxWx3 uint8 RGB, forward-diffused (noised) version
    """
    if t is None:
        # t = np.random.randint(0, _DIFF_T)
        t_max = 250   # try 150–300
        t = np.random.randint(0, t_max)

    a_bar = float(_alpha_bars[t])
    # Convert to float [0,1]
    x0 = img_u8.astype(np.float32) / 255.0
    eps = np.random.randn(*x0.shape).astype(np.float32)

    xt = (np.sqrt(a_bar) * x0) + (np.sqrt(1.0 - a_bar) * eps)
    xt = np.clip(xt, 0.0, 1.0)
    return (xt * 255.0).astype(np.uint8)

    
import re, os, shutil
from transformers import TrainerCallback


import numpy as np
import cv2
import albumentations as A


import numpy as np
import cv2
import albumentations as A

import copy
import torch.nn.functional as F


from transformers import Trainer

from sklearn.mixture import GaussianMixture

from sklearn.mixture import GaussianMixture
import numpy as np
import cv2
import albumentations as A

IGNORE = 255

import os, re

import os, re, shutil
from transformers import TrainerCallback

def fill_added_pixels_directional(img, old_mask01, added_mask01, src_yx, patch_r=3):
    """
    Fill ONLY added pixels using the local mean color around src_yx inside old_mask01.
    src_yx: (y,x) picked inside the original CC near the endpoint.
    """
    img2 = img.copy()
    old = (old_mask01 > 0).astype(np.uint8)
    added = (added_mask01 > 0) & (old == 0)
    if not added.any():
        return img2

    y, x = src_yx
    H, W = old.shape
    y0, y1 = max(0, y-patch_r), min(H, y+patch_r+1)
    x0, x1 = max(0, x-patch_r), min(W, x+patch_r+1)

    region = img[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
    if region.size == 0:
        return img2

    # mean/median color from that local neighborhood
    col = np.median(region, axis=0).astype(np.uint8)

    ay, ax = np.where(added)
    img2[ay, ax] = col
    return img2



def fill_added_pixels_from_old(
    img_uint8,
    old_mask01,
    new_mask01,
    context_dilate=6,          # include some neighborhood around old GT
    brown_hue=(5, 35),         # HSV hue range for brown/orange (OpenCV hue 0..179)
    min_sat=35,
    min_val=30,
    fallback_to_dilated=True
):
    """
    Fill newly-added GT pixels by copying from nearest *brown-ish* pixels
    around the original GT. This keeps extensions looking like stain/tissue,
    not random bluish background.

    - Uses a dilated old mask to define a 'source region'
    - Filters that source region to only brown-ish pixels in HSV
    - Uses distanceTransformWithLabels to copy from nearest valid source pixel
    """
    img2 = img_uint8.copy()
    old = (old_mask01 > 0).astype(np.uint8)
    new = (new_mask01 > 0).astype(np.uint8)
    added = (new == 1) & (old == 0)
    if not added.any():
        return img2

    H, W = old.shape

    # 1) build source region = old GT dilated
    if context_dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*context_dilate+1, 2*context_dilate+1))
        src_region = cv2.dilate(old, k, iterations=1)
    else:
        src_region = old.copy()

    # 2) pick only brown-ish pixels inside src_region
    hsv = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)
    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    brown = (
        (src_region == 1) &
        (h >= brown_hue[0]) & (h <= brown_hue[1]) &
        (s >= min_sat) &
        (v >= min_val)
    ).astype(np.uint8)

    # If too strict, fallback to src_region (still better than old-only)
    if brown.sum() < 10 and fallback_to_dilated:
        brown = src_region.astype(np.uint8)

    # If still empty, fallback to old mask (last resort)
    if brown.sum() < 1:
        brown = old.astype(np.uint8)

    # 3) nearest-neighbor copy from valid source pixels
    inv = (brown == 0).astype(np.uint8)  # zeros are "targets" for distanceTransformWithLabels
    _, labels = cv2.distanceTransformWithLabels(inv, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)

    lab = labels.astype(np.int64)
    idx = lab[added] - 1
    idx = np.clip(idx, 0, H * W - 1)
    yy = (idx // W).astype(np.int64)
    xx = (idx % W).astype(np.int64)

    ay, ax = np.where(added)
    img2[ay, ax] = img2[yy, xx]
    return img2


class OffsetCheckpointNamer(TrainerCallback):
    """
    After each checkpoint is saved, rename:
      output_dir/checkpoint-STEP  ->  output_dir/checkpoint-(STEP + offset)

    This is useful when you "continue training" but want step numbers to keep increasing
    even though you are NOT using resume_from_checkpoint (fresh optimizer/LR).
    """
    def __init__(self, output_dir: str, offset: int):
        self.output_dir = output_dir
        self.offset = int(offset)

    def on_save(self, args, state, control, **kwargs):
        # state.global_step is the step that was just saved as checkpoint-<global_step>
        step = int(state.global_step)
        src = os.path.join(self.output_dir, f"checkpoint-{step}")
        if not os.path.isdir(src):
            return control

        new_step = step + self.offset
        dst = os.path.join(self.output_dir, f"checkpoint-{new_step}")

        # If destination already exists, don't clobber it
        if os.path.exists(dst):
            print(f"[OffsetCheckpointNamer] dst exists, skipping: {dst}")
            return control

        # Rename/move folder
        shutil.move(src, dst)

        # Also fix trainer_state.json path inside the checkpoint if it exists (optional)
        ts = os.path.join(dst, "trainer_state.json")
        if os.path.isfile(ts):
            try:
                with open(ts, "r", encoding="utf-8") as f:
                    s = f.read()
                # This is a light-touch fix; safe if it doesn't match.
                s2 = re.sub(r'"global_step"\s*:\s*\d+', f'"global_step": {new_step}', s)
                if s2 != s:
                    with open(ts, "w", encoding="utf-8") as f:
                        f.write(s2)
            except Exception as e:
                print(f"[OffsetCheckpointNamer] trainer_state.json update failed: {e}")

        print(f"[OffsetCheckpointNamer] {src} -> {dst}")
        return control

    


class ReduceLROnPlateauCallback(TrainerCallback):
    def __init__(self, monitor="eval_mean_iou", factor=0.5, patience=5, min_lr=1e-7, threshold=1e-4):
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.threshold = threshold
        self.best = None
        self.bad_count = 0
        self.trainer = None  # <-- add

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or self.monitor not in metrics or self.trainer is None:
            return control

        val = float(metrics[self.monitor])

        if self.best is None or val > self.best + self.threshold:
            self.best = val
            self.bad_count = 0
            return control

        self.bad_count += 1
        if self.bad_count < self.patience:
            return control

        opt = self.trainer.optimizer
        for pg in opt.param_groups:
            pg["lr"] = max(self.min_lr, float(pg["lr"]) * self.factor)

        print(f"[ReduceLROnPlateau] {self.monitor} plateaued. LR -> {opt.param_groups[0]['lr']:.3e}")
        self.bad_count = 0
        return control


class MaskedHSVShift(A.DualTransform):
    """
    Apply HSV shift only inside GT. Useful for 'brownify' behavior.
    hue_shift: small (brown/orange region)
    sat_shift: increase saturation for stain-like look
    val_shift: small darker or brighter
    """
    def __init__(
        self,
        hue_shift=(-6, 6),
        sat_shift=(5, 35),
        val_shift=(-10, 10),
        feather_px=6,
        p=0.5
    ):
        super().__init__(p=p)
        self.hue_shift = hue_shift
        self.sat_shift = sat_shift
        self.val_shift = val_shift
        self.feather_px = int(feather_px)

    @property
    def targets_as_params(self):
        return ["image", "mask"]

    def __call__(self, force_apply=False, **data):
        if not (force_apply or np.random.rand() < self.p):
            data["_masked_hsv_ok"] = 0
            return data

        img = data["image"].copy()  # RGB uint8
        msk_u8 = data["mask"].astype(np.uint8)
        ignore = (msk_u8 == IGNORE)
        gt = (msk_u8 == 1)
        if not gt.any():
            data["_masked_hsv_ok"] = 0
            return data

        dh = int(np.random.randint(self.hue_shift[0], self.hue_shift[1] + 1))
        ds = int(np.random.randint(self.sat_shift[0], self.sat_shift[1] + 1))
        dv = int(np.random.randint(self.val_shift[0], self.val_shift[1] + 1))

        # alpha w/ feather
        alpha = gt.astype(np.float32)
        if self.feather_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*self.feather_px+1, 2*self.feather_px+1))
            dil = cv2.dilate(alpha, k, 1)
            ero = cv2.erode(alpha, k, 1)
            band = np.clip(dil - ero, 0, 1)
            blur_k = max(3, 2*self.feather_px + 1)
            if blur_k % 2 == 0: blur_k += 1
            band = cv2.GaussianBlur(band, (blur_k, blur_k), 0)
            alpha = np.clip(ero + band, 0, 1)

        alpha3 = alpha[..., None]

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int16)
        hsv2 = hsv.copy()
        hsv2[..., 0] = (hsv2[..., 0] + dh) % 180
        hsv2[..., 1] = np.clip(hsv2[..., 1] + ds, 0, 255)
        hsv2[..., 2] = np.clip(hsv2[..., 2] + dv, 0, 255)

        rgb2 = cv2.cvtColor(hsv2.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
        base = img.astype(np.float32)

        out = np.clip(alpha3 * rgb2 + (1 - alpha3) * base, 0, 255).astype(np.uint8)
        out[ignore] = img[ignore]

        data["image"] = out
        data["_masked_hsv_ok"] = 1
        data["_masked_hsv_shift"] = (dh, ds, dv)
        return data

    
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

def max_checkpoint_step(out_dir: str) -> int:
    """Return the largest N from checkpoint-N in out_dir, or 0 if none."""
    if not os.path.isdir(out_dir):
        return 0

    mx = 0
    for name in os.listdir(out_dir):
        m = re.match(r"checkpoint-(\d+)$", name)
        if m:
            mx = max(mx, int(m.group(1)))

    return mx  # <<< YOU WERE MISSING THIS

import numpy as np
import cv2
import albumentations as A

IGNORE = 255

class MaskedRGBShift(A.DualTransform):
    """
    Apply RGBShift only inside GT (mask==1). Keeps IGNORE pixels unchanged.
    Optionally feather edges for realism.
    """
    def __init__(
        self,
        r_shift_limit=(-10, 30),
        g_shift_limit=(-10, 20),
        b_shift_limit=(-10, 10),
        feather_px=6,         # 0 = hard edge, 4-12 looks nice
        p=0.5
    ):
        super().__init__(p=p)
        self.r_shift_limit = r_shift_limit
        self.g_shift_limit = g_shift_limit
        self.b_shift_limit = b_shift_limit
        self.feather_px = int(feather_px)

    @property
    def targets_as_params(self):
        return ["image", "mask"]

    def __call__(self, force_apply=False, **data):
        if not (force_apply or np.random.rand() < self.p):
            data["_masked_rgb_ok"] = 0
            return data

        img = data["image"].copy()  # uint8 RGB
        msk_u8 = data["mask"].astype(np.uint8)
        ignore = (msk_u8 == IGNORE)
        gt = (msk_u8 == 1)

        if not gt.any():
            data["_masked_rgb_ok"] = 0
            return data

        # random shifts (same for whole region)
        r = int(np.random.randint(self.r_shift_limit[0], self.r_shift_limit[1] + 1))
        g = int(np.random.randint(self.g_shift_limit[0], self.g_shift_limit[1] + 1))
        b = int(np.random.randint(self.b_shift_limit[0], self.b_shift_limit[1] + 1))

        # build alpha (optionally feather boundary)
        alpha = gt.astype(np.float32)
        if self.feather_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*self.feather_px+1, 2*self.feather_px+1))
            # make a soft band around GT
            dil = cv2.dilate(alpha, k, 1)
            ero = cv2.erode(alpha, k, 1)
            band = np.clip(dil - ero, 0, 1)

            # blur band to get smooth feather
            blur_k = max(3, 2*self.feather_px + 1)
            if blur_k % 2 == 0: blur_k += 1
            band = cv2.GaussianBlur(band, (blur_k, blur_k), 0)

            alpha = np.clip(ero + band, 0, 1)

        alpha3 = alpha[..., None]

        img_f = img.astype(np.float32)
        shifted = img_f.copy()
        shifted[..., 0] = np.clip(shifted[..., 0] + r, 0, 255)
        shifted[..., 1] = np.clip(shifted[..., 1] + g, 0, 255)
        shifted[..., 2] = np.clip(shifted[..., 2] + b, 0, 255)

        out = np.clip(alpha3 * shifted + (1 - alpha3) * img_f, 0, 255).astype(np.uint8)

        # don't touch IGNORE pixels (optional, but consistent with your logic)
        out[ignore] = img[ignore]

        data["image"] = out
        data["_masked_rgb_ok"] = 1
        data["_masked_rgb_shift"] = (r, g, b)
        return data

    

import copy
import torch
import torch.nn.functional as F
from transformers import Trainer
import copy
import torch
import torch.nn.functional as F
from transformers import Trainer

# set this to your ignore id (you used 255)
IGNORE = 255

class EMATeacherAmbiguityIgnoreTrainer(Trainer):
    """
    EMA teacher that masks ambiguous pixels:
      - If teacher is high-confidence and disagrees with GT, set that pixel to IGNORE
      - This prevents weight updates from those pixels (no gradient there)

    Expected batch:
      - pixel_values: (B,3,H,W)
      - labels:       (B,H,W) with {0,1,IGNORE}
    """

    def __init__(
        self,
        *args,
        ema_decay=0.999,
        warmup_steps=0,

        # High-confidence thresholds:
        # p(class=1) > tau_pos  => confident positive
        # p(class=1) < 1-tau_neg => confident negative  (i.e., p(class=0) > tau_neg)
        tau_pos=0.97,
        tau_neg=0.97,

        # optional: only ignore if teacher is ALSO very certain overall (helps with noisy teacher early)
        min_conf=None,   # e.g. 0.90, or None to disable

        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.ema_decay = float(ema_decay)
        self.warmup_steps = int(warmup_steps)
        self.tau_pos = float(tau_pos)
        self.tau_neg = float(tau_neg)
        self.min_conf = None if min_conf is None else float(min_conf)

        # EMA teacher = copy of student
        self.teacher = copy.deepcopy(self.model).eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self._teacher_on_device = False

    def _place_model_on_device(self):
        super()._place_model_on_device()
        device = next(self.model.parameters()).device
        self.teacher.to(device)
        self._teacher_on_device = True

    @torch.no_grad()
    def _ema_update(self):
        d = self.ema_decay

        if not self._teacher_on_device:
            device = next(self.model.parameters()).device
            self.teacher.to(device)
            self._teacher_on_device = True

        msd = self.model.state_dict()
        tsd = self.teacher.state_dict()

        for k in tsd.keys():
            t = tsd[k]
            m = msd[k]
            if m.device != t.device:
                m = m.to(t.device)

            if torch.is_floating_point(t):
                t.mul_(d).add_(m, alpha=1.0 - d)
            else:
                t.copy_(m)

    def training_step(self, model, inputs):
        # update teacher every step (after step 0)
        if self.state.global_step > 0:
            self._ema_update()
        return super().training_step(model, inputs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        pixel_values = inputs["pixel_values"]

        # ---- mask ambiguous pixels using teacher disagreement ----
        if self.state.global_step >= self.warmup_steps:
            with torch.no_grad():
                t_out = self.teacher(pixel_values=pixel_values)
                t_probs = torch.softmax(t_out.logits, dim=1)  # (B,C,h,w)

                # class-1 prob at teacher resolution
                p1 = t_probs[:, 1:2]  # (B,1,h,w)

                # upsample to label size
                p1 = F.interpolate(
                    p1,
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                ).squeeze(1)  # (B,H,W)

                # confident positive / negative
                conf_pos = p1 > self.tau_pos
                conf_neg = p1 < (1.0 - self.tau_neg)

                if self.min_conf is not None:
                    # optional: require max prob across classes to be high too
                    pmax = torch.max(t_probs, dim=1).values  # (B,h,w)
                    pmax = F.interpolate(
                        pmax.unsqueeze(1),
                        size=labels.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    ).squeeze(1)  # (B,H,W)
                    strong = pmax >= self.min_conf
                    conf_pos = conf_pos & strong
                    conf_neg = conf_neg & strong

                # disagreements with GT (ignore only where GT is not already IGNORE)
                valid_gt = (labels != IGNORE)
                disagree_pos = valid_gt & (labels == 0) & conf_pos  # teacher says 1, GT says 0
                disagree_neg = valid_gt & (labels == 1) & conf_neg  # teacher says 0, GT says 1

                if (disagree_pos.any() or disagree_neg.any()):
                    labels = labels.clone()
                    labels[disagree_pos | disagree_neg] = IGNORE
                    inputs["labels"] = labels

        # ---- normal supervised loss on remaining pixels ----
        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

    
class FillBlackNearest(A.ImageOnlyTransform):
    def __init__(self, p=1.0):
        super().__init__(p=p)

    def apply(self, img, **params):
        return fill_black_nearest(img)

   
def _nearest_from_mask(img, mask01):
    """
    For every pixel, find nearest pixel where mask01==1 and return its RGB.
    Uses OpenCV DIST_LABEL_PIXEL trick (fast, no scipy).
    """
    H, W = mask01.shape
    old = (mask01 > 0).astype(np.uint8)
    inv = (old == 0).astype(np.uint8)  # zeros are targets (old==1)

    # distance + labels of nearest zero pixel in inv => nearest old==1 pixel
    dist, labels = cv2.distanceTransformWithLabels(
        inv, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
    )

    lab = labels.astype(np.int64)
    idx = lab - 1
    idx = np.clip(idx, 0, H * W - 1)
    yy = (idx // W).astype(np.int64)
    xx = (idx % W).astype(np.int64)

    src = img[yy, xx]  # (H,W,3)
    return dist.astype(np.float32), src.astype(np.float32)
class BridgeNearbyEndpoints(A.DualTransform):
    """
    Connect disjoint GT fragments that are close (like a broken strip).
    - Uses ONLY positive pixels (mask==1)
    - Preserves IGNORE pixels (mask==255)
    - Optionally fills image on newly added GT pixels
    """
    def __init__(
        self,
        max_gap_px=18,
        max_bridges=3,
        line_thickness=(1, 2),
        min_cc_area=20,
        fill_image=True,
        p=0.5
    ):
        super().__init__(p=p)
        self.max_gap_px = int(max_gap_px)
        self.max_bridges = int(max_bridges)
        self.line_thickness = tuple(map(int, line_thickness))
        self.min_cc_area = int(min_cc_area)
        self.fill_image = bool(fill_image)

    @property
    def targets_as_params(self):
        return ["image", "mask"]

    def __call__(self, force_apply=False, **data):
        if not (force_apply or np.random.rand() < self.p):
            data["_bridge_ok"] = 0
            return data

        img = data["image"]

        # --- IMPORTANT: split mask into pos vs ignore ---
        msk_u8 = data["mask"].astype(np.uint8)          # expected 0/1/255
        ignore = (msk_u8 == IGNORE)
        msk = (msk_u8 == 1).astype(np.uint8)            # ONLY positives

        if int(msk.sum()) == 0:
            data["_bridge_ok"] = 0
            return data

        H, W = msk.shape
        old = msk.copy()

        nlabels, labels = cv2.connectedComponents(msk, connectivity=8)
        if nlabels <= 2:
            data["_bridge_ok"] = 0
            return data

        # collect CC masks + endpoints
        comps = []
        for lab in range(1, nlabels):
            cc = (labels == lab).astype(np.uint8)
            if int(cc.sum()) < self.min_cc_area:
                continue
            sk = _morph_skeleton(cc)
            eps = _endpoints_from_skeleton(sk)  # list[(y,x)]
            if len(eps) == 0:
                continue
            comps.append((lab, cc, eps))

        if len(comps) < 2:
            data["_bridge_ok"] = 0
            return data

        # candidate endpoint pairs across different CCs
        cand = []
        for i in range(len(comps)):
            _, _, eps_i = comps[i]
            for j in range(i + 1, len(comps)):
                _, _, eps_j = comps[j]
                for (y1, x1) in eps_i:
                    for (y2, x2) in eps_j:
                        d = float(np.hypot(y1 - y2, x1 - x2))
                        if d <= self.max_gap_px:
                            cand.append((d, (x1, y1), (x2, y2)))

        if not cand:
            data["_bridge_ok"] = 0
            return data

        cand.sort(key=lambda t: t[0])

        out = msk.copy()
        used = 0
        for _, p1, p2 in cand:
            if used >= self.max_bridges:
                break

            th = int(np.random.randint(self.line_thickness[0], self.line_thickness[1] + 1))
            tmp = np.zeros((H, W), np.uint8)
            cv2.line(tmp, p1, p2, color=1, thickness=th, lineType=cv2.LINE_AA)

            before = int(out.sum())
            out = np.maximum(out, tmp)
            after = int(out.sum())

            if after > before:
                used += 1

        if used == 0:
            data["_bridge_ok"] = 0
            return data

        # fill image for newly added GT pixels
        if self.fill_image:
            img2 = fill_added_pixels_from_old(img, old, out)
        else:
            img2 = img

        # restore IGNORE in the final mask
        out_u8 = out.astype(np.uint8)
        out_u8[ignore] = IGNORE

        data["image"] = img2
        data["mask"] = out_u8
        data["_bridge_ok"] = 1
        data["_bridges_done"] = used
        return data


class DiffuseGTIntoNeighborhood(A.DualTransform):
    """
    Make GT look like it diffuses/spreads (foggy halo) into nearby region.
    - Image: pulls color from nearest GT pixel and blends outward with a soft alpha.
    - Mask: expands consistently (dilate or alpha-threshold).

    Works best for stain-like sparse positives.
    """
    def __init__(
        self,
        radius_range=(6, 30),          # how far diffusion can reach
        sigma_frac=(0.35, 0.75),       # sigma = frac * radius
        alpha_power=(0.8, 1.4),        # shape of falloff
        max_alpha=(0.35, 0.85),        # strength of diffusion
        mask_mode="dilate",            # "dilate" or "alpha_thresh"
        mask_thresh=(0.25, 0.45),      # used if mask_mode="alpha_thresh"
        p=0.5,
    ):
        super().__init__(p=p)
        self.radius_range = radius_range
        self.sigma_frac = sigma_frac
        self.alpha_power = alpha_power
        self.max_alpha = max_alpha
        self.mask_mode = str(mask_mode)
        self.mask_thresh = mask_thresh

    @property
    def targets_as_params(self):
        return ["image", "mask"]

    def __call__(self, force_apply=False, **data):
        if not (force_apply or np.random.rand() < self.p):
            data["_diffuse_ok"] = 0
            return data

        img = data["image"].copy()
        # msk = (data["mask"] > 0).astype(np.uint8)
        msk_u8 = data["mask"].astype(np.uint8)
        ignore = (msk_u8 == IGNORE)
        msk = (msk_u8 == 1).astype(np.uint8)


        if msk.sum() == 0:
            data["_diffuse_ok"] = 0
            return data

        R = int(np.random.randint(self.radius_range[0], self.radius_range[1] + 1))
        frac = float(np.random.uniform(self.sigma_frac[0], self.sigma_frac[1]))
        sigma = max(1.0, frac * R)
        pwr = float(np.random.uniform(self.alpha_power[0], self.alpha_power[1]))
        amax = float(np.random.uniform(self.max_alpha[0], self.max_alpha[1]))

        dist, src = _nearest_from_mask(img, msk)  # dist to GT, src RGB from nearest GT pixel

        # halo region (outside GT but within R)
        outside = (msk == 0)
        halo = outside & (dist <= R)
        if not halo.any():
            data["_diffuse_ok"] = 0
            return data

        # soft alpha falloff
        d = dist.copy()
        alpha = np.zeros_like(d, dtype=np.float32)
        alpha[halo] = np.exp(-(d[halo] ** 2) / (2.0 * (sigma ** 2)))
        alpha[halo] = np.clip(alpha[halo], 0, 1) ** pwr
        alpha[halo] *= amax

        alpha3 = alpha[..., None]

        img_f = img.astype(np.float32)
        # blend only in halo; keep original inside GT
        img_f = (1.0 - alpha3) * img_f + alpha3 * src
        img_out = np.clip(img_f, 0, 255).astype(np.uint8)

        # mask expansion
        if self.mask_mode == "alpha_thresh":
            t = float(np.random.uniform(self.mask_thresh[0], self.mask_thresh[1]))
            msk_out = np.maximum(msk, (alpha > t).astype(np.uint8))
        else:
            # dist<=R is equivalent to dilation by R in Euclidean sense
            msk_out = np.maximum(msk, (dist <= R).astype(np.uint8))

        out_u8 = msk_out.astype(np.uint8)
        out_u8[ignore] = IGNORE
        data["mask"] = out_u8

        data["image"] = img_out
        # data["mask"] = msk_out
        data["_diffuse_ok"] = 1
        data["_diffuse_R"] = R
        data["_diffuse_sigma"] = sigma
        data["_diffuse_amax"] = amax
        data["_diffuse_mode"] = self.mask_mode
        return data

    
import numpy as np
import cv2

import numpy as np
import cv2
import albumentations as A

IGNORE = 255

class WhiteTissueDropout(A.DualTransform):
    """
    Simulate missing tissue / white voids / white strips seen in histology.
    - Paints 1..N white blobs or long strips on the IMAGE.
    - Marks the same pixels as IGNORE in the MASK (so no supervision there).
    """
    def __init__(
        self,
        n_shapes=(1, 3),
        mode_probs=(0.6, 0.4),  # (blob, strip)
        blob_radius=(18, 90),
        strip_thickness=(18, 80),
        strip_length_frac=(0.5, 1.3),  # fraction of patch width
        whiteness=(235, 255),          # white-ish, not pure sometimes
        blur_ksize=(7, 19),            # soften edges
        p=0.35
    ):
        super().__init__(p=p)
        self.n_shapes = n_shapes
        self.mode_probs = mode_probs
        self.blob_radius = blob_radius
        self.strip_thickness = strip_thickness
        self.strip_length_frac = strip_length_frac
        self.whiteness = whiteness
        self.blur_ksize = blur_ksize

    @property
    def targets_as_params(self):
        return ["image", "mask"]

    def __call__(self, force_apply=False, **data):
        if not (force_apply or np.random.rand() < self.p):
            data["_white_dropout_ok"] = 0
            return data

        img = data["image"].copy()                # uint8 RGB
        msk = data["mask"].copy().astype(np.uint8)  # 0/1/255
        H, W = msk.shape

        occ = np.zeros((H, W), np.uint8)

        n = np.random.randint(self.n_shapes[0], self.n_shapes[1] + 1)
        for _ in range(n):
            mode = np.random.choice(["blob", "strip"], p=np.array(self.mode_probs)/np.sum(self.mode_probs))

            if mode == "blob":
                r = np.random.randint(self.blob_radius[0], self.blob_radius[1] + 1)
                cy = np.random.randint(0, H)
                cx = np.random.randint(0, W)
                cv2.circle(occ, (cx, cy), r, 255, thickness=-1)

            else:  # strip
                th = np.random.randint(self.strip_thickness[0], self.strip_thickness[1] + 1)
                length = int(np.random.uniform(self.strip_length_frac[0], self.strip_length_frac[1]) * W)
                length = np.clip(length, 20, int(2.0*W))
                cx = np.random.randint(0, W)
                cy = np.random.randint(0, H)
                angle = np.random.uniform(0, 180)

                # draw a thick line across
                dx = int(np.cos(np.deg2rad(angle)) * length / 2)
                dy = int(np.sin(np.deg2rad(angle)) * length / 2)
                p1 = (int(cx - dx), int(cy - dy))
                p2 = (int(cx + dx), int(cy + dy))
                cv2.line(occ, p1, p2, 255, thickness=th, lineType=cv2.LINE_AA)

                # ---- deform the white mask (makes shapes irregular) ----
        if np.random.rand() < 0.85:  # deformation probability

            alpha = np.random.uniform(120, 420)   # strength
            sigma = np.random.uniform(6, 14)      # smoothness

            aug = A.ElasticTransform(
                alpha=float(alpha),
                sigma=float(sigma),
                interpolation=cv2.INTER_NEAREST,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,          # <-- replaces value
                fill_mask=0,     # <-- explicitly for mask
                p=1.0
            )


            out = aug(image=occ, mask=occ)
            occ = out["image"]


        # soften edges
        k = int(np.random.randint(self.blur_ksize[0], self.blur_ksize[1] + 1))
        if k % 2 == 0: k += 1
        occ_blur = cv2.GaussianBlur(occ, (k, k), 0)

        # threshold to final mask of affected pixels
        occ01 = (occ_blur > 40).astype(np.uint8)

        # paint white-ish
        # white = int(np.random.randint(self.whiteness[0], self.whiteness[1] + 1))
        # pixels you are DROPPING (occlusions you drew)
        drop = (occ01 == 1)

        # choose one whiteness value for this whole dropout
        w = int(np.random.randint(self.whiteness[0], self.whiteness[1] + 1))

        # paint dropout pixels white-ish
        img[drop] = (w, w, w)

        # ignore supervision there
        msk[drop] = IGNORE

        # ALSO ignore pre-existing "flat white" holes already in the image (optional)
        flat_white = (img.mean(axis=-1) > 245) & (img.std(axis=-1) < 5)
        msk[flat_white] = IGNORE


        data["image"] = img
        data["mask"] = msk
        data["_white_dropout_ok"] = 1
        data["_white_dropout_px"] = int(occ01.sum())
        return data

    

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

import numpy as np
import cv2
import albumentations as A

def _morph_skeleton(binary01: np.ndarray) -> np.ndarray:
    """
    Morphological skeletonization (no ximgproc required).
    Input: binary uint8 {0,1}
    Output: skeleton uint8 {0,1}
    """
    img = (binary01 > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break

    return (skel > 0).astype(np.uint8)

def _endpoints_from_skeleton(skel01: np.ndarray) -> np.ndarray:
    """
    Endpoints = skeleton pixels with exactly 1 neighbor in 8-connectivity.
    Returns list of (y,x).
    """
    s = (skel01 > 0).astype(np.uint8)
    # count neighbors via convolution
    k = np.ones((3,3), np.uint8)
    neigh = cv2.filter2D(s, -1, k)  # includes self
    # endpoint: self=1 and neigh==2 (self + 1 neighbor)
    ep = (s == 1) & (neigh == 2)
    ys, xs = np.where(ep)
    return list(zip(ys.tolist(), xs.tolist()))

def _pca_direction(points_yx: np.ndarray) -> np.ndarray:
    """
    PCA direction (unit vector) from a set of points (N,2) in (y,x) coords.
    Returns direction in (dy,dx).
    """
    if len(points_yx) < 2:
        return None
    pts = points_yx.astype(np.float32)
    pts_mean = pts.mean(axis=0, keepdims=True)
    X = pts - pts_mean
    # covariance
    C = (X.T @ X) / max(1, (len(pts) - 1))
    w, v = np.linalg.eigh(C)  # eigenvectors columns
    d = v[:, np.argmax(w)]    # principal axis in (y,x)
    norm = np.linalg.norm(d) + 1e-8
    d = d / norm
    return d  # (dy,dx)


class GrowAlongSkeleton(A.DualTransform):
    """
    Grow/elongate thin GT structures by extending skeleton endpoints.

    - Mask: extend endpoints along local PCA direction
    - Image: fill ONLY newly added pixels (either directional median, or brown-ish nearest)
    - Preserves IGNORE=255 pixels
    """
    def __init__(
        self,
        p_choose_cc=0.5,
        min_cc_area=25,
        endpoint_len_range=(6, 20),
        thickness_range=(1, 3),
        endpoint_count_cap=4,
        endpoint_radius=12,
        fill_mode="directional",   # "directional" | "brown_nearest" | "none"
        backtrack_px=6,
        patch_r=4,
        p=0.5
    ):
        super().__init__(p=p)
        self.p_choose_cc = float(p_choose_cc)
        self.min_cc_area = int(min_cc_area)
        self.endpoint_len_range = tuple(map(int, endpoint_len_range))
        self.thickness_range = tuple(map(int, thickness_range))
        self.endpoint_count_cap = int(endpoint_count_cap)
        self.endpoint_radius = int(endpoint_radius)

        self.fill_mode = str(fill_mode).lower()
        self.backtrack_px = int(backtrack_px)
        self.patch_r = int(patch_r)

    @property
    def targets_as_params(self):
        return ["image", "mask"]

    def __call__(self, force_apply=False, **data):
        if not (force_apply or np.random.rand() < self.p):
            data["_grow_ok"] = 0
            data["_grow_added_px"] = 0
            return data

        img = data["image"]
        msk_u8 = data["mask"].astype(np.uint8)
        ignore = (msk_u8 == IGNORE)
        msk = (msk_u8 == 1).astype(np.uint8)

        H, W = msk.shape
        nlabels, labels = cv2.connectedComponents(msk, connectivity=8)
        if nlabels <= 1:
            data["_grow_ok"] = 0
            data["_grow_added_px"] = 0
            return data

        old_msk = msk.copy()
        out_msk = msk.copy()

        # store per-endpoint "source points" for directional filling
        src_points = []  # list[(srcy, srcx)]

        added_total = 0
        grew_any = False

        for lab in range(1, nlabels):
            cc = (labels == lab).astype(np.uint8)
            area = int(cc.sum())
            if area < self.min_cc_area:
                continue
            if np.random.rand() > self.p_choose_cc:
                continue

            skel = _morph_skeleton(cc)
            endpoints = _endpoints_from_skeleton(skel)
            if len(endpoints) == 0:
                continue

            if len(endpoints) > self.endpoint_count_cap:
                pick_idx = np.random.choice(len(endpoints), self.endpoint_count_cap, replace=False)
                endpoints = [endpoints[i] for i in pick_idx]

            sy, sx = np.where(skel > 0)
            if len(sy) == 0:
                continue
            skel_pts = np.stack([sy, sx], axis=1)

            # CC centroid for outward direction sign
            cy, cx = np.mean(np.where(cc > 0), axis=1)

            for (ey, ex) in endpoints:
                dy = skel_pts[:, 0] - ey
                dx = skel_pts[:, 1] - ex
                keep = (dy*dy + dx*dx) <= (self.endpoint_radius * self.endpoint_radius)
                local = skel_pts[keep]
                dvec = _pca_direction(local)  # (dy,dx) unit
                if dvec is None:
                    continue

                # choose sign "outward" from centroid -> endpoint
                to_ep = np.array([ey - cy, ex - cx], dtype=np.float32)
                if (to_ep @ dvec) < 0:
                    dvec = -dvec

                L = int(np.random.randint(self.endpoint_len_range[0], self.endpoint_len_range[1] + 1))
                th = int(np.random.randint(self.thickness_range[0], self.thickness_range[1] + 1))

                y2 = int(np.clip(round(ey + dvec[0] * L), 0, H - 1))
                x2 = int(np.clip(round(ex + dvec[1] * L), 0, W - 1))

                tmp = np.zeros((H, W), np.uint8)
                cv2.line(tmp, (ex, ey), (x2, y2), color=1, thickness=th, lineType=cv2.LINE_8)

                before = int(out_msk.sum())
                out_msk = np.maximum(out_msk, tmp)
                after = int(out_msk.sum())
                added = max(0, after - before)

                if added > 0:
                    grew_any = True
                    added_total += added

                    # backtrack a bit into the original CC so we sample a realistic stain color
                    srcy = int(np.clip(round(ey - dvec[0] * self.backtrack_px), 0, H - 1))
                    srcx = int(np.clip(round(ex - dvec[1] * self.backtrack_px), 0, W - 1))
                    src_points.append((srcy, srcx))

        if not grew_any:
            data["_grow_ok"] = 0
            data["_grow_added_px"] = 0
            return data

        # --- fill only the newly-added pixels ---
        img2 = img.copy()
        if self.fill_mode == "directional":
            # do a few passes so different growth segments can pull different local colors
            # (cheap + usually looks better than one global fill)
            for (srcy, srcx) in (src_points[:10] if len(src_points) > 10 else src_points):
                img2 = fill_added_pixels_directional(
                    img2,
                    old_mask01=old_msk,
                    added_mask01=out_msk,   # function internally only fills pixels where out=1 and old=0
                    src_yx=(srcy, srcx),
                    patch_r=self.patch_r
                )

        elif self.fill_mode == "brown_nearest":
            img2 = fill_added_pixels_from_old(img2, old_msk, out_msk)

        elif self.fill_mode == "none":
            pass

        # restore IGNORE
        out_u8 = out_msk.astype(np.uint8)
        out_u8[ignore] = IGNORE

        data["image"] = img2
        data["mask"] = out_u8
        data["_grow_ok"] = 1
        data["_grow_added_px"] = int(added_total)
        return data


    
from sklearn.mixture import GaussianMixture

from sklearn.mixture import GaussianMixture
import numpy as np
import cv2

def quick_gmm_sanity_map(
    img_u8,
    seed_mask01=None,
    n_components=5,          # <= 5 is much more stable
    sample_max=25000,
    reg_covar=1e-2,          # bump from 1e-3 to 1e-2
    max_iter=50,
    random_state=0,
):
    H, W = img_u8.shape[:2]

    rgb = img_u8.reshape(-1, 3).astype(np.float64)
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY).reshape(-1, 1).astype(np.float64)
    X = np.concatenate([rgb, gray], axis=1)

    N = X.shape[0]
    if sample_max is not None and N > sample_max:
        idx = np.random.choice(N, size=sample_max, replace=False)
        X_fit = X[idx]
    else:
        X_fit = X

    # ---- retry logic: if it collapses, reduce K and/or increase reg ----
    for K in [n_components, max(2, n_components - 2), 2]:
        for reg in [reg_covar, reg_covar * 10, reg_covar * 100]:
            try:
                gmm = GaussianMixture(
                    n_components=K,
                    covariance_type="diag",
                    reg_covar=reg,
                    max_iter=max_iter,
                    n_init=2,
                    random_state=random_state,
                )
                gmm.fit(X_fit)
                prob = gmm.predict_proba(X)  # (N,K)

                # choose structure component
                if seed_mask01 is not None and seed_mask01.sum() > 0:
                    seed = (seed_mask01.reshape(-1) > 0)
                    comp_score = prob[seed].mean(axis=0)
                    struct_comp = int(np.argmax(comp_score))
                else:
                    means = gmm.means_
                    struct_comp = int(np.argmin(means[:, -1]))  # darkest in gray

                p_struct = prob[:, struct_comp].reshape(H, W).astype(np.float32)

                p8 = np.clip(p_struct * 255.0, 0, 255).astype(np.uint8)
                thr, _ = cv2.threshold(p8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                thr_f = float(thr) / 255.0
                auto = (p_struct >= thr_f).astype(np.uint8)

                return p_struct, auto, thr_f

            except ValueError:
                pass

    # if everything fails, return zeros (don’t crash your viz)
    return np.zeros((H, W), np.float32), np.zeros((H, W), np.uint8), 0.0


class ComponentScalePaste(A.DualTransform):
    """
    Extract ONE CC, resize it, and paste it at a random location.

    Context-aware paste:
      - IMAGE paste includes CC + nearby context (alpha defined by context_mode)
      - MASK paste includes ONLY the CC (labels), NOT the context halo
    """
    def __init__(
        self,
        p_choose=0.7,
        min_cc_area=25,
        margin=8,
        scale_range=(0.8, 1.8),
        max_scale=None,

        # --- context ---
        context_radius_px=18,
        context_mode="dilate",   # "dilate" | "distance" | "rect"

        # --- seam ---
        feather_blend=True,
        feather_width=18,

        # --- destination rules ---
        allow_overlap=True,
        require_background=False,
        bg_max_pixels=10,
        max_tries=80,
        p=0.5
    ):
        super().__init__(p=p)
        self.p_choose = float(p_choose)
        self.min_cc_area = int(min_cc_area)
        self.margin = int(margin)
        self.scale_range = tuple(map(float, scale_range))
        self.max_scale = float(max_scale) if max_scale is not None else None

        self.context_radius_px = int(context_radius_px)
        self.context_mode = str(context_mode).lower()

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
        # msk = (data["mask"] > 0).astype(np.uint8)
        msk_u8 = data["mask"].astype(np.uint8)
        ignore = (msk_u8 == IGNORE)
        msk = (msk_u8 == 1).astype(np.uint8)

        H, W = msk.shape

        nlabels, labels = cv2.connectedComponents(msk, connectivity=8)
        if nlabels <= 1 or (np.random.rand() > self.p_choose):
            data["_scale_paste_ok"] = 0
            return data

        # eligible CCs
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

        # isolate CC mask inside bbox
        cc = (labels[y0:y1+1, x0:x1+1] == lab).astype(np.uint8)

        # ROI image includes CC + surrounding pixels (context)
        roi_img = img[y0:y1+1, x0:x1+1].copy()
        roi_cc  = cc

        # choose scale
        s = float(np.random.uniform(self.scale_range[0], self.scale_range[1]))
        if self.max_scale is not None:
            s = min(s, self.max_scale)

        h1 = max(2, int(round(h0 * s)))
        w1 = max(2, int(round(w0 * s)))
        h1 = min(h1, H)
        w1 = min(w1, W)

        # resize ROI image + CC mask
        roi_img_r = cv2.resize(roi_img, (w1, h1), interpolation=cv2.INTER_LINEAR)
        roi_cc_r  = cv2.resize(roi_cc,  (w1, h1), interpolation=cv2.INTER_NEAREST)
        roi_cc_r  = (roi_cc_r > 0).astype(np.uint8)

        # --- alpha for IMAGE paste (CC + context) ---
        if self.context_mode == "rect":
            alpha2d = np.ones((h1, w1), np.float32)

        elif self.context_mode == "distance":
            r = float(max(1, self.context_radius_px))
            dist = cv2.distanceTransform((1 - roi_cc_r).astype(np.uint8), cv2.DIST_L2, 3)
            alpha2d = np.clip(1.0 - (dist / r), 0.0, 1.0).astype(np.float32)

        else:
            k = int(max(1, self.context_radius_px))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k + 1, 2*k + 1))
            halo = cv2.dilate(roi_cc_r, kernel, iterations=1).astype(np.float32)
            alpha2d = halo

        alpha = alpha2d[..., None].astype(np.float32)  # (h1,w1,1)

        # optional feather (softens rectangle boundary)
        if self.feather_blend:
            f = feather_alpha(h1, w1, feather=self.feather_width).astype(np.float32)
            alpha = alpha * f

        # pick destination
        dy0 = dx0 = None
        for _ in range(self.max_tries):
            dy0_try = np.random.randint(0, H - h1 + 1)
            dx0_try = np.random.randint(0, W - w1 + 1)

            if self.require_background:
                if int(msk[dy0_try:dy0_try+h1, dx0_try:dx0_try+w1].sum()) > self.bg_max_pixels:
                    continue

            if not self.allow_overlap:
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

        # paste IMAGE
        base = img2[dy0:dy1+1, dx0:dx1+1].astype(np.float32)
        paste = roi_img_r.astype(np.float32)
        img2[dy0:dy1+1, dx0:dx1+1] = np.clip(alpha * paste + (1 - alpha) * base, 0, 255).astype(np.uint8)

        # paste MASK (labels only)
        msk2[dy0:dy1+1, dx0:dx1+1] = np.maximum(msk2[dy0:dy1+1, dx0:dx1+1], roi_cc_r)

        data["image"] = img2
        data["mask"] = msk2
        data["_scale_paste_ok"] = 1
        data["_scale_factor"] = s
        data["_paste_box"] = (dy0, dy1, dx0, dx1)
        data["_context_mode"] = self.context_mode
        data["_context_radius_px"] = self.context_radius_px
        return data

class ComponentSwapPaste(A.DualTransform):
    """
    Pick ONE CC ROI and move it to a new random location.
    Source rectangle gets destination background (swap background).

    Context-aware move:
      - IMAGE move includes CC + nearby context (alpha defined by context_mode)
      - MASK move includes ONLY the CC (labels), NOT the context halo
    """
    def __init__(
        self,
        p_choose=0.35,
        min_cc_area=25,
        margin=8,

        # --- context ---
        context_radius_px=18,
        context_mode="dilate",   # "dilate" | "distance" | "rect"

        # --- seam ---
        feather_blend=True,
        feather_width=18,

        # --- placement ---
        allow_overlap=False,
        require_background=False,
        bg_max_pixels=10,
        max_tries=50,
        p=0.5
    ):
        super().__init__(p=p)
        self.p_choose = float(p_choose)
        self.min_cc_area = int(min_cc_area)
        self.margin = int(margin)

        self.context_radius_px = int(context_radius_px)
        self.context_mode = str(context_mode).lower()

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
        # msk = (data["mask"] > 0).astype(np.uint8)
        msk_u8 = data["mask"].astype(np.uint8)   # 0 / 1 / 255
        ignore = (msk_u8 == IGNORE)              # bool mask for ignore pixels
        msk = (msk_u8 == 1).astype(np.uint8)     # ONLY positives (foreground)

        H, W = msk.shape

        nlabels, labels = cv2.connectedComponents(msk, connectivity=8)
        if nlabels <= 1:
            data["_swap_ok"] = 0
            return data

        labs, areas = [], []
        for lab in range(1, nlabels):
            area = int((labels == lab).sum())
            if area >= self.min_cc_area:
                labs.append(lab)
                areas.append(area)

        if (not labs) or (np.random.rand() > self.p_choose):
            data["_swap_ok"] = 0
            return data

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
        if h <= 1 or w <= 1 or h > H or w > W:
            data["_swap_ok"] = 0
            return data

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

            if (not self.allow_overlap) and overlaps(src_box, dst_box):
                continue

            if self.require_background:
                if int(msk[dy0_try:dy0_try+h, dx0_try:dx0_try+w].sum()) > self.bg_max_pixels:
                    continue

            dy0, dx0 = dy0_try, dx0_try
            break

        if dy0 is None:
            data["_swap_ok"] = 0
            return data

        dy1, dx1 = dy0 + h - 1, dx0 + w - 1

        src_img = img[y0:y1+1, x0:x1+1].copy()
        dst_img = img[dy0:dy1+1, dx0:dx1+1].copy()

        # CC-only labels in source ROI
        src_cc = (labels[y0:y1+1, x0:x1+1] == lab).astype(np.uint8)

        # --- alpha for IMAGE move (CC + context) ---
        if self.context_mode == "rect":
            alpha2d = np.ones((h, w), np.float32)

        elif self.context_mode == "distance":
            r = float(max(1, self.context_radius_px))
            dist = cv2.distanceTransform((1 - src_cc).astype(np.uint8), cv2.DIST_L2, 3)
            alpha2d = np.clip(1.0 - (dist / r), 0.0, 1.0).astype(np.float32)

        else:
            k = int(max(1, self.context_radius_px))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k + 1, 2*k + 1))
            halo = cv2.dilate(src_cc, kernel, iterations=1).astype(np.float32)
            alpha2d = halo

        alpha = alpha2d[..., None].astype(np.float32)

        if self.feather_blend:
            f = feather_alpha(h, w, feather=self.feather_width).astype(np.float32)
            alpha = alpha * f

        img2 = img.copy()

        # 1) source becomes destination background (swap bg)
        img2[y0:y1+1, x0:x1+1] = dst_img

        # 2) destination receives source ROI (with alpha)
        base = img2[dy0:dy1+1, dx0:dx1+1].astype(np.float32)
        paste = src_img.astype(np.float32)
        img2[dy0:dy1+1, dx0:dx1+1] = np.clip(alpha * paste + (1 - alpha) * base, 0, 255).astype(np.uint8)

        # MASK move (labels only)
        msk2 = msk.copy()
        msk2[y0:y1+1, x0:x1+1] = 0
        msk2[dy0:dy1+1, dx0:dx1+1] = np.maximum(msk2[dy0:dy1+1, dx0:dx1+1], src_cc)
        
        # MASK move (labels only), keep IGNORE untouched
        msk2 = (msk_u8 == 1).astype(np.uint8)      # start from positives only
        msk2[y0:y1+1, x0:x1+1] = 0                 # remove source CC
        msk2[dy0:dy1+1, dx0:dx1+1] = np.maximum(
            msk2[dy0:dy1+1, dx0:dx1+1], src_cc
        )

        out_u8 = msk2.astype(np.uint8)
        out_u8[ignore] = IGNORE                    # restore ignore pixels
        data["mask"] = out_u8



        data["image"] = img2
        data["_swap_ok"] = 1
        data["_swap_lab"] = lab
        data["_swap_src_box"] = (y0, y1, x0, x1)
        data["_swap_dst_box"] = (dy0, dy1, dx0, dx1)
        data["_context_mode"] = self.context_mode
        data["_context_radius_px"] = self.context_radius_px
        return data

import numpy as np, cv2, albumentations as A

import cv2
import numpy as np





class RandomMaskThickness(A.DualTransform):
    """
    Randomly dilate or erode the GT mask to simulate thicker/thinner annotations.

    thickness_px_range: (min,max) radius in pixels
      - dilation uses +r
      - erosion uses -r

    p_dilate: probability of choosing dilation (else erosion)
    """
    def __init__(self, thickness_px_range=(1, 3), p_dilate=0.5, fill_image_on_dilate=True, p=0.5):
        super().__init__(p=p)
        self.thickness_px_range = thickness_px_range
        self.p_dilate = float(p_dilate)
        self.fill_image_on_dilate = bool(fill_image_on_dilate)

    @property
    def targets_as_params(self):
        return ["image", "mask"]

    def __call__(self, force_apply=False, **data):
        if not (force_apply or np.random.rand() < self.p):
            return data

        img = data["image"]
        msk_u8 = data["mask"].astype(np.uint8)
        ignore = (msk_u8 == IGNORE)
        msk = (msk_u8 == 1).astype(np.uint8)

        r = int(np.random.randint(self.thickness_px_range[0], self.thickness_px_range[1] + 1))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r + 1, 2*r + 1))

        do_dilate = (np.random.rand() < self.p_dilate)
        old = msk.copy()

        if do_dilate:
            new = cv2.dilate(msk, k, iterations=1)
            if self.fill_image_on_dilate:
                img2 = fill_added_pixels_from_old(img, old, new)
            else:
                img2 = img
        else:
            new = cv2.erode(msk, k, iterations=1)
            img2 = img  # erosion removes label; image doesn't need changes

        out_u8 = new.astype(np.uint8)
        out_u8[ignore] = IGNORE
        data["mask"] = out_u8

        data["image"] = img2
        data["_thickness_r"] = r if do_dilate else -r
        return data


class MultiCopyPaste(A.DualTransform):
    """
    Copy connected components many times with new geometry.

    You control:
      - how many copies
      - scaling
      - elastic deformation
      - rotation
      - overlap rules
    """

    def __init__(
        self,
        copies_range=(10, 50),     # <-- set (100,100) for fixed 100
        min_cc_area=25,

        # geometry
        scale_range=(0.7, 1.6),
        rotate_range=(-25, 25),

        elastic_alpha_range=(0, 400),
        elastic_sigma_range=(0, 6),

        # placement
        allow_overlap=True,
        require_background=False,
        bg_max_pixels=5,
        max_tries=60,

        feather_blend=True,
        feather_width=12,

        p=0.5
    ):
        super().__init__(p=p)

        self.copies_range = copies_range
        self.min_cc_area = min_cc_area

        self.scale_range = scale_range
        self.rotate_range = rotate_range

        self.elastic_alpha_range = elastic_alpha_range
        self.elastic_sigma_range = elastic_sigma_range

        self.allow_overlap = allow_overlap
        self.require_background = require_background
        self.bg_max_pixels = bg_max_pixels
        self.max_tries = max_tries

        self.feather_blend = feather_blend
        self.feather_width = feather_width

    @property
    def targets_as_params(self):
        return ["image", "mask"]

    def __call__(self, force_apply=False, **data):

        if not (force_apply or np.random.rand() < self.p):
            data["_copies_done"] = 0
            return data

        img = data["image"]
        msk = (data["mask"] > 0).astype(np.uint8)
        msk_u8 = data["mask"].astype(np.uint8)
        ignore = (msk_u8 == IGNORE)
        msk = (msk_u8 == 1).astype(np.uint8)   # ONLY positives



        H, W = msk.shape
        img2 = img.copy()
        msk2 = msk.copy()

        # connected components
        nlabels, labels = cv2.connectedComponents(msk, connectivity=8)

        # collect eligible CCs
        ccs = []
        for lab in range(1, nlabels):
            cc = (labels == lab).astype(np.uint8)
            if cc.sum() >= self.min_cc_area:
                ccs.append(cc)

        if len(ccs) == 0:
            data["_copies_done"] = 0
            return data

        # how many copies
        N = np.random.randint(
            self.copies_range[0],
            self.copies_range[1] + 1
        )

        copies_done = 0

        for _ in range(N):

            cc = random.choice(ccs)

            ys, xs = np.where(cc > 0)
            y0, y1 = ys.min(), ys.max()
            x0, x1 = xs.min(), xs.max()

            roi_img = img[y0:y1+1, x0:x1+1].copy()
            roi_msk = cc[y0:y1+1, x0:x1+1].copy()

            # ---- geometry transform ----
            scale = np.random.uniform(*self.scale_range)
            rot   = np.random.uniform(*self.rotate_range)

            h0, w0 = roi_msk.shape
            h1 = int(h0 * scale)
            w1 = int(w0 * scale)

            if h1 < 2 or w1 < 2:
                continue
            
            h1 = int(h0 * scale)
            w1 = int(w0 * scale)

            h1 = min(h1, H - 2)
            w1 = min(w1, W - 2)
            if h1 < 2 or w1 < 2:
                continue


            roi_img = cv2.resize(roi_img, (w1, h1))
            roi_msk = cv2.resize(roi_msk, (w1, h1), interpolation=cv2.INTER_NEAREST)

            # elastic warp
            alpha = np.random.uniform(*self.elastic_alpha_range)
            # sigma = np.random.uniform(*self.elastic_sigma_range)
            sigma = float(np.random.uniform(*self.elastic_sigma_range))
            sigma = max(1.0, sigma)  # <-- IMPORTANT: Albumentations requires sigma >= 1

            if alpha > 0:
                aug = A.ElasticTransform(alpha=float(alpha), sigma=float(sigma), p=1.0)


            if alpha > 0 and sigma > 0:
                aug = A.ElasticTransform(
                    alpha=alpha,
                    sigma=sigma,
                    p=1.0
                )
                out = aug(image=roi_img, mask=roi_msk)
                roi_img, roi_msk = out["image"], out["mask"]

            # rotate
            M = cv2.getRotationMatrix2D((w1//2, h1//2), rot, 1)
            roi_img = cv2.warpAffine(
                roi_img, M, (w1, h1),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101
            )
            roi_msk = cv2.warpAffine(
                roi_msk, M, (w1, h1),
                flags=cv2.INTER_NEAREST
            )

            # ---- placement ----
            placed = False
            for _ in range(self.max_tries):

                y = np.random.randint(0, H - h1)
                x = np.random.randint(0, W - w1)

                region = msk2[y:y+h1, x:x+w1]

                if self.require_background:
                    if region.sum() > self.bg_max_pixels:
                        continue

                if not self.allow_overlap:
                    if region.sum() > 0:
                        continue

                placed = True
                break

            if not placed:
                continue

            # ---- paste ----
            if self.feather_blend:
                a = feather_alpha(h1, w1, self.feather_width)
            else:
                a = roi_msk[..., None]

            base = img2[y:y+h1, x:x+w1].astype(np.float32)
            paste = roi_img.astype(np.float32)

            img2[y:y+h1, x:x+w1] = (
                a * paste + (1-a) * base
            ).astype(np.uint8)

            msk2[y:y+h1, x:x+w1] = np.maximum(
                msk2[y:y+h1, x:x+w1],
                roi_msk
            )

            copies_done += 1

            # at end:
        out_u8 = msk2.astype(np.uint8)
        out_u8[ignore] = IGNORE
        data["mask"] = out_u8

        data["image"] = img2
        # data["mask"]  = msk2
        data["_copies_done"] = copies_done

        return data

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

# ---------------- IO HELPERS ----------------
import os, re

def list_pairs(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith((".tif", ".tiff"))]

    unann, ann = {}, {}

    for f in files:
        f_stripped = f.strip()

        # --- unannotated ---
        m1 = re.match(r"^image\s*(\d+)\s*unannotated\.tif(f)?$", f_stripped, flags=re.IGNORECASE)
        if m1:
            unann[int(m1.group(1))] = os.path.join(folder, f)
            continue

        # --- annotated variants ---
        # 2016 style: "Image 6_FLAT.tif" or "Image 6 _FLAT.tif"
        m2a = re.match(r"^image\s*(\d+)\s*_?flat\.tif(f)?$", f_stripped, flags=re.IGNORECASE)

        # 2017 style: "IMAGE 13 annotated_FLAT.tif" or "IMAGE 13_annotated_FLAT.tif"
        m2b = re.match(r"^image\s*(\d+)\s*_?annotated_?flat\.tif(f)?$", f_stripped, flags=re.IGNORECASE)

        if m2a:
            ann[int(m2a.group(1))] = os.path.join(folder, f)
            continue
        if m2b:
            ann[int(m2b.group(1))] = os.path.join(folder, f)
            continue

    keys = sorted(set(unann.keys()) & set(ann.keys()))
    return [(k, unann[k], ann[k]) for k in keys]

def fill_black_nearest(img):
    mask = (img.sum(axis=2) == 0).astype(np.uint8)
    if mask.sum() == 0:
        return img

    inv = (mask == 0).astype(np.uint8)
    dist, labels = cv2.distanceTransformWithLabels(
        inv, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
    )

    H, W = mask.shape
    idx = labels.astype(np.int64) - 1
    idx = np.clip(idx, 0, H * W - 1)
    yy = (idx // W).astype(np.int64)
    xx = (idx %  W).astype(np.int64)

    img2 = img.copy()
    img2[mask == 1] = img2[yy[mask == 1], xx[mask == 1]]
    return img2


def list_pairs_multi(folders):
    all_pairs = []
    for fd in folders:
        if not os.path.isdir(fd):
            print("WARN: missing folder:", fd)
            continue
        pairs = list_pairs(fd)
        if len(pairs) == 0:
            print("WARN: no pairs found in:", fd)
        all_pairs.extend(pairs)
    return all_pairs

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

import numpy as np
import cv2

IGNORE = 255

def draw_border(
    rgb,
    msk,
    color=(255, 0, 0),
    thickness=2,
    black_thr=8,
    white_thr=245,
):
    """
    Draw GT border, but DO NOT draw inside:
      - near-black holes in the IMAGE (e.g., cutout/rotate padding artifacts)
      - near-white holes/strips in the IMAGE (missing tissue / tears)

    Also ignores IGNORE=255 pixels in the mask.

    Args:
      rgb: HxWx3 uint8
      msk: HxW uint8 (0/1 or 0/1/255)
      color: BGR or RGB? (cv2.drawContours expects BGR if you display with cv2;
             if you display with matplotlib, treat as RGB. Keep consistent.)
      thickness: contour thickness
      black_thr: pixels with all channels < black_thr are treated as holes
      white_thr: pixels with all channels > white_thr are treated as holes
    """
    out = rgb.copy()

    # 1) contour source mask (binary), excluding IGNORE
    m = (msk > 0).astype(np.uint8)
    if msk.dtype != np.uint8:
        msk_u8 = msk.astype(np.uint8)
    else:
        msk_u8 = msk
    m[msk_u8 == IGNORE] = 0

    # 2) hole detection from the IMAGE
    black_holes = (rgb[..., 0] < black_thr) & (rgb[..., 1] < black_thr) & (rgb[..., 2] < black_thr)
    white_holes = (rgb[..., 0] > white_thr) & (rgb[..., 1] > white_thr) & (rgb[..., 2] > white_thr)

    holes = black_holes | white_holes

    # 3) remove hole pixels from contour source so we don’t draw there
    m[holes] = 0

    # optional: reduce speck contours (tiny islands)
    # m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
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

from sklearn.mixture import GaussianMixture

from sklearn.mixture import GaussianMixture
import numpy as np
import cv2
import albumentations as A

IGNORE = 255

class GMMPromoteWholeTissue(A.DualTransform):
    """
    If GT dots sit on a coherent tissue/stain region, promote the *entire connected region*
    (under a posterior threshold) to GT.

    - Fit K-comp GMM in ROI around GT
    - Select structure component by GT posterior
    - Threshold p_struct -> candidate tissue mask
    - Keep only candidate CCs that intersect GT (or GT dilated)
    - Promote those CCs to GT
    - Optionally: mark low-confidence band as IGNORE
    """
    def __init__(
        self,
        margin=24,
        min_cc_area=25,
        max_rois=4,
        roi_max_size=256,

        n_components=5,            # <<< IMPORTANT (your observation)
        cov_type="diag",
        reg_covar=1e-4,
        max_iter=120,
        n_init=2,

        # tissue thresholding
        seed_prob=0.85,            # candidate tissue pixels (lower = more inclusive)
        ignore_prob=0.70,          # optional ignore band lower bound
        use_ignore_band=True,

        # limit explosion
        intersect_dilate=7,        # CC must intersect GT dilated by this
        max_promote_px=8000,       # safety cap per ROI

        connect_close=True,
        close_kernel=3,

        p=0.35
    ):
        super().__init__(p=p)
        self.margin = int(margin)
        self.min_cc_area = int(min_cc_area)
        self.max_rois = int(max_rois)
        self.roi_max_size = int(roi_max_size)

        self.n_components = int(n_components)
        self.cov_type = cov_type
        self.reg_covar = float(reg_covar)
        self.max_iter = int(max_iter)
        self.n_init = int(n_init)

        self.seed_prob = float(seed_prob)
        self.ignore_prob = float(ignore_prob)
        self.use_ignore_band = bool(use_ignore_band)

        self.intersect_dilate = int(intersect_dilate)
        self.max_promote_px = int(max_promote_px)

        self.connect_close = bool(connect_close)
        self.close_kernel = int(close_kernel)

    @property
    def targets_as_params(self):
        return ["image", "mask"]

    def __call__(self, force_apply=False, **data):
        if not (force_apply or np.random.rand() < self.p):
            data["_gmm_promote_ok"] = 0
            return data

        img = data["image"]                          # uint8 RGB
        msk = data["mask"].copy().astype(np.uint8)   # 0/1/255 possible
        H, W = msk.shape

        pos = (msk == 1).astype(np.uint8)
        if pos.sum() == 0:
            data["_gmm_promote_ok"] = 0
            return data

        nlabels, labels = cv2.connectedComponents(pos, connectivity=8)

        rois = []
        for lab in range(1, nlabels):
            cc = (labels == lab).astype(np.uint8)
            area = int(cc.sum())
            if area < self.min_cc_area:
                continue
            ys, xs = np.where(cc > 0)
            y0, y1 = ys.min(), ys.max()
            x0, x1 = xs.min(), xs.max()
            y0 = max(0, y0 - self.margin); x0 = max(0, x0 - self.margin)
            y1 = min(H - 1, y1 + self.margin); x1 = min(W - 1, x1 + self.margin)

            if (y1 - y0 + 1) > self.roi_max_size or (x1 - x0 + 1) > self.roi_max_size:
                cy = (y0 + y1) // 2
                cx = (x0 + x1) // 2
                half = self.roi_max_size // 2
                y0 = max(0, cy - half); y1 = min(H - 1, cy + half)
                x0 = max(0, cx - half); x1 = min(W - 1, cx + half)

            rois.append((area, y0, y1, x0, x1))

        if not rois:
            data["_gmm_promote_ok"] = 0
            return data

        rois.sort(reverse=True, key=lambda t: t[0])
        rois = rois[: self.max_rois]

        added_total, ignored_total = 0, 0

        for _, y0, y1, x0, x1 in rois:
            roi_img = img[y0:y1+1, x0:x1+1]
            roi_pos = pos[y0:y1+1, x0:x1+1]

            if roi_pos.sum() < self.min_cc_area:
                continue

            # features: RGB + gray
            rgb = roi_img.reshape(-1, 3).astype(np.float32)
            gray = cv2.cvtColor(roi_img, cv2.COLOR_RGB2GRAY).reshape(-1, 1).astype(np.float32)
            X = np.concatenate([rgb, gray], axis=1).astype(np.float64)

            gmm = GaussianMixture(
                n_components=self.n_components,
                covariance_type=self.cov_type,
                reg_covar=self.reg_covar,
                max_iter=self.max_iter,
                n_init=self.n_init,
                random_state=None
            )
            gmm.fit(X)

            prob = gmm.predict_proba(X)  # (N,K)

            roi_pos_flat = roi_pos.reshape(-1) > 0
            if roi_pos_flat.sum() == 0:
                continue

            # choose structure component by GT posterior
            comp_score = prob[roi_pos_flat].mean(axis=0)
            k_struct = int(np.argmax(comp_score))
            p_struct = prob[:, k_struct].reshape(roi_pos.shape)

            # candidate tissue region
            cand = (p_struct >= self.seed_prob).astype(np.uint8)

            if self.connect_close and cand.sum() > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.close_kernel, self.close_kernel))
                cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, k, iterations=1)

            if cand.sum() == 0:
                continue

            # Only keep CCs of cand that intersect GT (dilated)
            if self.intersect_dilate > 0:
                kd = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2*self.intersect_dilate + 1, 2*self.intersect_dilate + 1)
                )
                pos_d = cv2.dilate(roi_pos, kd, iterations=1)
            else:
                pos_d = roi_pos

            n2, lab2 = cv2.connectedComponents(cand, connectivity=8)
            keep = np.zeros_like(cand, dtype=np.uint8)
            for c in range(1, n2):
                cc = (lab2 == c).astype(np.uint8)
                if (cc & pos_d).any():
                    keep = np.maximum(keep, cc)

            if keep.sum() == 0:
                continue

            # safety cap
            if int(keep.sum()) > self.max_promote_px:
                # if too big, don’t promote (prevents runaway)
                continue

            # promote to GT, but don't overwrite IGNORE
            roi_msk = msk[y0:y1+1, x0:x1+1]
            add = (keep == 1) & (roi_msk == 0)
            if add.any():
                roi_msk[add] = 1
                added_total += int(add.sum())

            if self.use_ignore_band:
                ign = (p_struct >= self.ignore_prob) & (p_struct < self.seed_prob) & (roi_msk == 0)
                if ign.any():
                    roi_msk[ign] = IGNORE
                    ignored_total += int(ign.sum())

            msk[y0:y1+1, x0:x1+1] = roi_msk

        data["mask"] = msk
        data["_gmm_promote_ok"] = 1
        data["_gmm_promote_added_px"] = int(added_total)
        data["_gmm_promote_ignored_px"] = int(ignored_total)
        return data

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

class FillBlackNearest(A.ImageOnlyTransform):
    def __init__(self, p=1.0):
        super().__init__(p=p)

    def apply(self, img, **params):
        return fill_black_nearest(img)

   
def fill_black_nearest(img):

    mask = (img.sum(axis=2) == 0).astype(np.uint8)
    if mask.sum() == 0:
        return img

    inv = (mask == 0).astype(np.uint8)

    dist, labels = cv2.distanceTransformWithLabels(
        inv,
        cv2.DIST_L2,
        3,
        labelType=cv2.DIST_LABEL_PIXEL
    )

    H, W = mask.shape
    lab = labels.astype(np.int64)

    yy = (lab // W).clip(0, H-1)
    xx = (lab %  W).clip(0, W-1)

    img2 = img.copy()
    img2[mask == 1] = img2[yy[mask == 1], xx[mask == 1]]

    return img2


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
        self.elastic_alpha_range = (40.0, float(elastic_alpha))
        self.elastic_sigma_range = (1.0, float(elastic_sigma))
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
        # mask = (data["mask"] > 0).astype(np.uint8)
        msk_u8 = data["mask"].astype(np.uint8)
        ignore = (msk_u8 == IGNORE)
        mask = (msk_u8 == 1).astype(np.uint8)

        
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

        out_u8 = mask2.astype(np.uint8)
        out_u8[ignore] = IGNORE
        data["mask"] = out_u8

        data["image"] = img2
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


    # ---------------- TRAIN AUG ----------------

    ROI_MARGIN = 14

    ROI_MARGIN = 14
    PAD = 16  # keep your PAD if you already have one; otherwise define

    train_aug = A.Compose([
        # ---- SAFE GLOBAL GEOMETRY ----
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),

        A.Affine(
            translate_percent={"x": (-0.06, 0.06), "y": (-0.06, 0.06)},
            scale=(0.96, 1.06),
            rotate=(-18, 18),
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            mode=cv2.BORDER_REFLECT_101,
            p=0.40
        ),

        # ---- LABEL / MASK HANDLING ----
        WhiteTissueDropout(
            p=0.25  # was 0.35 (too many large white voids can change distribution)
        ),

        GMMPromoteWholeTissue(
            margin=ROI_MARGIN,
            min_cc_area=80,          # ↑ reduces tiny noisy CC promotions
            max_rois=3,              # ↓ fewer regions promoted
            roi_max_size=224,        # ↓ smaller ROIs to prevent massive fills
            n_components=4,          # ↓ slightly simpler
            seed_prob=0.80,
            ignore_prob=0.75,
            intersect_dilate=5,      # ↓ less aggressive merging
            max_promote_px=4500,     # ↓ huge reduction to prevent dense GT
            connect_close=True,
            close_kernel=3,
            p=0.35                   # ↓ less frequent
        ),

        # ---- COLOR / ACQUISITION REALISM (MOVE BEFORE COPY-PASTE) ----
        # Keep these mild; we want stain stability.
        A.OneOf([
            A.RandomBrightnessContrast(0.06, 0.06),
            A.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.04, hue=0.015),
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MotionBlur(blur_limit=3),
            A.ISONoise(color_shift=(0.01, 0.10), intensity=(0.06, 0.18)),
        ], p=0.45),

        A.RGBShift(r_shift_limit=8, g_shift_limit=6, b_shift_limit=8, p=0.35),

        # Higher quality JPEG to avoid stain destruction
        A.ImageCompression(quality_lower=85, quality_upper=98, p=0.20),

        # ---- COPY/PASTE (NOW AFTER COLOR) ----
        # The big fix: no heavy feathering.
        MultiCopyPaste(
            copies_range=(5, 20),
            scale_range=(0.90, 1.25),
            rotate_range=(-8, 8),
            elastic_alpha_range=(0, 180),
            elastic_sigma_range=(0, 5),
            allow_overlap=False,     # helps avoid dense clumps
            feather_blend=False,
            feather_width=2,         # ✅ was 10; this preserves brown stain
            p=0.92
        ),

        ComponentScalePaste(
            p_choose=0.55,
            min_cc_area=140,         # ↑ avoid tiny specks
            margin=ROI_MARGIN,
            scale_range=(0.95, 1.90),# ↓ was up to 2.3 (creates unrealistic big blobs)
            context_mode="distance",
            context_radius_px=14,    # ↓ softer context pull
            feather_blend=True,
            feather_width=3,         # ✅ keep stain
            p=0.20
        ),

        # ---- DIFFUSION LOOK (MAKE IT SUBTLE; PREVENT DENSE GT) ----
        DiffuseGTIntoNeighborhood(
            radius_range=(4, 10),        # ↓ was (6,14)
            sigma_frac=(0.50, 0.80),
            alpha_power=(1.0, 1.4),
            max_alpha=(0.12, 0.28),      # ✅ was (0.20,0.45) -> too strong / densifies
            mask_mode="alpha_thresh",
            mask_thresh=(0.75, 0.90),    # ✅ stricter threshold -> less expansion
            p=0.35                        # ↓ was 0.55
        ),

        # ---- CONNECTIVITY (TIGHT + RARE) ----
        BridgeNearbyEndpoints(
            max_gap_px=28,           # ↓ was 50 (created long fake vessels)
            max_bridges=1,           # ↓ was 2
            line_thickness=(1, 1),
            fill_image=True,
            p=0.30
        ),

        GrowAlongSkeleton(
            p_choose_cc=0.35,        # ↓
            min_cc_area=60,         # ↑ avoid tiny growth
            endpoint_len_range=(1, 280),  # ↓ was up to 50
            thickness_range=(1, 4),      # ↓ was up to 6
            endpoint_count_cap=1,        # ↓ was 2
            endpoint_radius=10,          # ↓
            p=0.22
        ),

        # ---- LOCAL CC WARP (MILD + RARE) ----
        ProbabilisticComponentWarp(
            p_choose=0.15,
            min_cc_area=120,
            max_pick=3,
            prefer_elongated=True,
            margin=ROI_MARGIN,
            pad=PAD,
            elastic_alpha=220,       # ↓ was 580
            elastic_sigma=6,
            alpha_affine=0,
            feather_blend=True,
            feather_width=8,         # ↓ was 14
            p=0.9
        ),

        # ---- MASKED COLOR SHIFT (KEEP SMALL; DON’T KILL STAIN) ----
        MaskedRGBShift(
            r_shift_limit=(-30, 30),
            g_shift_limit=(-30, 30),
            b_shift_limit=(-30, 30),
            feather_px=4,
            p=0.35
        ),

        # ---- PATCH NOISE (IMAGE ONLY) ----
        PatchGaussianNoise(
            num_patches_range=(1, 2),
            patch_size_range=(48, 120),  # slightly smaller
            noise_std_range=(5, 18),
            p=0.22
        ),
    ])

    
    val_aug = A.Compose([])  # NO randomness, NO tensorization



    def transforms(example_batch, augmentations, make_second_view: bool):
        images, images2, labels = [], [], []

        for img, lbl in zip(example_batch["pixel_values"], example_batch["label"]):
            img = np.array(Image.fromarray(np.uint8(img)).convert("RGB"))  # HWC
            lbl = np.array(Image.fromarray(np.uint8(lbl)).convert("L"))    # HW

            lbl = remap_labels(lbl).astype(np.uint8)  # binary before aug

            aug = augmentations(image=img, mask=lbl)

            aug_img = aug["image"]
            aug_msk = aug["mask"].astype(np.uint8)

            # Ensure black holes don't become training supervision
            holes = (aug_img[...,0] < 8) & (aug_img[...,1] < 8) & (aug_img[...,2] < 8)
            aug_msk = aug_msk.copy()
            aug_msk[holes] = IGNORE

            images.append(aug_img)
            labels.append(aug_msk)

            # ---- second view: diffusion-like forward noise (image-only), same mask ----
            if make_second_view:
                img2 = diffusion_forward_u8(aug_img)   # << key line
                images2.append(img2)

        enc = processor(images, labels, return_tensors="pt")  # gives pixel_values + labels

        if make_second_view:
            enc2 = processor(images2, return_tensors="pt")    # image-only
            enc["pixel_values_2"] = enc2["pixel_values"]      # attach second view

        return enc
    



    train_ds.set_transform(lambda ex: transforms(ex, train_aug, make_second_view=False))
    eval_ds.set_transform(lambda ex: transforms(ex, val_aug, make_second_view=False))

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

        load_best_model_at_end=False,
        push_to_hub=args.push_to_hub,
        report_to=["none"],
        seed=args.seed
            
    )

    import re
    
    def latest_checkpoint(out_dir: str):
        """
        Return the checkpoint-* folder with the most recent modification time.
        (Not the largest number.)
        """
        if not os.path.isdir(out_dir):
            return None

        ckpts = []

        for name in os.listdir(out_dir):
            m = re.match(r"checkpoint-(\d+)$", name)
            if not m:
                continue

            path = os.path.join(out_dir, name)
            if os.path.isdir(path):
                mtime = os.path.getmtime(path)
                ckpts.append((mtime, path))

        if not ckpts:
            return None

        ckpts.sort(reverse=True)  # newest first
        return ckpts[0][1]


    
    ckpt = latest_checkpoint(args.output_dir)

    if ckpt:
        print("Loading WEIGHTS ONLY from:", ckpt)
        model = SegformerForSemanticSegmentation.from_pretrained(
            ckpt, num_labels=2, id2label=id2label, label2id=label2id
        )
        set_dropout(model, hidden=0.3, attn=0.3, classifier=0.3)
    else:
        model = SegformerForSemanticSegmentation.from_pretrained(
            args.model_id, num_labels=2, id2label=id2label, label2id=label2id
        )

    trainer = EMATeacherAmbiguityIgnoreTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        ema_decay=0.999,
        tau_pos=0.97,
        tau_neg=0.97,       # optional but recommended
        warmup_steps=500,
        # min_conf=0.95,    # optional
    )


    offset = max_checkpoint_step(args.output_dir)
    print(f"[checkpoint offset] max existing checkpoint step in {args.output_dir} = {offset}")
    trainer.add_callback(OffsetCheckpointNamer(args.output_dir, offset))
    lr_cb = ReduceLROnPlateauCallback(
        monitor="eval_mean_iou",
        factor=0.7,
        patience=3,
        min_lr=1e-7
    )
    trainer.add_callback(lr_cb)
    lr_cb.trainer = trainer



        
#     if ckpt:
#         print("Loading WEIGHTS ONLY from:", ckpt)
#         model = SegformerForSemanticSegmentation.from_pretrained(
#             ckpt,
#             num_labels=2,
#             id2label=id2label,
#             label2id=label2id,
#         )
#         set_dropout(model, hidden=0.3, attn=0.3, classifier=0.3)
#         trainer.model = model
#     else:
#         print("No checkpoint found. Starting from base model:", args.model_id)




    trainer.train(resume_from_checkpoint=False)  # fresh LR/optimizer



if __name__ == "__main__":
    main()
