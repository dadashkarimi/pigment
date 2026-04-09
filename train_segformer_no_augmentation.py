#!/usr/bin/env python3
import os
import re
import argparse
import shutil
import numpy as np
import torch
from torch import nn
from datasets import load_dataset, concatenate_datasets
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
import evaluate
import copy
import torch.nn.functional as F
from PIL import Image

IGNORE = 255


# ---------- Callbacks ----------

class OffsetCheckpointNamer(TrainerCallback):
    """
    After each checkpoint is saved, rename:
      output_dir/checkpoint-STEP  ->  output_dir/checkpoint-(STEP + offset)
    """
    def __init__(self, output_dir: str, offset: int):
        self.output_dir = output_dir
        self.offset = int(offset)

    def on_save(self, args, state, control, **kwargs):
        step = int(state.global_step)
        src = os.path.join(self.output_dir, f"checkpoint-{step}")
        if not os.path.isdir(src):
            return control

        new_step = step + self.offset
        dst = os.path.join(self.output_dir, f"checkpoint-{new_step}")

        if os.path.exists(dst):
            print(f"[OffsetCheckpointNamer] dst exists, skipping: {dst}")
            return control

        shutil.move(src, dst)

        ts = os.path.join(dst, "trainer_state.json")
        if os.path.isfile(ts):
            try:
                with open(ts, "r", encoding="utf-8") as f:
                    s = f.read()
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
        self.trainer = None

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


# ---------- EMA Trainer ----------

class EMATeacherAmbiguityIgnoreTrainer(Trainer):
    """
    EMA teacher that masks ambiguous pixels:
      - If teacher is high-confidence and disagrees with GT, set that pixel to IGNORE.
    """

    def __init__(
        self,
        *args,
        ema_decay=0.999,
        warmup_steps=0,
        tau_pos=0.97,
        tau_neg=0.97,
        min_conf=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.ema_decay = float(ema_decay)
        self.warmup_steps = int(warmup_steps)
        self.tau_pos = float(tau_pos)
        self.tau_neg = float(tau_neg)
        self.min_conf = None if min_conf is None else float(min_conf)

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
        if self.state.global_step > 0:
            self._ema_update()
        return super().training_step(model, inputs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        pixel_values = inputs["pixel_values"]

        if self.state.global_step >= self.warmup_steps:
            with torch.no_grad():
                t_out = self.teacher(pixel_values=pixel_values)
                t_probs = torch.softmax(t_out.logits, dim=1)

                p1 = t_probs[:, 1:2]
                p1 = F.interpolate(
                    p1,
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                ).squeeze(1)

                conf_pos = p1 > self.tau_pos
                conf_neg = p1 < (1.0 - self.tau_neg)

                if self.min_conf is not None:
                    pmax = torch.max(t_probs, dim=1).values
                    pmax = F.interpolate(
                        pmax.unsqueeze(1),
                        size=labels.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    ).squeeze(1)
                    strong = pmax >= self.min_conf
                    conf_pos = conf_pos & strong
                    conf_neg = conf_neg & strong

                valid_gt = (labels != IGNORE)
                disagree_pos = valid_gt & (labels == 0) & conf_pos
                disagree_neg = valid_gt & (labels == 1) & conf_neg

                if disagree_pos.any() or disagree_neg.any():
                    labels = labels.clone()
                    labels[disagree_pos | disagree_neg] = IGNORE
                    inputs["labels"] = labels

        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


# ---------- Utilities ----------

def remap_labels(labels: np.ndarray) -> np.ndarray:
    labels = labels.copy()
    labels[(labels >= 0) & (labels <= 227)] = 0
    labels[(labels >= 228) & (labels <= 255)] = 1
    return labels


def set_dropout(model, hidden=0.2, attn=0.2, classifier=0.3):
    if hasattr(model.config, "hidden_dropout_prob"):
        model.config.hidden_dropout_prob = hidden
    if hasattr(model.config, "attention_probs_dropout_prob"):
        model.config.attention_probs_dropout_prob = attn
    if hasattr(model.config, "classifier_dropout_prob"):
        model.config.classifier_dropout_prob = classifier
    print("Dropout settings:")
    print("  hidden    :", getattr(model.config, "hidden_dropout_prob", None))
    print("  attention :", getattr(model.config, "attention_probs_dropout_prob", None))
    print("  classifier:", getattr(model.config, "classifier_dropout_prob", None))


def max_checkpoint_step(out_dir: str) -> int:
    if not os.path.isdir(out_dir):
        return 0
    mx = 0
    for name in os.listdir(out_dir):
        m = re.match(r"checkpoint-(\d+)$", name)
        if m:
            mx = max(mx, int(m.group(1)))
    return mx


def latest_checkpoint(out_dir: str):
    if not os.path.isdir(out_dir):
        return None
    ckpts = []
    for name in os.listdir(out_dir):
        m = re.match(r"checkpoint-(\d+)$", name)
        if m:
            path = os.path.join(out_dir, name)
            if os.path.isdir(path):
                ckpts.append((os.path.getmtime(path), path))
    if not ckpts:
        return None
    ckpts.sort(reverse=True)
    return ckpts[0][1]


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


# ---------- Argument Parsing ----------

def parse_args():
    p = argparse.ArgumentParser(description="Train SegFormer on 1+ HF datasets (no augmentation).")
    p.add_argument("--model_id",         type=str, required=True)
    p.add_argument("--dataset_ids",      type=str, nargs="+", required=True)
    p.add_argument("--output_dir",       type=str, required=True)
    p.add_argument("--learning_rate",    type=float, default=1e-5)
    p.add_argument("--num_epochs",       type=int,   default=10)
    p.add_argument("--max_steps",        type=int,   default=-1)
    p.add_argument("--train_batch_size", type=int,   default=8)
    p.add_argument("--eval_batch_size",  type=int,   default=8)
    p.add_argument("--save_steps",       type=int,   default=200)
    p.add_argument("--eval_steps",       type=int,   default=200)
    p.add_argument("--logging_steps",    type=int,   default=50)
    p.add_argument("--push_to_hub",      action="store_true")
    p.add_argument("--seed",             type=int,   default=42)
    return p.parse_args()


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

    # ---- Build a fixed, balanced eval subset ----
    def _example_has_pos(ex) -> bool:
        lbl = np.array(Image.fromarray(np.uint8(ex["label"])).convert("L"))
        lbl = remap_labels(lbl)
        return bool((lbl > 0).any())

    pos_idx, neg_idx = [], []
    MAX_SCAN = min(len(eval_ds), 20000)

    for i in range(MAX_SCAN):
        if _example_has_pos(eval_ds[i]):
            pos_idx.append(i)
        else:
            neg_idx.append(i)

    print(f"[eval scan] scanned={MAX_SCAN} pos={len(pos_idx)} neg={len(neg_idx)}")

    if len(pos_idx) == 0:
        print("WARNING: No positive pixels found in eval_ds. "
              "Class-1 metrics will be NaN. Check your labels/remap or eval split.")
    else:
        n_pos = min(2000, len(pos_idx))
        n_neg = min(2000, len(neg_idx))
        keep = pos_idx[:n_pos] + neg_idx[:n_neg]
        eval_ds = eval_ds.select(keep)
        print(f"[eval subset] using {len(eval_ds)} examples: pos={n_pos} neg={n_neg}")

    # ---- Processor ----
    processor = SegformerImageProcessor(do_resize=True, do_normalize=True)

    # ---- Transform (no augmentation) ----
    def transforms(example_batch):
        images, labels = [], []

        for img, lbl in zip(example_batch["pixel_values"], example_batch["label"]):
            img = np.array(Image.fromarray(np.uint8(img)).convert("RGB"))
            lbl = np.array(Image.fromarray(np.uint8(lbl)).convert("L"))
            lbl = remap_labels(lbl).astype(np.uint8)

            images.append(img)
            labels.append(lbl)

        return processor(images, labels, return_tensors="pt")

    train_ds.set_transform(transforms)
    eval_ds.set_transform(transforms)

    # ---- Model ----
    id2label = {0: "normal", 1: "abnormality"}
    label2id = {v: k for k, v in id2label.items()}

    # ---- Training args ----
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        load_best_model_at_end=False,
        push_to_hub=args.push_to_hub,
        report_to=["none"],
        seed=args.seed,
    )

    # ---- Load model (resume from checkpoint if available) ----
    ckpt = latest_checkpoint(args.output_dir)

    if ckpt:
        print("Loading weights from checkpoint:", ckpt)
        model = SegformerForSemanticSegmentation.from_pretrained(
            ckpt, num_labels=2, id2label=id2label, label2id=label2id
        )
        set_dropout(model, hidden=0.3, attn=0.3, classifier=0.3)
    else:
        print("No checkpoint found. Starting from base model:", args.model_id)
        model = SegformerForSemanticSegmentation.from_pretrained(
            args.model_id, num_labels=2, id2label=id2label, label2id=label2id
        )

    # ---- Trainer ----
    trainer = EMATeacherAmbiguityIgnoreTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        ema_decay=0.999,
        tau_pos=0.97,
        tau_neg=0.97,
        warmup_steps=500,
    )

    offset = max_checkpoint_step(args.output_dir)
    print(f"[checkpoint offset] max existing step in {args.output_dir} = {offset}")
    trainer.add_callback(OffsetCheckpointNamer(args.output_dir, offset))

    lr_cb = ReduceLROnPlateauCallback(
        monitor="eval_mean_iou",
        factor=0.7,
        patience=3,
        min_lr=1e-7,
    )
    trainer.add_callback(lr_cb)
    lr_cb.trainer = trainer

    trainer.train(resume_from_checkpoint=False)


if __name__ == "__main__":
    main()
