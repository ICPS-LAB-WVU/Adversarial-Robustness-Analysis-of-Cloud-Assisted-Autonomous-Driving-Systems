#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_adversarial_detection.py

Generate adversarial images (FGSM surrogate / FGSM_GT / PGD_GT) and log *dense*
predictions for later computation of:
  - PR curves (needs near-all predictions; use --log-conf 0.001)
  - F1 / Precision / Recall @ any confidence threshold (offline sweep)
  - Recall @ IoU thresholds (0.25/0.5/0.75 etc.)
  - Confusion matrix (IoU-matched)
  - mAP@0.5, mAP@0.75, mAP@[.5:.95] (offline AP)  <-- requires dense preds

Key fixes vs your script:
  1) Separates NMS IoU (inference) from evaluation IoU (offline matching).
  2) Logs predictions with --log-conf (default 0.001) so PR/mAP is possible.
  3) Adds loss_mode for GT-based attacks: full / cls / box / dfl / box_dfl.
  4) Fixes PGD grad bookkeeping (last-step grad is captured correctly).
  5) Writes consistent CSVs: ground_truth.csv, predictions.csv, summary.csv, dataset_check.csv.
"""

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss


# -------------------------
# Args / IO
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path, default=Path("dataset/images"))
    p.add_argument("--labels", type=Path, default=Path("dataset/labels"))
    p.add_argument("--weights", type=Path, default=Path("weight/yolo.pt"))
    p.add_argument("--out", type=Path, default=Path("offline_attacks_v2"))

    p.add_argument("--imgsz", type=int, default=640)

    # Logging/inference knobs (for building PR/mAP later)
    p.add_argument("--log-conf", type=float, default=0.001,
                   help="Minimum conf for logging predictions. Set tiny (0.001) for PR/mAP.")
    p.add_argument("--nms-iou", type=float, default=0.7,
                   help="NMS IoU used during YOLO predict() post-processing.")

    # Eps / attacks
    p.add_argument("--eps", type=float, nargs="+", default=[0.01, 0.02, 0.05])
    p.add_argument("--pgd-alpha", type=float, default=0.007)
    p.add_argument("--pgd-iters", type=int, default=10)

    # Loss mode for GT-based attacks
    p.add_argument("--loss-mode", type=str, default="full",
                   choices=["full", "cls", "box", "dfl", "box_dfl"],
                   help="Which YOLOv8 loss component to maximize for FGSM_GT/PGD_GT.")

    # Dataset / run control
    p.add_argument("--max-images", type=int, default=0, help="0 = no limit")
    p.add_argument("--include-adv", action="store_true",
                   help="Include *_adv.* images from dataset/images")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def list_images(img_dir: Path, include_adv: bool) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    out = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in exts:
            continue
        if (not include_adv) and "_adv" in p.stem:
            continue
        out.append(p)
    return out


# -------------------------
# Labels / batch building
# -------------------------
def load_yolo_labels(label_path: Path, img_idx: int, device: torch.device, min_wh=0.0):
    """
    YOLO txt label format: cls x y w h (normalized).
    Returns tensor [N,6] = [img_idx, cls, x, y, w, h].
    """
    if not label_path.exists():
        return torch.zeros((0, 6), dtype=torch.float32, device=device)

    rows = []
    with label_path.open("r") as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) != 5:
                continue
            cls, x, y, w, h = map(float, parts)
            if w < min_wh or h < min_wh or w > 1.0 or h > 1.0:
                continue
            rows.append([img_idx, cls, x, y, w, h])

    if not rows:
        return torch.zeros((0, 6), dtype=torch.float32, device=device)

    return torch.tensor(rows, dtype=torch.float32, device=device)


def build_single_batch(img_tensor: torch.Tensor, labels_tensor: torch.Tensor):
    """
    Constructs YOLOv8 loss batch dict expected by v8DetectionLoss.
    bboxes are normalized xywh here (as YOLO labels).
    """
    if labels_tensor.numel() == 0:
        return {
            "img": img_tensor,
            "bboxes": torch.zeros((0, 4), device=img_tensor.device),
            "cls": torch.zeros((0, 1), device=img_tensor.device),
            "batch_idx": torch.zeros((0,), dtype=torch.long, device=img_tensor.device),
        }

    t = labels_tensor.clone()
    t[:, 0] = 0  # single image batch idx
    return {
        "img": img_tensor,
        "bboxes": t[:, 2:6],       # x,y,w,h normalized
        "cls": t[:, 1:2].float(),  # class id
        "batch_idx": t[:, 0].long()
    }


def yolo_labels_to_xyxy(labels: torch.Tensor, imgsz: int):
    rows = []
    if labels.numel() == 0:
        return rows
    for _, cls, x, y, w, h in labels.detach().cpu().numpy().tolist():
        x1 = (x - w / 2.0) * imgsz
        y1 = (y - h / 2.0) * imgsz
        x2 = (x + w / 2.0) * imgsz
        y2 = (y + h / 2.0) * imgsz
        rows.append((int(cls), x1, y1, x2, y2))
    return rows


# -------------------------
# YOLO loss plumbing
# -------------------------
def make_v8_loss(core_model):
    loss = v8DetectionLoss(core_model)

    # Ensure hyp exists and has stable defaults
    try:
        hyp_dict = loss.hyp if isinstance(loss.hyp, dict) else vars(loss.hyp)
        hyp = dict(hyp_dict)
    except Exception:
        hyp = {}

    # Typical-ish weights; keep yours if you want
    hyp.setdefault("box", 7.5)
    hyp.setdefault("cls", 0.5)
    hyp.setdefault("dfl", 1.5)
    loss.hyp = SimpleNamespace(**hyp)
    return loss


def freeze_bn_(m: torch.nn.Module):
    for mod in m.modules():
        if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
            mod.eval()
            for p in mod.parameters(recurse=False):
                p.requires_grad_(False)


def select_loss_component(total_loss, loss_items, mode: str):
    """
    Ultralytics v8DetectionLoss typically returns:
      total_loss: tensor scalar or tensor
      loss_items: tensor-like [box, cls, dfl] (often shape [3])
    This function robustly extracts the desired component.
    """
    if mode == "full":
        return total_loss.sum() if total_loss.numel() > 1 else total_loss

    # Make loss_items a flat tensor if possible
    li = loss_items
    if li is None:
        # fallback: if items unavailable, use total (but warn upstream)
        return total_loss.sum() if total_loss.numel() > 1 else total_loss

    if isinstance(li, (list, tuple)):
        li = torch.stack([x if torch.is_tensor(x) else torch.tensor(float(x), device=total_loss.device) for x in li])
    elif torch.is_tensor(li):
        li = li.flatten()
    else:
        # unknown structure
        return total_loss.sum() if total_loss.numel() > 1 else total_loss

    # Heuristic mapping: [box, cls, dfl]
    box = li[0] if li.numel() > 0 else total_loss
    cls = li[1] if li.numel() > 1 else total_loss
    dfl = li[2] if li.numel() > 2 else total_loss

    if mode == "box":
        return box
    if mode == "cls":
        return cls
    if mode == "dfl":
        return dfl
    if mode == "box_dfl":
        return box + dfl

    # default safe fallback
    return total_loss.sum() if total_loss.numel() > 1 else total_loss


# -------------------------
# Adversarial generators
# -------------------------
def fgsm_surrogate(core_model, x_clean: torch.Tensor, eps: float):
    """
    No-GT surrogate FGSM: uses a differentiable objective on raw head outputs.
    NOTE: This is NOT 'cls-only' YOLO loss. It's a surrogate to produce a gradient.
    Objective chosen to *reduce* activations overall (tends to suppress detections):
        loss = head.mean()
        adv  = x - eps * sign(grad)  (gradient descent on head mean)
    """
    x = x_clean.clone().detach().requires_grad_(True)
    was_training = core_model.training
    try:
        core_model.train()
        freeze_bn_(core_model)
        preds = core_model(x)
        head = preds[0] if isinstance(preds, (list, tuple)) else preds

        loss = head.mean()
        core_model.zero_grad(set_to_none=True)
        loss.backward()

        if x.grad is None:
            return None, float(loss.detach().cpu().item()), 0.0, 0.0, 0.0, 0.0

        grad = x.grad.detach()
        gmean = float(grad.abs().mean().item())
        gmax = float(grad.abs().max().item())

        # Gradient descent step to suppress head activations
        adv = torch.clamp(x - eps * grad.sign(), 0.0, 1.0).detach()

        d = (adv - x_clean).abs()
        dmean = float(d.mean().item())
        dmax = float(d.max().item())
        return adv, float(loss.detach().cpu().item()), gmean, gmax, dmean, dmax
    finally:
        core_model.train(was_training)


def fgsm_gt(core_model, loss_fn, x_clean: torch.Tensor, batch: dict, eps: float, loss_mode: str):
    if batch["bboxes"].numel() == 0:
        return None, None, 0.0, 0.0, 0.0, 0.0

    x = x_clean.clone().detach().requires_grad_(True)
    was_training = core_model.training
    try:
        core_model.train()
        freeze_bn_(core_model)

        preds = core_model(x)
        total_loss, loss_items = loss_fn(preds, batch)
        loss = select_loss_component(total_loss, loss_items, loss_mode)

        core_model.zero_grad(set_to_none=True)
        loss.backward()

        if x.grad is None:
            return None, float(loss.detach().cpu().item()), 0.0, 0.0, 0.0, 0.0

        grad = x.grad.detach()
        gmean = float(grad.abs().mean().item())
        gmax = float(grad.abs().max().item())

        adv = torch.clamp(x + eps * grad.sign(), 0.0, 1.0).detach()

        d = (adv - x_clean).abs()
        dmean = float(d.mean().item())
        dmax = float(d.max().item())

        return adv, float(loss.detach().cpu().item()), gmean, gmax, dmean, dmax
    finally:
        core_model.train(was_training)


def pgd_gt(core_model, loss_fn, x_clean: torch.Tensor, batch: dict,
           eps: float, alpha: float, iters: int, loss_mode: str):
    if batch["bboxes"].numel() == 0:
        return None, None, 0.0, 0.0, 0.0, 0.0

    x_adv = x_clean.clone().detach()
    last_grad = None
    last_loss_val = None

    was_training = core_model.training
    try:
        core_model.train()
        freeze_bn_(core_model)

        # Random start inside eps-ball (often stronger)
        rand = torch.empty_like(x_adv).uniform_(-eps, eps)
        x_adv = torch.clamp(x_adv + rand, 0.0, 1.0)

        for _ in range(iters):
            x_adv = x_adv.clone().detach().requires_grad_(True)

            preds = core_model(x_adv)
            total_loss, loss_items = loss_fn(preds, batch)
            loss = select_loss_component(total_loss, loss_items, loss_mode)

            core_model.zero_grad(set_to_none=True)
            loss.backward()

            if x_adv.grad is None:
                return None, float(loss.detach().cpu().item()), 0.0, 0.0, 0.0, 0.0

            last_grad = x_adv.grad.detach()
            last_loss_val = float(loss.detach().cpu().item())

            # Gradient ascent to maximize loss
            x_adv = x_adv + alpha * last_grad.sign()

            # Project back to eps-ball around x_clean
            perturb = torch.clamp(x_adv - x_clean, min=-eps, max=eps)
            x_adv = torch.clamp(x_clean + perturb, 0.0, 1.0).detach()

        grad = last_grad
        gmean = float(grad.abs().mean().item()) if grad is not None else 0.0
        gmax = float(grad.abs().max().item()) if grad is not None else 0.0

        d = (x_adv - x_clean).abs()
        dmean = float(d.mean().item())
        dmax = float(d.max().item())

        return x_adv.detach(), last_loss_val, gmean, gmax, dmean, dmax
    finally:
        core_model.train(was_training)


# -------------------------
# Image utils / prediction
# -------------------------
def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    arr = (x[0].detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def save_image(path: Path, img: Image.Image):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=95)


def predict_yolo(yolo: YOLO, img: Image.Image, log_conf: float, imgsz: int, nms_iou: float):
    # This is post-NMS output; dense logging requires tiny log_conf.
    out = yolo.predict(source=np.array(img), conf=log_conf, imgsz=imgsz, iou=nms_iou, verbose=False)
    return out[0] if out else None


def write_preds(pred_w, img_path: Path, attack: str, eps: float, res, imgsz: int):
    if res is None or res.boxes is None:
        return 0
    n = len(res.boxes)
    if n == 0:
        return 0

    xyxy = res.boxes.xyxy.detach().cpu().numpy()
    cls = res.boxes.cls.detach().cpu().numpy()
    conf = res.boxes.conf.detach().cpu().numpy()

    for b, c, s in zip(xyxy, cls, conf):
        x1, y1, x2, y2 = b.tolist()
        pred_w.writerow({
            "image": str(img_path),
            "attack": attack,
            "eps": eps,
            "cls": int(c),
            "conf": f"{float(s):.6f}",
            "x1": f"{float(x1):.3f}",
            "y1": f"{float(y1):.3f}",
            "x2": f"{float(x2):.3f}",
            "y2": f"{float(y2):.3f}",
            "imgsz": imgsz,
        })
    return int(n)


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.out.mkdir(parents=True, exist_ok=True)
    dataset_check_path = args.out / "dataset_check.csv"
    preds_path = args.out / "predictions.csv"
    gts_path = args.out / "ground_truth.csv"
    summary_path = args.out / "summary.csv"

    images = list_images(args.images, include_adv=args.include_adv)
    if args.max_images > 0:
        images = images[: args.max_images]

    # Load model
    model = YOLO(str(args.weights)).to(device)
    core = model.model.to(device)
    loss_fn = make_v8_loss(core)

    transform = T.Compose([T.Resize((args.imgsz, args.imgsz)), T.ToTensor()])

    with dataset_check_path.open("w", newline="") as f_check, \
         preds_path.open("w", newline="") as f_pred, \
         gts_path.open("w", newline="") as f_gt, \
         summary_path.open("w", newline="") as f_sum:

        check_w = csv.DictWriter(f_check, fieldnames=[
            "image", "label", "label_exists", "n_labels",
        ])
        pred_w = csv.DictWriter(f_pred, fieldnames=[
            "image", "attack", "eps", "cls", "conf", "x1", "y1", "x2", "y2", "imgsz",
        ])
        gt_w = csv.DictWriter(f_gt, fieldnames=[
            "image", "cls", "x1", "y1", "x2", "y2", "imgsz",
        ])
        sum_w = csv.DictWriter(f_sum, fieldnames=[
            "image", "attack", "eps", "loss_mode",
            "n_gt", "n_pred",
            "loss_value",
            "grad_mean", "grad_max",
            "delta_mean", "delta_max",
            "log_conf", "nms_iou",
            "notes",
        ])

        check_w.writeheader()
        pred_w.writeheader()
        gt_w.writeheader()
        sum_w.writeheader()

        for img_path in images:
            label_path = args.labels / f"{img_path.stem}.txt"
            labels = load_yolo_labels(label_path, img_idx=0, device=device)

            check_w.writerow({
                "image": str(img_path),
                "label": str(label_path),
                "label_exists": int(label_path.exists()),
                "n_labels": int(labels.shape[0]),
            })

            # Write GT (in imgsz pixel space)
            gt_rows = yolo_labels_to_xyxy(labels, args.imgsz)
            for cls, x1, y1, x2, y2 in gt_rows:
                gt_w.writerow({
                    "image": str(img_path),
                    "cls": cls,
                    "x1": f"{x1:.3f}",
                    "y1": f"{y1:.3f}",
                    "x2": f"{x2:.3f}",
                    "y2": f"{y2:.3f}",
                    "imgsz": args.imgsz,
                })

            # Load image tensor
            pil = Image.open(img_path).convert("RGB")
            x_clean = transform(pil).unsqueeze(0).to(device)
            batch = build_single_batch(x_clean, labels)

            # --------------
            # Clean logging
            # --------------
            clean_img = tensor_to_pil(x_clean)
            res_clean = predict_yolo(model, clean_img, args.log_conf, args.imgsz, args.nms_iou)
            n_pred_clean = write_preds(pred_w, img_path, "clean", 0.0, res_clean, args.imgsz)

            sum_w.writerow({
                "image": str(img_path),
                "attack": "clean",
                "eps": 0.0,
                "loss_mode": "",
                "n_gt": int(labels.shape[0]),
                "n_pred": int(n_pred_clean),
                "loss_value": "",
                "grad_mean": "",
                "grad_max": "",
                "delta_mean": "",
                "delta_max": "",
                "log_conf": args.log_conf,
                "nms_iou": args.nms_iou,
                "notes": "",
            })

            # ----------------------------
            # Attacks for each eps
            # ----------------------------
            for eps in args.eps:
                if eps <= 0.0:
                    continue

                # ---- FGSM surrogate (no-GT) ----
                fgsm_adv, loss_val, gmean, gmax, dmean, dmax = fgsm_surrogate(core, x_clean, eps)
                if fgsm_adv is None:
                    sum_w.writerow({
                        "image": str(img_path),
                        "attack": "fgsm_sur",
                        "eps": eps,
                        "loss_mode": "surrogate",
                        "n_gt": int(labels.shape[0]),
                        "n_pred": 0,
                        "loss_value": f"{loss_val:.6e}" if loss_val is not None else "",
                        "grad_mean": "",
                        "grad_max": "",
                        "delta_mean": "",
                        "delta_max": "",
                        "log_conf": args.log_conf,
                        "nms_iou": args.nms_iou,
                        "notes": "no_grad",
                    })
                else:
                    fgsm_path = args.out / "adv" / "fgsm_sur" / f"eps_{eps:.3f}" / img_path.name
                    save_image(fgsm_path, tensor_to_pil(fgsm_adv))
                    res_fgsm = predict_yolo(model, tensor_to_pil(fgsm_adv), args.log_conf, args.imgsz, args.nms_iou)
                    n_pred_fgsm = write_preds(pred_w, img_path, "fgsm_sur", eps, res_fgsm, args.imgsz)

                    sum_w.writerow({
                        "image": str(img_path),
                        "attack": "fgsm_sur",
                        "eps": eps,
                        "loss_mode": "surrogate",
                        "n_gt": int(labels.shape[0]),
                        "n_pred": int(n_pred_fgsm),
                        "loss_value": f"{loss_val:.6e}" if loss_val is not None else "",
                        "grad_mean": f"{gmean:.6e}",
                        "grad_max": f"{gmax:.6e}",
                        "delta_mean": f"{dmean:.6e}",
                        "delta_max": f"{dmax:.6e}",
                        "log_conf": args.log_conf,
                        "nms_iou": args.nms_iou,
                        "notes": "",
                    })

                # ---- FGSM GT (YOLO loss component) ----
                fgsm_gt_adv, loss_val, gmean, gmax, dmean, dmax = fgsm_gt(
                    core, loss_fn, x_clean, batch, eps, args.loss_mode
                )
                note = ""
                if fgsm_gt_adv is None:
                    note = "no_gt_or_no_grad"
                else:
                    fgsm_gt_path = args.out / "adv" / "fgsm_gt" / args.loss_mode / f"eps_{eps:.3f}" / img_path.name
                    save_image(fgsm_gt_path, tensor_to_pil(fgsm_gt_adv))
                    res_fgsm_gt = predict_yolo(model, tensor_to_pil(fgsm_gt_adv), args.log_conf, args.imgsz, args.nms_iou)
                    n_pred_fgsm_gt = write_preds(pred_w, img_path, "fgsm_gt", eps, res_fgsm_gt, args.imgsz)

                    sum_w.writerow({
                        "image": str(img_path),
                        "attack": "fgsm_gt",
                        "eps": eps,
                        "loss_mode": args.loss_mode,
                        "n_gt": int(labels.shape[0]),
                        "n_pred": int(n_pred_fgsm_gt),
                        "loss_value": f"{loss_val:.6e}" if loss_val is not None else "",
                        "grad_mean": f"{gmean:.6e}",
                        "grad_max": f"{gmax:.6e}",
                        "delta_mean": f"{dmean:.6e}",
                        "delta_max": f"{dmax:.6e}",
                        "log_conf": args.log_conf,
                        "nms_iou": args.nms_iou,
                        "notes": note,
                    })

                if fgsm_gt_adv is None:
                    sum_w.writerow({
                        "image": str(img_path),
                        "attack": "fgsm_gt",
                        "eps": eps,
                        "loss_mode": args.loss_mode,
                        "n_gt": int(labels.shape[0]),
                        "n_pred": 0,
                        "loss_value": f"{loss_val:.6e}" if loss_val is not None else "",
                        "grad_mean": "",
                        "grad_max": "",
                        "delta_mean": "",
                        "delta_max": "",
                        "log_conf": args.log_conf,
                        "nms_iou": args.nms_iou,
                        "notes": note,
                    })

                # ---- PGD GT (YOLO loss component) ----
                pgd_adv, loss_val, gmean, gmax, dmean, dmax = pgd_gt(
                    core, loss_fn, x_clean, batch, eps, args.pgd_alpha, args.pgd_iters, args.loss_mode
                )
                note = ""
                if pgd_adv is None:
                    note = "no_gt_or_no_grad"
                    sum_w.writerow({
                        "image": str(img_path),
                        "attack": "pgd_gt",
                        "eps": eps,
                        "loss_mode": args.loss_mode,
                        "n_gt": int(labels.shape[0]),
                        "n_pred": 0,
                        "loss_value": f"{loss_val:.6e}" if loss_val is not None else "",
                        "grad_mean": "",
                        "grad_max": "",
                        "delta_mean": "",
                        "delta_max": "",
                        "log_conf": args.log_conf,
                        "nms_iou": args.nms_iou,
                        "notes": note,
                    })
                else:
                    pgd_path = args.out / "adv" / "pgd_gt" / args.loss_mode / f"eps_{eps:.3f}" / img_path.name
                    save_image(pgd_path, tensor_to_pil(pgd_adv))
                    res_pgd = predict_yolo(model, tensor_to_pil(pgd_adv), args.log_conf, args.imgsz, args.nms_iou)
                    n_pred_pgd = write_preds(pred_w, img_path, "pgd_gt", eps, res_pgd, args.imgsz)

                    sum_w.writerow({
                        "image": str(img_path),
                        "attack": "pgd_gt",
                        "eps": eps,
                        "loss_mode": args.loss_mode,
                        "n_gt": int(labels.shape[0]),
                        "n_pred": int(n_pred_pgd),
                        "loss_value": f"{loss_val:.6e}" if loss_val is not None else "",
                        "grad_mean": f"{gmean:.6e}",
                        "grad_max": f"{gmax:.6e}",
                        "delta_mean": f"{dmean:.6e}",
                        "delta_max": f"{dmax:.6e}",
                        "log_conf": args.log_conf,
                        "nms_iou": args.nms_iou,
                        "notes": note,
                    })


if __name__ == "__main__":
    main()
