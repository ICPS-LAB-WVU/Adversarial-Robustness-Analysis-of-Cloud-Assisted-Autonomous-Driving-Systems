# ======================= ICC ADVERSARY v4 ==========================
# Real-time adversary for Duckietown YOLO (Ultralytics)
#
# Adds TRUE white-box attacks using the model's *training loss*:
#   - wb_fgsm:  1-step FGSM on detection loss
#   - wb_pgd:   multi-step PGD on detection loss
#
# If you don't have GT labels at runtime, it can auto-build a pseudo-GT batch
# from the model's clean detections (self-training style) so the loss is valid.
#
# Keeps your surrogate modes too:
#   - conf_suppress
#   - class_flip
#   - suppress_stop
#
# Control via /tmp/icc_attack_ctrl.json, e.g.:
# {
#   "enable": true,
#   "mode": "wb_fgsm",
#   "eps": 0.02,
#   "alpha": 0.005,
#   "iters": 3,
#   "conf": 0.25,
#   "max_det": 30,
#   "use_pseudo_gt": true
# }
# ------------------------------------------------------------------

import os
import json
import time
import numpy as np
import torch
from types import SimpleNamespace
from ultralytics import YOLO

# =============================================================
# Runtime control (/tmp/icc_attack_ctrl.json)
# =============================================================
CTRL_FILE = "/tmp/icc_attack_ctrl.json"

def read_ctrl():
    try:
        with open(CTRL_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"enable": False}

def attack_enabled():
    """Used by Receive_store.py to quickly check if attack is on."""
    return bool(read_ctrl().get("enable", False))

# =============================================================
# Model loading
# =============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.environ.get("YOLO_WEIGHTS", "weight/yolo.pt")

_yolo = YOLO(MODEL_PATH)
_core = _yolo.model.to(DEVICE)
_core.eval()

print(f"[ICC-ADV] Loaded YOLO from {MODEL_PATH} on {DEVICE}", flush=True)

# =============================================================
# Helpers: safe criterion init + safe forward
# =============================================================
_CRIT = None

def _init_criterion():
    """
    Initialize Ultralytics criterion with robust hyp handling.
    Ultralytics losses expect hyp fields like .box/.cls/.dfl.
    """
    global _CRIT
    if _CRIT is not None:
        return _CRIT

    # Some forks store model.args as dict (fine), but loss.hyp is usually a SimpleNamespace internally.
    # Still, a few forks regress to dict. We guard that.
    try:
        if hasattr(_core, "init_criterion"):
            crit = _core.init_criterion()
        elif hasattr(_core, "criterion"):
            crit = _core.criterion
        else:
            raise RuntimeError("Model has no init_criterion() / criterion")

        # If crit has hyp and it's a dict, convert to SimpleNamespace
        if hasattr(crit, "hyp") and isinstance(crit.hyp, dict):
            crit.hyp = SimpleNamespace(**crit.hyp)

        # If crit lacks hyp entirely (some e2e variants), inject sensible defaults
        if not hasattr(crit, "hyp"):
            crit.hyp = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

        # If hyp exists but missing fields, fill them
        for k, v in {"box": 7.5, "cls": 0.5, "dfl": 1.5}.items():
            if not hasattr(crit.hyp, k):
                setattr(crit.hyp, k, v)

        _CRIT = crit
        print(f"[ICC-ADV] criterion={crit.__class__.__name__} hyp(box/cls/dfl)="
              f"{crit.hyp.box}/{crit.hyp.cls}/{crit.hyp.dfl}", flush=True)
        return _CRIT

    except Exception as e:
        print(f"[ICC-ADV] WARNING: init_criterion failed ({e}). "
              f"Falling back to surrogate-only.", flush=True)
        _CRIT = None
        return None

def _forward_train_safe(x):
    """
    Forward that works for Ultralytics models in gradient mode.
    Some models return dict/tuple; criterion handles typical outputs.
    """
    # keep eval() (we want deterministic BN), but allow grads
    return _core(x)

def _bgr_to_tensor(img_bgr, device=DEVICE):
    x = torch.from_numpy(img_bgr).float().to(device)          # H W C
    x = x.permute(2, 0, 1).unsqueeze(0) / 255.0               # 1 3 H W in [0,1]
    return x

def _tensor_to_bgr(x):
    out = (x[0].detach().permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
    return out

# =============================================================
# Build batch for Ultralytics loss (GT or pseudo-GT)
# =============================================================
def _xyxy_to_xywhn(xyxy, h, w, device):
    """
    xyxy in pixels -> normalized xywh (0..1), float32
    """
    x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    bw = (x2 - x1).clamp(min=1.0)
    bh = (y2 - y1).clamp(min=1.0)
    cx = x1 + 0.5 * bw
    cy = y1 + 0.5 * bh
    xywh = torch.stack([cx / w, cy / h, bw / w, bh / h], dim=1)
    return xywh.to(device=device, dtype=torch.float32)

def build_batch_from_dets(dets, img_tensor):
    """
    Convert Ultralytics Results -> a minimal 'batch' dict the criterion can use.
    NOTE: this is pseudo-GT (teacher labels) unless you supply real GT.
    """
    device = img_tensor.device
    _, _, H, W = img_tensor.shape

    if dets is None or dets.boxes is None or len(dets.boxes) == 0:
        # empty targets
        return {
            "img": img_tensor,
            "batch_idx": torch.zeros((0,), device=device, dtype=torch.int64),
            "cls": torch.zeros((0, 1), device=device, dtype=torch.float32),
            "bboxes": torch.zeros((0, 4), device=device, dtype=torch.float32),
        }

    # Use detected boxes as pseudo-GT
    xyxy = dets.boxes.xyxy.detach()
    cls  = dets.boxes.cls.detach()

    if xyxy.device != device:
        xyxy = xyxy.to(device)
        cls  = cls.to(device)

    bboxes = _xyxy_to_xywhn(xyxy, H, W, device)
    cls = cls.view(-1, 1).float()

    batch_idx = torch.zeros((bboxes.shape[0],), device=device, dtype=torch.int64)  # single image batch

    return {
        "img": img_tensor,
        "batch_idx": batch_idx,
        "cls": cls,
        "bboxes": bboxes,
    }

def build_batch_from_yolo_txt_labels(labels_tensor, img_tensor):
    """
    If you have YOLO txt labels already as tensor:
      labels_tensor: Nx6 [img_idx, class, cx, cy, w, h] normalized
    Convert to criterion batch format.
    """
    device = img_tensor.device
    if labels_tensor is None or labels_tensor.numel() == 0:
        return {
            "img": img_tensor,
            "batch_idx": torch.zeros((0,), device=device, dtype=torch.int64),
            "cls": torch.zeros((0, 1), device=device, dtype=torch.float32),
            "bboxes": torch.zeros((0, 4), device=device, dtype=torch.float32),
        }

    lt = labels_tensor.to(device=device)
    # expected normalized xywh already
    batch_idx = lt[:, 0].long()
    cls = lt[:, 1:2].float()
    bboxes = lt[:, 2:6].float()

    return {"img": img_tensor, "batch_idx": batch_idx, "cls": cls, "bboxes": bboxes}

# =============================================================
# Surrogate losses (kept from your v3)
# =============================================================
def loss_conf_suppress(preds):
    head = preds[0] if isinstance(preds, (list, tuple)) else preds
    return -head.mean()

def loss_class_flip(preds, target_cls=0):
    head = preds[0] if isinstance(preds, (list, tuple)) else preds
    class_logits = head[..., 5:]
    class_probs = torch.softmax(class_logits, dim=-1)
    target_prob = class_probs[..., target_cls]
    return -target_prob.mean()

def loss_suppress_stop(preds, stop_cls=2, obj_thresh=0.3):
    head = preds[0] if isinstance(preds, (list, tuple)) else preds
    obj_logit = head[..., 4]
    cls_logits = head[..., 5:]
    obj_prob = obj_logit.sigmoid()
    cls_probs = torch.softmax(cls_logits, dim=-1)
    stop_prob = cls_probs[..., stop_cls]
    top_cls = cls_probs.argmax(dim=-1)
    mask = (top_cls == stop_cls) & (obj_prob > obj_thresh)

    if mask.any():
        focused = (stop_prob * obj_prob)[mask]
        return -focused.mean()
    return torch.tensor(0.0, device=head.device, requires_grad=True)

# =============================================================
# White-box detection-loss attacks (Ultralytics criterion)
# =============================================================
def _wb_det_loss(img_tensor, batch):
    """
    Compute total detection loss using Ultralytics criterion.
    Returns scalar loss.
    """
    crit = _init_criterion()
    if crit is None:
        # hard fail to surrogate route
        raise RuntimeError("No criterion available")

    preds = _forward_train_safe(img_tensor)
    loss, _items = crit(preds, batch)

    # Ultralytics sometimes returns loss as tensor vector; force scalar
    if isinstance(loss, torch.Tensor) and loss.numel() > 1:
        loss = loss.sum()
    return loss

def _fgsm(x, eps, grad):
    return (x + eps * grad.sign()).clamp(0.0, 1.0)

def _pgd(x0, eps, alpha, iters, grad_fn):
    # random init inside eps-ball (optional) — keep OFF for repeatability
    x = x0.clone().detach()
    for _ in range(iters):
        x.requires_grad_(True)
        loss = grad_fn(x)

        _core.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad.zero_()
        loss.backward()

        g = x.grad
        if g is None:
            g = torch.zeros_like(x)

        x = x.detach() + alpha * g.sign()
        delta = (x - x0).clamp(-eps, eps)
        x = (x0 + delta).clamp(0.0, 1.0).detach()
    return x

# =============================================================
# MAIN ATTACK ENGINE
# =============================================================
def attack_frame(img_bgr, force_attack=False, labels_tensor=None, dets=None):
    """
    Apply adversarial perturbation to one BGR frame (H,W,3 uint8).
    Called from Receive_store.py.

    labels_tensor (optional):
      - If provided, must be Nx6 YOLO normalized labels: [img_idx, cls, cx, cy, w, h]
      - If not provided and use_pseudo_gt=True, we build pseudo-GT from clean detections.
    """
    ctrl = read_ctrl()

    if not ctrl.get("enable", False) and not force_attack:
        attack_frame.last_stats = {
            "attack_mode": "none",
            "adv_mean_diff": 0.0,
            "adv_max_diff": 0.0,
            "adv_compute_time": 0.0,
            "attack_used_pseudo_gt": False,
        }
        return img_bgr

    mode = str(ctrl.get("mode", "conf_suppress")).strip()
    EPS = float(ctrl.get("eps", 0.03))
    ALPHA = float(ctrl.get("alpha", 0.01))
    ITERS = int(ctrl.get("iters", 2))
    target_cls = int(ctrl.get("target_cls", 0))

    conf_th = float(ctrl.get("conf", 0.25))
    max_det = int(ctrl.get("max_det", 30))
    use_pseudo_gt = bool(ctrl.get("use_pseudo_gt", True))

    t_start = time.time()

    # Convert to tensor on DEVICE
    x0 = _bgr_to_tensor(img_bgr, DEVICE)
    x = x0.clone().detach()

    # ---------------- Surrogate modes ----------------
    if mode in ("conf_suppress", "class_flip", "suppress_stop"):
        if mode == "conf_suppress":
            loss_fn = loss_conf_suppress
        elif mode == "class_flip":
            loss_fn = lambda p: loss_class_flip(p, target_cls)
        else:
            loss_fn = lambda p: loss_suppress_stop(p, stop_cls=2, obj_thresh=0.3)

        try:
            for _ in range(ITERS):
                x.requires_grad_(True)
                preds = _forward_train_safe(x)
                loss = loss_fn(preds)

                _core.zero_grad(set_to_none=True)
                if x.grad is not None:
                    x.grad.zero_()
                loss.backward()

                grad = x.grad if x.grad is not None else torch.zeros_like(x)
                grad = torch.sign(grad + 1e-12)

                x = x.detach() + ALPHA * grad
                delta = (x - x0).clamp(-EPS, EPS)
                x = (x0 + delta).clamp(0.0, 1.0).detach()

            out = _tensor_to_bgr(x)

            diff = (x - x0).abs()
            attack_frame.last_stats = {
                "attack_mode": mode,
                "adv_mean_diff": float(diff.mean().cpu()),
                "adv_max_diff": float(diff.max().cpu()),
                "adv_compute_time": float(time.time() - t_start),
                "attack_used_pseudo_gt": False,
                "wb_loss": None,
            }
            print(f"[ICC-ADV] mode={mode} mean={attack_frame.last_stats['adv_mean_diff']:.6f} "
                  f"max={attack_frame.last_stats['adv_max_diff']:.6f} "
                  f"time={attack_frame.last_stats['adv_compute_time']:.4f}", flush=True)
            return out

        except Exception as e:
            print("[ICC-ADV] ERROR in surrogate attack_frame:", repr(e), flush=True)
            attack_frame.last_stats = {
                "attack_mode": "error",
                "error": str(e),
                "adv_mean_diff": 0.0,
                "adv_max_diff": 0.0,
                "adv_compute_time": float(time.time() - t_start),
                "attack_used_pseudo_gt": False,
            }
            return img_bgr

    # ---------------- TRUE white-box modes ----------------
    if mode not in ("wb_fgsm", "wb_pgd"):
        print(f"[ICC-ADV] UNKNOWN MODE: {mode}", flush=True)
        attack_frame.last_stats = {
            "attack_mode": "none",
            "adv_mean_diff": 0.0,
            "adv_max_diff": 0.0,
            "adv_compute_time": 0.0,
            "attack_used_pseudo_gt": False,
        }
        return img_bgr

    try:
        # Build batch
        used_pseudo = False

        if labels_tensor is not None and isinstance(labels_tensor, torch.Tensor) and labels_tensor.numel() > 0:
            batch = build_batch_from_yolo_txt_labels(labels_tensor, x0)
        else:
            if not use_pseudo_gt:
                # no labels and pseudo disabled -> cannot do proper WB loss
                raise RuntimeError("No GT labels provided and use_pseudo_gt=False (cannot compute WB det loss).")

            # compute clean detections if not provided
            if dets is None:
                dets = _yolo.predict(
                    img_bgr, conf=conf_th, imgsz=320, max_det=max_det,
                    verbose=False, device=str(DEVICE), half=False
                )[0]
            batch = build_batch_from_dets(dets, x0)
            used_pseudo = True

        # Define grad_fn(x): loss(x, batch)
        def grad_fn(x_in):
            return _wb_det_loss(x_in, batch)

        if mode == "wb_fgsm":
            x_adv = x0.clone().detach().requires_grad_(True)
            loss = grad_fn(x_adv)

            _core.zero_grad(set_to_none=True)
            if x_adv.grad is not None:
                x_adv.grad.zero_()
            loss.backward()

            g = x_adv.grad if x_adv.grad is not None else torch.zeros_like(x_adv)
            x_out = _fgsm(x_adv.detach(), EPS, g.detach())
            wb_loss_val = float(loss.detach().cpu())

        else:
            x_out = _pgd(x0, EPS, ALPHA, ITERS, grad_fn)
            # compute final loss for logging only
            with torch.no_grad():
                wb_loss_val = float(grad_fn(x_out).detach().cpu())

        out = _tensor_to_bgr(x_out)

        diff = (x_out - x0).abs()
        attack_frame.last_stats = {
            "attack_mode": mode,
            "adv_mean_diff": float(diff.mean().cpu()),
            "adv_max_diff": float(diff.max().cpu()),
            "adv_compute_time": float(time.time() - t_start),
            "attack_used_pseudo_gt": used_pseudo,
            "wb_loss": wb_loss_val,
            "eps": EPS,
            "alpha": ALPHA,
            "iters": ITERS if mode == "wb_pgd" else 1,
        }

        print(
            f"[ICC-ADV] mode={mode} loss={wb_loss_val:.4f} pseudoGT={used_pseudo} "
            f"mean={attack_frame.last_stats['adv_mean_diff']:.6f} "
            f"max={attack_frame.last_stats['adv_max_diff']:.6f} "
            f"time={attack_frame.last_stats['adv_compute_time']:.4f}",
            flush=True,
        )
        return out

    except Exception as e:
        print("[ICC-ADV] ERROR in WB attack_frame:", repr(e), flush=True)
        attack_frame.last_stats = {
            "attack_mode": "error",
            "error": str(e),
            "adv_mean_diff": 0.0,
            "adv_max_diff": 0.0,
            "adv_compute_time": float(time.time() - t_start),
            "attack_used_pseudo_gt": False,
        }
        return img_bgr

# Debug metadata storage
attack_frame.last_stats = {}

def get_attack_metadata():
    return attack_frame.last_stats

