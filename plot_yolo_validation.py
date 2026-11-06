#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import cv2
import numpy as np
import yaml
import matplotlib.pyplot as plt


# ----------------------------
# Helpers
# ----------------------------
def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def resolve_path(base_dir, p):
    """Resolve relative paths in YOLO data.yaml style."""
    if p is None:
        return None
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(base_dir, p))

def list_images(images_dir):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(images_dir, e)))
    files = sorted(files)
    return files

def image_id_from_path(img_path):
    # Ultralytics COCO json uses stem without extension
    # Example: c10005_jpg.rf.xxx  (stem)
    return os.path.splitext(os.path.basename(img_path))[0]

def read_img_wh(img_path):
    im = cv2.imread(img_path)
    if im is None:
        return None
    h, w = im.shape[:2]
    return w, h

def yolo_label_to_xyxy(line, w, h):
    # line: "cls cx cy bw bh" normalized
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cx, cy, bw, bh = map(float, parts[1:5])
    x1 = (cx - bw / 2.0) * w
    y1 = (cy - bh / 2.0) * h
    x2 = (cx + bw / 2.0) * w
    y2 = (cy + bh / 2.0) * h
    return [x1, y1, x2, y2]

def coco_xywh_to_xyxy(b):
    # b = [x, y, w, h]
    x, y, bw, bh = b
    return [x, y, x + bw, y + bh]

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union

def match_detections(gt_boxes, pred_boxes, iou_thr):
    """
    Greedy matching by best IoU.
    gt_boxes: list of xyxy
    pred_boxes: list of xyxy
    Returns: TP, FP, FN
    """
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)

    gt_used = [False] * len(gt_boxes)
    pred_used = [False] * len(pred_boxes)

    # precompute iou matrix
    ious = np.zeros((len(gt_boxes), len(pred_boxes)), dtype=np.float32)
    for i, g in enumerate(gt_boxes):
        for j, p in enumerate(pred_boxes):
            ious[i, j] = iou_xyxy(g, p)

    # greedily assign highest IoU pairs
    while True:
        i, j = np.unravel_index(np.argmax(ious), ious.shape)
        best = ious[i, j]
        if best < iou_thr:
            break
        if gt_used[i] or pred_used[j]:
            ious[i, j] = -1.0
            continue
        gt_used[i] = True
        pred_used[j] = True
        ious[i, :] = -1.0
        ious[:, j] = -1.0

    tp = sum(gt_used)
    fp = len(pred_boxes) - sum(pred_used)
    fn = len(gt_boxes) - sum(gt_used)
    return tp, fp, fn


# ----------------------------
# Core metric computation
# ----------------------------
def build_gt_from_labels(val_images_dir, labels_dir):
    """
    Build GT dict: image_id -> list of GT boxes (xyxy in pixels)
    Uses actual image width/height by reading the val images.
    """
    img_files = list_images(val_images_dir)
    if not img_files:
        raise RuntimeError(f"No images found in val_images_dir: {val_images_dir}")

    gt = {}
    missing_labels = 0

    for img_path in img_files:
        img_id = image_id_from_path(img_path)
        wh = read_img_wh(img_path)
        if wh is None:
            continue
        w, h = wh
        label_path = os.path.join(labels_dir, img_id + ".txt")
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    b = yolo_label_to_xyxy(line, w, h)
                    if b is not None:
                        boxes.append(b)
        else:
            missing_labels += 1
        gt[img_id] = boxes

    return gt, len(img_files), missing_labels

def load_predictions_json(pred_json_path, conf_thr):
    """
    Ultralytics predictions.json format:
    [{"image_id": "...", "category_id": int, "bbox":[x,y,w,h], "score":float}, ...]
    Return dict image_id -> list of pred boxes (xyxy), filtered by conf.
    """
    with open(pred_json_path, "r") as f:
        preds = json.load(f)

    out = defaultdict(list)
    for r in preds:
        score = float(r.get("score", 0.0))
        if score < conf_thr:
            continue
        img_id = r["image_id"]
        b = coco_xywh_to_xyxy(r["bbox"])
        out[img_id].append(b)
    return out

def compute_micro_metrics(gt_dict, pred_dict, iou_thr):
    tp = fp = fn = 0
    n_pred = 0
    for img_id, gt_boxes in gt_dict.items():
        pred_boxes = pred_dict.get(img_id, [])
        n_pred += len(pred_boxes)
        tpi, fpi, fni = match_detections(gt_boxes, pred_boxes, iou_thr)
        tp += tpi
        fp += fpi
        fn += fni

    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    return prec, rec, f1, tp, fp, fn, n_pred


# ----------------------------
# Runs parsing (attack/eps)
# ----------------------------
def parse_run_name(run_name):
    """
    Expected run dirs:
      clean
      fgsm_sur_eps0010
      fgsm_gt_full_eps0020
      pgd_gt_full_eps0050
    Returns: (family, eps_float, label_for_plot)
    """
    if run_name == "clean":
        return ("clean", None, "Clean")

    # epsXXXX
    m = re.search(r"eps(\d+)", run_name)
    if not m:
        return (None, None, None)
    eps_tag = m.group(1)
    # eps0010 -> 0.010
    eps = float(eps_tag) / 1000.0

    if run_name.startswith("fgsm_sur"):
        fam = "fgsm_sur"
        lab = f"FGSM ε={eps:.3f}"
    elif run_name.startswith("fgsm_gt_full"):
        fam = "fgsm_gt"
        lab = f"FGSM_GT ε={eps:.3f}"
    elif run_name.startswith("pgd_gt_full"):
        fam = "pgd_gt"
        lab = f"PGD ε={eps:.3f}"
    else:
        fam = "other"
        lab = run_name

    return fam, eps, lab


# ----------------------------
# Plotting
# ----------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[SAVED] {path}")

def plot_bar(metrics_by_label, out_path, bar_conf, bar_iou):
    labels = list(metrics_by_label.keys())
    precs = [metrics_by_label[k]["P"] for k in labels]
    recs  = [metrics_by_label[k]["R"] for k in labels]

    x = np.arange(len(labels))
    width = 0.38
    plt.figure(figsize=(14, 5))
    plt.bar(x - width/2, precs, width, label="Precision")
    plt.bar(x + width/2, recs,  width, label="Recall")

    for i, v in enumerate(precs):
        plt.text(i - width/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    for i, v in enumerate(recs):
        plt.text(i + width/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)

    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Metric Value")
    plt.title(f"Precision/Recall @ conf={bar_conf}, IoU={bar_iou}")
    plt.legend()
    savefig(out_path)

def plot_fnrate_vs_eps(series, out_path, bar_conf, bar_iou):
    plt.figure(figsize=(10, 5))
    for name, points in series.items():
        points = sorted(points, key=lambda t: t[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.plot(xs, ys, marker="o", label=name)
    plt.ylim(0, 1.0)
    plt.xlabel("ε")
    plt.ylabel("FN rate = FN/(TP+FN)")
    plt.title(f"False Negative Rate vs ε @ conf={bar_conf}, IoU={bar_iou}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig(out_path)

def plot_recall_vs_eps_conf(series, out_path, conf_list, fixed_iou):
    # series[fam][conf] -> list of (eps, recall)
    plt.figure(figsize=(12, 10))
    fam_order = ["fgsm_sur", "fgsm_gt", "pgd_gt"]
    titles = {"fgsm_sur": "FGSM (surrogate)", "fgsm_gt": "FGSM_GT (full loss)", "pgd_gt": "PGD_GT (full loss)"}

    for idx, fam in enumerate(fam_order, 1):
        plt.subplot(3, 1, idx)
        for conf in conf_list:
            pts = sorted(series.get(fam, {}).get(conf, []), key=lambda t: t[0])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            plt.plot(xs, ys, marker="o", label=f"conf={conf}")
        plt.ylim(0, 1.0)
        plt.title(titles[fam])
        plt.ylabel("Recall")
        plt.grid(True, alpha=0.3)
        plt.legend()

    plt.xlabel("ε")
    plt.suptitle(f"Recall vs ε at multiple confidence thresholds (IoU={fixed_iou})", y=1.02)
    savefig(out_path)

def plot_recall_vs_eps_iou(series, out_path, iou_list, fixed_conf):
    # series[fam][iou] -> list of (eps, recall)
    plt.figure(figsize=(12, 10))
    fam_order = ["fgsm_sur", "fgsm_gt", "pgd_gt"]
    titles = {"fgsm_sur": "FGSM (surrogate)", "fgsm_gt": "FGSM_GT (full loss)", "pgd_gt": "PGD_GT (full loss)"}

    for idx, fam in enumerate(fam_order, 1):
        plt.subplot(3, 1, idx)
        for iou in iou_list:
            pts = sorted(series.get(fam, {}).get(iou, []), key=lambda t: t[0])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            plt.plot(xs, ys, marker="o", label=f"IoU={iou}")
        plt.ylim(0, 1.0)
        plt.title(titles[fam])
        plt.ylabel("Recall")
        plt.grid(True, alpha=0.3)
        plt.legend()

    plt.xlabel("ε")
    plt.suptitle(f"Recall vs ε at multiple IoU thresholds (conf={fixed_conf})", y=1.02)
    savefig(out_path)

def plot_survival_vs_eps(series, out_path, fixed_conf):
    plt.figure(figsize=(10, 5))
    for name, points in series.items():
        points = sorted(points, key=lambda t: t[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.plot(xs, ys, marker="o", label=name)
    plt.xlabel("ε")
    plt.ylabel("Detection survival rate (n_pred_adv / n_pred_clean)")
    plt.title(f"Detection Survival vs ε @ conf={fixed_conf}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig(out_path)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", required=True, help="runs_adv directory containing subfolders with predictions.json")
    ap.add_argument("--data_yaml", required=True, help="clean data yaml used for YOLO val (to locate val images + labels)")
    ap.add_argument("--bar_conf", type=float, default=0.25)
    ap.add_argument("--bar_iou", type=float, default=0.5)
    ap.add_argument("--conf_list", type=float, nargs="+", default=[0.25, 0.5, 0.8])
    ap.add_argument("--iou_list", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    runs_dir = os.path.abspath(args.runs_dir)
    data_yaml = os.path.abspath(args.data_yaml)
    data_base = os.path.dirname(data_yaml)

    data = load_yaml(data_yaml)

    # YOLO data yaml may have: path + val + train...
    root_path = data.get("path", data_base)
    root_path = resolve_path(data_base, root_path)
    val_rel = data.get("val")
    if val_rel is None:
        raise RuntimeError("Your data yaml has no 'val:' entry.")
    val_images_dir = resolve_path(root_path, val_rel)

    # labels are usually sibling of images: replace ".../images" -> ".../labels"
    if val_images_dir.endswith("/images"):
        labels_dir = val_images_dir[:-len("/images")] + "/labels"
    else:
        # fallback: common structure dataset/labels
        labels_dir = resolve_path(root_path, "labels")

    if args.out_dir is None:
        out_dir = os.path.join(runs_dir, "_plots")
    else:
        out_dir = os.path.abspath(args.out_dir)
    ensure_dir(out_dir)

    gt_dict, n_imgs, missing_labels = build_gt_from_labels(val_images_dir, labels_dir)
    gt_boxes_total = sum(len(v) for v in gt_dict.values())

    print(f"[INFO] GT images: {n_imgs}  GT boxes: {gt_boxes_total}")
    print(f"[INFO] val_images_dir: {val_images_dir}")
    print(f"[INFO] labels_dir:     {labels_dir}")
    if missing_labels:
        print(f"[WARN] Missing label files for {missing_labels} images (treated as background).")
    print(f"[INFO] out_dir:        {out_dir}")

    # enumerate run subfolders that contain predictions.json
    run_subdirs = sorted([d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))])
    run_infos = []
    for r in run_subdirs:
        pred_json = os.path.join(runs_dir, r, "predictions.json")
        if os.path.exists(pred_json):
            fam, eps, label = parse_run_name(r)
            run_infos.append((r, fam, eps, label, pred_json))

    # Compute bar metrics at bar_conf / bar_iou
    bar_metrics = {}
    # Also build series:
    fnrate_series = {"FGSM (sur)": [], "FGSM_GT": [], "PGD_GT": []}
    survival_series = {"FGSM (sur)": [], "FGSM_GT": [], "PGD_GT": []}
    recall_vs_eps_conf = defaultdict(lambda: defaultdict(list))  # fam -> conf -> [(eps, recall)]
    recall_vs_eps_iou  = defaultdict(lambda: defaultdict(list))  # fam -> iou  -> [(eps, recall)]

    # Need clean n_pred for survival
    clean_pred_counts_by_conf = {}

    # First compute clean counts for survival at each conf in conf_list + bar_conf
    confs_needed = sorted(set(args.conf_list + [args.bar_conf]))
    for (rname, fam, eps, label, pred_json) in run_infos:
        if fam != "clean":
            continue
        for conf in confs_needed:
            preds = load_predictions_json(pred_json, conf_thr=conf)
            n_pred = sum(len(v) for v in preds.values())
            clean_pred_counts_by_conf[conf] = n_pred

    # Process each run
    for (rname, fam, eps, label, pred_json) in run_infos:
        preds_bar = load_predictions_json(pred_json, conf_thr=args.bar_conf)
        P, R, F1, TP, FP, FN, n_pred = compute_micro_metrics(gt_dict, preds_bar, iou_thr=args.bar_iou)

        if label is not None:
            bar_metrics[label] = {"P": P, "R": R}

        # for eps series (skip clean)
        if fam in ("fgsm_sur", "fgsm_gt", "pgd_gt") and eps is not None:
            fn_rate = FN / (TP + FN + 1e-9)
            if fam == "fgsm_sur":
                fnrate_series["FGSM (sur)"].append((eps, fn_rate))
            elif fam == "fgsm_gt":
                fnrate_series["FGSM_GT"].append((eps, fn_rate))
            elif fam == "pgd_gt":
                fnrate_series["PGD_GT"].append((eps, fn_rate))

            # survival uses n_pred ratio at bar_conf
            clean_npred = clean_pred_counts_by_conf.get(args.bar_conf, None)
            if clean_npred is not None and clean_npred > 0:
                surv = n_pred / clean_npred
                if fam == "fgsm_sur":
                    survival_series["FGSM (sur)"].append((eps, surv))
                elif fam == "fgsm_gt":
                    survival_series["FGSM_GT"].append((eps, surv))
                elif fam == "pgd_gt":
                    survival_series["PGD_GT"].append((eps, surv))

            # recall vs eps @ conf list (IoU fixed at bar_iou)
            for conf in args.conf_list:
                preds_c = load_predictions_json(pred_json, conf_thr=conf)
                _, Rc, _, TPc, FPc, FNc, _ = compute_micro_metrics(gt_dict, preds_c, iou_thr=args.bar_iou)
                recall_vs_eps_conf[fam][conf].append((eps, Rc))

            # recall vs eps @ iou list (conf fixed at bar_conf)
            preds_fixed = load_predictions_json(pred_json, conf_thr=args.bar_conf)
            for iou in args.iou_list:
                _, Ri, _, TPi, FPi, FNi, _ = compute_micro_metrics(gt_dict, preds_fixed, iou_thr=iou)
                recall_vs_eps_iou[fam][iou].append((eps, Ri))

    # Ensure bar order: Clean first, then grouped attacks
    def bar_sort_key(k):
        if k == "Clean":
            return (0, 0)
        if k.startswith("FGSM ε="):
            return (1, float(k.split("=")[-1]))
        if k.startswith("FGSM_GT ε="):
            return (2, float(k.split("=")[-1]))
        if k.startswith("PGD ε="):
            return (3, float(k.split("=")[-1]))
        return (9, 0)

    bar_metrics_sorted = dict(sorted(bar_metrics.items(), key=lambda kv: bar_sort_key(kv[0])))

    plot_bar(bar_metrics_sorted, os.path.join(out_dir, "metrics_bar.png"), args.bar_conf, args.bar_iou)
    plot_fnrate_vs_eps(fnrate_series, os.path.join(out_dir, "fnrate_vs_eps.png"), args.bar_conf, args.bar_iou)
    plot_recall_vs_eps_conf(recall_vs_eps_conf, os.path.join(out_dir, "recall_vs_eps_conf.png"), args.conf_list, args.bar_iou)
    plot_recall_vs_eps_iou(recall_vs_eps_iou, os.path.join(out_dir, "recall_vs_eps_iou.png"), args.iou_list, args.bar_conf)
    plot_survival_vs_eps(survival_series, os.path.join(out_dir, "survival_vs_eps.png"), args.bar_conf)


if __name__ == "__main__":
    main()

