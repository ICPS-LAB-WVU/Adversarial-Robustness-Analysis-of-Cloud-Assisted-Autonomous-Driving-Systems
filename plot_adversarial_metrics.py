#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_adversarial_metrics.py

Reads:
  - ground_truth.csv
  - predictions.csv

Computes detection metrics via:
  - confidence filtering at tau_conf
  - class-aware, one-to-one greedy IoU matching per image

Outputs:
  - metrics_bar.png
  - recall_vs_eps_conf.png   (ONE plot: FGSM_GT + PGD_GT, multiple conf thresholds)
  - recall_vs_eps_iou.png    (ONE plot: FGSM_GT + PGD_GT, multiple IoU thresholds)
  - fnrate_vs_eps.png
  - survival_vs_eps.png
  - metrics_summary.csv  (table for paper)

Usage:
  python plot_adversarial_metrics.py --root . \
    --bar-conf 0.25 --bar-iou 0.5 \
    --conf-list 0.25 0.5 0.8 \
    --iou-list 0.25 0.5 0.75 \
    --include-f1
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# IoU + matching
# -------------------------
def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return (inter / denom) if denom > 0 else 0.0

def match_by_confidence(gt_boxes, pred_boxes, iou_thr):
    """
    YOLO/COCO-style one-to-one matching:
    - predictions are assumed sorted by descending confidence already
    - for each prediction, match the best unmatched GT with IoU >= iou_thr
    Returns: TP, FP, FN
    """
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)

    gt_used = [False] * len(gt_boxes)
    tp = 0
    fp = 0

    for p in pred_boxes:
        best_iou = 0.0
        best_gi = -1
        for gi, g in enumerate(gt_boxes):
            if gt_used[gi]:
                continue
            i = iou_xyxy(g, p)
            if i > best_iou:
                best_iou = i
                best_gi = gi
        if best_iou >= iou_thr and best_gi >= 0:
            tp += 1
            gt_used[best_gi] = True
        else:
            fp += 1

    fn = sum(1 for u in gt_used if not u)
    return tp, fp, fn


def compute_metrics_micro(gt_df, pred_df, conf_thr, iou_thr, eval_images=None):
    """
    Micro-averaged across all classes:
      - filter preds by conf_thr
      - per image + per class: match predictions sorted by confidence (YOLO-like)
    eval_images:
      - None => use all images in gt_df union pred_df
      - list/set => restrict evaluation to these images only
    """
    pred_f = pred_df[pred_df["conf"] >= conf_thr].copy()

    if eval_images is not None:
        eval_images = set(eval_images)
        gt_df = gt_df[gt_df["image"].isin(eval_images)]
        pred_f = pred_f[pred_f["image"].isin(eval_images)]

    gt_grp = gt_df.groupby(["image", "cls"])
    pred_grp = pred_f.groupby(["image", "cls"])
    all_keys = set(gt_grp.groups.keys()) | set(pred_grp.groups.keys())

    TP = FP = FN = 0
    n_gt_total = 0
    n_pred_total = 0

    for key in all_keys:
        if key in gt_grp.groups:
            g = gt_grp.get_group(key)
            gt_boxes = g[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32).tolist()
        else:
            gt_boxes = []

        if key in pred_grp.groups:
            p = pred_grp.get_group(key).sort_values("conf", ascending=False)
            pred_boxes = p[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32).tolist()
        else:
            pred_boxes = []

        n_gt_total += len(gt_boxes)
        n_pred_total += len(pred_boxes)

        tp, fp, fn = match_by_confidence(gt_boxes, pred_boxes, iou_thr)
        TP += tp
        FP += fp
        FN += fn

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fn_rate = (FN / (TP + FN)) if (TP + FN) > 0 else 0.0

    return {
        "TP": TP, "FP": FP, "FN": FN,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fn_rate": fn_rate,
        "n_gt": n_gt_total,
        "n_pred": n_pred_total,
    }



def normalize_attack_name(a: str) -> str:
    # keep consistent naming across older logs
    a = str(a).strip().lower()
    aliases = {
        "fgsm": "fgsm_sur",
        "fgsm_no_gt": "fgsm_sur",
        "pgd": "pgd_gt",
        "pgd_rand": "pgd_gt",
    }
    return aliases.get(a, a)


def label_for(atk, eps):
    if atk == "clean":
        return "Clean"
    name = {"fgsm_sur": "FGSM_sur", "fgsm_gt": "FGSM", "pgd_gt": "PGD"}.get(atk, atk.upper())
    return f"{name} ε={eps:.3f}"


def build_order(attacks_present, eps_list):
    order = [("clean", 0.0)]
    for e in eps_list:
        for atk in ["fgsm_sur", "fgsm_gt", "pgd_gt"]:
            if atk in attacks_present:
                order.append((atk, e))
    return order


# -------------------------
# Plot helpers
# -------------------------
def plot_bar(rows, out_path: Path, show_f1=False, title=None):
    labels = [r["label"] for r in rows]
    prec = [r["precision"] for r in rows]
    rec = [r["recall"] for r in rows]
    f1 = [r["f1"] for r in rows]

    x = np.arange(len(labels))
    width = 0.35 if not show_f1 else 0.25

    plt.figure(figsize=(14, 7))
    plt.bar(x - width / 2, prec, width, label="Precision")
    plt.bar(x + width / 2, rec, width, label="Recall")

    if show_f1:
        plt.bar(x + 1.5 * width, f1, width, label="F1")

    def annotate(vals, xoffs):
        for xi, v in zip(x, vals):
            plt.text(xi + xoffs, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=10)

    annotate(prec, -width / 2)
    annotate(rec, +width / 2)
    if show_f1:
        annotate(f1, +1.5 * width)

    plt.ylim(0, 1.05)
    plt.ylabel("Metric Value")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.legend()
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_single_metric_vs_eps(eps_list, y_by_attack, ylabel, out_path: Path, title: str):
    plt.figure(figsize=(10, 6))
    for atk, ys in y_by_attack.items():
        name = {"fgsm_gt": "FGSM", "pgd_gt": "PGD"}.get(atk, atk)
        plt.plot(eps_list, ys, marker="o", label=name)
    plt.ylim(0, 1.0)
    plt.xlabel("ε")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_recall_oneplot(eps_list, curves, out_path: Path, title: str, xlabel="ε", ylabel="Recall"):
    """
    One figure, multiple lines.

    curves: list of dicts with keys:
      - y: list/array of y values over eps_list
      - label: legend string
      - ls: line style (e.g., '-', '--', ':')
      - marker: marker (e.g., 'o', 's')
    """
    plt.figure(figsize=(10, 6))
    for c in curves:
        plt.plot(
            eps_list,
            c["y"],
            marker=c.get("marker", "o"),
            linestyle=c.get("ls", "-"),
            label=c["label"],
        )
    plt.ylim(0, 1.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -------------------------
# Main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("."), help="Folder containing CSV logs")
    p.add_argument("--bar-conf", type=float, default=0.25)
    p.add_argument("--bar-iou", type=float, default=0.5)
    p.add_argument("--conf-list", type=float, nargs="+", default=[0.25, 0.5, 0.8])
    p.add_argument("--iou-list", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    p.add_argument("--include-f1", action="store_true")
    p.add_argument("--bar-mode", choices=["fixed", "best_f1"], default="fixed")
    p.add_argument("--sweep-steps", type=int, default=201)
    

    return p.parse_args()

def main():
    args = parse_args()
    root = args.root

    gt_path = root / "offline_attacks" / "ground_truth.csv"
    pred_path = root / "offline_attacks" / "predictions.csv"
    assert gt_path.exists(), f"Missing {gt_path}"
    assert pred_path.exists(), f"Missing {pred_path}"

    gt = pd.read_csv(gt_path)
    pr = pd.read_csv(pred_path)

    # Clean dtypes
    for col in ["x1", "y1", "x2", "y2"]:
        gt[col] = gt[col].astype(float)
        pr[col] = pr[col].astype(float)
    gt["cls"] = gt["cls"].astype(int)
    pr["cls"] = pr["cls"].astype(int)
    pr["conf"] = pr["conf"].astype(float)
    pr["eps"] = pr["eps"].astype(float)
    pr["attack"] = pr["attack"].apply(normalize_attack_name)

    # Stats
    print("GT images:", gt["image"].nunique(), "GT boxes:", len(gt))
    print("Pred clean images:", pr[pr.attack == "clean"]["image"].nunique(),
          "Pred clean boxes:", len(pr[pr.attack == "clean"]))
    print("GT max coord:", gt[["x2", "y2"]].max().to_dict())
    print("Pred max coord:", pr[pr.attack == "clean"][["x2", "y2"]].max().to_dict())

    attacks_present = sorted(set(pr["attack"].unique()))
    eps_list = sorted(set(pr.loc[pr["attack"] != "clean", "eps"].unique().tolist()))
    order = build_order(set(attacks_present), eps_list)

    # -------------------------
    # Build a consistent eval image set
    # -------------------------
    gt_images = set(gt["image"].unique().tolist())
    eval_images = set(gt_images)

    # Intersect with each prediction subset's image set (so comparisons are fair)
    for atk, eps in order:
        if atk == "clean":
            pred_sub = pr[pr["attack"] == "clean"]
        else:
            pred_sub = pr[(pr["attack"] == atk) & (np.isclose(pr["eps"], eps))]

        pred_imgs = set(pred_sub["image"].unique().tolist())
        eval_images &= pred_imgs if len(pred_imgs) > 0 else eval_images

    eval_images = sorted(list(eval_images))
    print(f"[INFO] Using eval_images intersection size = {len(eval_images)}")
    if len(eval_images) == 0:
        raise RuntimeError("eval_images is empty. Your GT/pred image keys do not overlap.")

    # -------------------------
    # Helpers
    # -------------------------
    def best_f1_metrics(gt_df, pred_df, iou_thr, steps=201):
        best = None
        for conf in np.linspace(0.0, 1.0, steps):
            m = compute_metrics_micro(gt_df, pred_df, conf_thr=float(conf), iou_thr=iou_thr, eval_images=eval_images)
            if best is None or m["f1"] > best["f1"]:
                best = {"conf_thr": float(conf), **m}
        return best

    # -------------------------
    # 1) Bar chart + summary table
    # -------------------------
    metrics_rows = []
    summary_rows = []

    print("\n=== Metrics (micro-avg) ===")
    print(f"    bar_mode={args.bar_mode}  bar_conf={args.bar_conf:.3f}  bar_iou={args.bar_iou:.3f}")

    for atk, eps in order:
        if atk == "clean":
            pred_sub = pr[pr["attack"] == "clean"]
        else:
            pred_sub = pr[(pr["attack"] == atk) & (np.isclose(pr["eps"], eps))]

        lab = label_for(atk, eps)

        if args.bar_mode == "best_f1":
            m = best_f1_metrics(gt, pred_sub, iou_thr=args.bar_iou, steps=args.sweep_steps)
            used_conf = m["conf_thr"]
        else:
            m = compute_metrics_micro(gt, pred_sub, conf_thr=args.bar_conf, iou_thr=args.bar_iou, eval_images=eval_images)
            used_conf = args.bar_conf

        metrics_rows.append({"label": lab, **m})

        summary_rows.append({
            "attack": atk,
            "eps": eps,
            "label": lab,
            "conf_thr": used_conf,
            "iou_thr": args.bar_iou,
            "TP": m["TP"], "FP": m["FP"], "FN": m["FN"],
            "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
            "fn_rate": m["fn_rate"],
            "n_gt": m["n_gt"], "n_pred": m["n_pred"],
            "eval_images": len(eval_images),
        })

        print("{:<18s}  conf={:.3f}  P={:.3f} R={:.3f} F1={:.3f}  TP={} FP={} FN={}  n_pred={}".format(
            lab, used_conf, m["precision"], m["recall"], m["f1"], m["TP"], m["FP"], m["FN"], m["n_pred"]
        ))

    # Survival rate relative to clean
    clean_n_pred = next((r["n_pred"] for r in summary_rows if r["attack"] == "clean"), None)
    for r in summary_rows:
        r["survival_rate"] = (float(r["n_pred"]) / float(clean_n_pred)) if clean_n_pred else np.nan

    summary_df = pd.DataFrame(summary_rows)
    summary_out = root / "offline_attacks" / "metrics_summary.csv"
    summary_df.to_csv(summary_out, index=False)
    print(f"\nSaved: {summary_out}")

    bar_out = root / "offline_attacks" / "metrics_bar.png"
    plot_bar(
        metrics_rows,
        out_path=bar_out,
        show_f1=args.include_f1,
        title=f"Precision/Recall @ mode={args.bar_mode}, conf={args.bar_conf:g}, IoU={args.bar_iou:g}"
    )
    print(f"Saved: {bar_out}")

    # -------------------------
    # 2) Recall vs eps for multiple conf thresholds (fixed IoU=bar_iou)
    # -------------------------
    fixed_iou = args.bar_iou

    style_by_attack = {
        "fgsm_gt": {"ls": "-",  "marker": "o", "name": "FGSM_GT"},
        "pgd_gt":  {"ls": "--", "marker": "s", "name": "PGD_GT"},
    }

    curves = []
    for atk in ["fgsm_gt", "pgd_gt"]:
        if atk not in attacks_present:
            continue
        st = style_by_attack[atk]

        for conf in args.conf_list:
            ys = []
            for eps in eps_list:
                pred_sub = pr[(pr["attack"] == atk) & (np.isclose(pr["eps"], eps))]
                m = compute_metrics_micro(gt, pred_sub, conf_thr=conf, iou_thr=fixed_iou, eval_images=eval_images)
                ys.append(m["recall"])

            curves.append({
                "y": ys,
                "label": f'{st["name"]}  conf={conf:g}',
                "ls": st["ls"],
                "marker": st["marker"],
            })

    conf_out = root / "offline_attacks" / "recall_vs_eps_conf.png"
    plot_recall_oneplot(
        eps_list=eps_list,
        curves=curves,
        out_path=conf_out,
        title=f"Recall vs ε @ multiple confidence thresholds (IoU={fixed_iou:g})",
    )
    print(f"Saved: {conf_out}")

    # -------------------------
    # 3) Recall vs eps for multiple IoU thresholds (fixed conf=bar_conf)
    # -------------------------
    fixed_conf = args.bar_conf

    curves = []
    for atk in ["fgsm_gt", "pgd_gt"]:
        if atk not in attacks_present:
            continue
        st = style_by_attack[atk]

        for iou in args.iou_list:
            ys = []
            for eps in eps_list:
                pred_sub = pr[(pr["attack"] == atk) & (np.isclose(pr["eps"], eps))]
                m = compute_metrics_micro(gt, pred_sub, conf_thr=fixed_conf, iou_thr=iou, eval_images=eval_images)
                ys.append(m["recall"])

            curves.append({
                "y": ys,
                "label": f'{st["name"]}  IoU={iou:g}',
                "ls": st["ls"],
                "marker": st["marker"],
            })

    iou_out = root / "offline_attacks" / "recall_vs_eps_iou.png"
    plot_recall_oneplot(
        eps_list=eps_list,
        curves=curves,
        out_path=iou_out,
        title=f"Recall vs ε @ multiple IoU thresholds (conf={fixed_conf:g})",
    )
    print(f"Saved: {iou_out}")

    # -------------------------
    # 4) FN-rate vs eps (fixed conf/iou)
    # -------------------------
    fnrate_by_attack = {}
    for atk in ["fgsm_sur", "fgsm_gt", "pgd_gt"]:
        if atk not in attacks_present:
            continue
        ys = []
        for eps in eps_list:
            pred_sub = pr[(pr["attack"] == atk) & (np.isclose(pr["eps"], eps))]
            m = compute_metrics_micro(gt, pred_sub, conf_thr=args.bar_conf, iou_thr=args.bar_iou, eval_images=eval_images)
            ys.append(m["fn_rate"])
        fnrate_by_attack[atk] = ys

    fn_out = root / "offline_attacks" / "fnrate_vs_eps.png"
    plot_single_metric_vs_eps(
        eps_list=eps_list,
        y_by_attack=fnrate_by_attack,
        ylabel="FN rate = FN/(TP+FN)",
        out_path=fn_out,
        title=f"False Negative Rate vs ε @ conf={args.bar_conf:g}, IoU={args.bar_iou:g}"
    )
    print(f"Saved: {fn_out}")

    # -------------------------
    # 5) Survival rate vs eps (fixed conf/iou)
    # -------------------------
    clean_pred_sub = pr[pr["attack"] == "clean"]
    clean_m = compute_metrics_micro(gt, clean_pred_sub, conf_thr=args.bar_conf, iou_thr=args.bar_iou, eval_images=eval_images)
    clean_np = clean_m["n_pred"] if clean_m["n_pred"] > 0 else None

    survival_by_attack = {}
    for atk in ["fgsm_sur", "fgsm_gt", "pgd_gt"]:
        if atk not in attacks_present:
            continue
        ys = []
        for eps in eps_list:
            pred_sub = pr[(pr["attack"] == atk) & (np.isclose(pr["eps"], eps))]
            m = compute_metrics_micro(gt, pred_sub, conf_thr=args.bar_conf, iou_thr=args.bar_iou, eval_images=eval_images)
            ys.append(float(m["n_pred"]) / float(clean_np) if clean_np else np.nan)
        survival_by_attack[atk] = ys

    surv_out = root / "offline_attacks" / "survival_vs_eps.png"
    plot_single_metric_vs_eps(
        eps_list=eps_list,
        y_by_attack=survival_by_attack,
        ylabel="Detection survival rate (n_pred_adv / n_pred_clean)",
        out_path=surv_out,
        title=f"Detection Survival vs ε @ conf={args.bar_conf:g}"
    )
    print(f"Saved: {surv_out}")

    print("\nInterpretation guide:")
    print("- If Recall drops more at higher conf thresholds -> confidence suppression dominates (more FN due to conf < τ).")
    print("- If Recall drops more at higher IoU thresholds -> localization damage dominates (localization damage dominates).")
    print("- FN-rate plot directly supports the 'FN increases under attack' claim.")
    print("- Survival plot shows whether the detector collapses (few boxes) vs hallucination (many boxes).")
    print("\nDone.")

if __name__ == "__main__":
    main()


