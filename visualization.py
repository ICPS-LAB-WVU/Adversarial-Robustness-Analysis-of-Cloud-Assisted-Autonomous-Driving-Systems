# visualization.py
import cv2
import numpy as np
from itertools import islice
from class_labels import CLASS_NAMES


font  = cv2.FONT_HERSHEY_SIMPLEX
red   = (0, 0, 255)
green = (0, 255, 0)
white = (255, 255, 255)
orange = (0, 165, 255)  
blue = (255, 200, 0)
def viz_lane_follow(rs_bgr, boxes, seg, err_px, v, omega, mode="unknown",
                    alpha=0.35, max_boxes=20, font=cv2.FONT_HERSHEY_SIMPLEX,
                    show_mode=True, tracked_objs=None):

    vis = rs_bgr.copy()
    H, W = vis.shape[:2]
    # ── segmentation mask (white & yellow lanes) ───────────────────────────
    if seg is not None and seg.ndim == 3 and seg.shape[0] >= 4:
        white_m  = seg[2] > 0
        yellow_m = seg[3] > 0
        if white_m.any() or yellow_m.any():           # don’t darken if empty
            seg_rgb = np.zeros_like(vis)
            seg_rgb[white_m]  = (255, 255, 255)
            seg_rgb[yellow_m] = (0,   255, 255)
            vis = cv2.addWeighted(vis, 1 - alpha, seg_rgb, alpha, 0)

    # ── YOLO boxes ─────────────────────────────────────────────────────────
    for (x1, y1, x2, y2, cls, conf) in islice(boxes, max_boxes):
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(vis, p1, p2, (0, 255, 0), 2)
        label = CLASS_NAMES.get(int(cls), f"cls{int(cls)}")
        # place label near the bottom of the box to avoid overlapping top HUD
        label_y = min(p2[1] + 18, H - 10)
        cv2.putText(vis, f"{label}:{conf:.2f}", (p1[0], label_y),
                    font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # ── tracked bot overlays ──────────────────────────────────────────────
    if tracked_objs:
        for track in tracked_objs:
            try:
                tx, ty, tw, th = map(int, track.tlwh)
                tid = int(track.track_id)
            except Exception:
                continue

            p1 = (max(0, tx), max(0, ty))
            p2 = (min(W - 1, tx + tw), min(H - 1, ty + th))
            cv2.rectangle(vis, p1, p2, orange, 2)
            label_y = max(p1[1] - 8, 15)
            cv2.putText(
                vis,
                f"ID {tid}",
                (p1[0], label_y),
                font,
                0.5,
                orange,
                2,
                cv2.LINE_AA,
            )

    # ── text read-out ──────────────────────────────────────────────────────
    cv2.putText( vis,
            f"err = {err_px:+.1f} px",
            (10, 30),
            font,
            0.8,
            red,
            2,
            cv2.LINE_AA,)
   
    cv2.putText(
            vis,
            f"Velocity = {v:.2f} m/s   Omega = {omega:.2f} rad/s",
            (10, 60),
            font,
            0.8,
            white,
            2,
            cv2.LINE_AA,
        )
    # ── lane mode (top right corner) ───────────────────────────────────────
    # ── lane mode label (top right) ────────────────────────────────────────
   #
   #  return vis
    # ── lane mode label (top right, optional) ──────────────────────────────
    if show_mode:
        label_text = f"{mode}"
        mode_color = (
            green if mode == "two-way" else
            orange if mode == "one-way" else
            blue if mode == "memory" else
            (0, 0, 255)  # default: red
        )
        text_size, _ = cv2.getTextSize(label_text, font, 0.8, 2)
        text_w = text_size[0]
        cv2.putText(
            vis,
            label_text,
            (W - text_w - 10, 30),
            font,
            0.8,
            mode_color,
            2,
            cv2.LINE_AA,
        )
    
    return vis
