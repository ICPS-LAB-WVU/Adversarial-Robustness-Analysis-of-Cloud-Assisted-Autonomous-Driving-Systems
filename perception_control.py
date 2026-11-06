#!/usr/bin/env python3
"""
perception_control.py — lane, obstacle, and intersection handling
Integrated YOLO + UNet + ByteTrack + PID + Traffic light FSM
"""

import time, math, os, cv2, numpy as np, torch
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker
from segmentation.unet_model import UNet
from collections import deque
# ===================== ADVERSARIAL HOOK (PLUG-IN) =====================
try:
    import adversarial_attacks as adv
    _ADV_AVAILABLE = True
    print("[ADV] ICC adversary loaded")
except Exception as e:
    adv = None
    _ADV_AVAILABLE = False
    print(f"[ADV] No adversary loaded: {e}")

def attack_enabled():
    return _ADV_AVAILABLE and adv.attack_enabled()

def attack_frame(img_bgr, force_attack=False):
    if not attack_enabled():
        return img_bgr
    return adv.attack_frame(img_bgr, force_attack)

def get_attack_metadata():
    if not _ADV_AVAILABLE:
        return {}
    return adv.get_attack_metadata()
# =====================================================================


BOT_ARRIVALS   = {}   # track_id -> first time seen near its stop line
BOT_LAST_SEEN  = {}   # track_id -> last time seen anywhere
BOT_STALE_SEC  = 6.0  # drop a bot if not seen for this many seconds


# ========================== INTERSECTION CONSTANTS ==========================
STOP_WAIT_SEC = 3.0                   # Required full stop
STOP_REGION_FRAC = 0.15               # Bot near its stop sign
INTERSECTION_REGION_FRAC = 0.40       # Bot crossing
# ===========================================================================


# ====================== BYTE TRACKER SETUP =================================
byte_cfg = {
    'track_thresh': 0.5,
    'match_thresh': 0.8,
    'track_buffer': 5,
    'mot20': False
}
_tracker = BYTETracker(args=type("ByteArgs", (), byte_cfg)())
# ===========================================================================


# ========================== MODEL LOAD =====================================
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEG_WEIGHTS  = os.environ.get("SEG_WEIGHTS", "weight/segment_model_old.pth")
YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "weight/yolo.pt")

_unet = UNet(3,4).to(_device).eval()
_unet.load_state_dict(torch.load(SEG_WEIGHTS, map_location=_device))
_yolo = YOLO(YOLO_WEIGHTS).to(_device)

from class_labels import CLASS_NAMES

BASE_SPEED   = 0.3
P_GAIN       = 0.04
I_GAIN       = 0.0
D_GAIN       = 0.005

LANE_INNER_WIDTH_M = 0.2032
# ===========================================================================



# ======================== INTERSECTION HELPER ================================
# ======================== INTERSECTION HELPER ================================
def other_bots_in_intersection(tracked_objs, img_h, my_arrival_time):
    """
    FCFS intersection logic (for THIS bot):

    Returns True if:
      - some other bot arrived at its stop line BEFORE me and is still waiting, or
      - any bot is currently crossing the intersection center.

    Args:
        tracked_objs    : list of STrack (only Bot class, from ByteTrack)
        img_h           : image height
        my_arrival_time : time.time() when *this* bot entered INTERSECTION_WAIT
    """
    global BOT_ARRIVALS, BOT_LAST_SEEN

    if tracked_objs is None or len(tracked_objs) == 0:
        print("[INT] No tracked bots → intersection clear")
        return False

    now = time.time()
    wait_thr = STOP_REGION_FRAC * img_h      # height threshold = "at stop line"
    mid_low  = 0.35 * img_h                  # center region
    mid_high = 0.65 * img_h

    # --- Update arrival and last-seen times for all tracked bots ---
    for obj in tracked_objs:
        tid = int(obj.track_id)
        x, y, w, h = map(int, obj.tlwh)
        cy = y + h / 2.0

        BOT_LAST_SEEN[tid] = now

        # If bot is clearly near its own stop line, give/keep an arrival timestamp
        if h >= wait_thr:
            if tid not in BOT_ARRIVALS:
                BOT_ARRIVALS[tid] = now
                print(f"[INT] Bot {tid} arrival recorded at {BOT_ARRIVALS[tid]:.2f} (h={h} ≥ {wait_thr:.1f})")
        # If not in the stop-line region, we DON'T erase its arrival_time:
        # it might have just started moving; crossing is handled separately.

    # --- Prune stale bots that disappeared for too long ---
    stale_ids = [tid for tid, t_last in BOT_LAST_SEEN.items()
                 if now - t_last > BOT_STALE_SEC]
    for tid in stale_ids:
        print(f"[INT] Removing stale bot {tid}")
        BOT_LAST_SEEN.pop(tid, None)
        BOT_ARRIVALS.pop(tid, None)

    # --- 1) If ANY bot is crossing the intersection center → we must wait ---
    for obj in tracked_objs:
        tid = int(obj.track_id)
        x, y, w, h = map(int, obj.tlwh)
        cy = y + h / 2.0

        if mid_low <= cy <= mid_high:
            print(f"[INT] Bot {tid} currently crossing center (cy={cy:.1f}, mid=[{mid_low:.1f},{mid_high:.1f}])")
            return True

    # --- 2) FCFS: check if any bot arrived BEFORE us at its stop line ---
    if my_arrival_time is None:
        # Shouldn't happen, but be robust
        print("[INT][WARN] my_arrival_time is None; treating as latest arrival")
        my_arrival_time = now

    for tid, arr_t in BOT_ARRIVALS.items():
        if arr_t < my_arrival_time - 1e-3:   # small epsilon
            # Someone was in the stop region before we were
            print(f"[INT] Bot {tid} arrived earlier ({arr_t:.2f} < my {my_arrival_time:.2f}) → I yield")
            return True

    print("[INT] No earlier-arrived bots and nobody crossing → I have priority")
    return False
# ============================================================================



# ============================ INFERENCE =====================================
@torch.no_grad()
def infer(frame_bgr):
    """
    Returns:
      err_px, lane_w_px, boxes[N,6], seg(4x640x640),
      resized_img, fallback_one_way, mode, tracked_objs
    """
    img = cv2.resize(frame_bgr, (640, 640))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t   = torch.from_numpy(rgb.transpose(2,0,1)[None]).float().to(_device)

    # UNet segmentation
    try:
        seg = _unet(t)[0].cpu().numpy().astype(np.uint8)
        _, white_m, yellow_m = seg[1], seg[2], seg[3]
    except:
        white_m = np.zeros((640,640),np.uint8)
        yellow_m = np.zeros((640,640),np.uint8)
        seg = np.zeros((4,640,640),np.uint8)

    # YOLO detection
    dets = _yolo.predict(img, conf=0.25, imgsz=320, verbose=False)[0]

    boxes = []
    for b in dets.boxes:
        x1,y1,x2,y2 = map(int, b.xyxy[0])
        boxes.append([x1,y1,x2,y2,int(b.cls[0]),float(b.conf[0])])
    boxes = np.array(boxes, dtype=np.float32) if len(boxes)>0 else np.zeros((0,6),np.float32)

    # BYTE TRACK input (★ FIXED: track ONLY Bot class ★)
    try:
        img_h, img_w = img.shape[:2]

        # Extract raw YOLO detections
        if len(dets.boxes):
            cls_np  = dets.boxes.cls.cpu().numpy()
            bot_mask = (cls_np == 0)

            # Debug: show raw detection counts
            #print(f"[BT] YOLO raw counts: Bots={np.sum(bot_mask)}, Total={len(cls_np)}")

            # Track only Bots
            if np.any(bot_mask):
                xyxy = dets.boxes.xyxy.cpu().numpy()[bot_mask]
                conf = dets.boxes.conf.cpu().numpy()[bot_mask].reshape(-1, 1)
                det_arr = np.concatenate([xyxy, conf], axis=1).astype(np.float32)
                det_tensor = torch.from_numpy(det_arr).float().cpu()
            else:
                det_tensor = torch.zeros((0, 5), dtype=torch.float32)
        else:
            det_tensor = torch.zeros((0, 5), dtype=torch.float32)

        # Update tracker
        tracked_objs = _tracker.update(det_tensor, (img_h, img_w), (img_h, img_w))

        # Debug: print track info
        #print(f"[BT] Active tracks: {len(tracked_objs)}")
        for t in tracked_objs:
            print(f"[BT] Track ID={t.track_id}, tlwh={t.tlwh}")

    except Exception as e:
        print(f"[BT] ERROR: {e}")
        tracked_objs = []


    # ----------------------- Lane estimation -----------------------
    H, W = white_m.shape
    cx_img = W/2
    roi_h = int(0.2*H)
    LANE_OFFSET_PX = 200

    yellow_roi = yellow_m[H-roi_h:H,:]
    y_pts = cv2.findNonZero(yellow_roi)
    yx = float(np.median(y_pts[:,0,0])) if y_pts is not None else np.nan

    white_roi = white_m[int(0.7*H):H,:]
    w_pts = cv2.findNonZero(white_roi)
    wx = float(np.median(w_pts[:,0,0])) if w_pts is not None else np.nan

    # lane width memory
    if not hasattr(infer, "last_lane_w"):
        infer.last_lane_w = 400
    if not hasattr(infer, "last_err"):
        infer.last_err = 0
    if not hasattr(infer, "last_lane_cx"):
        infer.last_lane_cx = cx_img

    if not np.isnan(yx) and not np.isnan(wx):
        lane_w_px = abs(wx - yx)
        infer.last_lane_w = 0.8*infer.last_lane_w + 0.2*lane_w_px
    else:
        lane_w_px = infer.last_lane_w

    # lane center logic
    if not np.isnan(yx) and not np.isnan(wx):
        lane_cx = 0.5*(yx+wx)
        mode = "two-way"
    elif np.isnan(yx) and not np.isnan(wx):
        lane_cx = max(0, wx - LANE_OFFSET_PX)
        mode = "one-way-left"
    elif np.isnan(wx) and not np.isnan(yx):
        lane_cx = min(W, yx + LANE_OFFSET_PX)
        mode = "one-way-right"
    else:
        # both missing
        if abs(infer.last_err) < 5:
            lane_cx = cx_img
            mode = "memory"
        else:
            alpha = np.clip(0.05 + 0.002*abs(infer.last_err), 0.1, 0.6)
            lane_cx = (1-alpha)*infer.last_lane_cx + alpha*cx_img
            mode = "memory"

    err_px = lane_cx - cx_img
    infer.last_err = err_px
    infer.last_lane_cx = lane_cx

    return (
        float(err_px),
        float(lane_w_px),
        boxes,
        seg,
        img,
        False,
        mode,
        tracked_objs
    )
# ============================================================================



# ================================ CONTROLLER =================================
class LaneFollower:
    CLOSE_H_FRAC = 0.06
    STOP_H_FRAC  = 0.15
    TL_COOLDOWN  = 5.0

    GREEN_MIN_PCT = 0.03
    RED_MAX_PCT   = 0.08

    MIN_SPEED = 0.1

    def __init__(self):
        self.state = "DRIVING"
        self.state_start = time.time()
        self.last_stop_time = 0
        self.last_left_tl = 0

        self.last_err = 0
        self.last_time = time.time()

        # FCFS: when did *we* arrive at the intersection stop line?
        self.my_arrival_time = None

    # --------------------------- TRAFFIC LIGHT HELPER ------------------------
    def _find_tl_box(self, boxes):
        for x1,y1,x2,y2,cls,conf in boxes:
            if int(cls)==3 and conf>0.5:
                return (x1,y1,x2,y2)
        return None

    # ------------------------------- FSM UPDATE ------------------------------
    def update(self, err_px, lane_w_px, boxes, img_shape, last_cam_bgr=None,
               fallback=False, tracked_objs=None):

        now = time.time()
        h_img, w_img = img_shape
        v = BASE_SPEED
        omega = 0

        # ---------------- INTERSECTION_WAIT STATE ----------------
        if self.state == "INTERSECTION_WAIT":
            waited = now - self.state_start
            print(f"[FSM] INTERSECTION_WAIT: waited={waited:.2f}s, my_arrival_time={self.my_arrival_time}")

            # Ensure we have an arrival time
            if self.my_arrival_time is None:
                self.my_arrival_time = self.state_start

            # mandatory full stop
            if waited < STOP_WAIT_SEC:
                return 0.0, 0.0

            # any other bot with earlier arrival or currently crossing?
            conflict = other_bots_in_intersection(tracked_objs, h_img, self.my_arrival_time)
            if conflict:
                return 0.0, 0.0

            # intersection clear & I am earliest → go
            print("[FSM] INTERSECTION → DRIVING (FCFS: I go now)")
            self.state = "DRIVING"
            self.last_stop_time = now
            self.my_arrival_time = None
            return BASE_SPEED, 0.0

        # ---------------- TRAFFIC LIGHT WAIT STATE ----------------
        if self.state == "TL_RED_WAIT":
            tl_box = self._find_tl_box(boxes)
            if tl_box is None:
                print("[TL] No TL box in WAIT state")
                return 0,0

            # Ensure integer indices
            x1,y1,x2,y2 = map(int, tl_box)

            # Clamp to image bounds
            x1 = max(0, min(x1, w_img-1))
            x2 = max(0, min(x2, w_img-1))
            y1 = max(0, min(y1, h_img-1))
            y2 = max(0, min(y2, h_img-1))

            if x2 <= x1 or y2 <= y1:
                print("[TL] Invalid crop region")
                return 0, 0

            crop = last_cam_bgr[y1:y2, x1:x2]

            if crop.size == 0:
                print("[TL] Empty crop; skipping")
                return 0,0

            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            g = cv2.inRange(hsv, np.array([40,70,50]), np.array([90,255,255]))
            r1= cv2.inRange(hsv,np.array([0,70,50]), np.array([10,255,255]))
            r2= cv2.inRange(hsv,np.array([160,70,50]), np.array([180,255,255]))
            red = cv2.bitwise_or(r1,r2)

            g_pct = cv2.countNonZero(g) / (crop.size/3.0)
            r_pct = cv2.countNonZero(red) / (crop.size/3.0)

            #print(f"[TL] g_pct={g_pct:.3f} r_pct={r_pct:.3f}")

            # Still red/yellow
            if g_pct < self.GREEN_MIN_PCT or r_pct > self.RED_MAX_PCT:
               # print("[TL] still red/yellow → waiting")
                return 0,0

            print("[FSM] Green → DRIVING")
            self.state = "DRIVING"
            self.last_left_tl = now
            self.last_err = 0
            self.last_time = now
            return BASE_SPEED, 0.0


        # ====================== DRIVING STATE ======================
        if not math.isnan(err_px):
            dt = max(now - self.last_time, 1e-3)
            p = err_px * P_GAIN
            d = (err_px - self.last_err)/dt * D_GAIN
            i = err_px * I_GAIN * dt
            omega = -(p+i+d)

            self.last_err = err_px
            self.last_time = now

        v = max(self.MIN_SPEED, min(v, BASE_SPEED))

        # ----- STOP SIGN DETECTION → ENTER INTERSECTION_WAIT -----
        if now > self.last_stop_time + 5.0:
            for x1,y1,x2,y2,cls,conf in boxes:
                if int(cls)==2 and conf>0.6:
                    if (y2-y1) > 0.15*h_img:
                        print("[FSM] DRIVING → INTERSECTION_WAIT")
                        self.state = "INTERSECTION_WAIT"
                        self.state_start = now
                        self.my_arrival_time = now   # record my arrival
                        return 0,0

        # ------------- Traffic Light detection -----------------
        if now - self.last_left_tl > self.TL_COOLDOWN:
            for x1,y1,x2,y2,cls,conf in boxes:
                if int(cls)==3 and conf>0.6:
                    tl_h = y2 - y1

                    # Two-stage geometry (calibrated for Duckietown @ 640px)
                    TL_SLOW_FRAC =  self.CLOSE_H_FRAC 
                    TL_STOP_FRAC = self.STOP_H_FRAC 

                    slow_px = TL_SLOW_FRAC * h_img
                    stop_px = TL_STOP_FRAC * h_img

                    print(f"[TL] tl_h={tl_h:.1f} slow={slow_px:.1f} stop={stop_px:.1f}")

                    # ----- HARD STOP ZONE (near stop line) -----
                    if tl_h >= stop_px:
                        print("[FSM] DRIVING → TL_RED_WAIT")
                        self.state = "TL_RED_WAIT"
                        self.state_start = now
                        return 0,0

                    # ----- SLOW DOWN ZONE (approach) -----
                    elif tl_h >= slow_px:
                        ratio = (tl_h - slow_px) / max(stop_px - slow_px, 1.0)
                        v = max(self.MIN_SPEED, BASE_SPEED * (1.0 - ratio))
                        print(f"[TL] slowing down, v={v:.3f}")
                        break


        # ------------- Bot too close in same lane? -------------
        if not math.isnan(lane_w_px):
            half_lane = lane_w_px/2.5
            for x1,y1,x2,y2,cls,conf in boxes:
                if int(cls)==0 and conf>0.5:
                    obs_cx = (x1+x2)/2.0
                    obs_err = obs_cx - (w_img/2.0)
                    if abs(obs_err - err_px) < half_lane:
                        if (y2-y1) > 0.35*h_img:
                            print("[FSM] Bot front → STOP")
                            return 0,0

        return v, omega
# Singleton controller
_FOLLOWER = LaneFollower()

