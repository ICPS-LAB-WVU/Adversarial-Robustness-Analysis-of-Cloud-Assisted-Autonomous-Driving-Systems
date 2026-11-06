#!/usr/bin/env python3
"""
gpu_server.py — vehicle-cloud TCP server and controller

Captures full telemetry for network-delay, bandwidth-limit, and adversarial perturbation experiments.
Now includes persistent connections, fallback control logic, and timeout recovery.
"""
import socket
import csv
import threading
import os
import math
import time
import json
import cv2
import numpy as np
from collections import deque
import copy 
import perception_control as lf
from visualization import viz_lane_follow


# -------- ICC adversary import (NEW) --------
try:
    import adversarial_attacks as adv
    print("[ADV] ICC adversary loaded", flush=True)
except Exception as e:
    print("[ADV] No ICC adversary loaded:", e, flush=True)
    adv = None

# --------------------------------------------

# ---------------- Attack Parameters ----------------
ATTACK_MODE = os.environ.get("ATTACK_MODE", "none").lower()
ADV_EPS = float(os.environ.get("ADV_EPS", "0.03"))
ADV_PGD_ALPHA = float(os.environ.get("ADV_PGD_ALPHA", "0.007"))
ADV_PGD_ITERS = int(os.environ.get("ADV_PGD_ITERS", "10"))

print(f"[server][INIT] ATTACK_MODE={ATTACK_MODE}, EPS={ADV_EPS}, "
      f"ALPHA={ADV_PGD_ALPHA}, ITERS={ADV_PGD_ITERS}", flush=True)

# ---------------- Configuration ----------------
PORT = int(os.environ.get("GPU_SERVER_PORT", "5001"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")
LOG_ENABLED = os.environ.get("LOG_ENABLED", "1") != "0"
LOG_PATH = os.environ.get("LANE_LOG_PATH", os.path.join(OUTPUT_DIR, "lane_logs_robust.csv"))
NET_PROFILE = os.environ.get("NET_PROFILE", "none")
TC_CMD = os.environ.get("TC_CMD", "")
EXPERIMENT_ID = os.environ.get("EXPERIMENT_ID", "exp_unknown")
RUN_ID = os.environ.get("RUN_ID", "run_0")
TRIAL_ID = os.environ.get("TRIAL_ID", "trial_0")
SERVER_TAG = os.environ.get("SERVER_TAG", "")
SERVER_HOSTNAME = socket.gethostname()

SHOW_GUI = os.environ.get("SHOW_GUI", "1") == "1"
SAVE_VIDEO = os.environ.get("SAVE_VIDEO", "1") == "1"
FRAME_RATE = float(os.environ.get("FRAME_RATE", "15.0"))

frame_qs = {}              # maps vehicle -> deque of latest frames
followers = {}             # maps vehicle -> dedicated LaneFollower()
active_conns = {}          # maps vehicle -> current connection socket
frame_qs_lock = threading.Lock()

# ---------------- Logging Setup ----------------
try:
    BASE_COLS = list(lf.LOG_COLUMNS)
except Exception:
    BASE_COLS = []

SERVER_FIELDS = [
    "frame_id", "t_gen", "t_server",
    "network_delay", "network_jitter", "reply_rtt", "packet_loss_flag",
    "processing_delay", "control_delay", "aoi_server",
    "frame_interval", "total_server_latency",
    "v_cmd", "omega_cmd",
    "yolo_stop_conf", "yolo_stop_hit",
    "yolo_light_conf", "yolo_light_hit",
    "yolo_bot_conf", "yolo_bot_hit",
    "n_boxes_before", "n_boxes_after", "attack_disrupted",
    "attack_mode", "adv_mean_diff", "adv_max_diff", "adv_grad_mean", "adv_grad_max",
    "attack_used_surrogate", "adv_compute_time",
    "payload_bytes", "bandwidth_utilization",
    "server_proc_start", "server_proc_end", "server_proc_wall",
    "server_infer_start", "server_infer_end", "server_infer_wall",
    "reply_send_time", "reply_success", "infer_error",
    "experiment_id", "run_id", "trial_id", "net_profile", "tc_cmd",
    "server_hostname", "server_tag"
]

LANE_FIELDS = ["t_s", "err_px", "err_m", "velocity", "omega"]
_FIELDNAMES = SERVER_FIELDS + [f for f in LANE_FIELDS if f not in SERVER_FIELDS]

_log_lock = threading.Lock()
_log_fh = None
_log_wr = None

def _open_logger_once():
    global _log_fh, _log_wr
    if not LOG_ENABLED:
        return
    if _log_wr is not None:
        return
    os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
    new_file = (not os.path.exists(LOG_PATH)) or os.stat(LOG_PATH).st_size == 0
    _log_fh = open(LOG_PATH, "a", newline="")
    _log_wr = csv.DictWriter(_log_fh, fieldnames=_FIELDNAMES)
    if new_file:
        _log_wr.writeheader()
        _log_fh.flush()
    print("[logger] writing to", os.path.abspath(LOG_PATH), flush=True)

def log_row_from_lf(v, omega, extra_row=None):
    if not LOG_ENABLED:
        return True
    _open_logger_once()
    if _log_wr is None:
        return False

    try:
        row = lf.pop_pending_row() or {}
    except Exception:
        row = {}

    if not row and not extra_row:
        return True

    row["v_cmd"] = float(v)
    row["omega_cmd"] = float(omega)
    if extra_row:
        row.update(extra_row)

    for key, val in {
        "experiment_id": EXPERIMENT_ID,
        "run_id": RUN_ID,
        "trial_id": TRIAL_ID,
        "net_profile": NET_PROFILE,
        "tc_cmd": TC_CMD,
        "server_hostname": SERVER_HOSTNAME,
        "server_tag": SERVER_TAG,
    }.items():
        row.setdefault(key, val)

    safe_row = {k: row.get(k, math.nan) for k in _FIELDNAMES}
    with _log_lock:
        try:
            _log_wr.writerow(safe_row)
            _log_fh.flush()
            os.fsync(_log_fh.fileno())
        except Exception as e:
            print("[logger] write error:", e, flush=True)
            return False
    return True

# ---------------- YOLO summary extractor ----------------
def _extract_yolo_summaries(boxes):
    stop_conf = 0.0; stop_hit = 0
    light_conf = 0.0; light_hit = 0
    bot_conf = 0.0; bot_hit = 0
    try:
        for b in boxes:
            if len(b) < 6:
                continue
            x1,y1,x2,y2,cls,conf = b
            cls = int(cls)
            conf = float(conf)
            if cls == 2:
                stop_conf = max(stop_conf, conf)
                stop_hit |= int(conf > 0.6)
            elif cls == 3:
                light_conf = max(light_conf, conf)
                light_hit |= int(conf > 0.6)
            elif cls == 0:
                bot_conf = max(bot_conf, conf)
                bot_hit |= int(conf > 0.6)
    except Exception:
        pass
    return stop_conf, stop_hit, light_conf, light_hit, bot_conf, bot_hit

# ---------------- Run metadata ----------------
def _write_run_metadata():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    meta = {
        "experiment_id": EXPERIMENT_ID,
        "run_id": RUN_ID,
        "trial_id": TRIAL_ID,
        "net_profile": NET_PROFILE,
        "tc_cmd": TC_CMD,
        "server_hostname": SERVER_HOSTNAME,
        "server_tag": SERVER_TAG,
        "start_time": time.time(),
        "frame_rate": FRAME_RATE
    }
    try:
        with open(os.path.join(OUTPUT_DIR, "run_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print("[meta] run metadata written", flush=True)
    except Exception as e:
        print("[meta] failed to write:", e, flush=True)

# ---------------- ServerStation ----------------

class ServerStation:
    def __init__(self, frame_q):
        self.frame_q = frame_q
        self.frame_rate = FRAME_RATE
        self._video_writer = None
        self._prev_network_delay = None
        self._prev_frame_time = None
        self._last_disconnect_info = (None, 0.0)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self._init_video_writer_if_needed()
        self.infer_lock = threading.Lock()

    def _init_video_writer_if_needed(self):
        if not SAVE_VIDEO:
            return
        try:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            video_path = os.path.join(OUTPUT_DIR, "debug_lane_follow.avi")
            self._video_writer = cv2.VideoWriter(video_path, fourcc, FRAME_RATE, (640, 640))
            if not self._video_writer.isOpened():
                print("[VIDEO] failed to open writer", flush=True)
                self._video_writer = None
        except Exception as e:
            print("[VIDEO] init error:", e, flush=True)

    def _write_video_frame(self, frame):
        if self._video_writer is not None:
            try:
                self._video_writer.write(frame)
            except Exception as e:
                print("[VIDEO] write error:", e, flush=True)

    def _send_safe_reply(self, conn, frame_id, v, omega):
        try:
            t_now = time.time()
            payload = f"UNKNOWN,{frame_id},{v:.3f},{omega:.3f},{t_now:.6f},0.000\n".encode()

            conn.sendall(payload)
        except Exception as e:
            print(f"[net] failed to send safe reply: {e}", flush=True)

    # ------------------------------------------------------------------
    # Handle each client connection robustly
    # ------------------------------------------------------------------
    def handle_client(self, conn, addr):
        client_ip = addr[0]
        print(f"[+] Connection from {addr}", flush=True)
        self._prev_network_delay = None
        self._prev_frame_time = None
        SAFE_SPEED = 0.1
        last_v, last_omega = 0.0, 0.0
        prev_frame_id = None
        session_vehicle = None
        conn.settimeout(10.0)
        fp = conn.makefile("rb")
        idle_start = time.time()
        client_follower = None

        try:
            if hasattr(lf, "clone_follower"):
                client_follower = lf.clone_follower()
            else:
                client_follower = copy.deepcopy(lf._FOLLOWER)
        except Exception:
            client_follower = lf._FOLLOWER  # last resort; still works but shared

        try:
            while True:
                # Read frame header with adaptive timeout
                try:
                    header = fp.readline()
                    if not header:
                        print("[net] client disconnected", flush=True)
                        break
                    idle_start = time.time()
                except socket.timeout:
                    idle_duration = time.time() - idle_start
                    if idle_duration > 30:
                        print(f"[net] idle for {idle_duration:.1f}s — closing", flush=True)
                        break
                    print(f"[net] idle {idle_duration:.1f}s — waiting...", flush=True)
                    continue
                except Exception as e:
                    print(f"[net] header read error: {e}", flush=True)
                    break

                # Parse header safely
                try:
                    img_size_str, vehicle, frame_id_str, t_gen_str = header.decode().strip().split(",", 3)
                    img_size = int(img_size_str)
                    frame_id = int(frame_id_str)
                    t_gen = float(t_gen_str)
                except Exception:
                    continue

                # FIRST TIME WE KNOW VEHICLE NAME → bind the connection to this vehicle
                if session_vehicle is None:
                    session_vehicle = vehicle
                    with frame_qs_lock:
                        if session_vehicle not in frame_qs:
                            frame_qs[session_vehicle] = deque(maxlen=2)
                    self.frame_q = frame_qs[session_vehicle]
                    print(f"[bind] {client_ip} → veh_name={session_vehicle}", flush=True)

                    # follower creation — per vehicle
                    if session_vehicle not in followers:
                        try:
                            followers[session_vehicle] = lf.clone_follower()
                        except Exception:
                            followers[session_vehicle] = copy.deepcopy(lf._FOLLOWER)

                    client_follower = followers[session_vehicle]

                # Read JPEG payload
                data = fp.read(img_size)
                if len(data) != img_size:
                    self._send_safe_reply(conn, frame_id, last_v, last_omega)
                    continue

                arr = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    self._send_safe_reply(conn, frame_id, last_v, last_omega)
                    continue

                img = cv2.resize(img, (640, 640))
                server_proc_start = time.time()

                # ----------------- ICC ADVERSARY (UPDATED) -----------------
                # ----------------- ICC ADVERSARY (UPDATED) -----------------
                adv_stats = {
                    "attack_mode": "none",
                    "adv_mean_diff": math.nan,
                    "adv_max_diff": math.nan,
                    "adv_compute_time": math.nan,
                }

                # Only run adversary if module loaded AND ICC says enable=True
                if adv is not None and adv.attack_enabled():
                    try:
                        img_before = img.copy()

                        # attack_frame already reads /tmp/icc_attack_ctrl.json internally
                        img_adv = adv.attack_frame(img, force_attack=False)
                        meta = adv.get_attack_metadata() or {}

                        # If attack produced a modified frame
                        if img_adv is not None:
                            img = img_adv

                        # Compute perturbation strength
                        diff = np.abs(img.astype(np.float32) - img_before.astype(np.float32))
                        adv_stats.update({
                            "adv_mean_diff": float(np.mean(diff)),
                            "adv_max_diff": float(np.max(diff)),
                        })
                        adv_stats.update(meta)

                        # Only print when attack is ON
                        mode = adv_stats.get("attack_mode", "none")
                        if mode not in (None, "none"):
                            print(
                                "[ICC-ADV] mode={m} mean={mean:.4f} max={mx:.4f} time={t:.4f}".format(
                                    m=mode,
                                    mean=adv_stats.get("adv_mean_diff", math.nan),
                                    mx=adv_stats.get("adv_max_diff", math.nan),
                                    t=adv_stats.get("adv_compute_time", math.nan),
                                ),
                                flush=True,
                            )

                    except Exception as e:
                        print("[ADV] attack error:", e, flush=True)
                else:
                    # When disabled, explicitly declare no attack
                    adv_stats["attack_mode"] = "none"
                # ------------------------------------------------------------


                # Inference
                server_infer_start = time.time()
                with self.infer_lock:
                    try:
                        err_px, lane_w_px, boxes, seg, rs_bgr, fb, mode, tracked_objs = lf.infer(img)
                        infer_error = 0
                    except Exception as e:
                        print(f"[infer]  error: previous infer failed", flush=True)
                        err_px, lane_w_px, boxes, seg, rs_bgr, fb, mode = (
                            math.nan, math.nan, [], None, img, True, "error"
                        )
                        infer_error = 1
                server_infer_end = time.time()

                # Control logic
                try:
                    if infer_error:
                        v, omega = SAFE_SPEED, last_omega
                        print("[infer] inference failed; using fallback command", flush=True)
                    else:
                        v, omega = client_follower.update(
                            err_px, lane_w_px, boxes, rs_bgr.shape[:2],
                            last_cam_bgr=rs_bgr, fallback=fb, tracked_objs=tracked_objs
                        )
                except Exception as e:
                    print(f"[ctrl] follower error: {e}", flush=True)
                    v, omega = last_v, last_omega
                last_v, last_omega = v, omega

                # Timing and metrics
                t_server = time.time()
                network_delay = server_infer_start - t_gen
                total_latency = t_server - server_proc_start
                aoi_server = t_server - t_gen

                # Send control reply  <<< here use VEHICLE NAME NOT IP
                payload = f"{session_vehicle},{frame_id},{v:.3f},{omega:.3f},{t_server:.6f},{aoi_server:.6f}\n".encode()

                reply_success = 0
                try:
                    conn.sendall(payload)
                    reply_success = 1
                except Exception as e:
                    print(f"[net] send error: {e}", flush=True)
                    if isinstance(frame_id, int):
                        self._send_safe_reply(conn, frame_id, last_v, last_omega)

                # Persist server-side telemetry for network experiment runs.
                # The optional client timing line is currently not used here.
                if LOG_ENABLED:
                    row = {
                        "frame_id": frame_id,
                        "t_gen": t_gen,
                        "t_server": t_server,
                        "network_delay": network_delay,
                        "processing_delay": server_infer_end - server_infer_start,
                        "control_delay": t_server - server_infer_end,
                        "aoi_server": aoi_server,
                        "server_proc_start": server_proc_start,
                        "server_proc_end": t_server,
                        "server_proc_wall": t_server - server_proc_start,
                        "server_infer_start": server_infer_start,
                        "server_infer_end": server_infer_end,
                        "server_infer_wall": server_infer_end - server_infer_start,
                        "payload_bytes": img_size,
                        "reply_send_time": time.time(),
                        "infer_error": infer_error,
                        "reply_success": reply_success,
                    }
                    row.update(adv_stats)
                    log_row_from_lf(v, omega, extra_row=row)

                # Visualization
                try:
                    gui_frame = viz_lane_follow(
                        rs_bgr,
                        boxes,
                        seg,
                        err_px,
                        v,
                        omega,
                        mode=mode,
                        show_mode=True,
                        tracked_objs=tracked_objs,
                    )
                    if isinstance(gui_frame, np.ndarray):
                        with frame_qs_lock:
                            q = frame_qs.setdefault(session_vehicle, deque(maxlen=2))
                            q.append(gui_frame)
                    self._write_video_frame(gui_frame)

                except Exception:
                    pass

        except Exception as e:
            print(f"[net] handler exception: {e}", flush=True)
        finally:
            self._last_disconnect_info = (client_ip, time.time())
            try: fp.close()
            except: pass
            try: conn.shutdown(socket.SHUT_RDWR)
            except: pass
            try: conn.close()
            except: pass
            print(f"[-] Connection closed for {addr}", flush=True)



    # ------------------------------------------------------------------
    # Server main loop
    # ------------------------------------------------------------------
    def serve(self, host="0.0.0.0", port=PORT):
        """Main server loop with reconnect storm protection."""
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    try:
                        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                    except Exception:
                        pass
                    s.bind((host, port))
                    s.listen(5)
                    print(f"[server] Listening on {host}:{port}", flush=True)

                    while True:
                        conn, addr = s.accept()
                        client_ip = addr[0]
                        last_ip, last_time = self._last_disconnect_info
                        if client_ip == last_ip and (time.time() - last_time) < 0.5:
                            print(f"[server] Rapid reconnect from {client_ip}, delaying", flush=True)
                            conn.close()
                            time.sleep(0.5)
                            continue
                        print(f"[server] Accepted connection from {client_ip}:{addr[1]}", flush=True)
                        threading.Thread(target=self.handle_client, args=(conn, addr), daemon=True).start()
            except OSError as e:
                if e.errno == 98:
                    print(f"[server][WARN] Port {port} in use — retrying in 3s...", flush=True)
                    time.sleep(3)
                    continue
                print(f"[server][ERR] Socket error: {e}", flush=True)
                time.sleep(3)
            except Exception as e:
                print(f"[server][FATAL] Unexpected error: {e}", flush=True)
                time.sleep(3)
# ---------------- main ----------------
# ---------------- main ----------------
if __name__ == "__main__":
    _write_run_metadata()
    server = ServerStation(frame_qs)
    threading.Thread(target=server.serve, daemon=True).start()
    print(f"[server] started; LOG={LOG_ENABLED}, VIDEO={SAVE_VIDEO}, GUI={SHOW_GUI}", flush=True)

    if SHOW_GUI:
        windows = {}   # maps ip → window name

    try:
        while True:
            if SHOW_GUI:
                with frame_qs_lock:
                    for vehicle_ip, q in frame_qs.items():
                        if len(q) == 0:
                            continue
                        frame = q[-1]

                        # create window if first time
                        if vehicle_ip not in windows:
                            win_name = f"LF_{vehicle_ip}"
                            windows[vehicle_ip] = win_name
                            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                            cv2.resizeWindow(win_name, 960, 540)

                        cv2.imshow(windows[vehicle_ip], frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break
            else:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("[server] interrupted", flush=True)
    finally:
        if SAVE_VIDEO and server._video_writer:
            server._video_writer.release()
        if SHOW_GUI:
            cv2.destroyAllWindows()
        print("[server] exit", flush=True)
