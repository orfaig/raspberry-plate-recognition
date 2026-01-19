#!/usr/bin/env python3
# coding: utf-8
"""
Calibration project (student exercise): QR target tracking + drive-to-target.

Goal
----
1) Detect QR and draw its quad.
2) Apply simple calibration filters:
   - min bbox width/height
   - square-ish aspect ratio range
3) Compute target center -> yaw (deg) using camera intrinsics.
4) Path planning: steer by yaw and drive forward to the target.
5) Stop when target is "reached" (QR bbox width >= TARGET_RATIO * frame width).
6) Count ONLY accepted detections (after filters + cooldown).
7) Visualization: reuse "plate-tracker style" HUD (arrow + status + wheel PWMs).

Notes
-----
- This is intentionally simple and parameterized for student tuning.
- No OCR, no ONNX. QR only.
"""

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"   # Raspberry Pi desktop OpenCV windows

import cv2
import time
import threading
import numpy as np
import argparse
from flask import Flask, Response

from YB_Pcb_Car_control import YB_Pcb_Car
from helper_function import pixel_to_angles


# ============================================================
# PARAMETERS (all in one place)
# ============================================================
def build_args():
    p = argparse.ArgumentParser()

    # camera
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--w", type=int, default=640)
    p.add_argument("--h", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)

    # web stream
    p.add_argument("--port", type=int, default=8080)

    # mode
    p.add_argument("--mode", choices=["remote", "debug"], default="debug",
                   help="debug shows local cv2 window; remote disables it.")

    # ===== calibration knobs (student tuning) =====
    p.add_argument("--qr_min_w", type=int, default=30,
                   help="Reject if QR bbox width < this.")
    p.add_argument("--qr_min_h", type=int, default=30,
                   help="Reject if QR bbox height < this.")
    p.add_argument("--qr_min_aspect", type=float, default=0.65,
                   help="Reject if (w/h) < this.")
    p.add_argument("--qr_max_aspect", type=float, default=1.35,
                   help="Reject if (w/h) > this.")

    # accepted-detection counting
    p.add_argument("--cooldown_ms", type=int, default=600,
                   help="Count again only if last count was more than this many ms ago.")
    p.add_argument("--show_text_s", type=float, default=1.5,
                   help="Show last decoded text for this many seconds.")
    p.add_argument("--debug_metrics", action="store_true",
                   help="Show bbox size + aspect for calibration (on-screen).")

    # ===== camera intrinsics (pixel_to_angles) =====
    p.add_argument("--fx", type=float, default=450.0)
    p.add_argument("--fy", type=float, default=600.0)
    p.add_argument("--cx", type=float, default=320.0)
    p.add_argument("--cy", type=float, default=240.0)

        # ===== path planning / driving =====
    # --speed (float, PWM units, default 80.0)
    # Base forward speed per wheel (0..100). This is the nominal PWM before steering correction.
    # Higher = faster approach but more overshoot/less stability; lower = slower but steadier.
    p.add_argument("--speed", type=float, default=80.0, help="Base speed PWM (0..100).")

    # --yaw_deadband_deg (float, degrees, default 5.0)
    # If |yaw| is within this range, drive straight (L=R) to avoid twitchy micro-corrections.
    # Larger = smoother but less precise centering; smaller = more precise but can oscillate.
    p.add_argument("--yaw_deadband_deg", type=float, default=5.0)

    # --steer_kp (float, default 0.020)
    # Proportional steering gain. Converts yaw error (deg) into a steering correction:
    # steer = steer_kp * yaw_deg
    # Higher = more aggressive turning (can oscillate); lower = smoother but may not correct enough.
    p.add_argument("--steer_kp", type=float, default=0.020)

    # --steer_limit (float, default 0.25)
    # Clamp on steering correction magnitude: steer in [-steer_limit, +steer_limit].
    # Limits max left/right PWM difference (prevents extreme commands).
    # Higher = sharper turns; lower = gentler turns (might not turn enough).
    p.add_argument("--steer_limit", type=float, default=0.25)

    # --yaw_filter_alpha (float, 0..1, default 0.30)
    # Low-pass filter factor for yaw to reduce jitter from detection noise:
    # yaw_filt = (1-alpha)*yaw_filt + alpha*yaw_deg
    # Higher = more responsive but noisier; lower = smoother but more lag.
    p.add_argument("--yaw_filter_alpha", type=float, default=0.30)

    # --lost_timeout_s (float, seconds, default 1.0)
    # Grace period after losing detection: keep moving using last filtered yaw for this long.
    # Larger = drives through short dropouts (risk: driving blind longer);
    # smaller = stops sooner (safer but more stop/start).
    p.add_argument("--lost_timeout_s", type=float, default=1.0)


    # stop condition
    p.add_argument("--target_ratio", type=float, default=0.60,
                   help="Stop when bbox_w >= target_ratio * frame_width")
    
    

    return p.parse_args()


# ============================================================
# FLASK STREAM (MJPEG)
# ============================================================
app = Flask(__name__)
_latest_jpeg = None
_latest_lock = threading.Lock()
_stop_event = threading.Event()


def publish_frame(frame_bgr):
    global _latest_jpeg
    ok, jpg = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return
    with _latest_lock:
        _latest_jpeg = jpg.tobytes()


def mjpeg_generator():
    while not _stop_event.is_set():
        with _latest_lock:
            data = _latest_jpeg
        if data is None:
            time.sleep(0.05)
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
        time.sleep(0.03)


@app.route("/")
def index():
    return "<h3>Robot HUD Stream</h3><img src='/video' style='max-width:100%;height:auto;'/>"


@app.route("/video")
def video():
    return Response(mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


def run_flask(port: int):
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False, threaded=True)


# ============================================================
# CAMERA
# ============================================================
def open_camera(args):
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.h)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    if cap.isOpened():
        return cap
    cap.release()
    return None


# ============================================================
# QR GEOMETRY + FILTERS
# ============================================================
def points_to_quad(points):
    if points is None:
        return None
    pts = points[0] if len(points.shape) == 3 else points
    return np.asarray(pts, dtype=np.float32).reshape(4, 2)


def quad_bbox(pts):
    xs = pts[:, 0]
    ys = pts[:, 1]
    x1 = int(np.min(xs)); y1 = int(np.min(ys))
    x2 = int(np.max(xs)); y2 = int(np.max(ys))
    return x1, y1, x2, y2


def draw_qr_quad(frame, pts, ok):
    color = (0, 255, 0) if ok else (0, 0, 255)
    p = pts.astype(int)
    cv2.polylines(frame, [p], True, color, 2)
    for (x, y) in p:
        cv2.circle(frame, (int(x), int(y)), 4, (255, 255, 255), -1)


def passes_square_filters(pts, args):
    x1, y1, x2, y2 = quad_bbox(pts)
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    aspect = bw / float(bh)

    size_ok = (bw >= args.qr_min_w) and (bh >= args.qr_min_h)
    aspect_ok = (args.qr_min_aspect <= aspect <= args.qr_max_aspect)

    ok = size_ok and aspect_ok
    metrics = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "bw": bw, "bh": bh, "aspect": aspect,
               "size_ok": size_ok, "aspect_ok": aspect_ok}
    return ok, metrics


# ============================================================
# CAR CONTROL (same style as plate code)
# ============================================================
class CarControl:
    def __init__(self, max_pwm=100):
        self.car = YB_Pcb_Car()
        self.max_pwm = int(max_pwm)

    def stop(self):
        self.car.Car_Stop()

    def drive_pwm(self, left_pwm, right_pwm):
        left_pwm = int(np.clip(left_pwm, -self.max_pwm, self.max_pwm))
        right_pwm = int(np.clip(right_pwm, -self.max_pwm, self.max_pwm))
        self.car.Control_Car(left_pwm, right_pwm)


# ============================================================
# PATH PLANNER (steer by yaw, drive forward)
# ============================================================
class PathPlanner:
    def __init__(self, args):
        self.base_speed = float(args.speed)
        self.Kp = float(args.steer_kp)
        self.yaw_deadband_deg = float(args.yaw_deadband_deg)
        self.steer_limit = float(args.steer_limit)
        self.yaw_alpha = float(args.yaw_filter_alpha)
        self.lost_timeout_s = float(args.lost_timeout_s)

        self.active = True
        self.has_lock = False
        self.yaw_filt = 0.0
        self.lost_since = None

    def reset(self):
        self.active = True
        self.has_lock = False
        self.yaw_filt = 0.0
        self.lost_since = None

    def _compute_pwm(self, yaw_used_deg):
        base = float(np.clip(self.base_speed, 0.0, 100.0))

        if abs(yaw_used_deg) <= self.yaw_deadband_deg:
            return base, base

        steer = self.Kp * float(yaw_used_deg)
        steer = float(np.clip(steer, -self.steer_limit, self.steer_limit))

        left_pwm = base * (1.0 + steer)
        right_pwm = base * (1.0 - steer)

        # keep a minimum turning power when moving
        MIN_TURN = 70.0
        MAX_PWM = 100.0
        left_pwm = max(MIN_TURN, min(MAX_PWM, left_pwm))
        right_pwm = max(MIN_TURN, min(MAX_PWM, right_pwm))
        return left_pwm, right_pwm

    def step(self, now, yaw_deg, has_detection):
        if not self.active:
            return 0.0, 0.0

        if has_detection and yaw_deg is not None:
            self.has_lock = True
            self.lost_since = None

            self.yaw_filt = (1.0 - self.yaw_alpha) * self.yaw_filt + self.yaw_alpha * float(yaw_deg)
            return self._compute_pwm(self.yaw_filt)

        # no detection
        if not self.has_lock:
            return 0.0, 0.0

        if self.lost_since is None:
            self.lost_since = now

        if (now - self.lost_since) <= self.lost_timeout_s:
            return self._compute_pwm(self.yaw_filt)

        self.has_lock = False
        return 0.0, 0.0


# ============================================================
# HUD (reuse style: arrow + status + wheels)
# ============================================================
def draw_hud(frame, yaw_deg, status_text, left_pwm, right_pwm,
             accepted_count, decoded_text=""):
    vis = frame.copy()
    h, w = vis.shape[:2]
    bar_h = int(h * 0.23)
    font = cv2.FONT_HERSHEY_SIMPLEX

    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)

    # arrow
    cx = w // 2
    cy = int(bar_h * 0.70)
    arrow_len = int(bar_h * 0.6)

    yaw_disp = 0.0 if yaw_deg is None else float(np.clip(yaw_deg, -45, 45))
    theta = np.radians(yaw_disp)
    ex = int(cx + arrow_len * np.sin(theta))
    ey = int(cy - arrow_len * np.cos(theta))
    cv2.arrowedLine(vis, (cx, cy), (ex, ey), (0, 255, 0), 4, tipLength=0.25)

    # text
    cv2.putText(vis, f"Yaw: {yaw_disp:.1f} deg", (10, bar_h - 10),
                font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    if status_text.startswith("TARGET"):
        status_color = (255, 0, 0)
    else:
        status_color = (255, 255, 255)
    cv2.putText(vis, f"Status: {status_text}", (10, bar_h + 25),
                font, 0.7, status_color, 2, cv2.LINE_AA)

    cv2.putText(vis, f"L:{left_pwm:.0f} R:{right_pwm:.0f}", (w - 170, bar_h - 10),
                font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(vis, f"Accepted detections: {accepted_count}", (10, bar_h + 55),
                font, 0.75, (0, 255, 255), 2, cv2.LINE_AA)

    if decoded_text:
        cv2.putText(vis, decoded_text[:60], (10, bar_h + 85),
                    font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    return vis


def draw_debug_metrics(vis, metrics, args):
    font = cv2.FONT_HERSHEY_SIMPLEX
    txt1 = f"bbox: {metrics['bw']}x{metrics['bh']}  aspect={metrics['aspect']:.2f}"
    txt2 = f"min_w={args.qr_min_w} min_h={args.qr_min_h}  aspect_range=[{args.qr_min_aspect:.2f},{args.qr_max_aspect:.2f}]"
    cv2.putText(vis, txt1, (10, 140), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, txt2, (10, 170), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def maybe_show_debug_window(args, frame_bgr):
    if args.mode != "debug":
        return False
    cv2.imshow("camera", frame_bgr)
    k = cv2.waitKey(1) & 0xFF
    return (k == ord('q') or k == 27)


# ============================================================
# MAIN LOOP
# ============================================================
def main_robot(args):
    K = np.array([[args.fx, 0.0, args.cx],
                  [0.0, args.fy, args.cy],
                  [0.0, 0.0, 1.0]], dtype=np.float32)

    cap = open_camera(args)
    if cap is None:
        print("Error: cannot open camera")
        _stop_event.set()
        return

    detector = cv2.QRCodeDetector()

    car = CarControl(max_pwm=100)
    planner = PathPlanner(args)

    # counting state
    accepted_count = 0
    last_count_t = 0.0
    cooldown_s = max(0.0, args.cooldown_ms / 1000.0)

    # text preview state
    last_text = ""
    last_text_t = 0.0

    fail_count = 0

    try:
        while not _stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                fail_count += 1
                if fail_count > 10:
                    print("Camera read failed, trying to reopen...")
                    cap.release()
                    time.sleep(1.0)
                    cap = open_camera(args)
                    if cap is None:
                        print("Error: cannot reopen camera, stopping.")
                        break
                    fail_count = 0
                continue
            fail_count = 0

            now = time.time()

            # --- QR detection/decoding ---
            text, points, _ = detector.detectAndDecode(frame)

            has_det = False
            yaw_deg = None
            bb_w = None
            metrics = None

            if points is not None:
                pts = points_to_quad(points)
                ok, metrics = passes_square_filters(pts, args)
                draw_qr_quad(frame, pts, ok)

                if ok:
                    # center of bbox as target center (simple + stable)
                    x1, y1, x2, y2 = metrics["x1"], metrics["y1"], metrics["x2"], metrics["y2"]
                    bb_w = metrics["bw"]

                    u = (x1 + x2) / 2.0
                    v = (y1 + y2) / 2.0
                    yaw_deg, _ = pixel_to_angles(u, v, K)
                    has_det = True

                    # count accepted detections with cooldown
                    if (now - last_count_t) >= cooldown_s:
                        accepted_count += 1
                        last_count_t = now

            # decoded text preview (optional)
            if text:
                last_text = text
                last_text_t = now

            show_text = ""
            if last_text and (now - last_text_t) <= float(args.show_text_s):
                show_text = last_text

            # stop condition (target reached)
            h, w = frame.shape[:2]
            if has_det and bb_w is not None and bb_w >= float(args.target_ratio) * w:
                status = "TARGET REACHED!"
                left_pwm = 0.0
                right_pwm = 0.0
                car.stop()
                planner.active = False
            else:
                left_pwm, right_pwm = planner.step(now, yaw_deg, has_det)

                if left_pwm == 0.0 and right_pwm == 0.0:
                    status = "STOPPED"
                    car.stop()
                else:
                    status = "ACTIVE"
                    car.drive_pwm(left_pwm, right_pwm)

            vis_hud = draw_hud(frame, yaw_deg, status, left_pwm, right_pwm,
                               accepted_count=accepted_count,
                               decoded_text=show_text)

            if args.debug_metrics and metrics is not None:
                draw_debug_metrics(vis_hud, metrics, args)

            publish_frame(vis_hud)

            if maybe_show_debug_window(args, vis_hud):
                break

            time.sleep(0.01)

    finally:
        car.stop()
        cap.release()
        cv2.destroyAllWindows()
        _stop_event.set()


# ============================================================
# ENTRY
# ============================================================
if __name__ == "__main__":
    args = build_args()

    # robot logic in background
    t = threading.Thread(target=main_robot, args=(args,), daemon=True)
    t.start()

    # web stream server (main thread)
    run_flask(args.port)
