#!/usr/bin/env python3
# coding: utf-8

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"   # important for Raspberry Pi desktop

import cv2
import threading
import time
import numpy as np
import argparse
from flask import Flask, Response

from YB_Pcb_Car_control import YB_Pcb_Car
from helper_function import pixel_to_angles


# ------------------------
# FLASK STREAM (WORKING VERSION)
# ------------------------
app = Flask(__name__)

latest_jpeg = None
running = True


def generate_stream():
    """Generator for MJPEG stream."""
    global latest_jpeg, running
    while running:
        if latest_jpeg is not None:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   latest_jpeg + b"\r\n")
        else:
            time.sleep(0.02)


@app.route("/")
def index():
    return """
    <html>
      <head>
        <title>Raspbot Camera</title>
        <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate" />
        <meta http-equiv="Pragma" content="no-cache" />
        <meta http-equiv="Expires" content="0" />
        <script>
          function reloadStream() {
            const img = document.getElementById('cam');
            img.src = '/video_feed?ts=' + new Date().getTime();
          }
          function onStreamError() {
            setTimeout(reloadStream, 1000);
          }
          window.onload = function() {
            setInterval(reloadStream, 30000);
          }
        </script>
      </head>
      <body>
        <h1>Raspbot Camera Stream</h1>
        <img id="cam" src="/video_feed" onerror="onStreamError()" />
        <p>
          <button onclick="location.reload()">Refresh page</button>
          <button onclick="reloadStream()">Refresh video</button>
        </p>
        <p>Press 'q' on the Raspberry Pi window to quit.</p>
      </body>
    </html>
    """


@app.route("/video_feed")
def video_feed():
    return Response(generate_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


def run_flask():
    app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)


# ------------------------
# CAMERA INTRINSICS (SAME AS YOUR PREVIOUS CODE)
# ------------------------
FX = 450.0
FY = 600.0
CX = 320.0
CY = 240.0
K = np.array([[FX, 0.0, CX],
              [0.0, FY, CY],
              [0.0, 0.0, 1.0]], dtype=np.float32)


# ------------------------
# ROBOT / TRACKING PARAMS
# ------------------------
YAW_DEADBAND_DEG = 5.0
STEER_KP = 0.020
STEER_LIMIT = 0.25
YAW_FILTER_ALPHA = 0.30

LOST_TIMEOUT_S = 1.0          # shorter for QR to avoid “ghost driving”
MAX_SPEED_PWM = 100.0
MIN_TURN_SPEED_PWM = 70.0

TARGET_RATIO = 0.6            # bb_w >= TARGET_RATIO * frame_w => reached


# ------------------------
# DRAWING (YOUR WORKING QR POLY DRAW)
# ------------------------
def draw_qr_bbox(frame, points):
    """
    points: ndarray shape (1,4,2) or (4,2) from OpenCV QRCodeDetector
    Draw polygon + corner points.
    """
    if points is None:
        return frame

    pts = points
    if len(pts.shape) == 3:
        pts = pts[0]

    pts = pts.astype(int)

    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    for (x, y) in pts:
        cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)

    return frame


def points_to_bbox(points):
    if points is None:
        return None
    pts = points[0] if len(points.shape) == 3 else points
    pts = np.asarray(pts, dtype=np.float32)
    xs = pts[:, 0]
    ys = pts[:, 1]
    x1 = int(np.min(xs)); y1 = int(np.min(ys))
    x2 = int(np.max(xs)); y2 = int(np.max(ys))
    return x1, y1, x2, y2, pts


# ------------------------
# CAR CONTROL
# ------------------------
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


# ------------------------
# PATH PLANNER (STEER BY YAW)
# ------------------------
class PathPlanner:
    def __init__(self,
                 base_speed=80,
                 Kp=STEER_KP,
                 yaw_deadband_deg=YAW_DEADBAND_DEG,
                 steer_limit=STEER_LIMIT,
                 yaw_alpha=YAW_FILTER_ALPHA,
                 lost_timeout_s=LOST_TIMEOUT_S):
        self.base_speed = float(base_speed)
        self.Kp = float(Kp)
        self.yaw_deadband_deg = float(yaw_deadband_deg)
        self.steer_limit = float(steer_limit)
        self.yaw_alpha = float(yaw_alpha)
        self.lost_timeout_s = float(lost_timeout_s)

        self.has_lock = False
        self.yaw_filt = 0.0
        self.lost_since = None

    def reset(self):
        self.has_lock = False
        self.yaw_filt = 0.0
        self.lost_since = None

    def _compute_pwm(self, yaw_used_deg):
        base = min(self.base_speed, MAX_SPEED_PWM)

        if abs(yaw_used_deg) <= self.yaw_deadband_deg:
            return base, base

        steer = self.Kp * yaw_used_deg
        steer = float(np.clip(steer, -self.steer_limit, self.steer_limit))

        left_pwm = base * (1.0 + steer)
        right_pwm = base * (1.0 - steer)

        # ensure some minimum turning power
        left_pwm = max(MIN_TURN_SPEED_PWM, min(MAX_SPEED_PWM, left_pwm))
        right_pwm = max(MIN_TURN_SPEED_PWM, min(MAX_SPEED_PWM, right_pwm))
        return left_pwm, right_pwm

    def step(self, now, yaw_deg, has_detection):
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


# ------------------------
# SIMPLE HUD (STATUS + WHEELS)
# ------------------------
def draw_status(frame, status, yaw_deg, left_pwm, right_pwm, qr_text):
    vis = frame
    h, w = vis.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # top bar
    bar_h = 55
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    vis[:] = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)

    yaw_disp = 0.0 if yaw_deg is None else float(np.clip(yaw_deg, -45, 45))
    cv2.putText(vis, f"Status: {status}", (10, 22), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, f"Yaw: {yaw_disp:.1f} deg", (10, 48), font, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, f"L:{left_pwm:.0f} R:{right_pwm:.0f}", (w - 180, 48), font, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    if qr_text:
        cv2.putText(vis, qr_text[:60], (10, bar_h + 25),
                    font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


# ------------------------
# MAIN
# ------------------------
def main():
    global latest_jpeg, running

    p = argparse.ArgumentParser()
    p.add_argument("--speed", type=float, default=80)
    p.add_argument("--target_qr", type=str, default=None,
                   help="If set, robot moves ONLY when decoded QR text matches this exactly.")
    p.add_argument("--qr_min_w", type=int, default=60)
    p.add_argument("--qr_min_h", type=int, default=60)
    p.add_argument("--qr_min_aspect", type=float, default=0.80)
    p.add_argument("--qr_max_aspect", type=float, default=1.25)
    args = p.parse_args()

    # start Flask in background thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # open camera in MAIN THREAD
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Error: cannot open camera")
        running = False
        return

    detector = cv2.QRCodeDetector()

    car = CarControl(max_pwm=100)
    planner = PathPlanner(base_speed=args.speed)

    print("Camera started. Press 'q' in the window to quit.")
    print("From phone, open:  http://<PI_IP>:8080")

    fail_count = 0
    last_text = ""
    last_seen_t = 0.0

    try:
        while running:
            ret, frame = cap.read()

            if not ret:
                fail_count += 1
                if fail_count > 10:
                    print("Camera read failed, trying to reopen...")
                    cap.release()
                    time.sleep(1.0)
                    cap = cv2.VideoCapture(0)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    if not cap.isOpened():
                        print("Error: cannot reopen camera, stopping.")
                        running = False
                        break
                    fail_count = 0
                continue
            else:
                fail_count = 0

            now = time.time()

            # --- QR detection/decoding ---
            text, points, _ = detector.detectAndDecode(frame)

            # Always draw bbox when points exist (visual feedback)
            if points is not None:
                draw_qr_bbox(frame, points)

            yaw_deg = None
            has_det = False
            bb_w = None

            # STRICT CONTROL: move only when QR is REALLY decoded (non-empty text)
            # Optional: also require exact match to --target_qr
            if text:
                if (args.target_qr is None) or (text == args.target_qr):
                    box = points_to_bbox(points)
                    if box is not None:
                        x1, y1, x2, y2, _pts = box
                        bb_w = x2 - x1
                        bb_h = y2 - y1
                        aspect = bb_w / max(1, bb_h)

                        # geometry filters (reduce junk)
                        if (bb_w >= args.qr_min_w and bb_h >= args.qr_min_h and
                                args.qr_min_aspect <= aspect <= args.qr_max_aspect):
                            u = (x1 + x2) / 2.0
                            v = (y1 + y2) / 2.0
                            yaw_deg, _ = pixel_to_angles(u, v, K)
                            has_det = True

                # keep showing text briefly even if it stops decoding
                last_text = text
                last_seen_t = now

            # show decoded text briefly (same behavior as your good code)
            show_text = ""
            if last_text and (now - last_seen_t) < 1.5:
                show_text = last_text

            # TARGET REACHED -> stop if QR fills enough width
            h, w = frame.shape[:2]
            if has_det and bb_w is not None and bb_w >= TARGET_RATIO * w:
                car.stop()
                planner.reset()
                status = "TARGET REACHED (STOP)"
                left_pwm = 0.0
                right_pwm = 0.0
            else:
                left_pwm, right_pwm = planner.step(now, yaw_deg, has_det)
                if left_pwm == 0 and right_pwm == 0:
                    car.stop()
                    status = "STOP"
                else:
                    car.drive_pwm(left_pwm, right_pwm)
                    status = "TRACK"

            # HUD
            draw_status(frame, status, yaw_deg, left_pwm, right_pwm, show_text)

            # encode for web (with bbox + HUD)
            ok, buffer = cv2.imencode(".jpg", frame)
            if ok:
                latest_jpeg = buffer.tobytes()

            # show on Pi screen
            cv2.imshow("Raspbot QR Tracker (press q to quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                running = False
                break

    finally:
        car.stop()
        cap.release()
        cv2.destroyAllWindows()
        running = False
        print("Stopped.")


if __name__ == "__main__":
    main()
