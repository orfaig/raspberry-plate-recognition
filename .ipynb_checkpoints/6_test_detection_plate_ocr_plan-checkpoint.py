#!/usr/bin/env python
# coding: utf-8
import os
# os.environ["QT_QPA_PLATFORM"] = "xcb"

import time
import sys
import cv2
import pytesseract
import onnx
import onnxruntime as onnxr
import numpy as np
import threading
import argparse
import select
import re
from flask import Flask, Response

from YB_Pcb_Car_control import YB_Pcb_Car
from remote_fucntion import get_ir_key

from helper_function import (
    bbox_center, pixel_to_angles,
    run_onnx_inference, decode_yolov8,
    free_camera, keyboard_start_pressed
)

# ======================
# TARGET PLATE
# ======================
TARGET_PLATE_DEFAULT = "91-179-18"

# ======================
# TESSERACT
# ======================
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

CONFIG_TESS = (
    "--oem 1 --psm 7 "
    "-c tessedit_char_whitelist=0123456789 "
    "-c classify_bln_numeric_mode=1 "
    "-c user_defined_dpi=200 "
    "-c load_system_dawg=0 -c load_freq_dawg=0"
)

# ======================
# DETECTION / OCR
# ======================
ONNX_PATH = "/home/pi/Yahboom_project/uveye/code/raspberry-plate-recognition/yolov8n-license_plate.onnx"

CONF_THRES = 0.2
IOU_THRES = 0.2
INFER_PERIOD_S = 0.10

# OCR every N frames (frame-based, like your “good one”)
OCR_EVERY_N_FRAMES = 3
MARGIN_X = 0.10
MARGIN_Y = 0.25

TARGET_RATIO = 0.6  # bbox width >= TARGET_RATIO * frame_width

# bbox filters (plate-ish)
MIN_W = 40
MIN_H = 15
MIN_ASPECT = 2.0
MAX_ASPECT = 7.0

# ======================
# TRACKING / STEERING
# ======================
YAW_DEADBAND_DEG = 5.0
STEER_KP = 0.020
STEER_LIMIT = 0.25
YAW_FILTER_ALPHA = 0.30
LOST_TIMEOUT_S = 3.0

MAX_SPEED_PWM = 100.0
MIN_TURN_SPEED_PWM = 70.0

# ======================
# CAMERA INTRINSICS
# ======================
FX = 450.0
FY = 600.0
CX = 320.0
CY = 240.0

K = np.array([[FX,   0.0, CX],
              [0.0,  FY,  CY],
              [0.0,  0.0, 1.0]], dtype=np.float32)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# ======================
# FLASK STREAM
# ======================
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

# ======================
# CAMERA
# ======================
def open_camera():
    tries = [
        (0, cv2.CAP_V4L2),
        (0, cv2.CAP_ANY),
        ("/dev/video0", cv2.CAP_V4L2),
        ("/dev/video0", cv2.CAP_ANY),
    ]
    for src, backend in tries:
        cap = cv2.VideoCapture(src, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            print(f"[INFO] Camera opened: src={src} backend={backend}")
            return cap
        cap.release()
    print("[ERROR] Cannot open camera.")
    return None

# ======================
# OCR HELPERS  (FIXED)
# ======================
def normalize_plate_il(raw: str):
    digits = re.sub(r"\D", "", raw or "")
    if len(digits) == 7:
        return f"{digits[:2]}-{digits[2:5]}-{digits[5:]}"
    if len(digits) == 8:
        return f"{digits[:3]}-{digits[3:5]}-{digits[5:]}"
    return None

def crop_plate(frame, box):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    mx = int(w * MARGIN_X)
    my = int(h * MARGIN_Y)
    return frame[
        max(0, y1 - my):min(frame.shape[0], y2 + my),
        max(0, x1 - mx):min(frame.shape[1], x2 + mx)
    ]

def ocr_plate_il(plate_bgr):
    if plate_bgr is None or plate_bgr.size == 0:
        return None

    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)
    gray = cv2.resize(gray, None, fx=2.4, fy=2.4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    for img in (bin_img, 255 - bin_img):
        text = pytesseract.image_to_string(img, config=CONFIG_TESS)  # <<< FIX: CONFIG_TESS (not CONFIG_TESSERACT)
        plate = normalize_plate_il(text)
        if plate:
            return plate
    return None

# ======================
# PERCEPTION
# ======================
onnxr.set_default_logger_severity(3)
onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)
session = onnxr.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
_inp = session.get_inputs()[0]
print(f"[INFO] ONNX input name={_inp.name} shape={_inp.shape}")

class Perception:
    def __init__(self, target_plate):
        self.target_plate = target_plate
        self.last_infer_time = 0.0
        self.detections = []
        self.plate_text = None      # locked forever once target found

        self.last_guess = ""
        self.last_guess_t = 0.0

        self.frame_count = 0
        self.last_box = None        # fallback box display

    def process_frame(self, frame, now):
        fh, fw = frame.shape[:2]

        if now - self.last_infer_time >= INFER_PERIOD_S:
            outputs = run_onnx_inference(frame)  # uses your helper (model-size correct)
            self.detections = decode_yolov8(outputs, fw, fh,
                                            conf_thres=CONF_THRES,
                                            iou_thres=IOU_THRES)
            self.last_infer_time = now

        vis = frame.copy()

        # draw last box fallback
        if not self.detections and self.last_box:
            x1, y1, x2, y2 = self.last_box
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if not self.detections:
            return vis, None, None, False, None, self.plate_text, self.last_guess

        best = max(self.detections, key=lambda d: d["confidence"])
        (x1, y1, x2, y2) = best["bbox"]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        self.last_box = (x1, y1, x2, y2)

        bb_w = x2 - x1
        bb_h = y2 - y1
        aspect = bb_w / max(1, bb_h)

        # filters
        if bb_w < MIN_W or bb_h < MIN_H or not (MIN_ASPECT <= aspect <= MAX_ASPECT):
            # still draw box to help debugging
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            return vis, None, None, False, None, self.plate_text, self.last_guess

        u, v = bbox_center((x1, y1, x2, y2))
        yaw_deg, pitch_deg = pixel_to_angles(u, v, K)

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(vis, (int(u), int(v)), 4, (0, 0, 255), -1)

        # ===== OCR every N frames, stop forever after target found =====
        if self.plate_text is None:
            self.frame_count += 1
            if (self.frame_count % OCR_EVERY_N_FRAMES) == 0:
                crop = crop_plate(frame, (x1, y1, x2, y2))
                plate = ocr_plate_il(crop)

                if plate:
                    self.last_guess = plate
                    self.last_guess_t = now

                if plate and plate == self.target_plate:
                    self.plate_text = plate
                    print(f"[TARGET_PLATE_FOUND] {plate}", flush=True)

        return vis, yaw_deg, pitch_deg, True, bb_w, self.plate_text, self.last_guess

# ======================
# CAR CONTROL
# ======================
class CarControl:
    def __init__(self, max_pwm=100):
        self.car = YB_Pcb_Car()
        self.max_pwm = max_pwm

    def stop(self):
        self.car.Car_Stop()

    def drive_pwm(self, left_pwm, right_pwm):
        left_pwm = int(np.clip(left_pwm, -self.max_pwm, self.max_pwm))
        right_pwm = int(np.clip(right_pwm, -self.max_pwm, self.max_pwm))
        self.car.Control_Car(left_pwm, right_pwm)

# ======================
# PATH PLANNER
# ======================
class PathPlanner:
    def __init__(self, base_speed):
        self.base_speed = float(base_speed)
        self.active = True
        self.has_lock = False
        self.yaw_filt = 0.0
        self.lost_since = None

    def reset(self):
        self.active = True
        self.has_lock = False
        self.yaw_filt = 0.0
        self.lost_since = None

    def _compute_pwm(self, yaw_deg):
        base = min(self.base_speed, MAX_SPEED_PWM)

        if abs(yaw_deg) <= YAW_DEADBAND_DEG:
            return base, base

        steer = STEER_KP * float(yaw_deg)
        steer = float(np.clip(steer, -STEER_LIMIT, STEER_LIMIT))

        left_pwm = base * (1.0 + steer)
        right_pwm = base * (1.0 - steer)

        left_pwm = max(MIN_TURN_SPEED_PWM, min(MAX_SPEED_PWM, left_pwm))
        right_pwm = max(MIN_TURN_SPEED_PWM, min(MAX_SPEED_PWM, right_pwm))
        return left_pwm, right_pwm

    def step(self, now, yaw_deg, has_detection):
        if not self.active:
            return 0.0, 0.0

        if has_detection and yaw_deg is not None:
            self.has_lock = True
            self.lost_since = None
            self.yaw_filt = (1.0 - YAW_FILTER_ALPHA) * self.yaw_filt + YAW_FILTER_ALPHA * float(yaw_deg)
            return self._compute_pwm(self.yaw_filt)

        if not self.has_lock:
            return 0.0, 0.0

        if self.lost_since is None:
            self.lost_since = now

        if (now - self.lost_since) <= LOST_TIMEOUT_S:
            return self._compute_pwm(self.yaw_filt)

        self.has_lock = False
        return 0.0, 0.0

# ======================
# HUD
# ======================
def draw_hud(frame, yaw_deg, status_text, left_pwm, right_pwm, plate_text=None, last_guess=None):
    vis = frame.copy()
    h, w = vis.shape[:2]
    bar_h = int(h * 0.23)
    font = cv2.FONT_HERSHEY_SIMPLEX

    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)

    cx = w // 2
    cy = int(bar_h * 0.70)
    arrow_len = int(bar_h * 0.6)

    yaw_disp = 0.0 if yaw_deg is None else float(np.clip(yaw_deg, -45, 45))
    theta = np.radians(yaw_disp)
    ex = int(cx + arrow_len * np.sin(theta))
    ey = int(cy - arrow_len * np.cos(theta))
    cv2.arrowedLine(vis, (cx, cy), (ex, ey), (0, 255, 0), 4, tipLength=0.25)

    cv2.putText(vis, f"Yaw: {yaw_disp:.1f} deg", (10, bar_h - 8),
                font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    color = (255, 0, 0) if status_text == "TARGET REACHED!" else (255, 255, 255)
    cv2.putText(vis, f"Status: {status_text}", (10, bar_h + 25),
                font, 0.7, color, 2, cv2.LINE_AA)

    cv2.putText(vis, f"L:{left_pwm:.0f} R:{right_pwm:.0f}", (w - 170, bar_h - 10),
                font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    if plate_text:
        cv2.putText(vis, f"TARGET: {plate_text}", (10, bar_h + 60),
                    font, 0.9, (0, 255, 0), 3, cv2.LINE_AA)
    else:
        if last_guess:
            cv2.putText(vis, f"OCR: {last_guess}", (10, bar_h + 60),
                        font, 0.9, (0, 255, 255), 3, cv2.LINE_AA)
        else:
            cv2.putText(vis, "OCR: ----", (10, bar_h + 60),
                        font, 0.9, (255, 255, 255), 3, cv2.LINE_AA)

    return vis

# ======================
# START WAIT
# ======================
def stdin_pressed_p() -> bool:
    try:
        r, _, _ = select.select([sys.stdin], [], [], 0)
        if r:
            line = sys.stdin.readline()
            if not line:
                return False
            return line.strip().lower() == "p"
    except Exception:
        return False
    return False

def wait_for_play(start_input):
    KEY_PLAY = 0x15
    if start_input == "ir":
        print("[WAIT] Continue via IR PLAY")
    else:
        print("[WAIT] Continue via keyboard: 'p' + Enter")

    while True:
        if start_input == "ir":
            key = get_ir_key()
            key = 0 if key is None else int(key)
            if key == KEY_PLAY:
                print("[START] Resuming.")
                return
        else:
            if stdin_pressed_p() or keyboard_start_pressed():
                print("[START] Resuming.")
                return
        time.sleep(0.02)

# ======================
# DEBUG WINDOW
# ======================
def maybe_show_debug_window(args, frame_bgr):
    if args.mode != "debug":
        return False
    cv2.imshow("camera", frame_bgr)
    k = cv2.waitKey(1) & 0xFF
    return (k == ord('q') or k == 27)

# ======================
# MAIN LOOP
# ======================
def main(args):
    free_camera()

    perception = Perception(target_plate=args.target_plate)
    car = CarControl(max_pwm=100)
    planner = PathPlanner(base_speed=args.speed)

    wait_for_play(args.start_input)

    cap = open_camera()
    if cap is None:
        return

    bad_reads = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                bad_reads += 1
                time.sleep(0.05)
                if bad_reads >= 30:
                    print("[WARN] Camera read failed, reopening...")
                    cap.release()
                    cap = open_camera()
                    bad_reads = 0
                    if cap is None:
                        return
                continue

            bad_reads = 0
            now = time.time()

            vis, yaw_deg, pitch_deg, has_det, bb_w, plate_text, last_guess = perception.process_frame(frame, now)
            h, w = frame.shape[:2]

            if perception.plate_text is not None and has_det and bb_w is not None and bb_w >= TARGET_RATIO * w:
                status = "TARGET REACHED!"
                left_pwm = 0.0
                right_pwm = 0.0
                car.stop()
                planner.active = False

                while True:
                    vis_hud = draw_hud(vis, yaw_deg, status, left_pwm, right_pwm,
                                       plate_text=perception.plate_text,
                                       last_guess=last_guess)
                    publish_frame(vis_hud)

                    if maybe_show_debug_window(args, vis_hud):
                        return

                    key = get_ir_key()
                    key = 0 if key is None else int(key)

                    if key == 0x15 or keyboard_start_pressed():
                        print("[RESUME] PLAY pressed -> resuming.")
                        planner.reset()
                        break

                    time.sleep(0.03)

                continue

            left_pwm, right_pwm = planner.step(now, yaw_deg, has_det)

            if left_pwm == 0.0 and right_pwm == 0.0:
                status = "STOPPED"
                car.stop()
            else:
                status = "ACTIVE"
                car.drive_pwm(left_pwm, right_pwm)

            if perception.plate_text is not None and status == "ACTIVE":
                status = "TARGET LOCKED"

            vis_hud = draw_hud(vis, yaw_deg, status, left_pwm, right_pwm,
                               plate_text=perception.plate_text,
                               last_guess=last_guess)

            publish_frame(vis_hud)

            if maybe_show_debug_window(args, vis_hud):
                break

            time.sleep(0.01)

    finally:
        car.stop()
        cap.release()
        _stop_event.set()
        cv2.destroyAllWindows()

# ======================
# ENTRY
# ======================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--speed", type=float, default=80)
    p.add_argument("--mode", choices=["remote", "debug"], default="debug")
    p.add_argument("--start_input", choices=["ir", "kbd"], default="kbd",
                   help="How to continue from WAIT: IR remote or keyboard.")
    p.add_argument("--target_plate", type=str, default=TARGET_PLATE_DEFAULT)
    args = p.parse_args()

    t = threading.Thread(target=main, args=(args,), daemon=True)
    t.start()

    app.run(host="0.0.0.0", port=8080, threaded=True)
