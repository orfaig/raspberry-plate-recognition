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
import subprocess
import argparse
import select
from YB_Pcb_Car_control import YB_Pcb_Car
from remote_fucntion import get_ir_key

from helper_function import (
    bbox_center, pixel_to_angles, iou_xyxy, nms_xyxy,
    preprocess_plate, extract_valid_plate,
    run_onnx_inference, decode_yolov8,
    crop_with_margin, free_camera,keyboard_start_pressed
)
from flask import Flask, Response
import threading
import re

CONFIG_TESSERACT = (
    "--oem 3 --psm 8 "
    "-c tessedit_char_whitelist=0123456789 "
    "-c classify_bln_numeric_mode=1 "
    "-c user_defined_dpi=300 "
    "-c load_system_dawg=0 -c load_freq_dawg=0"
)
OCR_UPSCALE = 2.0
PLATE_FALLBACK_MINLEN = 5
PLATE_FALLBACK_MAXLEN = 12
PLATE_STABILITY_N = 3            # must see same plate 3 times before locking
PLATE_STABILITY_WINDOW_S = 3.0   # within 3 seconds
PLATE_CONFIRM_N = 3
PLATE_COUNT_TTL_S = 10.0 

try:
    from plate_format.plate_format_ro import is_valid_plate, normalize_plate_format
except Exception:
    is_valid_plate = None
    normalize_plate_format = None




# steering / tracking hyperparams
YAW_DEADBAND_DEG = 5.0        # +/- deg => drive straight
STEER_KP = 0.020              # smaller = less aggressive pivot
STEER_LIMIT = 0.35            # max left/right differential (0..1)
YAW_FILTER_ALPHA = 0.35       # 0..1 (higher = more responsive, lower = smoother)

# lost-target grace period (seconds)
LOST_TIMEOUT_S = 3.0          # keep moving for this long after losing target
MAX_SPEED_PWM = 100.0          # never exceed this per wheel
MIN_TURN_SPEED_PWM = 70.0     # in a turn, never go below this per wheel

TARGET_PLATE_DEFAULT = "91-179-18"
OCR_PERIOD_S = 0.2   # was 1.0 (faster reaction; optional)

app = Flask(__name__)

_latest_jpeg = None
_latest_lock = threading.Lock()
_stop_event = threading.Event()

def open_camera():
    # try multiple ways (index + device path, V4L2 + ANY)
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

    print("[ERROR] Cannot open camera with OpenCV. Check /dev/video0 and permissions.")
    return None


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


# ------------------------
# CONFIG
# ------------------------
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

ONNX_PATH = "/home/pi/Yahboom_project/uveye/code/raspberry-plate-recognition/yolov8n-license_plate.onnx"

CONF_THRES = 0.2
IOU_THRES = 0.2
INFER_PERIOD_S = 0.1
# OCR_PERIOD_S = 1.0
MAX_MISSED_FRAMES = 10
TARGET_RATIO = 0.6   # width >= 90% of frame width => target reached

# camera intrinsics
FX = 450.0
FY = 600.0
CX = 320.0
CY = 240.0

K = np.array([[FX,   0.0, CX],
              [0.0,  FY,  CY],
              [0.0,  0.0, 1.0]], dtype=np.float32)

clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))

onnxr.set_default_logger_severity(3)
onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)

session = onnxr.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
inp = session.get_inputs()[0]

def normalize_plate_il(raw: str):
    digits = re.sub(r"\D", "", raw or "")
    if len(digits) == 7:
        return f"{digits[:2]}-{digits[2:5]}-{digits[5:]}"  # 91-179-18
    if len(digits) == 8:
        return f"{digits[:3]}-{digits[3:5]}-{digits[5:]}"  # 123-45-678
    return None

def ocr_plate_il(plate_bgr):
    if plate_bgr is None or plate_bgr.size == 0:
        return None

    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)
    gray = cv2.resize(gray, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # try a few binarizations
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adap = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 7)

    variants = [otsu, cv2.bitwise_not(otsu), adap, cv2.bitwise_not(adap)]

    # vote within this single call
    hits = {}
    for img in variants:
        raw = pytesseract.image_to_string(img, config=CONFIG_TESSERACT)
        norm = normalize_plate_il(raw)
        if norm:
            hits[norm] = hits.get(norm, 0) + 1

    if not hits:
        return None

    # return the most frequent candidate
    return max(hits.items(), key=lambda kv: kv[1])[0]


def _int_or_none(x):
    return x if isinstance(x, int) else None

MODEL_H = _int_or_none(inp.shape[2]) or 512
MODEL_W = _int_or_none(inp.shape[3]) or 512

def preprocess_plate_for_ocr(plate_bgr):
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)
    blur = cv2.bilateralFilter(gray, 11, 16, 16)
    thr = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 2
    )
    return thr

def ocr_plate_text(plate_bgr):
    # if plate_bgr is None or plate_bgr.size == 0:
    #     return None

    # if OCR_UPSCALE != 1.0:
    #     plate_bgr = cv2.resize(
    #         plate_bgr, None, fx=OCR_UPSCALE, fy=OCR_UPSCALE, interpolation=cv2.INTER_CUBIC
    #     )

    # proc = preprocess_plate_for_ocr(plate_bgr)
    # raw = pytesseract.image_to_string(proc, config=CONFIG_TESSERACT)
    # raw = raw.upper().strip()
    # raw = re.sub(r"[^A-Z0-9]", "", raw)  # keep only A-Z0-9

    # if not raw:
    #     return None

    # # If Romanian formatter exists and matches, return normalized
    # if is_valid_plate and normalize_plate_format and is_valid_plate(raw):
    #     return normalize_plate_format(raw)

    # # Fallback: show best guess even if not “valid”
    # if PLATE_FALLBACK_MINLEN <= len(raw) <= PLATE_FALLBACK_MAXLEN:
    #     return raw

    return None

# ------------------------
# PERCEPTION
# ------------------------
class Perception:
    def __init__(self, target_plate=None):
        self.last_infer_time = 0.0
        self.last_ocr_time = 0.0
        self.detections = []
        self.plate_text = None  # cached forever after first target read
        self.target_plate = target_plate

    def _try_ocr_once(self, frame_bgr, bbox, now):
        if self.plate_text is not None:
            return
        if now - self.last_ocr_time < OCR_PERIOD_S:
            return

        x1, y1, x2, y2 = bbox
        bb_w = x2 - x1
        bb_h = y2 - y1

        # mx = int(0.12 * bb_w)
        # my = int(0.35 * bb_h)
        mx = int(0.20 * bb_w)
        my = int(0.55 * bb_h)
        x1m = max(0, x1 - mx)
        y1m = max(0, y1 - my)
        x2m = min(frame_bgr.shape[1] - 1, x2 + mx)
        y2m = min(frame_bgr.shape[0] - 1, y2 + my)

        crop = frame_bgr[y1m:y2m, x1m:x2m]

        try:
            plate = ocr_plate_il(crop)
        except Exception:
            plate = None

        # IMMEDIATE lock if the target plate is seen once
        if plate and self.target_plate and plate == self.target_plate:
            self.plate_text = plate
            print(f"[TARGET_PLATE_FOUND] {plate}", flush=True)

        self.last_ocr_time = now





    def process_frame(self, frame, now):
        fh, fw = frame.shape[:2]

        # run inference
        if now - self.last_infer_time >= INFER_PERIOD_S:
            outputs = run_onnx_inference(frame)
            self.detections = decode_yolov8(outputs, fw, fh,
                                            conf_thres=CONF_THRES,
                                            iou_thres=IOU_THRES)
            self.last_infer_time = now

        vis = frame.copy()

        if not self.detections:
            return vis, None, None, False, None, self.plate_text

        best = max(self.detections, key=lambda d: d["confidence"])
        (x1, y1, x2, y2) = best["bbox"]

        bb_w = x2 - x1
        bb_h = y2 - y1
        aspect = bb_w / max(1, bb_h)

        MIN_W = 40
        MIN_H = 15
        MIN_ASPECT = 2.0
        MAX_ASPECT = 7.0

        if bb_w < MIN_W or bb_h < MIN_H or not (MIN_ASPECT <= aspect <= MAX_ASPECT):
            return frame.copy(), None, None, False, None, self.plate_text

        u, v = bbox_center(best["bbox"])
        yaw_deg, pitch_deg = pixel_to_angles(u, v, K)

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(vis, (int(u), int(v)), 4, (0, 0, 255), -1)

        # OCR only until we get a valid plate once
        self._try_ocr_once(frame, (x1, y1, x2, y2), now)

        return vis, yaw_deg, pitch_deg, True, bb_w, self.plate_text


# ------------------------
# CAR CONTROL
# ------------------------
class CarControl:
    def __init__(self, max_pwm=100):
        self.car = YB_Pcb_Car()
        self.max_pwm = max_pwm

    def stop(self):
        self.car.Car_Stop()

    def drive(self, left_norm, right_norm):
        left_pwm = int(left_norm * self.max_pwm)
        right_pwm = int(right_norm * self.max_pwm)
        self.car.Control_Car(left_pwm, right_pwm)

# ------------------------
# PATH PLANNER
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

        self.active = True
        self.has_lock = False

        self.last_yaw = 0.0
        self.yaw_filt = 0.0
        self.lost_since = None  # timestamp when we first lost detection

    def reset(self):
        self.active = True
        self.has_lock = False
        self.last_yaw = 0.0
        self.yaw_filt = 0.0
        self.lost_since = None

    def _compute_pwm(self, yaw_used_deg):
        # STOP logic stays outside this function; this only computes moving speeds.

        # Straight: both wheels same speed (capped)
        if abs(yaw_used_deg) <= self.yaw_deadband_deg:
            s = min(self.base_speed, MAX_SPEED_PWM)
            return s, s

        # Turning: compute differential, then clamp each wheel to [70..80]
        steer = self.Kp * yaw_used_deg
        steer = float(np.clip(steer, -self.steer_limit, self.steer_limit))

        # Use base_speed as the nominal; cap it to MAX
        base = min(self.base_speed, MAX_SPEED_PWM)

        left_pwm  = base * (1.0 + steer)
        right_pwm = base * (1.0 - steer)

        # Enforce min/max during turns
        left_pwm  = max(MIN_TURN_SPEED_PWM, min(MAX_SPEED_PWM, left_pwm))
        right_pwm = max(MIN_TURN_SPEED_PWM, min(MAX_SPEED_PWM, right_pwm))

        return left_pwm, right_pwm


    def step(self, now, yaw_deg, has_detection):
        if not self.active:
            return 0.0, 0.0

        if has_detection:
            self.has_lock = True
            self.lost_since = None

            if yaw_deg is not None:
                self.last_yaw = float(yaw_deg)
                # low-pass filter to reduce wobble
                self.yaw_filt = (1.0 - self.yaw_alpha) * self.yaw_filt + self.yaw_alpha * self.last_yaw

            return self._compute_pwm(self.yaw_filt)

        # no detection:
        if not self.has_lock:
            return 0.0, 0.0  # never move until we saw it at least once

        if self.lost_since is None:
            self.lost_since = now

        # keep moving for LOST_TIMEOUT_S using last filtered yaw
        if (now - self.lost_since) <= self.lost_timeout_s:
            return self._compute_pwm(self.yaw_filt)

        # lost too long -> stop and drop lock
        self.has_lock = False
        return 0.0, 0.0


# ------------------------
# HUD DRAW
# ------------------------
def draw_hud(frame, yaw_deg, status_text, left_pwm, right_pwm, plate_text=None, target_plate=None):
    vis = frame.copy()
    h, w = vis.shape[:2]
    bar_h = int(h * 0.23)
    font = cv2.FONT_HERSHEY_SIMPLEX

    overlay = vis.copy()
    cv2.rectangle(overlay, (0,0), (w,bar_h), (0,0,0), -1)
    vis = cv2.addWeighted(overlay,0.35,vis,0.65,0)

    cx = w // 2
    cy = int(bar_h * 0.70)
    arrow_len = int(bar_h * 0.6)

    yaw_disp = 0.0 if yaw_deg is None else max(-45, min(45, yaw_deg))
    theta = np.radians(yaw_disp)

    ex = int(cx + arrow_len * np.sin(theta))
    ey = int(cy - arrow_len * np.cos(theta))
    cv2.arrowedLine(vis,(cx,cy),(ex,ey),(0,255,0),4,tipLength=0.25)

    cv2.putText(vis,f"Yaw: {yaw_disp:.1f} deg",(10,bar_h-8),
                font,0.7,(0,255,255),2)

    if status_text == "TARGET REACHED!":
        cv2.putText(vis, f"Status: {status_text}", (10, bar_h + 25),
                    font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(vis, f"Status: {status_text}", (10, bar_h + 25),
                    font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    wheel_text = f"L:{left_pwm:.0f} R:{right_pwm:.0f}"
    cv2.putText(vis,wheel_text,(w-150,bar_h-10),
                font,0.7,(0,255,255),2)

    # plate box: yellow fill, black border
    if plate_text is not None:
            is_target = (target_plate is not None and plate_text == target_plate)
            label = f"PLATE: {plate_text or '----'}"

            (tw, th), base = cv2.getTextSize(label, font, 0.8, 2)
            x = 10
            y = bar_h + 60
            pad = 6
            tl = (x - pad, y - th - pad)
            br = (x + tw + pad, y + base + pad)

            bg = (0, 255, 255) if is_target else (0, 255, 255)  # red if target else yellow
            cv2.rectangle(vis, tl, br, bg, -1)
            cv2.rectangle(vis, tl, br, (0,0,0), 2)
            cv2.putText(vis, label, (x, y), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    return vis


# ------------------------
# IR WAIT
# ------------------------
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

def wait_for_play(args):
    KEY_PLAY = 0x15
    if args.start_input == "ir":
        print("[WAIT] Continue via IR PLAY")
    else:
        print("[WAIT] Continue via keyboard: 'p' + Enter")

    while True:
        if args.start_input == "ir":
            key = get_ir_key()
            key = 0 if key is None else int(key)
            if key != 0:
                print(f"[IR] key=0x{key:02X}")
            if key == KEY_PLAY:
                print("[START] Resuming.")
                return
        else:
            if stdin_pressed_p() or keyboard_start_pressed():
                print("[START] Resuming.")
                return
        time.sleep(0.02)


# ------------------------
# MAIN LOOP
# ------------------------
def main(args):
    free_camera()

    perception = Perception(target_plate=args.target_plate)

    car = CarControl()
    planner = PathPlanner(
        base_speed=args.speed,
        Kp=0.02,
        yaw_deadband_deg=5.0,
        steer_limit=0.25,
        yaw_alpha=0.30,
        lost_timeout_s=3.0
    )

    wait_for_play(args)

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
                if bad_reads >= 30:  # ~1.5s of failures
                    print("[WARN] Camera read failed, reopening...")
                    cap.release()
                    cap = open_camera()
                    bad_reads = 0
                    if cap is None:
                        return
                continue

            bad_reads = 0
            now = time.time()
            vis, yaw_deg, pitch_deg, has_det, bb_w, plate_text = perception.process_frame(frame, now)
            h, w = frame.shape[:2]

            # (rest of your loop unchanged...)


            # TARGET REACHED: stop and wait for PLAY to resume
            if has_det and bb_w is not None and bb_w >= TARGET_RATIO * w:
                status = "TARGET REACHED!"
                left_pwm = 0
                right_pwm = 0
                car.stop()
                planner.active = False
                while True:
                    # keep updating stream while waiting
                    vis_hud = draw_hud(vis, yaw_deg, status, left_pwm, right_pwm, plate_text, args.target_plate)


                    publish_frame(vis_hud)

                    # >>> ADD THIS LINE <<<
                    if maybe_show_debug_window(args, vis_hud):
                        return  # exit main cleanly if user presses q / ESC

                    key = get_ir_key()
                    key = 0 if key is None else int(key)

                    if key == 0x15 or keyboard_start_pressed():

                        print("[IR] PLAY pressed -> resuming.")
                        planner.reset()
                        planner.active = True
                        break

                    time.sleep(0.03)


                continue

            # normal tracking
            left_pwm, right_pwm = planner.step(now, yaw_deg, has_det)


            if left_pwm == 0 and right_pwm == 0:
                status = "STOPPED"
                car.stop()
            else:
                status = "ACTIVE"
                car.drive(left_pwm / 100.0, right_pwm / 100.0)

            vis_hud = draw_hud(vis, yaw_deg, status, left_pwm, right_pwm, plate_text, args.target_plate)


            publish_frame(vis_hud)
            if maybe_show_debug_window(args, vis_hud):
                 break
                

            # optional: stop loop if you want an IR key to exit (no keyboard on SSH)
            # if get_ir_key() == SOME_KEY: break

            time.sleep(0.01)

    finally:
        car.stop()
        cap.release()
        _stop_event.set()
        cv2.destroyAllWindows()

def maybe_show_debug_window(args, frame_bgr):
    if args.mode != "debug":
        return False  # no request to exit
    cv2.imshow("camera", frame_bgr)
    k = cv2.waitKey(1) & 0xFF
    return (k == ord('q') or k == 27)
# ------------------------
# ENTRY
# ------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--speed", type=float, default=80)
    p.add_argument("--mode", choices=["remote", "debug"], default="debug")
    p.add_argument("--start_input", choices=["ir", "kbd"], default="kbd",
                   help="How to continue from WAIT: IR remote or keyboard.")
    p.add_argument("--target_plate", type=str, default=TARGET_PLATE_DEFAULT)
    args = p.parse_args()

    # Start your robot logic in background
    t = threading.Thread(target=main, args=(args,), daemon=True)
    t.start()

    # Start the web stream server
    app.run(host="0.0.0.0", port=8080, threaded=True)
