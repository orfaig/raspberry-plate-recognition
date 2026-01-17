#!/usr/bin/env python3
# coding: utf-8
import os
# os.environ["QT_QPA_PLATFORM"] = "xcb"
import select
import sys
import time
import cv2
import pytesseract
import onnx
import onnxruntime as onnxr
import numpy as np
import threading
import argparse
import re
from flask import Flask, Response

from YB_Pcb_Car_control import YB_Pcb_Car
from remote_fucntion import get_ir_key

from helper_function import (
    bbox_center, pixel_to_angles,
    run_onnx_inference, decode_yolov8,
    free_camera, keyboard_start_pressed
)


# ------------------------
# OCR / PLATE CONFIG
# ------------------------
CONFIG_TESSERACT = (
    "--oem 3 --psm 7 "
    "-c tessedit_char_whitelist=0123456789 "
    "-c classify_bln_numeric_mode=1 "
    "-c user_defined_dpi=300 "
    "-c load_system_dawg=0 -c load_freq_dawg=0"
)

TARGET_PLATE_DEFAULT = "91-179-18"
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# ------------------------
# ROBOT / TRACKING PARAMS
# ------------------------
YAW_DEADBAND_DEG = 5.0
STEER_KP = 0.020
STEER_LIMIT = 0.35
YAW_FILTER_ALPHA = 0.35

LOST_TIMEOUT_S = 3.0
MAX_SPEED_PWM = 100.0
MIN_TURN_SPEED_PWM = 70.0

TARGET_RATIO = 0.6  # bb_w >= TARGET_RATIO * frame_w => reached

# ------------------------
# FLASK STREAM
# ------------------------
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

# ------------------------
# CAMERA
# ------------------------
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
    print("[ERROR] Cannot open camera with OpenCV. Check /dev/video0 and permissions.")
    return None

# ------------------------
# MODEL (PLATE ONNX)
# ------------------------
ONNX_PATH = "/home/pi/Yahboom_project/uveye/code/raspberry-plate-recognition/yolov8n-license_plate.onnx"

onnxr.set_default_logger_severity(3)
onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)
session = onnxr.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

# ------------------------
# CAMERA INTRINSICS
# ------------------------
FX = 450.0
FY = 600.0
CX = 320.0
CY = 240.0
K = np.array([[FX, 0.0, CX],
              [0.0, FY, CY],
              [0.0, 0.0, 1.0]], dtype=np.float32)

# ------------------------
# PLATE OCR HELPERS
# ------------------------
clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))

def normalize_plate_il(raw: str):
    digits = re.sub(r"\D", "", raw or "")
    if len(digits) == 7:
        return f"{digits[:2]}-{digits[2:5]}-{digits[5:]}"
    if len(digits) == 8:
        return f"{digits[:3]}-{digits[3:5]}-{digits[5:]}"
    return None

def ocr_plate_il(plate_bgr):
    if plate_bgr is None or plate_bgr.size == 0:
        return None

    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)
    gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adap = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 7)
    variants = [otsu, cv2.bitwise_not(otsu), adap, cv2.bitwise_not(adap)]

    hits = {}
    for img in variants:
        raw = pytesseract.image_to_string(img, config=CONFIG_TESSERACT)
        norm = normalize_plate_il(raw)
        if norm:
            hits[norm] = hits.get(norm, 0) + 1

    if not hits:
        return None
    return max(hits.items(), key=lambda kv: kv[1])[0]

# ------------------------
# PERCEPTION PARAMS (SEPARATED)
# ------------------------
class PlateParams:
    def __init__(self,
                 conf_thres=0.2,
                 iou_thres=0.2,
                 infer_period_s=0.1,
                 ocr_period_s=1.0,
                 min_w=40,
                 min_h=15,
                 min_aspect=2.0,
                 max_aspect=7.0):
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.infer_period_s = float(infer_period_s)
        self.ocr_period_s = float(ocr_period_s)
        self.min_w = int(min_w)
        self.min_h = int(min_h)
        self.min_aspect = float(min_aspect)
        self.max_aspect = float(max_aspect)

class QrParams:
    """
    IMPORTANT CHANGE:
      - detect_period_s controls how often we run QR detect() for bbox/yaw (driving)
      - decode_period_s controls how often we run detectAndDecode() for text (optional target filter / display)
    """
    def __init__(self,
                 detect_period_s=0.0,
                 decode_period_s=0.25,
                 min_w=50,
                 min_h=50,
                 min_aspect=0.85,
                 max_aspect=1.15):
        self.detect_period_s = float(detect_period_s)
        self.decode_period_s = float(decode_period_s)
        self.min_w = int(min_w)
        self.min_h = int(min_h)
        self.min_aspect = float(min_aspect)
        self.max_aspect = float(max_aspect)

# ------------------------
# QR HELPERS (BBOX + DRAW LIKE PLATE)
# ------------------------
def _qr_points_to_bbox(points):
    if points is None:
        return None
    pts = points[0] if len(points.shape) == 3 else points
    pts = np.asarray(pts, dtype=np.float32)
    xs = pts[:, 0]
    ys = pts[:, 1]
    x1 = int(np.min(xs)); y1 = int(np.min(ys))
    x2 = int(np.max(xs)); y2 = int(np.max(ys))
    return x1, y1, x2, y2, pts

def _draw_bbox_and_center(vis, x1, y1, x2, y2, u, v):
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.circle(vis, (int(u), int(v)), 4, (0, 0, 255), -1)

# ------------------------
# PERCEPTION: PLATE OR QR (NOT BOTH)
# ------------------------
class Perception:
    def __init__(self, task: str, plate_params: PlateParams, qr_params: QrParams,
                 target_plate: str | None = None, target_qr: str | None = None):
        self.task = task
        self.pp = plate_params
        self.qp = qr_params

        self.target_plate = target_plate
        self.target_qr = target_qr  # optional: drive ONLY when matched

        # plate state
        self.last_infer_time = 0.0
        self.last_ocr_time = 0.0
        self.detections = []
        self.plate_text = None

        # qr state
        self.qr = cv2.QRCodeDetector()
        self.last_qr_detect_time = 0.0
        self.last_qr_decode_time = 0.0
        self.qr_text = None

    def _try_plate_ocr_once(self, frame_bgr, bbox, now):
        if self.plate_text is not None:
            return
        if now - self.last_ocr_time < self.pp.ocr_period_s:
            return

        x1, y1, x2, y2 = bbox
        bb_w = x2 - x1
        bb_h = y2 - y1

        mx = int(0.12 * bb_w)
        my = int(0.35 * bb_h)
        x1m = max(0, x1 - mx)
        y1m = max(0, y1 - my)
        x2m = min(frame_bgr.shape[1] - 1, x2 + mx)
        y2m = min(frame_bgr.shape[0] - 1, y2 + my)

        crop = frame_bgr[y1m:y2m, x1m:x2m]

        try:
            plate = ocr_plate_il(crop)
        except Exception:
            plate = None

        if plate:
            if self.target_plate is None or plate == self.target_plate:
                self.plate_text = plate
                print(f"[PLATE] {plate}", flush=True)

        self.last_ocr_time = now

    def _process_plate(self, frame, now):
        fh, fw = frame.shape[:2]

        if now - self.last_infer_time >= self.pp.infer_period_s:
            outputs = run_onnx_inference(frame)
            self.detections = decode_yolov8(outputs, fw, fh,
                                            conf_thres=self.pp.conf_thres,
                                            iou_thres=self.pp.iou_thres)
            self.last_infer_time = now

        vis = frame.copy()
        if not self.detections:
            return vis, None, False, None, self.plate_text

        best = max(self.detections, key=lambda d: d["confidence"])
        x1, y1, x2, y2 = best["bbox"]

        bb_w = x2 - x1
        bb_h = y2 - y1
        aspect = bb_w / max(1, bb_h)

        if bb_w < self.pp.min_w or bb_h < self.pp.min_h:
            return frame.copy(), None, False, None, self.plate_text
        if not (self.pp.min_aspect <= aspect <= self.pp.max_aspect):
            return frame.copy(), None, False, None, self.plate_text

        u, v = bbox_center(best["bbox"])
        yaw_deg, _ = pixel_to_angles(u, v, K)

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(vis, (int(u), int(v)), 4, (0, 0, 255), -1)

        self._try_plate_ocr_once(frame, (x1, y1, x2, y2), now)

        return vis, yaw_deg, True, bb_w, self.plate_text

    def _process_qr(self, frame, now):
        """
        IMPORTANT CHANGE:
          - We always use detect() for bbox/yaw (so bbox is shown and wheels move like plate).
          - We decode text only periodically (optional) and only use it to gate driving if --target_qr is set.
        """
        # rate-limit detect() if requested
        if self.qp.detect_period_s > 0 and (now - self.last_qr_detect_time) < self.qp.detect_period_s:
            # keep last qr_text for HUD, but do NOT claim detection without fresh bbox
            return frame.copy(), None, False, None, self.qr_text

        self.last_qr_detect_time = now
        vis = frame.copy()

        ok = self.qr.detect(frame)
        if isinstance(ok, tuple):
            # some OpenCV builds return (retval, points)
            retval, points = ok
            if not retval:
                points = None
        else:
            # older builds may return just retval; then bbox requires detectAndDecode()
            points = None

        # If detect() didn't return points, fallback to detectAndDecode() just for points
        if points is None:
            _text, points, _ = self.qr.detectAndDecode(frame)

        if points is None:
            self.qr_text = None
            return vis, None, False, None, None

        box = _qr_points_to_bbox(points)
        if box is None:
            self.qr_text = None
            return vis, None, False, None, None

        x1, y1, x2, y2, pts = box
        bb_w = x2 - x1
        bb_h = y2 - y1
        aspect = bb_w / max(1, bb_h)

        # geometry filters (same idea as plate)
        if bb_w < self.qp.min_w or bb_h < self.qp.min_h:
            return vis, None, False, None, self.qr_text
        if not (self.qp.min_aspect <= aspect <= self.qp.max_aspect):
            return vis, None, False, None, self.qr_text

        # center + yaw
        u = (x1 + x2) / 2.0
        v = (y1 + y2) / 2.0
        yaw_deg, _ = pixel_to_angles(u, v, K)

        # SHOW bounding box EXACTLY like plate mode (rectangle + center dot)
        _draw_bbox_and_center(vis, x1, y1, x2, y2, u, v)

        # decode text only sometimes (optional display / optional gating)
        if now - self.last_qr_decode_time >= self.qp.decode_period_s:
            txt, _pts, _ = self.qr.detectAndDecode(frame)
            self.last_qr_decode_time = now
            self.qr_text = txt if txt else None

        # If user provided target_qr: only allow "has_det" when decoded matches.
        # But we STILL show bbox always (so visualization is stable).
        if self.target_qr is not None:
            if self.qr_text != self.target_qr:
                return vis, yaw_deg, False, bb_w, self.qr_text

        return vis, yaw_deg, True, bb_w, self.qr_text

    def process_frame(self, frame, now):
        if self.task == "plate":
            return self._process_plate(frame, now)
        return self._process_qr(frame, now)

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
        self.lost_since = None

    def reset(self):
        self.active = True
        self.has_lock = False
        self.last_yaw = 0.0
        self.yaw_filt = 0.0
        self.lost_since = None

    def _compute_pwm(self, yaw_used_deg):
        if abs(yaw_used_deg) <= self.yaw_deadband_deg:
            s = min(self.base_speed, MAX_SPEED_PWM)
            return s, s

        steer = self.Kp * yaw_used_deg
        steer = float(np.clip(steer, -self.steer_limit, self.steer_limit))
        base = min(self.base_speed, MAX_SPEED_PWM)

        left_pwm = base * (1.0 + steer)
        right_pwm = base * (1.0 - steer)

        left_pwm = max(MIN_TURN_SPEED_PWM, min(MAX_SPEED_PWM, left_pwm))
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
                self.yaw_filt = (1.0 - self.yaw_alpha) * self.yaw_filt + self.yaw_alpha * self.last_yaw
            return self._compute_pwm(self.yaw_filt)

        if not self.has_lock:
            return 0.0, 0.0

        if self.lost_since is None:
            self.lost_since = now

        if (now - self.lost_since) <= self.lost_timeout_s:
            return self._compute_pwm(self.yaw_filt)

        self.has_lock = False
        return 0.0, 0.0

# ------------------------
# HUD DRAW (PLATE OR QR)
# ------------------------
def draw_hud(frame, task, yaw_deg, status_text, left_pwm, right_pwm,
             plate_text=None, target_plate=None,
             qr_text=None):
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

    yaw_disp = 0.0 if yaw_deg is None else max(-45, min(45, yaw_deg))
    theta = np.radians(yaw_disp)
    ex = int(cx + arrow_len * np.sin(theta))
    ey = int(cy - arrow_len * np.cos(theta))
    cv2.arrowedLine(vis, (cx, cy), (ex, ey), (0, 255, 0), 4, tipLength=0.25)

    cv2.putText(vis, f"Task: {task}", (10, 25), font, 0.7, (255, 255, 255), 2)
    cv2.putText(vis, f"Yaw: {yaw_disp:.1f} deg", (10, bar_h - 8), font, 0.7, (0, 255, 255), 2)

    cv2.putText(vis, f"Status: {status_text}", (10, bar_h + 25),
                font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    wheel_text = f"L:{left_pwm:.0f} R:{right_pwm:.0f}"
    cv2.putText(vis, wheel_text, (w - 150, bar_h - 10), font, 0.7, (0, 255, 255), 2)

    if task == "plate":
        if plate_text is not None:
            label = f"PLATE: {plate_text or '----'}"
            (tw, th), base = cv2.getTextSize(label, font, 0.8, 2)
            x = 10
            y = bar_h + 60
            pad = 6
            tl = (x - pad, y - th - pad)
            br = (x + tw + pad, y + base + pad)
            cv2.rectangle(vis, tl, br, (0, 255, 255), -1)
            cv2.rectangle(vis, tl, br, (0, 0, 0), 2)
            cv2.putText(vis, label, (x, y), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

            if target_plate and plate_text == target_plate:
                cv2.putText(vis, "TARGET", (x + tw + 20, y), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    else:  # task == "qr"
        if qr_text is not None:
            label = f"QR: {qr_text[:40]}"
            (tw, th), base = cv2.getTextSize(label, font, 0.7, 2)
            x = 10
            y = bar_h + 60
            pad = 6
            tl = (x - pad, y - th - pad)
            br = (x + tw + pad, y + base + pad)
            cv2.rectangle(vis, tl, br, (255, 255, 255), -1)
            cv2.rectangle(vis, tl, br, (0, 0, 0), 2)
            cv2.putText(vis, label, (x, y), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

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
# DEBUG WINDOW
# ------------------------
def maybe_show_debug_window(args, frame_bgr):
    if args.mode != "debug":
        return False
    cv2.imshow("camera", frame_bgr)
    k = cv2.waitKey(1) & 0xFF
    return (k == ord('q') or k == 27)

# ------------------------
# MAIN LOOP
# ------------------------
def main(args):
    free_camera()

    plate_params = PlateParams(
        conf_thres=args.plate_conf,
        iou_thres=args.plate_iou,
        infer_period_s=args.plate_infer_period,
        ocr_period_s=args.plate_ocr_period,
        min_w=args.plate_min_w,
        min_h=args.plate_min_h,
        min_aspect=args.plate_min_aspect,
        max_aspect=args.plate_max_aspect,
    )
    qr_params = QrParams(
        detect_period_s=args.qr_detect_period,
        decode_period_s=args.qr_decode_period,
        min_w=args.qr_min_w,
        min_h=args.qr_min_h,
        min_aspect=args.qr_min_aspect,
        max_aspect=args.qr_max_aspect,
    )

    perception = Perception(
        task=args.task,
        plate_params=plate_params,
        qr_params=qr_params,
        target_plate=args.target_plate if args.task == "plate" else None,
        target_qr=args.target_qr if args.task == "qr" else None,
    )

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

            vis, yaw_deg, has_det, bb_w, text = perception.process_frame(frame, now)
            h, w = frame.shape[:2]

            # TARGET REACHED
            if has_det and bb_w is not None and bb_w >= TARGET_RATIO * w:
                status = "TARGET REACHED!"
                left_pwm = 0
                right_pwm = 0
                car.stop()
                planner.active = False

                while True:
                    vis_hud = draw_hud(
                        vis, args.task, yaw_deg, status, left_pwm, right_pwm,
                        plate_text=(text if args.task == "plate" else None),
                        target_plate=args.target_plate,
                        qr_text=(text if args.task == "qr" else None),
                    )
                    publish_frame(vis_hud)

                    if maybe_show_debug_window(args, vis_hud):
                        return

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

            vis_hud = draw_hud(
                vis, args.task, yaw_deg, status, left_pwm, right_pwm,
                plate_text=(text if args.task == "plate" else None),
                target_plate=args.target_plate,
                qr_text=(text if args.task == "qr" else None),
            )

            publish_frame(vis_hud)
            if maybe_show_debug_window(args, vis_hud):
                break

            time.sleep(0.01)

    finally:
        car.stop()
        cap.release()
        _stop_event.set()
        cv2.destroyAllWindows()

# ------------------------
# ENTRY
# ------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--speed", type=float, default=80)
    p.add_argument("--mode", choices=["remote", "debug"], default="debug")

    p.add_argument("--task", choices=["plate", "qr"], default="qr")
    p.add_argument("--start_input", choices=["ir", "kbd"], default="kbd",
                   help="How to continue from WAIT: IR remote or keyboard.")

    # targets
    p.add_argument("--target_plate", type=str, default=TARGET_PLATE_DEFAULT,
                   help="Used only when --task=plate (optional: lock when matched).")
    p.add_argument("--target_qr", type=str, default=None,
                   help="Used only when --task=qr (optional: drive only when matched).")

    # plate hyperparams
    p.add_argument("--plate_conf", type=float, default=0.2)
    p.add_argument("--plate_iou", type=float, default=0.2)
    p.add_argument("--plate_infer_period", type=float, default=0.1)
    p.add_argument("--plate_ocr_period", type=float, default=1.0)
    p.add_argument("--plate_min_w", type=int, default=40)
    p.add_argument("--plate_min_h", type=int, default=15)
    p.add_argument("--plate_min_aspect", type=float, default=2.0)
    p.add_argument("--plate_max_aspect", type=float, default=7.0)

    # qr hyperparams
    p.add_argument("--qr_detect_period", type=float, default=0.0,
                   help="How often to run QR bbox detection (0 = every frame).")
    p.add_argument("--qr_decode_period", type=float, default=0.25,
                   help="How often to decode QR text for HUD/target gating.")
    p.add_argument("--qr_min_w", type=int, default=35)
    p.add_argument("--qr_min_h", type=int, default=35)
    p.add_argument("--qr_min_aspect", type=float, default=0.65)
    p.add_argument("--qr_max_aspect", type=float, default=1.6)

    args = p.parse_args()

    # Robot logic in background
    t = threading.Thread(target=main, args=(args,), daemon=True)
    t.start()

    # Web stream server
    app.run(host="0.0.0.0", port=8080, threaded=True)
