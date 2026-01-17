#!/usr/bin/env python3
# coding: utf-8
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

from helper_function import (
    bbox_center, pixel_to_angles,
    run_onnx_inference, decode_yolov8,
    free_camera
)

# ------------------------
# Tesseract (digits-only)
# ------------------------
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
CONFIG_TESS = (
    "--oem 1 --psm 7 "
    "-c tessedit_char_whitelist=0123456789 "
    "-c classify_bln_numeric_mode=1 "
    "-c user_defined_dpi=200 "
    "-c load_system_dawg=0 -c load_freq_dawg=0"
)

# ------------------------
# Detection hyperparams (FAST always)
# ------------------------
CONF_THRES = 0.2
IOU_THRES = 0.2
INFER_PERIOD_S = 0.03   # bbox update cadence

# ------------------------
# OCR hyperparams (FAST)
# - tighter crop = faster + less noise
# - OCR only every N detected frames
# ------------------------
OCR_TRY_EVERY_N_FRAMES_DEFAULT = 1
MARGIN_X = 0.10
MARGIN_Y = 0.25

# ------------------------
# camera intrinsics (yaw only)
# ------------------------
FX = 450.0
FY = 600.0
CX = 320.0
CY = 240.0
K = np.array([[FX, 0.0, CX],
              [0.0, FY, CY],
              [0.0, 0.0, 1.0]], dtype=np.float32)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# ------------------------
# ONNX (plate detector)
# ------------------------
ONNX_PATH = "/home/pi/Yahboom_project/uveye/code/raspberry-plate-recognition/yolov8n-license_plate.onnx"
onnxr.set_default_logger_severity(3)
onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)
session = onnxr.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

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
            time.sleep(0.02)
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
        time.sleep(0.03)

@app.route("/")
def index():
    return "<h3>Plate Detect (fast) + optional OCR</h3><img src='/video' style='max-width:100%;height:auto;'/>"

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
    print("[ERROR] Cannot open camera.")
    return None

# ------------------------
# OCR (STRICT success only 7/8 digits formatted) + FAST preprocessing
# ------------------------
def normalize_plate_il_strict(raw: str):
    digits = re.sub(r"\D", "", raw or "")
    if len(digits) == 7:
        return f"{digits[:2]}-{digits[2:5]}-{digits[5:]}"
    if len(digits) == 8:
        return f"{digits[:3]}-{digits[3:5]}-{digits[5:]}"
    return None

def crop_plate_tight(frame_bgr, bbox):
    x1, y1, x2, y2 = bbox
    bb_w = x2 - x1
    bb_h = y2 - y1
    mx = int(MARGIN_X * bb_w)
    my = int(MARGIN_Y * bb_h)

    x1m = max(0, x1 - mx)
    y1m = max(0, y1 - my)
    x2m = min(frame_bgr.shape[1] - 1, x2 + mx)
    y2m = min(frame_bgr.shape[0] - 1, y2 + my)
    return frame_bgr[y1m:y2m, x1m:x2m]

def ocr_plate_fast_on_crop(crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0:
        return None

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)

    # fast: moderate upscale (avoid 4x)
    gray = cv2.resize(gray, None, fx=2.4, fy=2.4, interpolation=cv2.INTER_CUBIC)

    # fast: one blur + one otsu
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # try polarity both ways (2 passes total)
    for img in (bin_img, 255 - bin_img):
        raw = pytesseract.image_to_string(img, config=CONFIG_TESS)
        norm = normalize_plate_il_strict(raw)
        if norm:
            return norm
    return None

# ------------------------
# Perception (fast bbox always; OCR optional; OCR only on crop)
# ------------------------
class Perception:
    def __init__(self, ocr_enabled: bool, ocr_every_n_frames: int):
        self.ocr_enabled = bool(ocr_enabled)
        self.ocr_every_n_frames = max(1, int(ocr_every_n_frames))

        self.last_infer_time = 0.0
        self.detections = []

        self.last_bbox = None          # (x1,y1,x2,y2,u,v,yaw_deg)
        self.detect_frame_count = 0    # counts only valid detected frames

        self.plate_text = None         # last successful strict OCR
        self.plate_text_t = 0.0

    def process_frame(self, frame, now):
        fh, fw = frame.shape[:2]

        if now - self.last_infer_time >= INFER_PERIOD_S:
            outputs = run_onnx_inference(frame)
            self.detections = decode_yolov8(outputs, fw, fh,
                                            conf_thres=CONF_THRES,
                                            iou_thres=IOU_THRES)
            self.last_infer_time = now

        vis = frame.copy()
        yaw_deg = None
        drew = False

        if self.detections:
            best = max(self.detections, key=lambda d: d["confidence"])
            x1, y1, x2, y2 = best["bbox"]

            bb_w = x2 - x1
            bb_h = y2 - y1
            aspect = bb_w / max(1, bb_h)

            if bb_w >= 40 and bb_h >= 15 and (2.0 <= aspect <= 7.0):
                u, v = bbox_center(best["bbox"])
                yaw_deg, _ = pixel_to_angles(u, v, K)

                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(vis, (int(u), int(v)), 4, (0, 0, 255), -1)
                drew = True

                self.last_bbox = (x1, y1, x2, y2, u, v, yaw_deg)

                self.detect_frame_count += 1
                if self.ocr_enabled and (self.detect_frame_count % self.ocr_every_n_frames == 0):
                    crop = crop_plate_tight(frame, (x1, y1, x2, y2))
                    try:
                        plate = ocr_plate_fast_on_crop(crop)
                    except Exception:
                        plate = None

                    if plate:
                        self.plate_text = plate
                        self.plate_text_t = now
                        print(f"[PLATE] {plate}", flush=True)

        if (not drew) and (self.last_bbox is not None):
            x1, y1, x2, y2, u, v, yaw_deg = self.last_bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(vis, (int(u), int(v)), 4, (0, 0, 255), -1)

        return vis, yaw_deg, self.plate_text, self.plate_text_t, self.ocr_enabled

# ------------------------
# HUD
# ------------------------
def draw_hud(frame, yaw_deg, plate_text, plate_text_t, now, ocr_enabled, ocr_every_n):
    vis = frame.copy()
    h, w = vis.shape[:2]
    bar_h = int(h * 0.20)
    font = cv2.FONT_HERSHEY_SIMPLEX

    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)

    yaw_disp = 0.0 if yaw_deg is None else float(np.clip(yaw_deg, -45, 45))
    cv2.putText(vis, f"Yaw: {yaw_disp:.1f} deg", (10, bar_h - 10),
                font, 0.7, (0, 255, 255), 2)

    cv2.putText(vis, f"OCR: {'ON' if ocr_enabled else 'OFF'} (every {ocr_every_n} det frames)",
                (10, 22), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    if ocr_enabled and plate_text:
        age = now - (plate_text_t or now)
        label = f"PLATE: {plate_text} ({age:.1f}s)"
        (tw, th), base = cv2.getTextSize(label, font, 0.75, 2)
        x, y = 10, bar_h + 40
        pad = 6
        tl = (x - pad, y - th - pad)
        br = (x + tw + pad, y + base + pad)
        cv2.rectangle(vis, tl, br, (0, 255, 255), -1)
        cv2.rectangle(vis, tl, br, (0, 0, 0), 2)
        cv2.putText(vis, label, (x, y), font, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
    elif ocr_enabled:
        cv2.putText(vis, "PLATE: ----", (10, bar_h + 40),
                    font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return vis

# ------------------------
# MAIN
# ------------------------
def main(args):
    free_camera()
    cap = open_camera()
    if cap is None:
        return

    percep = Perception(
        ocr_enabled=not args.no_ocr,
        ocr_every_n_frames=args.ocr_every_n
    )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.005)
                continue

            now = time.time()
            vis, yaw_deg, plate_text, plate_text_t, ocr_enabled = percep.process_frame(frame, now)

            vis_hud = draw_hud(vis, yaw_deg, plate_text, plate_text_t, now,
                               ocr_enabled=ocr_enabled,
                               ocr_every_n=percep.ocr_every_n_frames)
            publish_frame(vis_hud)

            if args.mode == "debug":
                cv2.imshow("plate_detect_fast_crop_ocr", vis_hud)
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q') or k == 27:
                    break

    finally:
        cap.release()
        _stop_event.set()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["remote", "debug"], default="debug")
    p.add_argument("--no_ocr", action="store_true", help="Disable OCR completely (fastest).")
    p.add_argument("--ocr_every_n", type=int, default=OCR_TRY_EVERY_N_FRAMES_DEFAULT,
                   help="Run OCR once every N detected frames (default: 6).")
    args = p.parse_args()

    t = threading.Thread(target=main, args=(args,), daemon=True)
    t.start()

    app.run(host="0.0.0.0", port=8080, threaded=True)
