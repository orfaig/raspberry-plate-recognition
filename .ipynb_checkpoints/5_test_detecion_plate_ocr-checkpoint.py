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
    bbox_center,
    run_onnx_inference,
    decode_yolov8,
    free_camera
)

# ======================
# TARGET PLATE
# ======================
TARGET_PLATE = "91-179-18"

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
# DETECTION (FAST)
# ======================
CONF_THRES = 0.2
IOU_THRES = 0.2
INFER_PERIOD_S = 0.03

# ======================
# OCR
# ======================
OCR_EVERY_N_FRAMES = 3
MARGIN_X = 0.10
MARGIN_Y = 0.25

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# ======================
# ONNX
# ======================
ONNX_PATH = "/home/pi/Yahboom_project/uveye/code/raspberry-plate-recognition/yolov8n-license_plate.onnx"
onnxr.set_default_logger_severity(3)
session = onnxr.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

# ======================
# FLASK
# ======================
app = Flask(__name__)
_latest = None
_lock = threading.Lock()
_stop = threading.Event()

def publish(frame):
    global _latest
    ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if ok:
        with _lock:
            _latest = jpg.tobytes()

def stream():
    while not _stop.is_set():
        with _lock:
            data = _latest
        if data:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
        time.sleep(0.03)

@app.route("/")
def index():
    return "<h3>Plate Detection</h3><img src='/video'>"

@app.route("/video")
def video():
    return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

# ======================
# OCR HELPERS
# ======================
def normalize(raw):
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

def ocr_plate(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)
    gray = cv2.resize(gray, None, fx=2.4, fy=2.4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    for img in (bin_img, 255 - bin_img):
        text = pytesseract.image_to_string(img, config=CONFIG_TESS)
        plate = normalize(text)
        if plate:
            return plate
    return None

# ======================
# PERCEPTION
# ======================
class Perception:
    def __init__(self):
        self.last_infer = 0
        self.detections = []
        self.last_box = None
        self.frame_count = 0
        self.target_found = False
        self.plate = None

    def process(self, frame, now):
        if now - self.last_infer >= INFER_PERIOD_S:
            out = run_onnx_inference(frame)
            self.detections = decode_yolov8(
                out, frame.shape[1], frame.shape[0],
                CONF_THRES, IOU_THRES
            )
            self.last_infer = now

        vis = frame.copy()

        # draw last box fallback
        if not self.detections and self.last_box:
            x1, y1, x2, y2 = self.last_box
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if self.detections:
            best = max(self.detections, key=lambda d: d["confidence"])
            x1, y1, x2, y2 = best["bbox"]

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            self.last_box = (x1, y1, x2, y2)

            # ===== STOP OCR forever after target found =====
            if not self.target_found:
                self.frame_count += 1
                if self.frame_count % OCR_EVERY_N_FRAMES == 0:
                    crop = crop_plate(frame, (x1, y1, x2, y2))
                    plate = ocr_plate(crop)

                    # store last OCR guess (can be wrong) if you want to show it
                    if plate:
                        self.last_ocr = plate

                    # lock ONLY if target matches
                    if plate == TARGET_PLATE:
                        self.plate = plate
                        self.target_found = True
                        print(f"[TARGET FOUND] {plate}", flush=True)

        # ===== DISPLAY LOGIC =====
        # show target if found; otherwise show last OCR guess (optional)
        if self.target_found:
            cv2.putText(vis, f"TARGET: {self.plate}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0), 3)
        else:
            # if you DON'T want wrong guesses displayed, comment this block out
            txt = getattr(self, "last_ocr", None)
            if txt:
                cv2.putText(vis, f"OCR: {txt}",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 255, 255), 3)
            else:
                cv2.putText(vis, "OCR: ----",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (255, 255, 255), 3)

        return vis


# ======================
# MAIN
# ======================
def main():
    free_camera()
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    percep = Perception()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        vis = percep.process(frame, time.time())
        publish(vis)

        cv2.imshow("plate", vis)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    _stop.set()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    threading.Thread(target=main, daemon=True).start()
    app.run(host="0.0.0.0", port=8080, threaded=True)
