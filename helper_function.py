#------------------------
# Shared helpers
# ------------------------

# coding: utf-8
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"   # must be set before importing cv2

import time
import sys
import cv2
import pytesseract
import onnx
import onnxruntime as onnxr
import numpy as np
import subprocess


from YB_Pcb_Car_control import YB_Pcb_Car  # import the class directly


from plate_format.plate_format_ro import is_valid_plate, normalize_plate_format
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

CAMERA_DEV = "/dev/video0"

# ------------------------
# Configuration
# ------------------------
ONNX_PATH = "/home/pi/Yahboom_project/uveye/code/raspberry-plate-recognition/yolov8n-license_plate.onnx"
CONFIG_TESSERACT = (
    "--psm 6 --oem 1 "
    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


CONF_THRES = 0.2
IOU_THRES = 0.05
INFER_PERIOD_S = 0.1
OCR_PERIOD_S = 1.0
PLATE_DEDUPE_SECONDS = 0
MAX_MISSED_FRAMES = 10  # stop car if plate not seen for this many frames

# Approximate camera intrinsics for 640x480 (REPLACE with your calibrated values)
FX = 450.0
FY = 600.0
CX = 320.0
CY = 240.0

K = np.array([[FX,   0.0, CX],
              [0.0,  FY,  CY],
              [0.0,  0.0, 1.0]], dtype=np.float32)

print("[INFO] Camera intrinsics K:")
print(K)

# CLAHE preprocessing (contrast enhancement)
clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))

# ------------------------
# ONNX Runtime: CPU only + quieter logs
# ------------------------
onnxr.set_default_logger_severity(3)  # 0=verbose ... 3=warning ... 4=error

onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)

session = onnxr.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
inp = session.get_inputs()[0]
input_name = inp.name
output_names = [out.name for out in session.get_outputs()]

def _int_or_none(x):
    return x if isinstance(x, int) else None

MODEL_H = _int_or_none(inp.shape[2]) or 512
MODEL_W = _int_or_none(inp.shape[3]) or 512

print(f"[INFO] YOLO model input size: {MODEL_W}x{MODEL_H}")


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    u = 0.5 * (x1 + x2)
    v = 0.5 * (y1 + y2)
    return u, v

def pixel_to_angles(u, v, K):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    x = (u - cx) / fx
    y = (v - cy) / fy

    yaw_rad = np.arctan(x)
    pitch_rad = np.arctan(y)

    yaw_deg = np.degrees(yaw_rad)
    pitch_deg = np.degrees(pitch_rad)
    return yaw_deg, pitch_deg

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)

    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)

def nms_xyxy(boxes, scores, iou_thres=0.6):
    if len(boxes) == 0:
        return []
    boxes = boxes.astype(np.float32)
    scores = scores.astype(np.float32)

    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = iou_xyxy(boxes[i], boxes[rest])
        order = rest[ious <= iou_thres]
    return keep

def preprocess_plate(plate_crop):
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)
    blur = cv2.bilateralFilter(gray, 11, 16, 16)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return morph

def extract_valid_plate(plate_crop):
    raw_text = pytesseract.image_to_string(
        preprocess_plate(plate_crop), config=CONFIG_TESSERACT
    )
    raw_text = raw_text.strip().replace("\n", " ").replace("\f", "")
    raw_text = "".join(c for c in raw_text if c.isalnum() or c.isspace())
    if is_valid_plate(raw_text):
        return normalize_plate_format(raw_text)
    return None

def run_onnx_inference(frame_bgr):
    resized = cv2.resize(frame_bgr, (MODEL_W, MODEL_H), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))          # HWC -> CHW
    x = np.expand_dims(x, axis=0)           # -> NCHW
    return session.run(output_names, {input_name: x})

def decode_yolov8(outputs, frame_w, frame_h, conf_thres=0.25, iou_thres=0.6):
    out = outputs[0]
    pred = np.squeeze(out)

    if pred.ndim == 3:
        pred = pred[0]
    if pred.ndim != 2:
        return []

    if pred.shape[0] < pred.shape[1]:
        pred = pred.T

    N, C = pred.shape
    if C < 5:
        return []

    boxes_xywh = pred[:, 0:4]
    has_obj = (C >= 6)

    if has_obj:
        obj = pred[:, 4]
        cls_scores = pred[:, 5:]
        if cls_scores.shape[1] == 0:
            cls_scores = np.ones((N, 1), dtype=pred.dtype)
        if obj.max() > 1.2 or obj.min() < -0.2 or cls_scores.max() > 1.2 or cls_scores.min() < -0.2:
            obj = 1.0 / (1.0 + np.exp(-obj))
            cls_scores = 1.0 / (1.0 + np.exp(-cls_scores))
        cls_id = np.argmax(cls_scores, axis=1)
        cls_conf = cls_scores[np.arange(N), cls_id]
        conf = obj * cls_conf
    else:
        cls_scores = pred[:, 4:]
        if cls_scores.shape[1] == 0:
            return []
        if cls_scores.max() > 1.2 or cls_scores.min() < -0.2:
            cls_scores = 1.0 / (1.0 + np.exp(-cls_scores))
        cls_id = np.argmax(cls_scores, axis=1) if cls_scores.shape[1] > 1 else np.zeros((N,), dtype=np.int64)
        conf = cls_scores.max(axis=1) if cls_scores.shape[1] > 1 else cls_scores[:, 0]

    m = conf >= conf_thres
    if not np.any(m):
        return []

    boxes_xywh = boxes_xywh[m]
    conf = conf[m]
    cls_id = cls_id[m]

    if boxes_xywh[:, 0].max() <= 2.0 and boxes_xywh[:, 2].max() <= 2.0:
        boxes_xywh = boxes_xywh.copy()
        boxes_xywh[:, [0, 2]] *= float(MODEL_W)
        boxes_xywh[:, [1, 3]] *= float(MODEL_H)

    cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    keep = nms_xyxy(boxes_xyxy, conf, iou_thres=iou_thres)
    boxes_xyxy = boxes_xyxy[keep]
    conf = conf[keep]
    cls_id = cls_id[keep]

    sx = frame_w / float(MODEL_W)
    sy = frame_h / float(MODEL_H)
    boxes_xyxy[:, [0, 2]] *= sx
    boxes_xyxy[:, [1, 3]] *= sy

    detections = []
    for b, c, k in zip(boxes_xyxy, conf, cls_id):
        x1, y1, x2, y2 = b
        x1 = int(max(0, min(frame_w - 1, x1)))
        y1 = int(max(0, min(frame_h - 1, y1)))
        x2 = int(max(0, min(frame_w - 1, x2)))
        y2 = int(max(0, min(frame_h - 1, y2)))
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append({"bbox": (x1, y1, x2, y2), "confidence": float(c), "class_id": int(k)})

    return detections

def crop_with_margin(frame, bbox, margin=0.08):
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    dx = int((x2 - x1) * margin)
    dy = int((y2 - y1) * margin)
    x1 = max(0, x1 - dx)
    y1 = max(0, y1 - dy)
    x2 = min(w, x2 + dx)
    y2 = min(h, y2 + dy)
    if x2 - x1 < 20 or y2 - y1 < 10:
        return None
    return frame[y1:y2, x1:x2]


# ------------------------
# CAMERA / SYSTEM UTILITIES
# ------------------------
def free_camera():
    if not os.path.exists(CAMERA_DEV):
        print(f"[WARN] {CAMERA_DEV} does not exist. Nothing to free.")
        return

    print(f"[INFO] Checking for processes using {CAMERA_DEV}...")

    try:
        out = subprocess.check_output(
            ["fuser", CAMERA_DEV],
            stderr=subprocess.STDOUT
        )
        pids = out.decode().strip().split()
    except subprocess.CalledProcessError:
        print(f"[INFO] {CAMERA_DEV} is free.")
        return

    if not pids:
        print(f"[INFO] {CAMERA_DEV} is free.")
        return

    print(f"[INFO] {CAMERA_DEV} is BUSY. Killing: {pids}")
    for pid in pids:
        subprocess.run(["kill", "-9", pid], check=False)

    print("[INFO] Camera freed successfully.")
