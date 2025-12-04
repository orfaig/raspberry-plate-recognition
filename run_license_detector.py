# main script (fixed) good! Roy

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"   # must be set before importing cv2

import time
import sys
import cv2
import pytesseract
import onnx
import onnxruntime as onnxr
import numpy as np
from plate_format.plate_format_ro import is_valid_plate, normalize_plate_format
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
import subprocess

#CAM_DEVICES = ["/dev/video0", "/dev/video1", "/dev/video2"]
CAMERA_DEV = "/dev/video0"

# ------------------------
# Configuration
# ------------------------
ONNX_PATH = "yolov8n-license_plate.onnx"
CONFIG_TESSERACT = (
    "--psm 6 --oem 1 "
    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)

CONF_THRES = 0.2
IOU_THRES = 0.05
INFER_PERIOD_S = 0.1
OCR_PERIOD_S = 1.0
PLATE_DEDUPE_SECONDS = 0

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

def bbox_center(bbox):
    """Return (u, v) = center pixel of a bbox = (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = bbox
    u = 0.5 * (x1 + x2)
    v = 0.5 * (y1 + y2)
    return u, v


def pixel_to_angles(u, v, K):
    """
    Convert pixel (u, v) to yaw/pitch angles (in degrees)
    relative to camera optical axis, using intrinsics K.
    Positive yaw = to the right, positive pitch = down if v>cy.
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    x = (u - cx) / fx
    y = (v - cy) / fy

    yaw_rad = np.arctan(x)      # horizontal angle
    pitch_rad = np.arctan(y)    # vertical angle (simple model)

    yaw_deg = np.degrees(yaw_rad)
    pitch_deg = np.degrees(pitch_rad)
    return yaw_deg, pitch_deg


# Model input shape is typically [1, 3, H, W] and in your case H=W=512
def _int_or_none(x):
    return x if isinstance(x, int) else None

MODEL_H = _int_or_none(inp.shape[2]) or 512
MODEL_W = _int_or_none(inp.shape[3]) or 512

# ------------------------
# Utilities
# ------------------------
def iou_xyxy(a, b):
    # a: (4,), b: (N,4) => returns (N,)
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
    # YOLOv8 exports typically expect RGB NCHW float32 in 0..1
    resized = cv2.resize(frame_bgr, (MODEL_W, MODEL_H), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))          # HWC -> CHW
    x = np.expand_dims(x, axis=0)           # -> NCHW
    return session.run(output_names, {input_name: x})

def decode_yolov8(outputs, frame_w, frame_h, conf_thres=0.25, iou_thres=0.6):
    """
    Fixed decoder:
    - Accepts YOLOv8 exports with C=5 (1-class, no obj): [cx,cy,w,h,cls0]
    - Accepts exports with C>=6 (with obj): [cx,cy,w,h,obj,cls...]
    - Handles (1,C,N), (1,N,C), (C,N), (N,C)
    - Auto-sigmoid if scores look like logits
    - Handles normalized box coords (0..1) vs pixel coords
    """

    out = outputs[0]
    pred = np.squeeze(out)

    # Normalize to 2D
    if pred.ndim == 3:
        pred = pred[0]
    if pred.ndim != 2:
        return []

    # Make it (N, C)
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T

    N, C = pred.shape
    if C < 5:
        return []

    boxes_xywh = pred[:, 0:4]

    # Decide whether we have an explicit objectness column
    has_obj = (C >= 6)

    if has_obj:
        obj = pred[:, 4]
        cls_scores = pred[:, 5:]
        if cls_scores.shape[1] == 0:
            cls_scores = np.ones((N, 1), dtype=pred.dtype)
        # If logits, sigmoid
        if obj.max() > 1.2 or obj.min() < -0.2 or cls_scores.max() > 1.2 or cls_scores.min() < -0.2:
            obj = 1.0 / (1.0 + np.exp(-obj))
            cls_scores = 1.0 / (1.0 + np.exp(-cls_scores))
        cls_id = np.argmax(cls_scores, axis=1)
        cls_conf = cls_scores[np.arange(N), cls_id]
        conf = obj * cls_conf
    else:
        # No obj: scores start at column 4
        cls_scores = pred[:, 4:]
        if cls_scores.shape[1] == 0:
            return []
        if cls_scores.max() > 1.2 or cls_scores.min() < -0.2:
            cls_scores = 1.0 / (1.0 + np.exp(-cls_scores))
        cls_id = np.argmax(cls_scores, axis=1) if cls_scores.shape[1] > 1 else np.zeros((N,), dtype=np.int64)
        conf = cls_scores.max(axis=1) if cls_scores.shape[1] > 1 else cls_scores[:, 0]

    # Filter by confidence
    m = conf >= conf_thres
    if not np.any(m):
        return []

    boxes_xywh = boxes_xywh[m]
    conf = conf[m]
    cls_id = cls_id[m]

    # If coords look normalized, scale to model input size
    if boxes_xywh[:, 0].max() <= 2.0 and boxes_xywh[:, 2].max() <= 2.0:
        boxes_xywh = boxes_xywh.copy()
        boxes_xywh[:, [0, 2]] *= float(MODEL_W)
        boxes_xywh[:, [1, 3]] *= float(MODEL_H)

    # xywh -> xyxy
    cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # NMS
    keep = nms_xyxy(boxes_xyxy, conf, iou_thres=iou_thres)
    boxes_xyxy = boxes_xyxy[keep]
    conf = conf[keep]
    cls_id = cls_id[keep]

    # Scale to frame size (assumes you resized directly to MODEL_{W,H} without letterbox)
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
# def decode_yolov8(outputs, frame_w, frame_h, conf_thres=0.25, iou_thres=0.6):
#     # Supports common YOLOv8 ONNX shapes:
#     #   (1, 4+nc, N) or (1, N, 4+nc) or (4+nc, N) or (N, 4+nc)
#     out = outputs[0]
#     pred = np.squeeze(out)

#     if pred.ndim != 2:
#         return []

#     # Make it (N, 4+nc)
#     if pred.shape[0] < pred.shape[1]:
#         pred = pred.T

#     if pred.shape[1] < 6:  # must be at least 4 + 1 class prob (typical nc>=1)
#         return []

#     boxes_xywh = pred[:, 0:4]
#     cls_scores = pred[:, 4:]  # YOLOv8: class probabilities (no separate objectness)

#     conf = cls_scores.max(axis=1)
#     cls_id = cls_scores.argmax(axis=1)

#     m = conf >= conf_thres
#     if not np.any(m):
#         return []

#     boxes_xywh = boxes_xywh[m]
#     conf = conf[m]
#     cls_id = cls_id[m]

#     # xywh (center-based) in model-input pixel coordinates -> xyxy
#     cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
#     x1 = cx - w / 2.0
#     y1 = cy - h / 2.0
#     x2 = cx + w / 2.0
#     y2 = cy + h / 2.0
#     boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

#     # NMS (class-agnostic; fine for 1-class “plate” models)
#     keep = nms_xyxy(boxes_xyxy, conf, iou_thres=iou_thres)
#     boxes_xyxy = boxes_xyxy[keep]
#     conf = conf[keep]
#     cls_id = cls_id[keep]

#     # Scale boxes to frame size
#     sx = frame_w / float(MODEL_W)
#     sy = frame_h / float(MODEL_H)
#     boxes_xyxy[:, [0, 2]] *= sx
#     boxes_xyxy[:, [1, 3]] *= sy

#     detections = []
#     for b, c, k in zip(boxes_xyxy, conf, cls_id):
#         x1, y1, x2, y2 = b
#         x1 = int(max(0, min(frame_w - 1, x1)))
#         y1 = int(max(0, min(frame_h - 1, y1)))
#         x2 = int(max(0, min(frame_w - 1, x2)))
#         y2 = int(max(0, min(frame_h - 1, y2)))
#         if x2 <= x1 or y2 <= y1:
#             continue
#         detections.append({"bbox": (x1, y1, x2, y2), "confidence": float(c), "class_id": int(k)})
#     return detections

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
# Main
# ------------------------
def display_camera_with_detection():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # index 0 -> /dev/video0

    # Known-good webcam mode
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        raise SystemExit("Error: Could not open camera (is /dev/video0 busy?)")

    last_infer_time = 0.0
    last_ocr_time = 0.0
    last_seen_plate = {}  # plate -> timestamp

    detections = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            now = time.time()
            fh, fw = frame.shape[:2]

            # expire old dedupe entries
            for p in list(last_seen_plate.keys()):
                if now - last_seen_plate[p] > PLATE_DEDUPE_SECONDS:
                    del last_seen_plate[p]

            # inference
            outputs = run_onnx_inference(frame)
            out = outputs[0]
            pred = np.squeeze(out)
            
            boxes_xywh = pred[:, 0:4]
            
            cls_scores = pred[:, 4:] 
           # print(f"check: {boxes_xywh[0]}", flush=True)
    #        detections = decode_yolov8(outputs, fw, fh, conf_thres=CONF_THRES, iou_thres=IOU_THRES)
            if now - last_infer_time >= INFER_PERIOD_S:
                last_infer_time = now
                outputs = run_onnx_inference(frame)
                detections = decode_yolov8(outputs, fw, fh, conf_thres=CONF_THRES, iou_thres=IOU_THRES)

            # draw detections
            vis = frame.copy()
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                conf = det["confidence"]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    vis, f"{conf:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
                )

            # OCR throttling: run on best detection only

            if detections and (now - last_ocr_time >= OCR_PERIOD_S):
                last_ocr_time = now
                best = max(detections, key=lambda d: d["confidence"])
                  # Compute direction of plate center
                u, v = bbox_center(best["bbox"])
                yaw_deg, pitch_deg = pixel_to_angles(u, v, K)

                # This is the key value you will later send to the robot
                # yaw_deg > 0 => plate is to the right of the heading
                print(f"Plate center angle: yaw={yaw_deg:.2f} deg, pitch={pitch_deg:.2f} deg", flush=True)
                cv2.circle(vis, (int(u), int(v)), 4, (0, 0, 255), -1)
                cv2.putText(vis, f"yaw={yaw_deg:.1f} deg",(int(u) + 5, int(v) - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)


                crop = crop_with_margin(frame, best["bbox"], margin=0.10)
                if crop is not None:
                    plate = extract_valid_plate(crop)
                    if plate and plate not in last_seen_plate:
                        print(f"[{time.strftime('%H:%M:%S')}] License Plate: {plate}", flush=True)
                        last_seen_plate[plate] = now

            cv2.imshow("camera (q/esc to quit)", vis)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

def free_camera():
    """
    Kill any process that is using /dev/video0.
    Does NOT open the camera.
    Does NOT reload drivers.
    Safe to run before every script start.
    """
    if not os.path.exists(CAMERA_DEV):
        print(f"[WARN] {CAMERA_DEV} does not exist. Nothing to free.")
        return

    print(f"[INFO] Checking for processes using {CAMERA_DEV}...")

    try:
        # fuser prints PIDs that are using the device
        out = subprocess.check_output(
            ["fuser", CAMERA_DEV],
            stderr=subprocess.STDOUT
        )
        pids = out.decode().strip().split()
    except subprocess.CalledProcessError:
        # No PIDs → device is NOT busy
        print(f"[INFO] {CAMERA_DEV} is free.")
        return

    if not pids:
        print(f"[INFO] {CAMERA_DEV} is free.")
        return

    print(f"[INFO] {CAMERA_DEV} is BUSY. Killing: {pids}")

    for pid in pids:
        subprocess.run(["kill", "-9", pid], check=False)

    print("[INFO] Camera freed successfully.")

if __name__ == "__main__":
    #cam_index = reset_camera()
    free_camera()
    display_camera_with_detection()

    # Camera intrinsics for the resolution used (here 640x480).
    # Replace these with your calibrated values.
    FX = 1330.0   # example
    FY = 1330.0   # example
    CX = 960.0   # example: fw/2
    CY = 540.0   # example: fh/2


    K = np.array([[FX,   0.0, CX],
                [0.0,  FY,  CY],
                [0.0,  0.0, 1.0]], dtype=np.float32)
    

    scale_x = 640.0 / W_cal
    scale_y = 480.0 / H_cal
    FX = fx_cal * scale_x
    FY = fy_cal * scale_y
    CX = cx_cal * scale_x
    CY = cy_cal * scale_y