#!/usr/bin/env python
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
import argparse

from YB_Pcb_Car_control import YB_Pcb_Car  # import the class directly


from plate_format.plate_format_ro import is_valid_plate, normalize_plate_format

# ------------------------
# Tesseract configuration
# ------------------------
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

CAMERA_DEV = "/dev/video0"

# ------------------------
# Configuration
# ------------------------
ONNX_PATH = "/home/pi/Yahboom_project/uveye/code/raspberry-plate-recognition/yolov8n-license_plate.onnx"
CONFIG_TESSERACT = (
    "--psm 6 --oem 1 "
    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)

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

# ------------------------
# Shared helpers
# ------------------------
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
# PERCEPTION CLASS
# ------------------------
class Perception:
    def __init__(self, K, conf_thres, iou_thres,
                 infer_period_s, ocr_period_s, plate_dedupe_seconds):
        self.K = K
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.infer_period_s = infer_period_s
        self.ocr_period_s = ocr_period_s
        self.plate_dedupe_seconds = plate_dedupe_seconds

        self.last_infer_time = 0.0
        self.last_ocr_time = 0.0
        self.last_seen_plate = {}  # plate -> timestamp
        self.detections = []

    def update_detections(self, frame, now):
        if now - self.last_infer_time < self.infer_period_s:
            return  # reuse previous self.detections

        self.last_infer_time = now
        fh, fw = frame.shape[:2]

        outputs = run_onnx_inference(frame)
        self.detections = decode_yolov8(outputs, fw, fh,
                                        conf_thres=self.conf_thres,
                                        iou_thres=self.iou_thres)
        print(f"[PERCEPTION] {len(self.detections)} detection(s) this frame.")

    def expire_old_plates(self, now):
        for p in list(self.last_seen_plate.keys()):
            if now - self.last_seen_plate[p] > self.plate_dedupe_seconds:
                del self.last_seen_plate[p]

    def process_frame(self, frame, now):
        """
        Returns:
          vis_frame, yaw_deg, pitch_deg, has_detection (bool)
        """
        self.expire_old_plates(now)
        self.update_detections(frame, now)

        vis = frame.copy()
        fh, fw = frame.shape[:2]

        for det in self.detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                vis, f"{conf:.2f}", (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
            )

        if not self.detections:
            return vis, None, None, False

        best = max(self.detections, key=lambda d: d["confidence"])
        u, v = bbox_center(best["bbox"])
        yaw_deg, pitch_deg = pixel_to_angles(u, v, self.K)

        print(f"[PERCEPTION] Plate center pixel: u={u:.1f}, v={v:.1f}")
        print(f"[PERCEPTION] Angles: yaw={yaw_deg:.2f} deg, pitch={pitch_deg:.2f} deg")

        cv2.circle(vis, (int(u), int(v)), 4, (0, 0, 255), -1)
        cv2.putText(
            vis, f"yaw={yaw_deg:.1f} deg",
            (int(u) + 5, int(v) - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
        )

        # OCR throttled
        if now - self.last_ocr_time >= self.ocr_period_s:
            self.last_ocr_time = now
            crop = crop_with_margin(frame, best["bbox"], margin=0.10)
            if crop is not None:
                plate = extract_valid_plate(crop)
                if plate and plate not in self.last_seen_plate:
                    self.last_seen_plate[plate] = now
                    print(f"[PERCEPTION] [{time.strftime('%H:%M:%S')}] License Plate: {plate}")

        return vis, yaw_deg, pitch_deg, True

# ------------------------
# LOW-LEVEL CAR CONTROL
# ------------------------
class CarControl:
    """
    Low-level interface to the YB_Pcb_Car.
    Expects wheel commands in [-1, 1] and maps to motor PWM.
    """
    def __init__(self, max_pwm=120):
        self.car = YB_Pcb_Car()
        self.max_pwm = max_pwm

    def stop(self):
        self.car.Car_Stop()

    def drive(self, left_norm, right_norm):
        """
        left_norm, right_norm in [-1, 1]
        """
        left_norm = max(-1.0, min(1.0, left_norm))
        right_norm = max(-1.0, min(1.0, right_norm))

        left_pwm = int(left_norm * self.max_pwm)
        right_pwm = int(right_norm * self.max_pwm)

        # Use Control_Car which supports signed speeds
        self.car.Control_Car(left_pwm, right_pwm)

# ------------------------
# PATH PLANNING CLASS
# ------------------------
class PathPlanner:
    """
    High-level planner:
    - Uses yaw angle from Perception to generate (steer, speed).
    - Drives forward toward plate center.
    - Stops if plate is not seen for > max_missed_frames frames.
    """
    def __init__(self, base_speed=100, deadband=1.0, Kp=0.05, max_missed_frames=10,
                 move_enabled=True):
        self.base_speed = base_speed
        self.deadband = deadband
        self.Kp = Kp
        self.max_missed_frames = max_missed_frames
        self.move_enabled = move_enabled

        self.missed_frames = 0
        self.last_yaw = 0.0

    def toggle_move(self):
        self.move_enabled = not self.move_enabled
        print(f"[PATH] move_enabled -> {self.move_enabled}")

    def _compute_steering_from_yaw(self, yaw_deg):
        if yaw_deg is None:
            return 0.0

        if abs(yaw_deg) < self.deadband:
            return 0.0

        steer = self.Kp * yaw_deg
        steer = max(-1.0, min(1.0, steer))
        return steer

    def step(self, yaw_deg, has_detection):
        """
        Returns (steer, speed) in [-1,1] and [0,1].

        Behaviour:
        - If plate is detected: reset missed_frames, compute steer from yaw,
          move forward at base_speed.
        - If plate is NOT detected:
            - For the first max_missed_frames frames, keep going forward
              using last_yaw (to avoid micro drop-outs).
            - After that: speed = 0 (full stop).
        - If move_enabled is False: speed forced to 0.
        """
        if has_detection:
            self.missed_frames = 0
            self.last_yaw = yaw_deg if yaw_deg is not None else 0.0
            speed = self.base_speed
            steer = self._compute_steering_from_yaw(yaw_deg)
        else:
            self.missed_frames += 1
            print(f"[PATH] No plate detected. missed_frames={self.missed_frames}")

            if self.missed_frames > self.max_missed_frames:
                print("[PATH] Plate lost for too long -> stopping car.")
                speed = 0.0
                steer = 0.0
            else:
                speed = self.base_speed
                steer = self._compute_steering_from_yaw(self.last_yaw)

        if not self.move_enabled:
            speed = 0.0

        return steer, speed

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

# ------------------------
# MAIN LOOP
# ------------------------
def main(args):
    free_camera()

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        raise SystemExit("Error: Could not open camera (is /dev/video0 busy?)")

    perception = Perception(
        K=K,
        conf_thres=CONF_THRES,
        iou_thres=IOU_THRES,
        infer_period_s=INFER_PERIOD_S,
        ocr_period_s=OCR_PERIOD_S,
        plate_dedupe_seconds=PLATE_DEDUPE_SECONDS,
    )

    car_control = CarControl(max_pwm=120)  # adjust max_pwm if needed

    planner = PathPlanner(
        base_speed=args.speed,
        deadband=1.0,
        Kp=0.05,
        max_missed_frames=MAX_MISSED_FRAMES,
        move_enabled=args.auto_move,
    )

    print("[INFO] Press 'm' to toggle move_enabled, 'q' or ESC to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] Failed to read frame from camera.")
                time.sleep(0.05)
                continue

            now = time.time()

            vis, yaw_deg, pitch_deg, has_detection = perception.process_frame(frame, now)
        
            # High-level planning: get desired steer/speed
            steer, speed = planner.step(yaw_deg, has_detection)
         
            # Convert (steer, speed) -> left/right wheel in [-1,1]
            if speed <= 0.0:
                car_control.stop()
            else:
                left = speed * (1.0 - steer)
                right = speed * (1.0 + steer)
                car_control.drive(left, right)

            cv2.imshow("camera (q/esc to quit, m=toggle move)", vis)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                print("[INFO] Quit requested.")
                break
            elif k == ord("m"):
                planner.toggle_move()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        # Safety stop
        planner.move_enabled = False
        car_control.stop()
        print("[INFO] Camera and motors released.")

# ------------------------
# ENTRY POINT
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--auto-move",
        action="store_true",
        help="Start with movement enabled (otherwise you must press 'm')."
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=100,
        help="Base forward speed in [0,1] when tracking a plate."
    )
    args = parser.parse_args()

    main(args)
