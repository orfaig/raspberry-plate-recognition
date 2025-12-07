#!/usr/bin/env python
# coding: utf-8
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import time
import sys
import cv2
import pytesseract
import onnx
import onnxruntime as onnxr
import numpy as np
import subprocess
import argparse

from YB_Pcb_Car_control import YB_Pcb_Car
from remote_fucntion import get_ir_key

from helper_function import (
    bbox_center, pixel_to_angles, iou_xyxy, nms_xyxy,
    preprocess_plate, extract_valid_plate,
    run_onnx_inference, decode_yolov8,
    crop_with_margin, free_camera
)

# ------------------------
# CONFIG
# ------------------------
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

ONNX_PATH = "/home/pi/Yahboom_project/uveye/code/raspberry-plate-recognition/yolov8n-license_plate.onnx"

CONF_THRES = 0.3
IOU_THRES = 0.3
INFER_PERIOD_S = 0.1
OCR_PERIOD_S = 1.0
MAX_MISSED_FRAMES = 10
TARGET_RATIO = 0.50   # width >= 90% of frame width => target reached

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

def _int_or_none(x):
    return x if isinstance(x, int) else None

MODEL_H = _int_or_none(inp.shape[2]) or 512
MODEL_W = _int_or_none(inp.shape[3]) or 512

# ------------------------
# PERCEPTION
# ------------------------
class Perception:
    def __init__(self):
        self.last_infer_time = 0.0
        self.last_ocr_time = 0.0
        self.detections = []

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
            return vis, None, None, False, None

        # best
        best = max(self.detections, key=lambda d: d["confidence"])
        (x1, y1, x2, y2) = best["bbox"]
        bb_width = x2 - x1

        u, v = bbox_center(best["bbox"])
        yaw_deg, pitch_deg = pixel_to_angles(u, v, K)

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(vis, (int(u), int(v)), 4, (0, 0, 255), -1)

        return vis, yaw_deg, pitch_deg, True, bb_width

# ------------------------
# CAR CONTROL
# ------------------------
class CarControl:
    def __init__(self, max_pwm=120):
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
    def __init__(self, base_speed=100, Kp=0.05):
        self.base_speed = base_speed
        self.Kp = Kp
        self.missed = 0
        self.last_yaw = 0.0
        self.active = True

    def reset(self):
        self.missed = 0
        self.last_yaw = 0.0
        self.active = True

    def step(self, yaw_deg, has_detection):
        if not self.active:
            return 0.0, 0.0

        if has_detection:
            self.missed = 0
            if yaw_deg is not None:
                self.last_yaw = yaw_deg
            steer = np.tanh(self.Kp * self.last_yaw)

            # A1 smooth steering:
            # inner wheel slows to 30% of speed at max steering
            smooth_factor = 0.70
            left_pwm  = self.base_speed * (1.0 + steer * smooth_factor)
            right_pwm = self.base_speed * (1.0 - steer * smooth_factor)

        else:
            self.missed += 1
            if self.missed > MAX_MISSED_FRAMES:
                return 0.0, 0.0

            steer = np.tanh(self.Kp * self.last_yaw)
            smooth_factor = 0.70
            left_pwm  = self.base_speed * (1.0 + steer * smooth_factor)
            right_pwm = self.base_speed * (1.0 - steer * smooth_factor)

        # clamp: 0–100 (never reverse)
        left_pwm  = max(0, min(100, left_pwm))
        right_pwm = max(0, min(100, right_pwm))

        return left_pwm, right_pwm

# ------------------------
# HUD DRAW
# ------------------------
def draw_hud(frame, yaw_deg, status_text, left_pwm, right_pwm):
    vis = frame.copy()
    h, w = vis.shape[:2]
    bar_h = int(h * 0.23)
    font = cv2.FONT_HERSHEY_SIMPLEX

    overlay = vis.copy()
    cv2.rectangle(overlay, (0,0), (w,bar_h), (0,0,0), -1)
    vis = cv2.addWeighted(overlay,0.35,vis,0.65,0)

    # arrow
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
    cv2.putText(vis,f"Status: {status_text}",(10,bar_h+25),
                font,0.7,(255,255,255),2)

    wheel_text = f"L:{left_pwm:.0f} R:{right_pwm:.0f}"
    cv2.putText(vis,wheel_text,(w-200,bar_h+25),
                font,0.7,(0,255,255),2)

    return vis

# ------------------------
# IR WAIT
# ------------------------
def wait_for_play():
    KEY_PLAY = 0x15
    print("[IR] Waiting for PLAY...")
    while True:
        key = get_ir_key()
        if key == KEY_PLAY:
            print("[IR] PLAY pressed, resuming.")
            return
        time.sleep(0.02)

# ------------------------
# MAIN LOOP
# ------------------------
def main(args):
    free_camera()
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    cap.set(cv2.CAP_PROP_FPS,30)

    perception = Perception()
    car = CarControl()
    planner = PathPlanner(base_speed=args.speed, Kp=0.05)

    wait_for_play()

    TARGET_REACHED = False

    try:
        while True:

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            now = time.time()
            vis, yaw_deg, pitch_deg, has_det, bb_w = perception.process_frame(frame, now)
            h, w = frame.shape[:2]

            # check target reached
            if has_det and bb_w is not None and bb_w >= TARGET_RATIO * w:
                TARGET_REACHED = True
                car.stop()
                planner.active = False
                status = "GET THE TARGET!"
                left_pwm = 0
                right_pwm = 0

                vis_hud = draw_hud(vis, yaw_deg, status, left_pwm, right_pwm)
                cv2.imshow("camera", vis_hud)

                # wait for PLAY, then reset
                wait_for_play()
                planner.reset()
                TARGET_REACHED = False
                continue

            # normal tracking
            left_pwm, right_pwm = planner.step(yaw_deg, has_det)

            if left_pwm == 0 and right_pwm == 0:
                status = "STOPPED"
                car.stop()
            else:
                status = "ACTIVE"
                car.drive(left_pwm/100.0, right_pwm/100.0)

            vis_hud = draw_hud(vis, yaw_deg, status, left_pwm, right_pwm)
            cv2.imshow("camera", vis_hud)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q') or k == 27:
                break

    finally:
        car.stop()
        cap.release()
        cv2.destroyAllWindows()

# ------------------------
# ENTRY
# ------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--speed", type=float, default=100)
    args = p.parse_args()
    main(args)
