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
from remote_fucntion import get_ir_key

from plate_format.plate_format_ro import is_valid_plate, normalize_plate_format

from helper_function import (
    bbox_center, pixel_to_angles, iou_xyxy, nms_xyxy,
    preprocess_plate, extract_valid_plate,
    run_onnx_inference, decode_yolov8,
    crop_with_margin, free_camera
)

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

CONF_THRES = 0.3
IOU_THRES = 0.3
INFER_PERIOD_S = 0.1
OCR_PERIOD_S = 1.0
PLATE_DEDUPE_SECONDS = 0
MAX_MISSED_FRAMES = 10  # stop car if plate not seen for this many frames

# Approximate camera intrinsics for 640x480
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
        # cv2.putText(
        #     vis, f"yaw={yaw_deg:.1f} deg",
        #     (int(u) + 5, int(v) - 5),
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
        # )

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
# HUD DRAWING
# ------------------------
def draw_hud(frame, yaw_deg, status_text, left_pwm, right_pwm):
    vis = frame.copy()
    h, w = vis.shape[:2]
    bar_h = int(h * 0.23)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # top dark bar
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)

    # left/right labels
    heading_y = int(bar_h * 0.35)
    cv2.putText(vis, "LEFT", (10, heading_y), font, 0.7, (0, 200, 0), 2, cv2.LINE_AA)
    cv2.putText(vis, "RIGHT", (w - 90, heading_y), font, 0.7, (0, 200, 0), 2, cv2.LINE_AA)

    # yaw arrow
    cx = w // 2
    cy = int(bar_h * 0.70)
    arrow_len = int(bar_h * 0.6)

    if yaw_deg is None:
        yaw_disp = 0.0
    else:
        yaw_disp = max(-45.0, min(45.0, float(yaw_deg)))

    theta = np.radians(yaw_disp)
    ex = int(cx + arrow_len * np.sin(theta))
    ey = int(cy - arrow_len * np.cos(theta))
    cv2.arrowedLine(vis, (cx, cy), (ex, ey), (0, 255, 0), 4, tipLength=0.25)

    # yaw text
    cv2.putText(vis, f"Yaw: {yaw_disp:.1f} deg",
                (10, bar_h - 8), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    # status text
    cv2.putText(vis, f"Status: {status_text}",
                (10, bar_h + 25), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # NEW WHEEL SPEED DISPLAY
    wheel_text = f"L: {left_pwm:.0f}   R: {right_pwm:.0f}"
    cv2.putText(vis, wheel_text,
                (w - 220, bar_h + 25),
                font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    return vis


# ------------------------
# IR START (ONE-TIME PLAY)
# ------------------------
def wait_for_play_once():
    """
    Block here until PLAY is pressed once on the IR remote.
    After that, remote is not used anymore.
    """
    KEY_PLAY = 0x15   # update if your PLAY key code is different

    print("[IR] Waiting for PLAY (0x15) to activate...")
    while True:
        key = get_ir_key()
        if key is None:
            time.sleep(0.01)
            continue

        print(f"[IR] key=0x{key:02X}")
        if key == KEY_PLAY:
            print("[IR] PLAY pressed -> activating system.")
            return
        # any other key is ignored

# ------------------------
# MAIN LOOP
# ------------------------
def main(args):
    # Wait for IR PLAY once before starting anything
    wait_for_play_once()

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

    car_control = CarControl(max_pwm=120)

    # Force ACTIVE after PLAY: move_enabled=True
    planner = PathPlanner(
        base_speed=args.speed,
        deadband=1.0,
        Kp=0.05,
        max_missed_frames=MAX_MISSED_FRAMES,
        move_enabled=True,   # always active immediately after PLAY
    )

    print("[INFO] Activated after PLAY. Press 'm' to pause/resume, 'q' or ESC to quit.")
    print("[INFO] IR remote is only used once at startup (PLAY).")

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

            if speed <= 0.0:
                car_control.stop()
                left_pwm = 0
                right_pwm = 0
            else:
                # steering differential — scaled in PWM units (0–100)
                left_pwm  = speed * (1.0 + steer*0.7)
                right_pwm = speed * (1.0 - steer*0.7)

                # clamp for safety
                left_pwm  = max(-100, min(100, left_pwm))
                right_pwm = max(-100, min(100, right_pwm))

                # drive function expects normalized [-1..1]
                car_control.drive(left_pwm / 100.0, right_pwm / 100.0)


            # Status for HUD
            if not planner.move_enabled:
                status_text = "INACTIVE"
            elif speed <= 0.0:
                status_text = "STOPPED"
            else:
                status_text = "ACTIVE"

            # Draw heading arrow + yaw + status
            vis_hud = draw_hud(vis, yaw_deg, status_text, left_pwm, right_pwm)


            cv2.imshow("camera (q/esc to quit, m=toggle active)", vis_hud)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                print("[INFO] Quit requested.")
                break
            elif k == ord("m"):
                planner.toggle_move()

    finally:
        cap.release()
        cv2.destroyAllWindows()
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
        help="(Ignored; system is activated automatically after PLAY)."
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=100,
        help="Base forward speed in [0,1] when tracking a plate."
    )
    args = parser.parse_args()

    main(args)
