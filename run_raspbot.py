
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
import threading
from flask import Flask, Response

import RPi.GPIO as GPIO

from YB_Pcb_Car_control import YB_Pcb_Car  # import the class directly

from plate_format.plate_format_ro import is_valid_plate, normalize_plate_format
from helper_function import (
    bbox_center, pixel_to_angles, iou_xyxy, nms_xyxy,
    preprocess_plate, extract_valid_plate, run_onnx_inference,
    decode_yolov8, crop_with_margin, free_camera
)
import RPi.GPIO as GPIO
import time
from remote_fucntion import get_ir_key
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

CONF_THRES = 0.6
IOU_THRES = 0.05
INFER_PERIOD_S = 0.1
OCR_PERIOD_S = 1.0
PLATE_DEDUPE_SECONDS = 0
MAX_MISSED_FRAMES = 10  # used only for logging now (car no longer auto-stops on plate loss)

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
# GPIO / ULTRASONIC SETUP
# ------------------------
AvoidSensorLeft = 21     # Left infrared obstacle avoidance sensor pin (not used here)
AvoidSensorRight = 19    # Right infrared obstacle avoidance sensor pin (not used here)
Avoid_ON = 22            # Infrared obstacle avoidance sensor switch pin (not used here)

EchoPin = 18             # Ultrasonic echo pin
TrigPin = 16             # Ultrasonic trig pin

ULTRASONIC_ENABLED = False  # global flag

PIN = 36;   #Define IR pin

#Set the GPIO port to BIARD encoding mode

# GPIO.cleanup()
GPIO.setmode(GPIO.BOARD)

#Ignore the warning message
GPIO.setwarnings(False)


#The pin of the red external device needs to be set to input pull-up
GPIO.setup(PIN,GPIO.IN,GPIO.PUD_UP)
print("IR test start...")  #Initially print "IR test start".

def init_controller(enable: bool):

    PIN = 36  # IR pin (BOARD numbering)

    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    GPIO.setup(PIN, GPIO.IN, GPIO.PUD_UP)


def init_ultrasonic(enable: bool):
    """
    Initialize ultrasonic subsystem if 'enable' is True.
    Never crashes the process: on any error, ultrasonic is disabled gracefully.
    """
    global ULTRASONIC_ENABLED

    if not enable:
        print("[ULTRASONIC] Disabled by parameter.")
        ULTRASONIC_ENABLED = False
        return

    try:
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)

        # Only ultrasonic pins here; leave car pins to YB_Pcb_Car
        GPIO.setup(TrigPin, GPIO.OUT)
        GPIO.setup(EchoPin, GPIO.IN)

        ULTRASONIC_ENABLED = True
        print("[ULTRASONIC] Initialized successfully (Trig=16, Echo=18).")
    except Exception as e:
        ULTRASONIC_ENABLED = False
        print(f"[ULTRASONIC] Init failed: {e}. Ultrasonic disabled.")

def Distance():
    """
    Single ultrasonic reading in cm.
    Returns -1 on failure or if ultrasonic is disabled.
    """
    if not ULTRASONIC_ENABLED:
        return -1

    try:
        GPIO.output(TrigPin, GPIO.LOW)
        time.sleep(0.000002)
        GPIO.output(TrigPin, GPIO.HIGH)
        time.sleep(0.000015)
        GPIO.output(TrigPin, GPIO.LOW)

        t3 = time.time()
        while not GPIO.input(EchoPin):
            t4 = time.time()
            if (t4 - t3) > 0.03:
                return -1
        t1 = time.time()

        while GPIO.input(EchoPin):
            t5 = time.time()
            if (t5 - t1) > 0.03:
                return -1

        t2 = time.time()
        return ((t2 - t1) * 340.0 / 2.0) * 100.0  # cm
    except Exception as e:
        print(f"[ULTRASONIC] Distance read error: {e}")
        return -1

def Distance_test():
    """
    Robust ultrasonic reading:
    - Takes several samples, filters out invalid and extreme values.
    - Returns averaged distance in cm, or -1 if all failed or ultrasonic disabled.
    """
    if not ULTRASONIC_ENABLED:
        return -1

    readings = []
    attempts = 0
    while len(readings) < 3 and attempts < 10:
        d = Distance()
        attempts += 1
        if int(d) == -1:
            continue
        if int(d) >= 500 or int(d) == 0:
            continue
        readings.append(d)

    if len(readings) < 3:
        return -1

    distance = sum(readings[:3]) / 3.0
    return distance

# ------------------------
# IR REMOTE CLASS
# ------------------------
class IRRemote:
    """
    Simple NEC-protocol IR remote decoder (blocking for ~tens of ms only when a key is actually received).

    Mapping we use here:
      - 0x15 : PLAY / START / CONTINUE
      - 0x0D : PAUSE
      - 0x52 : '0' key, EXIT PROGRAM
    """
    def __init__(self, pin=36):
        self.pin = pin
        # GPIO.setmode(GPIO.BOARD)
        # GPIO.setwarnings(False)
        # GPIO.setup(self.pin, GPIO.IN, GPIO.PUD_UP)
        # print(f"[IR] Remote initialized on pin {self.pin} (BOARD).")

    def get_key(self):
        ir_repeat_cnt = 0
        while True:
            if GPIO.input(PIN) == 0:   #The signal emitted by the infrared remote control is detected
                ir_repeat_cnt = 0;
                count = 0
                while GPIO.input(PIN) == 0 and count < 200:   #Judge the boot code of 9ms high level pulse
                    count += 1
                    time.sleep(0.00006)

                count = 0
                while GPIO.input(PIN) == 1 and count < 80:   #Judge the boot code of 4.5ms low-level pulse
                    count += 1
                    time.sleep(0.00006)

                idx = 0
                cnt = 0
                data = [0,0,0,0]   #Define data used to store the address code, address inversion, signal code, and signal inversion of infrared signals
                for i in range(0,32):   #data[0],data[1],data[2],data[3] In total, 8bit*4=32
                    count = 0
                    while GPIO.input(PIN) == 0 and count < 10:   #Start decoding, used to filter the first 560us pulse of logic 0 and logic 1
                        count += 1
                        time.sleep(0.00006)

                    count = 0
                    while GPIO.input(PIN) == 1 and count < 40:   #After the 560us high-level pulse, check the remaining low-level pulse time length to determine whether it is logic 0 or logic 1
                        '''
                        Description:
                        According to the infrared NCE agreement:
                        The period of logic 1 is 2.25ms, and the pulse time is 0.56ms.Total period-pulse time = the value we set, the set value is slightly larger than the actual value.

                        The logic 0 period is 1.12 and the time is 0.56ms. Total period-pulse time = the value we set, the set value is slightly larger than the actual value.
                        
                        '''
                        count += 1
                        time.sleep(0.00006)

                    if count > 9:    
                        #This code is used to determine whether the currently received signal is logic 1 or logic 0.
                        #If count>9, it proves that the duration of the current low-level signal is greater than 560 (9*60=540us), which is logic 1.
                        #For example: when count=10, the low-level signal is 10*60=600us (greater than 560us)，it is logic 1.
                        
                        data[idx] |= 1<<cnt   
                    if cnt == 7:   #When cnt=7, one byte is full, and the next byte is ready to be stored.
                        cnt = 0
                        idx += 1
                    else:
                        cnt += 1  
                        
                if data[0]+data[1] == 0xFF and data[2]+data[3] == 0xFF:  #It is judged that the correct infrared remote control code value is received.
                    print("Get the key: 0x%02x" %data[2])   #Print the command code obtained
                    return data[2]
            else:
                if ir_repeat_cnt > 110: #Judge whether the infrared remote control button is released, because the repetition cycle time is 110ms, so here it should be set to 110*0.001.
                    ir_repeat_cnt = 0
                else:
                    time.sleep(0.001)
                    ir_repeat_cnt += 1
            
            return None


    def get_key2(self):
        """
        Non-infinite read of one full frame.
        Returns:
          key_code (int 0..255) or None if no valid frame was received.
        """
        if GPIO.input(self.pin) != 0:
            return None

        # Detected start of frame
        count = 0
        # 9 ms leading low (in their comments "high", but with pull-up it's active low)
        while GPIO.input(self.pin) == 0 and count < 200:
            count += 1
            time.sleep(0.00006)

        # 4.5 ms space
        count = 0
        while GPIO.input(self.pin) == 1 and count < 80:
            count += 1
            time.sleep(0.00006)

        data = [0, 0, 0, 0]
        idx = 0
        cnt = 0

        # 32 bits: addr, ~addr, cmd, ~cmd
        for i in range(32):
            count = 0
            # 560 us low
            while GPIO.input(self.pin) == 0 and count < 10:
                count += 1
                time.sleep(0.00006)

            # measure high time to distinguish 0 / 1
            count = 0
            while GPIO.input(self.pin) == 1 and count < 40:
                count += 1
                time.sleep(0.00006)

            if count > 9:
                # logic "1"
                data[idx] |= 1 << cnt

            if cnt == 7:
                cnt = 0
                idx += 1
            else:
                cnt += 1

        # Check NEC integrity
        if data[0] + data[1] == 0xFF and data[2] + data[3] == 0xFF:
            key = data[2]
            print(f"[IR] Get the key (command value): 0x{key:02X}") 
            return key

        return None

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
        self.current_plate = None  # for telemetry display

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
          vis_frame, yaw_deg, pitch_deg, has_detection (bool), plate_text
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
            return vis, None, None, False, self.current_plate

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
                if plate:
                    self.last_seen_plate[plate] = now
                    self.current_plate = plate
                    print(f"[PERCEPTION] [{time.strftime('%H:%M:%S')}] License Plate: {plate}")

        return vis, yaw_deg, pitch_deg, True, self.current_plate

# ------------------------
# LOW-LEVEL CAR CONTROL (PWM-BASED)
# ------------------------
class CarControl:
    """
    Low-level interface to the YB_Pcb_Car.
    Accepts PWM commands directly.
    """
    def __init__(self, max_pwm=120):
        self.car = YB_Pcb_Car()
        self.max_pwm = max_pwm

    def stop(self):
        self.car.Car_Stop()

    def drive(self, left_pwm, right_pwm):
        """
        left_pwm, right_pwm in approximately [-max_pwm, max_pwm]
        """
        left_pwm = int(max(-self.max_pwm, min(self.max_pwm, left_pwm)))
        right_pwm = int(max(-self.max_pwm, min(self.max_pwm, right_pwm)))

        print(f"[CAR] drive left_pwm={left_pwm}, right_pwm={right_pwm}")
        self.car.Control_Car(left_pwm, right_pwm)

# ------------------------
# PATH PLANNING CLASS (PWM SPEED)
# ------------------------
class PathPlanner:
    """
    High-level planner:
    - Uses yaw angle from Perception to generate (steer, speed_pwm).
    - Drives forward toward plate center.
    - Now continues to move forward searching even if no plate is detected,
      until PAUSE or EXIT is pressed (or ultrasonic stop).
    speed_pwm is in [0, 120] (PWM).
    """
    def __init__(self, base_speed=100, deadband=1.0, Kp=0.10, max_missed_frames=10):
        self.base_speed = base_speed   # PWM (0..120)
        self.deadband = deadband
        self.Kp = Kp
        self.max_missed_frames = max_missed_frames

        # Will be controlled by remote / keyboard
        self.move_enabled = False

        self.missed_frames = 0
        self.last_yaw = 0.0
        print(f"[PATH] init base_speed={self.base_speed}, move_enabled={self.move_enabled}")

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
        Returns (steer, speed_pwm) in [-1,1] and [0,120].
        No automatic stop when plate is lost: the car keeps moving
        forward (searching) as long as move_enabled is True.
        """
        if has_detection:
            self.missed_frames = 0
            self.last_yaw = yaw_deg if yaw_deg is not None else 0.0
            steer = self._compute_steering_from_yaw(yaw_deg)
            speed_pwm = self.base_speed
        else:
            self.missed_frames += 1
            print(f"[PATH] No plate detected. missed_frames={self.missed_frames}")
            # Keep moving forward using last_yaw (or straight if no history)
            steer = self._compute_steering_from_yaw(self.last_yaw)
            speed_pwm = self.base_speed

        if not self.move_enabled:
            speed_pwm = 0.0

        print(f"[PATH] step: move_enabled={self.move_enabled}, "
              f"yaw={yaw_deg}, steer={steer:.3f}, speed_pwm={speed_pwm:.1f}")
        return steer, speed_pwm

# ------------------------
# TELEMETRY CLASS (STREAM + HUD OVERLAY)
# ------------------------
class Telemetry:
    """
    Streams an enlarged, annotated camera view over HTTP (MJPEG) and
    draws a HUD for phone viewing.
    Output size fixed to 960x540 (16:9) for phones.
    """
    def __init__(self, host="0.0.0.0", port=8080, width=640, height=480):
        self.host = host
        self.port = port
        self.width = width
        self.height = height
        self.latest_frame = None

        self.app = Flask(__name__)

        @self.app.route("/video")
        def video_feed():
            return Response(self._mjpeg_generator(),
                            mimetype="multipart/x-mixed-replace; boundary=frame")

    def _mjpeg_generator(self):
        while True:
            if self.latest_frame is not None:
                ret, jpeg = cv2.imencode(".jpg", self.latest_frame)
                if not ret:
                    time.sleep(0.03)
                    continue
                frame = jpeg.tobytes()
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.03)

    def start(self):
        threading.Thread(
            target=self.app.run,
            kwargs={"host": self.host, "port": self.port, "debug": False, "use_reloader": False},
            daemon=True
        ).start()
        print(f"[TELEMETRY] MJPEG server running on http://{self.host}:{self.port}/video")

    def _draw_hud(self, frame, yaw_deg, speed_pwm, plate_text, distance_cm,
                  stop_distance_cm, status_text, last_ir_key):
        # Resize to fixed phone-friendly size (16:9)
        vis = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        h, w = vis.shape[:2]

        # Semi-transparent top bar
        bar_h = int(h * 0.2)
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
        alpha = 0.0
        vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)

        # Heading labels
        center_x = w // 2
        heading_y = int(bar_h * 0.3)
        cv2.putText(vis, "LEFT", (10, heading_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, "RIGHT", (w - 90, heading_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2, cv2.LINE_AA)

        # Heading arrow using yaw in [-90, 90], 0=up
        if yaw_deg is None:
            yaw_deg = 0.0
        heading_deg = max(-90.0, min(90.0, yaw_deg))
        heading_int = int(round(heading_deg))

        cx = center_x
        cy = int(bar_h * 0.65)
        arrow_len = int(bar_h * 0.7)
        theta = np.radians(heading_deg)

        ex = int(cx + arrow_len * np.sin(theta))
        ey = int(cy - arrow_len * np.cos(theta))

        arrow_color = (0, 255, 0)
        cv2.arrowedLine(vis, (cx, cy), (ex, ey), arrow_color, 4, tipLength=0.25)

        # Heading text (int degrees)
        cv2.putText(vis, f"Heading: {heading_int} deg",
                    (10, bar_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # License plate text
        plate_str = plate_text if plate_text else "N/A"
        cv2.putText(vis, f"Plate: {plate_str}",
                    (10, bar_h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

        # Yaw and speed
        yaw_txt = f"Yaw: {heading_int} deg"
        cv2.putText(vis, yaw_txt,
                    (10, bar_h + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(vis, f"Speed: {int(speed_pwm)} PWM",
                    (10, bar_h + 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Distance
        if distance_cm is not None and distance_cm > 0:
            dist_txt = f"Dist: {int(distance_cm)} cm"
            color = (0, 255, 0)
            if distance_cm < stop_distance_cm:
                color = (0, 0, 255)
                dist_txt += "  [STOP]"
            cv2.putText(vis, dist_txt,
                        (10, bar_h + 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        else:
            cv2.putText(vis, "Dist: N/A",
                        (10, bar_h + 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # Status on the left
        cv2.putText(vis, f"Status: {status_text}",
                    (10, bar_h + 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Last IR command value
        key_txt = "None" if last_ir_key is None else f"0x{last_ir_key:02X}"
        cv2.putText(vis, f"IR Cmd: {key_txt}",
                    (10, bar_h + 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        return vis

    def update(self, frame, yaw_deg, speed_pwm, plate_text, distance_cm,
               stop_distance_cm, status_text, last_ir_key):
        vis = self._draw_hud(
            frame, yaw_deg, speed_pwm, plate_text, distance_cm,
            stop_distance_cm, status_text, last_ir_key
        )
        self.latest_frame = vis
        return vis

# ------------------------
# MAIN LOOP
# ------------------------
def main(args):
    stop_distance_cm = args.min_distance

    # init_ultrasonic(args.ultrasonic)
   # init_controller(True)
    free_camera()

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        raise SystemExit("Error: Could not open camera (is /dev/video0 busy?)")

    # IR remote
    remote = IRRemote(pin=36)
    KEY_PLAY = 0x15   # start / continue
    KEY_PAUSE = 0x0d  # pause
    KEY_EXIT = 0x52   # '0' key -> exit

    last_ir_key = None  # remember last remote command value

    perception = Perception(
        K=K,
        conf_thres=CONF_THRES,
        iou_thres=IOU_THRES,
        infer_period_s=INFER_PERIOD_S,
        ocr_period_s=OCR_PERIOD_S,
        plate_dedupe_seconds=PLATE_DEDUPE_SECONDS,
    )

    car_control = CarControl(max_pwm=120)

    # Path planner: start DISABLED, activation via remote PLAY key
    planner = PathPlanner(
        base_speed=args.speed,      # PWM, like old code
        deadband=1.0,
        Kp=0.10,
        max_missed_frames=MAX_MISSED_FRAMES,
    )
    planner.move_enabled = False
    activated = False  # overall program activation state

    telemetry = Telemetry(host="0.0.0.0", port=8080, width=640, height=480)
    telemetry.start()

    print("[INFO] Controls:")
    print("   Remote PLAY (0x15)  : activate / continue movement")
    print("   Remote PAUSE (0x0D) : pause (car stops)")
    print("   Remote '0' (0x52)   : exit program")
    print("   Keyboard 'm'        : toggle move_enabled (debug)")
    print("   Keyboard 'q' / ESC  : exit program")
    print("[INFO] Initial status: NOT ACTIVE (car will not move until PLAY is pressed).")
    print("[INFO] Once PLAY is pressed, the car will keep moving forward and searching")
    print("       for a license plate until you press PAUSE or EXIT (or ultrasonic stop).")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] Failed to read frame from camera.")
                time.sleep(0.05)
                continue

            now = time.time()

            # Remote key handling
            key = remote.get_key()
            print(f"key={key}")
            if key is not None:
                last_ir_key = key
                if key == KEY_PLAY:
                    if not activated:
                        print("[IR] PLAY pressed (0x15) -> activating program, start moving.")
                        activated = True
                        planner.missed_frames = 0
                    else:
                        print("[IR] PLAY pressed (0x15) -> resume movement.")
                    planner.move_enabled = True
                elif key == KEY_PAUSE:
                    print("[IR] PAUSE pressed (0x0D) -> pausing movement.")
                    planner.move_enabled = False
                elif key == KEY_EXIT:
                    print("[IR] EXIT pressed (0x52, '0' key) -> exiting program.")
                    break
                else:
                    print(f"[IR] Unmapped command value: 0x{key:02X}")

            # Perception (run always so the detector keeps searching)
            vis, yaw_deg, pitch_deg, has_detection, plate_text = perception.process_frame(frame, now)

            # High-level planning: get desired steer/speed (PWM)
            steer, speed_pwm = planner.step(yaw_deg, has_detection)

            # If not activated at all yet, force zero speed (search only visually)
            if not activated:
                speed_pwm = 0.0

            # Ultrasonic distance and safety stop (only if ultrasonic is enabled)
            if ULTRASONIC_ENABLED:
                distance_cm = Distance_test()
                if distance_cm > 0 and distance_cm < stop_distance_cm:
                    print(f"[SAFETY] Obstacle at {distance_cm:.1f} cm < {stop_distance_cm} cm -> stopping car.")
                    speed_pwm = 0.0
            else:
                distance_cm = -1

            # Determine status text
            if not activated:
                status_text = "NOT ACTIVE"
            elif not planner.move_enabled or speed_pwm <= 0.0:
                status_text = "PAUSED"
            else:
                status_text = "RUNNING"

            # Debug output including command value
            cmd_str = "None" if last_ir_key is None else f"0x{last_ir_key:02X}"
            print(
                f"[DEBUG] steer={steer:.3f}, speed_pwm={speed_pwm:.1f}, "
                f"has_detection={has_detection}, dist={distance_cm:.1f} cm, "
                f"activated={activated}, move_enabled={planner.move_enabled}, "
                f"status={status_text}, last_ir_cmd={cmd_str}"
            )

            # Convert (steer, speed_pwm) -> left/right PWM
            if speed_pwm <= 0.0:
                car_control.stop()
            else:
                left_pwm = speed_pwm * (1.0 - steer)
                right_pwm = speed_pwm * (1.0 + steer)
                car_control.drive(left_pwm, right_pwm)

            # Telemetry overlay + streaming frame
            vis_tel = telemetry.update(
                vis, yaw_deg, speed_pwm, plate_text,
                distance_cm, stop_distance_cm, status_text, last_ir_key
            )

            # Local display (optional)
            cv2.imshow("camera (q/esc to quit, m=toggle move)", vis_tel)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                print("[INFO] Quit requested from keyboard.")
                break
            elif k == ord("m"):
                planner.toggle_move()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        planner.move_enabled = False
        car_control.stop()
        try:
            GPIO.cleanup()
        except Exception as e:
            print(f"[GPIO] cleanup error (ignored): {e}")
        print("[INFO] Camera, motors, and GPIO released.")

# ------------------------
# ENTRY POINT
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--speed",
        type=float,
        default=100.0,   # same as old script
        help="Base forward speed in PWM (0..120) when tracking/searching."
    )
    parser.add_argument(
        "--min-distance",
        type=float,
        default=20.0,
        help="Minimum distance in cm; below this the car stops (only if --ultrasonic enabled)."
    )
    parser.add_argument(
        "--ultrasonic",
        action="store_true",
        help="Enable ultrasonic distance safety stop. If not set, ultrasonic is disabled."
    )
    args = parser.parse_args()

    main(args)



# #!/usr/bin/env python
# # coding: utf-8
# import os
# os.environ["QT_QPA_PLATFORM"] = "xcb"   # must be set before importing cv2

# import time
# import sys
# import cv2
# import pytesseract
# import onnx
# import onnxruntime as onnxr
# import numpy as np
# import subprocess
# import argparse
# import threading
# from flask import Flask, Response

# import RPi.GPIO as GPIO

# from YB_Pcb_Car_control import YB_Pcb_Car  # import the class directly

# from plate_format.plate_format_ro import is_valid_plate, normalize_plate_format
# from helper_function import (
#     bbox_center, pixel_to_angles, iou_xyxy, nms_xyxy,
#     preprocess_plate, extract_valid_plate, run_onnx_inference,
#     decode_yolov8, crop_with_margin, free_camera
# )

# # ------------------------
# # Tesseract configuration
# # ------------------------
# pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# CAMERA_DEV = "/dev/video0"

# # ------------------------
# # Configuration
# # ------------------------
# ONNX_PATH = "/home/pi/Yahboom_project/uveye/code/raspberry-plate-recognition/yolov8n-license_plate.onnx"
# CONFIG_TESSERACT = (
#     "--psm 6 --oem 1 "
#     "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
# )

# CONF_THRES = 0.6
# IOU_THRES = 0.05
# INFER_PERIOD_S = 0.1
# OCR_PERIOD_S = 1.0
# PLATE_DEDUPE_SECONDS = 0
# MAX_MISSED_FRAMES = 10  # stop car if plate not seen for this many frames

# # Approximate camera intrinsics for 640x480 (REPLACE with your calibrated values)
# FX = 450.0
# FY = 600.0
# CX = 320.0
# CY = 240.0

# K = np.array([[FX,   0.0, CX],
#               [0.0,  FY,  CY],
#               [0.0,  0.0, 1.0]], dtype=np.float32)

# print("[INFO] Camera intrinsics K:")
# print(K)

# # ------------------------
# # ONNX Runtime: CPU only + quieter logs
# # ------------------------
# onnxr.set_default_logger_severity(3)  # 0=verbose ... 3=warning ... 4=error

# onnx_model = onnx.load(ONNX_PATH)
# onnx.checker.check_model(onnx_model)

# session = onnxr.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
# inp = session.get_inputs()[0]
# input_name = inp.name
# output_names = [out.name for out in session.get_outputs()]

# def _int_or_none(x):
#     return x if isinstance(x, int) else None

# MODEL_H = _int_or_none(inp.shape[2]) or 512
# MODEL_W = _int_or_none(inp.shape[3]) or 512

# print(f"[INFO] YOLO model input size: {MODEL_W}x{MODEL_H}")

# # ------------------------
# # GPIO / ULTRASONIC SETUP (OPTIONAL)
# # ------------------------
# AvoidSensorLeft = 21     # Left infrared obstacle avoidance sensor pin (not used here)
# AvoidSensorRight = 19    # Right infrared obstacle avoidance sensor pin (not used here)
# Avoid_ON = 22            # Infrared obstacle avoidance sensor switch pin (not used here)

# EchoPin = 18             # Ultrasonic echo pin
# TrigPin = 16             # Ultrasonic trig pin

# ULTRASONIC_ENABLED = False  # global flag

# def init_ultrasonic(enable: bool):
#     """
#     Initialize ultrasonic subsystem if 'enable' is True.
#     Never crashes the process: on any error, ultrasonic is disabled gracefully.
#     """
#     global ULTRASONIC_ENABLED

#     if not enable:
#         print("[ULTRASONIC] Disabled by parameter.")
#         ULTRASONIC_ENABLED = False
#         return

#     try:
#         GPIO.setmode(GPIO.BOARD)
#         GPIO.setwarnings(False)

#         # Only ultrasonic pins here; leave car pins to YB_Pcb_Car
#         GPIO.setup(TrigPin, GPIO.OUT)
#         GPIO.setup(EchoPin, GPIO.IN)

#         ULTRASONIC_ENABLED = True
#         print("[ULTRASONIC] Initialized successfully (Trig=16, Echo=18).")
#     except Exception as e:
#         ULTRASONIC_ENABLED = False
#         print(f"[ULTRASONIC] Init failed: {e}. Ultrasonic disabled.")

# def Distance():
#     """
#     Single ultrasonic reading in cm.
#     Returns -1 on failure or if ultrasonic is disabled.
#     """
#     if not ULTRASONIC_ENABLED:
#         return -1

#     try:
#         GPIO.output(TrigPin, GPIO.LOW)
#         time.sleep(0.000002)
#         GPIO.output(TrigPin, GPIO.HIGH)
#         time.sleep(0.000015)
#         GPIO.output(TrigPin, GPIO.LOW)

#         t3 = time.time()
#         while not GPIO.input(EchoPin):
#             t4 = time.time()
#             if (t4 - t3) > 0.03:
#                 return -1
#         t1 = time.time()

#         while GPIO.input(EchoPin):
#             t5 = time.time()
#             if (t5 - t1) > 0.03:
#                 return -1

#         t2 = time.time()
#         return ((t2 - t1) * 340.0 / 2.0) * 100.0  # cm
#     except Exception as e:
#         print(f"[ULTRASONIC] Distance read error: {e}")
#         return -1

# def Distance_test():
#     """
#     Robust ultrasonic reading:
#     - Takes several samples, filters out invalid and extreme values.
#     - Returns averaged distance in cm, or -1 if all failed or ultrasonic disabled.
#     """
#     if not ULTRASONIC_ENABLED:
#         return -1

#     readings = []
#     attempts = 0
#     while len(readings) < 3 and attempts < 10:
#         d = Distance()
#         attempts += 1
#         if int(d) == -1:
#             continue
#         if int(d) >= 500 or int(d) == 0:
#             continue
#         readings.append(d)

#     if len(readings) < 3:
#         return -1

#     distance = sum(readings[:3]) / 3.0
#     return distance

# # ------------------------
# # PERCEPTION CLASS
# # ------------------------
# class Perception:
#     def __init__(self, K, conf_thres, iou_thres,
#                  infer_period_s, ocr_period_s, plate_dedupe_seconds):
#         self.K = K
#         self.conf_thres = conf_thres
#         self.iou_thres = iou_thres
#         self.infer_period_s = infer_period_s
#         self.ocr_period_s = ocr_period_s
#         self.plate_dedupe_seconds = plate_dedupe_seconds

#         self.last_infer_time = 0.0
#         self.last_ocr_time = 0.0
#         self.last_seen_plate = {}  # plate -> timestamp
#         self.detections = []
#         self.current_plate = None  # for telemetry display

#     def update_detections(self, frame, now):
#         if now - self.last_infer_time < self.infer_period_s:
#             return  # reuse previous self.detections

#         self.last_infer_time = now
#         fh, fw = frame.shape[:2]

#         outputs = run_onnx_inference(frame)
#         self.detections = decode_yolov8(outputs, fw, fh,
#                                         conf_thres=self.conf_thres,
#                                         iou_thres=self.iou_thres)
#         print(f"[PERCEPTION] {len(self.detections)} detection(s) this frame.")

#     def expire_old_plates(self, now):
#         for p in list(self.last_seen_plate.keys()):
#             if now - self.last_seen_plate[p] > self.plate_dedupe_seconds:
#                 del self.last_seen_plate[p]

#     def process_frame(self, frame, now):
#         """
#         Returns:
#           vis_frame, yaw_deg, pitch_deg, has_detection (bool), plate_text
#         """
#         self.expire_old_plates(now)
#         self.update_detections(frame, now)

#         vis = frame.copy()

#         for det in self.detections:
#             x1, y1, x2, y2 = det["bbox"]
#             conf = det["confidence"]
#             cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(
#                 vis, f"{conf:.2f}", (x1, max(0, y1 - 6)),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
#             )

#         if not self.detections:
#             return vis, None, None, False, self.current_plate

#         best = max(self.detections, key=lambda d: d["confidence"])
#         u, v = bbox_center(best["bbox"])
#         yaw_deg, pitch_deg = pixel_to_angles(u, v, self.K)

#         print(f"[PERCEPTION] Plate center pixel: u={u:.1f}, v={v:.1f}")
#         print(f"[PERCEPTION] Angles: yaw={yaw_deg:.2f} deg, pitch={pitch_deg:.2f} deg")

#         cv2.circle(vis, (int(u), int(v)), 4, (0, 0, 255), -1)
#         cv2.putText(
#             vis, f"yaw={yaw_deg:.1f} deg",
#             (int(u) + 5, int(v) - 5),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
#         )

#         # OCR throttled
#         if now - self.last_ocr_time >= self.ocr_period_s:
#             self.last_ocr_time = now
#             crop = crop_with_margin(frame, best["bbox"], margin=0.10)
#             if crop is not None:
#                 plate = extract_valid_plate(crop)
#                 if plate:
#                     self.last_seen_plate[plate] = now
#                     self.current_plate = plate
#                     print(f"[PERCEPTION] [{time.strftime('%H:%M:%S')}] License Plate: {plate}")

#         return vis, yaw_deg, pitch_deg, True, self.current_plate

# # ------------------------
# # LOW-LEVEL CAR CONTROL (PWM-BASED, LIKE OLD CODE)
# # ------------------------
# class CarControl:
#     """
#     Low-level interface to the YB_Pcb_Car.
#     Accepts PWM commands directly.
#     """
#     def __init__(self, max_pwm=120):
#         self.car = YB_Pcb_Car()
#         self.max_pwm = max_pwm

#     def stop(self):
#         self.car.Car_Stop()

#     def drive(self, left_pwm, right_pwm):
#         """
#         left_pwm, right_pwm in approximately [-max_pwm, max_pwm]
#         """
#         left_pwm = int(max(-self.max_pwm, min(self.max_pwm, left_pwm)))
#         right_pwm = int(max(-self.max_pwm, min(self.max_pwm, right_pwm)))

#         print(f"[CAR] drive left_pwm={left_pwm}, right_pwm={right_pwm}")
#         self.car.Control_Car(left_pwm, right_pwm)

# # ------------------------
# # PATH PLANNING CLASS (PWM SPEED, LIKE OLD CODE)
# # ------------------------
# class PathPlanner:
#     """
#     High-level planner:
#     - Uses yaw angle from Perception to generate (steer, speed_pwm).
#     - Drives forward toward plate center.
#     - Stops if plate is not seen for > max_missed_frames frames.
#     speed_pwm is in [0, 120] (PWM).
#     """
#     def __init__(self, base_speed=100, deadband=1.0, Kp=0.10, max_missed_frames=10):
#         self.base_speed = base_speed   # PWM (0..120)
#         self.deadband = deadband
#         self.Kp = Kp
#         self.max_missed_frames = max_missed_frames

#         # IMPORTANT: start enabled, like old code
#         self.move_enabled = True

#         self.missed_frames = 0
#         self.last_yaw = 0.0
#         print(f"[PATH] init base_speed={self.base_speed}, move_enabled={self.move_enabled}")

#     def toggle_move(self):
#         self.move_enabled = not self.move_enabled
#         print(f"[PATH] move_enabled -> {self.move_enabled}")

#     def _compute_steering_from_yaw(self, yaw_deg):
#         if yaw_deg is None:
#             return 0.0
#         if abs(yaw_deg) < self.deadband:
#             return 0.0
#         steer = self.Kp * yaw_deg
#         steer = max(-1.0, min(1.0, steer))
#         return steer

#     def step(self, yaw_deg, has_detection):
#         """
#         Returns (steer, speed_pwm) in [-1,1] and [0,120].
#         """
#         if has_detection:
#             self.missed_frames = 0
#             self.last_yaw = yaw_deg if yaw_deg is not None else 0.0
#             speed_pwm = self.base_speed
#             steer = self._compute_steering_from_yaw(yaw_deg)
#         else:
#             self.missed_frames += 1
#             print(f"[PATH] No plate detected. missed_frames={self.missed_frames}")

#             if self.missed_frames > self.max_missed_frames:
#                 print("[PATH] Plate lost for too long -> stopping car.")
#                 speed_pwm = 0.0
#                 steer = 0.0
#             else:
#                 speed_pwm = self.base_speed
#                 steer = self._compute_steering_from_yaw(self.last_yaw)

#         if not self.move_enabled:
#             speed_pwm = 0.0

#         print(f"[PATH] step: move_enabled={self.move_enabled}, "
#               f"yaw={yaw_deg}, steer={steer:.3f}, speed_pwm={speed_pwm:.1f}")
#         return steer, speed_pwm

# # ------------------------
# # TELEMETRY CLASS (STREAM + HUD OVERLAY)
# # ------------------------
# class Telemetry:
#     """
#     Streams an enlarged, annotated camera view over HTTP (MJPEG) and
#     draws a HUD for phone viewing.
#     Output size fixed to 960x540 (16:9) for phones.
#     """
#     def __init__(self, host="0.0.0.0", port=8080, width=640, height=480):
#         self.host = host
#         self.port = port
#         self.width = width
#         self.height = height
#         self.latest_frame = None

#         self.app = Flask(__name__)

#         @self.app.route("/video")
#         def video_feed():
#             return Response(self._mjpeg_generator(),
#                             mimetype="multipart/x-mixed-replace; boundary=frame")

#     def _mjpeg_generator(self):
#         while True:
#             if self.latest_frame is not None:
#                 ret, jpeg = cv2.imencode(".jpg", self.latest_frame)
#                 if not ret:
#                     time.sleep(0.03)
#                     continue
#                 frame = jpeg.tobytes()
#                 yield (b"--frame\r\n"
#                        b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
#             time.sleep(0.03)

#     def start(self):
#         threading.Thread(
#             target=self.app.run,
#             kwargs={"host": self.host, "port": self.port, "debug": False, "use_reloader": False},
#             daemon=True
#         ).start()
#         print(f"[TELEMETRY] MJPEG server running on http://{self.host}:{self.port}/video")

#     def _draw_hud(self, frame, yaw_deg, speed_pwm, plate_text, distance_cm, stop_distance_cm):
#         # Resize to fixed phone-friendly size (16:9)
#         vis = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
#         h, w = vis.shape[:2]

#         # Semi-transparent top bar
#         bar_h = int(h * 0.2)
#         overlay = vis.copy()
#         cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
#         alpha = 0.0
#         vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)

#         # Heading labels
#         center_x = w // 2
#         heading_y = int(bar_h * 0.3)
#         cv2.putText(vis, "LEFT", (10, heading_y),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2, cv2.LINE_AA)
#         # cv2.putText(vis, "FRONT", (center_x - 45, heading_y),
#         #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (00, 200, 0), 3, cv2.LINE_AA)
#         cv2.putText(vis, "RIGHT", (w - 90, heading_y),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2, cv2.LINE_AA)

#         # Heading arrow using yaw in [-90, 90], 0=up
#         if yaw_deg is None:
#             yaw_deg = 0.0
#         heading_deg = max(-90.0, min(90.0, yaw_deg))
#         heading_int = int(round(heading_deg))

#         cx = center_x
#         cy = int(bar_h * 0.65)
#         arrow_len = int(bar_h * 0.7)
#         theta = np.radians(heading_deg)

#         ex = int(cx + arrow_len * np.sin(theta))
#         ey = int(cy - arrow_len * np.cos(theta))

#         arrow_color = (0, 255, 0)
#         cv2.arrowedLine(vis, (cx, cy), (ex, ey), arrow_color, 4, tipLength=0.25)

#         # Heading text (int degrees)
#         cv2.putText(vis, f"Heading: {heading_int} deg",
#                     (10, bar_h - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

#         # License plate text
#         plate_str = plate_text if plate_text else "N/A"
#         cv2.putText(vis, f"Plate: {plate_str}",
#                     (10, bar_h + 25),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

#         # Yaw and speed
#         yaw_txt = f"Yaw: {heading_int} deg"
#         cv2.putText(vis, yaw_txt,
#                     (10, bar_h + 55),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

#         cv2.putText(vis, f"Speed: {int(speed_pwm)} PWM",
#                     (10, bar_h + 85),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

#         # Distance
#         if distance_cm is not None and distance_cm > 0:
#             dist_txt = f"Dist: {int(distance_cm)} cm"
#             color = (0, 255, 0)
#             if distance_cm < stop_distance_cm:
#                 color = (0, 0, 255)
#                 dist_txt += "  [STOP]"
#             cv2.putText(vis, dist_txt,
#                         (10, bar_h + 115),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
#         else:
#             cv2.putText(vis, "Dist: N/A",
#                         (10, bar_h + 115),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

#         return vis

#     def update(self, frame, yaw_deg, speed_pwm, plate_text, distance_cm, stop_distance_cm):
#         vis = self._draw_hud(frame, yaw_deg, speed_pwm, plate_text, distance_cm, stop_distance_cm)
#         self.latest_frame = vis
#         return vis

# # ------------------------
# # MAIN LOOP
# # ------------------------
# def main(args):
#     stop_distance_cm = args.min_distance

#     init_ultrasonic(args.ultrasonic)
#     free_camera()

#     cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
#     cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#     cap.set(cv2.CAP_PROP_FPS, 30)

#     if not cap.isOpened():
#         raise SystemExit("Error: Could not open camera (is /dev/video0 busy?)")

#     perception = Perception(
#         K=K,
#         conf_thres=CONF_THRES,
#         iou_thres=IOU_THRES,
#         infer_period_s=INFER_PERIOD_S,
#         ocr_period_s=OCR_PERIOD_S,
#         plate_dedupe_seconds=PLATE_DEDUPE_SECONDS,
#     )

#     car_control = CarControl(max_pwm=120)

#     # Path planner: always start enabled, same as old code
#     planner = PathPlanner(
#         base_speed=args.speed,      # PWM, like old code
#         deadband=1.0,
#         Kp=0.10,
#         max_missed_frames=MAX_MISSED_FRAMES,
#     )

#     telemetry = Telemetry(host="0.0.0.0", port=8080, width=640, height=480)
#     telemetry.start()

#     print("[INFO] Press 'm' to toggle move_enabled, 'q' or ESC to quit.")



#     try:
#         print("start")
#         while True:
#             if GPIO.input(PIN) == 0:   #The signal emitted by the infrared remote control is detected
#                 ir_repeat_cnt = 0;
#                 count = 0
#                 while GPIO.input(PIN) == 0 and count < 200:   #Judge the boot code of 9ms high level pulse
#                     count += 1
#                     time.sleep(0.00006)

#                 count = 0
#                 while GPIO.input(PIN) == 1 and count < 80:   #Judge the boot code of 4.5ms low-level pulse
#                     count += 1
#                     time.sleep(0.00006)

#                 idx = 0
#                 cnt = 0
#                 data = [0,0,0,0]   #Define data used to store the address code, address inversion, signal code, and signal inversion of infrared signals
#                 for i in range(0,32):   #data[0],data[1],data[2],data[3] In total, 8bit*4=32
#                     count = 0
#                     while GPIO.input(PIN) == 0 and count < 10:   #Start decoding, used to filter the first 560us pulse of logic 0 and logic 1
#                         count += 1
#                         time.sleep(0.00006)

#                     count = 0
#                     while GPIO.input(PIN) == 1 and count < 40:   #After the 560us high-level pulse, check the remaining low-level pulse time length to determine whether it is logic 0 or logic 1
#                         '''
#                         Description:
#                         According to the infrared NCE agreement:
#                         The period of logic 1 is 2.25ms, and the pulse time is 0.56ms.Total period-pulse time = the value we set, the set value is slightly larger than the actual value.

#                         The logic 0 period is 1.12 and the time is 0.56ms. Total period-pulse time = the value we set, the set value is slightly larger than the actual value.
                        
#                         '''
#                         count += 1
#                         time.sleep(0.00006)

#                     if count > 9:    
#                         #This code is used to determine whether the currently received signal is logic 1 or logic 0.
#                         #If count>9, it proves that the duration of the current low-level signal is greater than 560 (9*60=540us), which is logic 1.
#                         #For example: when count=10, the low-level signal is 10*60=600us (greater than 560us)，it is logic 1.
                        
#                         data[idx] |= 1<<cnt   
#                     if cnt == 7:   #When cnt=7, one byte is full, and the next byte is ready to be stored.
#                         cnt = 0
#                         idx += 1
#                     else:
#                         cnt += 1  
                        
#                 if data[0]+data[1] == 0xFF and data[2]+data[3] == 0xFF:  #It is judged that the correct infrared remote control code value is received.
#                     print("Get the key: 0x%02x" %data[2])   #Print the command code obtained
#             else:
#                 if ir_repeat_cnt > 110: #Judge whether the infrared remote control button is released, because the repetition cycle time is 110ms, so here it should be set to 110*0.001.
#                     ir_repeat_cnt = 0
#                 else:
#                     time.sleep(0.001)
#                     ir_repeat_cnt += 1

#     except KeyboardInterrupt:
#         pass
#     print("Ending")
#     GPIO.cleanup()

#     try:
#         ################################################################################
#         while True:
#             ret, frame = cap.read()
#             if not ret or frame is None:
#                 print("[WARN] Failed to read frame from camera.")
#                 time.sleep(0.05)
#                 continue

#             now = time.time()

#             vis, yaw_deg, pitch_deg, has_detection, plate_text = perception.process_frame(frame, now)

#             # High-level planning: get desired steer/speed (PWM)
#             steer, speed_pwm = planner.step(yaw_deg, has_detection)

#             # Ultrasonic distance and safety stop (only if ultrasonic is enabled)
#             if ULTRASONIC_ENABLED:
#                 distance_cm = Distance_test()
#                 if distance_cm > 0 and distance_cm < stop_distance_cm:
#                     print(f"[SAFETY] Obstacle at {distance_cm:.1f} cm < {stop_distance_cm} cm -> stopping car.")
#                     speed_pwm = 0.0
#             else:
#                 distance_cm = -1

#             print(f"[DEBUG] steer={steer:.3f}, speed_pwm={speed_pwm:.1f}, "
#                   f"has_detection={has_detection}, dist={distance_cm:.1f} cm")

#             # Convert (steer, speed_pwm) -> left/right PWM
#             if speed_pwm <= 0.0:
#                 car_control.stop()
#             else:
#                 left_pwm = speed_pwm * (1.0 - steer)
#                 right_pwm = speed_pwm * (1.0 + steer)
#                 car_control.drive(left_pwm, right_pwm)

#             # Telemetry overlay + streaming frame
#             vis_tel = telemetry.update(vis, yaw_deg, speed_pwm, plate_text, distance_cm, stop_distance_cm)

#             # Local display (optional)
#             cv2.imshow("camera (q/esc to quit, m=toggle move)", vis_tel)
#             k = cv2.waitKey(1) & 0xFF
#             if k in (ord("q"), 27):
#                 print("[INFO] Quit requested.")
#                 break
#             elif k == ord("m"):
#                 planner.toggle_move()
#         ##########################################################################################################3#

#     finally:
#         cap.release()
#         cv2.destroyAllWindows()
#         planner.move_enabled = False
#         car_control.stop()
#         try:
#             GPIO.cleanup()
#         except Exception as e:
#             print(f"[GPIO] cleanup error (ignored): {e}")
#         print("[INFO] Camera, motors, and GPIO released.")

# # ------------------------
# # ENTRY POINT
# # ------------------------
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--speed",
#         type=float,
#         default=100.0,   # same as old script
#         help="Base forward speed in PWM (0..120) when tracking a plate."
#     )
#     parser.add_argument(
#         "--min-distance",
#         type=float,
#         default=20.0,
#         help="Minimum distance in cm; below this the car stops (only if --ultrasonic enabled)."
#     )
#     parser.add_argument(
#         "--ultrasonic",
#         action="store_true",
#         help="Enable ultrasonic distance safety stop. If not set, ultrasonic is disabled."
#     )
#     args = parser.parse_args()

#     main(args)
