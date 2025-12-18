import time
import sys
import cv2
import pytesseract
import onnx
import onnxruntime as onnxr
import numpy as np
from plate_format.plate_format_ro import is_valid_plate, normalize_plate_format

IMAGE_SIZE = 512
ONNX_PATH = "/home/pi/Yahboom_project/uveye/code/raspberry-plate-recognition/yolov8n-license_plate.onnx"
CONFIG_TESSERACT = '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

last_detected_plates = {}
max_plate_age_seconds = 10


# Out of the function to optimize performance
clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8,8))

# Create a session
onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)

session = onnxr.InferenceSession(ONNX_PATH)
input_name = session.get_inputs()[0].name
output_names = [out.name for out in session.get_outputs()]

def findIntersectionOverUnion(box1, box2):
    """Computes IoU between two boxes given as (cx, cy, w, h)."""
    box1_w, box1_h = box1[2]/2.0, box1[3]/2.0
    box2_w, box2_h = box2[2]/2.0, box2[3]/2.0

    b1_1, b1_2 = box1[0] - box1_w, box1[1] - box1_h
    b1_3, b1_4 = box1[0] + box1_w, box1[1] + box1_h
    b2_1, b2_2 = box2[0] - box2_w, box2[1] - box2_h
    b2_3, b2_4 = box2[0] + box2_w, box2[1] + box2_h

    x1, y1 = max(b1_1, b2_1), max(b1_2, b2_2)
    x2, y2 = min(b1_3, b2_3), min(b1_4, b2_4)

    intersect = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (b1_3 - b1_1) * (b1_4 - b1_2)
    area2 = (b2_3 - b2_1) * (b2_4 - b2_2)
    union = area1 + area2 - intersect

    return intersect / union if union > 0 else 0

def preprocess_plate(plate_crop):
    """Applies a set of preprocessing steps to enhance plate image for OCR: contrast enhancement, denoising, binarization, and deskewing."""
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)
    blur = cv2.bilateralFilter(gray, 11, 16, 16)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return morph

def extract_valid_plate(plate_crop):
    """Runs OCR on the plate image and returns a valid Romanian plate string if found."""
    raw_text = pytesseract.image_to_string(preprocess_plate(plate_crop), config=CONFIG_TESSERACT)
    raw_text = raw_text.strip().replace("\n", " ").replace("\f", "")
    raw_text = ''.join(c for c in raw_text if c.isalnum() or c.isspace())
    if is_valid_plate(raw_text):
        return normalize_plate_format(raw_text)
    return None

def run_onnx_inference(frame):
    """Preprocesses frame and runs ONNX inference."""
    resized = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
    input_tensor = resized.astype(np.float32) / 255.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, axis=0)
    return session.run(output_names, {input_name: input_tensor})

def postprocess_detections(outputs, conf_thres=0.25, iou_thres=0.7):
    """Filters detections using confidence and IoU thresholds."""
    detections = []
    for detection in outputs[0]:
        boxes = detection[:4, :]
        scores = detection[4:7, :]
        class_ids = np.argmax(scores, axis=0)
        confs = scores[class_ids, np.arange(scores.shape[1])]
        valid = confs >= conf_thres
        indices = np.where(valid)[0]

        flags = np.zeros(len(indices))
        for i, idx in enumerate(indices):
            if flags[i]: continue
            box, class_id, score = boxes[:, idx], class_ids[idx], confs[idx]

            for j, idx2 in enumerate(indices):
                if idx2 < idx or class_ids[idx2] != class_id:
                    continue
                if findIntersectionOverUnion(box, boxes[:, idx2]) >= iou_thres:
                    flags[j] = True

            detections.append({"bbox": box, "confidence": score, "class_id": class_id})
            flags[i] = True
    return detections

def extract_plate_box(frame, detection, x_scale, y_scale):
    """Extracts plate crop from frame based on detection."""
    x, y, w, h = detection["bbox"]
    x1, y1 = int((x - w / 2) * x_scale), int((y - h / 2) * y_scale)
    x2, y2 = int((x + w / 2) * x_scale), int((y + h / 2) * y_scale)
    if x2 - x1 < 60 or y2 - y1 < 20:
        return None, None
    return (x1, y1, x2, y2), frame[y1:y2, x1:x2]

def display_camera_with_detection():
    cap = cv2.VideoCapture(0)
    last_detection_time = 0
    conf_thres, iou_thres = 0.25, 0.7
    x_scale = y_scale = None
    ocr_interval = 3

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        if x_scale is None or y_scale is None:
            h, w = frame.shape[:2]
            x_scale = w / IMAGE_SIZE
            y_scale = h / IMAGE_SIZE

        now = time.time()
        if now - last_detection_time >= 0.5:
            outputs = run_onnx_inference(frame)
            detections = postprocess_detections(outputs, conf_thres, iou_thres)

            for det in detections:
                key, crop = extract_plate_box(frame, det, x_scale, y_scale)
                if crop is None or crop.size == 0 or key in last_detected_plates:
                    continue
                if now - last_detection_time > ocr_interval:
                    plate = extract_valid_plate(crop)
                    if plate:
                        sys.stdout.write(f"\n[{time.strftime('%H:%M:%S')}] License Plate: {plate}\n\n")
                        sys.stdout.flush()
                        last_detected_plates[key] = now
                        last_detection_time = now

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    display_camera_with_detection()
