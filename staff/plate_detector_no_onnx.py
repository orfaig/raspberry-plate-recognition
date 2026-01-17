import time
import cv2
import pytesseract
import numpy as np
from ultralytics import YOLO
from plate_format.plate_format_ro import is_valid_plate, normalize_plate_format

model = YOLO("yolov8n-license_plate.pt")

last_detected_plates = {}
max_plate_age_seconds = 10

def deskew_image(image, max_angle=10):
    """
    Straightens the image if it is tilted.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    if lines is None:
        return image

    angles = []
    for rho, theta in lines[:, 0]:
        angle_deg = (theta * 180 / np.pi) - 90
        if -max_angle <= angle_deg <= max_angle:
            angles.append(angle_deg)

    if not angles:
        return image

    median_angle = np.median(angles)
    if abs(median_angle) < 1:
        return image

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def preprocess_plate(plate_crop):
    """
    Applies a set of preprocessing steps to enhance plate image for OCR.
    Includes contrast enhancement, denoising, binarization, and deskewing.
    """
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)
    blur = cv2.bilateralFilter(gray, 11, 16, 16)
    color_for_deskew = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
    deskewed = deskew_image(color_for_deskew)
    deskewed_gray = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(deskewed_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return morph

def extract_valid_plate(plate_crop):
    """
    Runs OCR on the plate image and returns a valid Romanian plate string if found.
    """
    processed = preprocess_plate(plate_crop)
    config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    raw_text = pytesseract.image_to_string(processed, config=config)
    raw_text = raw_text.strip().replace("\n", " ").replace("\f", "")
    raw_text = ''.join(c for c in raw_text if c.isalnum() or c.isspace())

    if is_valid_plate(raw_text):
        return normalize_plate_format(raw_text)

    return None

def display_camera_with_detection():
    cap = cv2.VideoCapture(0)

    last_detection_time = 0
    ocr_interval_second = 3

    frame_count = 0
    detect_every_n_frames = 2

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        if frame_count % detect_every_n_frames != 0:
            time.sleep(0.01)
            continue

        current_time = time.time()

        resize_width = 640
        height, width = frame.shape[:2]
        scale = resize_width / width
        frame_small = cv2.resize(frame, (resize_width, int(height * scale)))

        results = model.predict(source=frame_small, conf=0.25, imgsz=640, verbose=False)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                width = x2 - x1
                height = y2 - y1

                if width < 60 or height < 20:
                    continue

                key = (x1, y1, x2, y2)
                if key in last_detected_plates and current_time - last_detected_plates[key] < max_plate_age_seconds:
                    continue

                if current_time - last_detection_time > ocr_interval_second:
                    plate_crop = frame[y1:y2, x1:x2]
                    if plate_crop.size == 0:
                        continue

                    plate = extract_valid_plate(plate_crop)
                    if plate:
                        print(f"[{time.strftime('%H:%M:%S')}] Plate detected : {plate}")
                        last_detected_plates[key] = current_time
                        last_detection_time = current_time

        time.sleep(0.02)

    cap.release()

if __name__ == "__main__":
    display_camera_with_detection()
