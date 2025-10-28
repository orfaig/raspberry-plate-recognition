# raspberry-plate-recognition

License plate detection using computer vision on Raspberry Pi (resource-optimized).

This project combines YOLOv8 for license plate detection and Tesseract OCR for text recognition. The system captures real-time video streams, detects license plates from the frames, and extracts the alphanumeric text printed on them.

It is designed to be compact and efficient enough to run on low-power devices such as the Raspberry Pi. The detection model can be optimized and exported to ONNX format for faster inference.

#### Key Features

- Lightweight and optimized for Raspberry Pi
- Real-time plate detection and OCR (in my case EasyOCR)
- Compatible with YOLOv8 / ONNX runtime

#### Dataset

The model uses a publicly available dataset from Kaggle:
https://www.kaggle.com/datasets/fareselmenshawii/license-plate-dataset

This dataset contains various license plate images suitable for vehicle detection and OCR model training. Additional datasets can be integrated for better generalization across countries or plate types.

## Installation

```bash
sudo apt update
sudo apt install tesseract-ocr
```

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python3 -m venv venv
source venv/bin/activate
python plate_detector_live.py
```

You can also try running the ONNX-based version, which consumes less power and offers better performance on compatible devices. However, depending on your Raspberry Pi version, ONNX may not work reliably. If that happens, use the default version instead.

---

## Training Your Own Model

If you want to adapt the detection to other license plate formats or countries:

- Collect or download a new dataset.
- Train a YOLOv8 model using the ultralytics package.
- Export the trained model to .pt or .onnx format.
- Replace the model file in the project folder.

Example :

```bash
yolo detect train data=dataset.yaml model=yolov8n.pt epochs=50 imgsz=640
```
