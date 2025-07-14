# raspberry-plate-recognition

License plate detection using computer vision on Raspberry Pi (resource-optimized).

This is the most minimalistic version of the project [https://github.com/Charlyhno-eng/license-plate-recognition-CV](https://github.com/Charlyhno-eng/license-plate-recognition-CV). The goal is to keep it as lightweight as possible to run efficiently on a Raspberry Pi.

Currently suitable for Romanian license plates. If you want other formats, please create them in the plate_format folder.

## Installation

- sudo apt install tesseract-ocr

- python3 -m venv venv
- source venv/bin/activate
- pip install -r requirements.txt

## Usage

- python3 -m venv venv
- source venv/bin/activate
- python plate_detector_live.py

You can try to run the ONNX-based version which consumes less power. However, it does not work on all devices (this was the case with my Raspberry Pi), in which case run the version without ONNX.
