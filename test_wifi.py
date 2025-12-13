# wifi_video_stream.py
# Standalone MJPEG video streaming over Wi-Fi to Chrome.
# Open on phone: http://<PI_WLAN_IP>:8080

from flask import Flask, Response
import cv2
import time

app = Flask(__name__)

# Change this if you use a different camera index.
CAMERA_INDEX = 0

# Optional: set a smaller resolution for smoother streaming
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

def open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return cap

def mjpeg_generator():
    cap = open_camera()
    last_ok = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                # camera hiccup: avoid tight loop
                time.sleep(0.1)
                # if it stays bad, exit
                if time.time() - last_ok > 3:
                    break
                continue

            last_ok = time.time()

            # Encode as JPEG
            ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                continue

            data = jpg.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n" +
                data + b"\r\n"
            )
    finally:
        cap.release()

@app.route("/")
def index():
    return (
        "<h3>Raspberry Pi MJPEG Stream</h3>"
        "<p>Open: <a href='/video'>/video</a></p>"
        "<img src='/video' style='max-width:100%;height:auto;'/>"
    )

@app.route("/video")
def video():
    return Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    # host=0.0.0.0 makes it reachable from your phone over Wi-Fi
    app.run(host="0.0.0.0", port=8080, threaded=True)
