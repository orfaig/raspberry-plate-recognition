#!/usr/bin/env python3
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"   # important for Raspberry Pi desktop

import cv2
import threading
import time
from flask import Flask, Response

app = Flask(__name__)

latest_jpeg = None
running = True


def generate_stream():
    """Generator for MJPEG stream."""
    global latest_jpeg, running
    while running:
        if latest_jpeg is not None:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   latest_jpeg + b"\r\n")
        else:
            time.sleep(0.02)


@app.route("/")
def index():
    return """
    <html>
      <head>
        <title>Raspbot Camera</title>
        <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate" />
        <meta http-equiv="Pragma" content="no-cache" />
        <meta http-equiv="Expires" content="0" />
        <script>
          function reloadStream() {
            const img = document.getElementById('cam');
            img.src = '/video_feed?ts=' + new Date().getTime();
          }
          function onStreamError() {
            setTimeout(reloadStream, 1000);
          }
          window.onload = function() {
            setInterval(reloadStream, 30000);
          }
        </script>
      </head>
      <body>
        <h1>Raspbot Camera Stream</h1>
        <img id="cam" src="/video_feed" onerror="onStreamError()" />
        <p>
          <button onclick="location.reload()">Refresh page</button>
          <button onclick="reloadStream()">Refresh video</button>
        </p>
        <p>Press 'q' on the Raspberry Pi window to quit.</p>
      </body>
    </html>
    """


@app.route("/video_feed")
def video_feed():
    return Response(generate_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


def run_flask():
    app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)


def draw_qr_bbox(frame, points):
    """
    points: ndarray shape (1,4,2) or (4,2) from OpenCV QRCodeDetector
    Draw polygon + corner points.
    """
    if points is None:
        return frame

    pts = points
    if len(pts.shape) == 3:
        pts = pts[0]

    pts = pts.astype(int)

    # draw polygon
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # draw corners
    for (x, y) in pts:
        cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)

    return frame


def main():
    global latest_jpeg, running

    # start Flask in background thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # open camera in MAIN THREAD
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Error: cannot open camera")
        running = False
        return

    detector = cv2.QRCodeDetector()

    print("Camera started. Press 'q' in the window to quit.")
    print("From phone, open:  http://10.2.0.17:8080")

    fail_count = 0
    last_text = ""
    last_seen_t = 0.0

    while running:
        ret, frame = cap.read()

        if not ret:
            fail_count += 1
            if fail_count > 10:
                print("Camera read failed, trying to reopen...")
                cap.release()
                time.sleep(1.0)
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)

                if not cap.isOpened():
                    print("Error: cannot reopen camera, stopping.")
                    running = False
                    break
                fail_count = 0
            continue
        else:
            fail_count = 0

        # --- QR detection (CPU-only) ---
        # If you only need bbox: use detect()
        # If you also want decoded text: use detectAndDecode()
        text, points, _ = detector.detectAndDecode(frame)

        if points is not None:
            draw_qr_bbox(frame, points)

        # show decoded text briefly (optional)
        now = time.time()
        if text:
            last_text = text
            last_seen_t = now

        if last_text and (now - last_seen_t) < 1.5:
            cv2.putText(frame, last_text[:60], (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # encode for web (with overlay/bbox)
        ok, buffer = cv2.imencode(".jpg", frame)
        if ok:
            latest_jpeg = buffer.tobytes()

        # show on Pi screen
        cv2.imshow("Raspbot Camera (press q to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            running = False
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == "__main__":
    main()
