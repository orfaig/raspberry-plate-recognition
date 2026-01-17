#!/usr/bin/env python3
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"   # important for Raspberry Pi desktop

import cv2
import threading
import time
from flask import Flask, Response

app = Flask(__name__)

latest_jpeg = None        # last encoded frame for web
running = True            # global flag for shutdown


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
    # simple HTML page with refresh button and basic auto-reload
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
            // force reload (bust cache)
            img.src = '/video_feed?ts=' + new Date().getTime();
          }
          function onStreamError() {
            // try to reload stream after 1 second if it breaks
            setTimeout(reloadStream, 1000);
          }
          window.onload = function() {
            // optional periodic refresh every 30 seconds
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
    # start web server for phone; Pi hotspot IP is 10.42.0.1
    app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)


def main():
    global latest_jpeg, running

    # start Flask in background thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # open camera in MAIN THREAD (for imshow)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Error: cannot open camera")
        running = False
        return

    print("Camera started. Press 'q' in the window to quit.")
    print("From phone, open:  http://10.2.0.17:8080")

    fail_count = 0

    while running:
        ret, frame = cap.read()

        if not ret:
            fail_count += 1
            # if camera fails too many times, try to reopen ("refresh camera")
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

        # encode for web
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
    # when main ends, running=False will naturally end the stream;
    # clients can hit Refresh in the browser to reconnect next time you start it.


if __name__ == "__main__":
    main()
