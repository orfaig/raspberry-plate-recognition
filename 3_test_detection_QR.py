#!/usr/bin/env python3
# coding: utf-8
"""
Student exercise: calibrate a QR detector using simple filters.

Goal
----
Count a detection ONLY when it passes calibration filters:
1) Minimum size (bbox width/height)
2) "Square-ish" shape (affine OK): width/height aspect ratio close to 1.0

How to calibrate (exercise steps)
---------------------------------
1) Put the QR on A4 paper. Start close to camera (big QR).
2) Increase distance until detection becomes unstable.
3) Adjust:
   - qr_min_w / qr_min_h so far detections are rejected (reduce false positives).
   - qr_min_aspect / qr_max_aspect to accept tilted square (but not rectangles).
4) Turn on --debug to see measured bbox + aspect on screen.

Suggested tuning:
- Start: min_w=min_h=50, aspect range 0.85..1.15
- If it rejects when the page is tilted: widen to 0.80..1.25
- If it accepts rectangles: tighten to 0.90..1.10
"""

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"   # important for Raspberry Pi desktop

import cv2
import threading
import time
import numpy as np
import argparse
from flask import Flask, Response


# ============================================================
# PARAMETERS (all in one place)
# ============================================================
def build_args():
    p = argparse.ArgumentParser()

    # camera
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--w", type=int, default=640)
    p.add_argument("--h", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)

    # web stream
    p.add_argument("--port", type=int, default=8080)

    # counting UI
    p.add_argument("--cooldown_ms", type=int, default=600,
                   help="Count again only if last count was more than this many ms ago.")
    p.add_argument("--show_text_s", type=float, default=1.5,
                   help="Show last decoded text for this many seconds.")
    p.add_argument("--debug", action="store_true",
                   help="Show bbox size + aspect ratio for calibration.")

    # calibration filters (the exercise knobs)
    p.add_argument("--qr_min_w", type=int, default=35,
                   help="Reject detection if bbox width < this.")
    p.add_argument("--qr_min_h", type=int, default=35,
                   help="Reject detection if bbox height < this.")
    p.add_argument("--qr_min_aspect", type=float, default=0.6,
                   help="Reject detection if (w/h) < this.")
    p.add_argument("--qr_max_aspect", type=float, default=1.,
                   help="Reject detection if (w/h) > this.")

    return p.parse_args()


# ============================================================
# FLASK STREAM (MJPEG)
# ============================================================
app = Flask(__name__)
latest_jpeg = None
running = True

def generate_stream():
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
          function onStreamError() { setTimeout(reloadStream, 1000); }
          window.onload = function() { setInterval(reloadStream, 30000); }
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

def run_flask(port: int):
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


# ============================================================
# DRAWING / FILTERS
# ============================================================
def points_to_quad(points):
    if points is None:
        return None
    pts = points[0] if len(points.shape) == 3 else points
    return np.asarray(pts, dtype=np.float32).reshape(4, 2)

def quad_bbox(pts):
    xs = pts[:, 0]
    ys = pts[:, 1]
    x1 = int(np.min(xs)); y1 = int(np.min(ys))
    x2 = int(np.max(xs)); y2 = int(np.max(ys))
    return x1, y1, x2, y2

def draw_qr_quad(frame, pts, ok):
    color = (0, 255, 0) if ok else (0, 0, 255)
    p = pts.astype(int)
    cv2.polylines(frame, [p], True, color, 2)
    for (x, y) in p:
        cv2.circle(frame, (int(x), int(y)), 4, (255, 255, 255), -1)

def passes_square_filters(pts, args):
    """
    Exercise: students calibrate these filters.
    - size filter: bbox width/height
    - aspect filter: bbox w/h close to 1.0 for a square-ish target
    """
    x1, y1, x2, y2 = quad_bbox(pts)
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    aspect = bw / float(bh)

    size_ok = (bw >= args.qr_min_w) and (bh >= args.qr_min_h)
    aspect_ok = (args.qr_min_aspect <= aspect <= args.qr_max_aspect)

    ok = size_ok and aspect_ok
    metrics = {"bw": bw, "bh": bh, "aspect": aspect, "size_ok": size_ok, "aspect_ok": aspect_ok}
    return ok, metrics

def draw_counter(frame, count, decoded_preview=""):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    bar_h = 70 if decoded_preview else 50
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    cv2.putText(frame, f"Accepted detections: {count}", (10, 32),
                font, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

    if decoded_preview:
        cv2.putText(frame, decoded_preview[:60], (10, 62),
                    font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

def draw_debug(frame, metrics, args):
    font = cv2.FONT_HERSHEY_SIMPLEX
    txt1 = f"bbox: {metrics['bw']}x{metrics['bh']}  aspect={metrics['aspect']:.2f}"
    txt2 = f"min_w={args.qr_min_w} min_h={args.qr_min_h}  aspect_range=[{args.qr_min_aspect:.2f},{args.qr_max_aspect:.2f}]"
    cv2.putText(frame, txt1, (10, 110), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, txt2, (10, 140), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


# ============================================================
# MAIN
# ============================================================
def main():
    global latest_jpeg, running

    args = build_args()

    # start Flask in background thread
    flask_thread = threading.Thread(target=run_flask, args=(args.port,), daemon=True)
    flask_thread.start()

    # open camera in MAIN THREAD
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.h)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        print("Error: cannot open camera")
        running = False
        return

    detector = cv2.QRCodeDetector()

    print("Camera started. Press 'q' in the window to quit.")
    print(f"From phone, open:  http://<PI_IP>:{args.port}")
    print("Calibration exercise: tune --qr_min_w/--qr_min_h/--qr_min_aspect/--qr_max_aspect")
    if args.debug:
        print("Debug enabled: showing bbox size + aspect on the video.")

    # detection counter state
    accepted_count = 0
    last_count_t = 0.0
    cooldown_s = max(0.0, args.cooldown_ms / 1000.0)

    # optional: show decoded text briefly
    last_text = ""
    last_text_t = 0.0

    fail_count = 0

    try:
        while running:
            ret, frame = cap.read()
            if not ret:
                fail_count += 1
                if fail_count > 10:
                    print("Camera read failed, trying to reopen...")
                    cap.release()
                    time.sleep(1.0)
                    cap = cv2.VideoCapture(args.cam)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.w)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.h)
                    cap.set(cv2.CAP_PROP_FPS, args.fps)
                    if not cap.isOpened():
                        print("Error: cannot reopen camera, stopping.")
                        running = False
                        break
                    fail_count = 0
                continue
            fail_count = 0

            now = time.time()

            # --- QR detection/decoding ---
            text, points, _ = detector.detectAndDecode(frame)

            metrics = None
            if points is not None:
                pts = points_to_quad(points)

                ok, metrics = passes_square_filters(pts, args)
                draw_qr_quad(frame, pts, ok)

                if ok and (now - last_count_t) >= cooldown_s:
                    accepted_count += 1
                    last_count_t = now

            # decoded text preview
            if text:
                last_text = text
                last_text_t = now

            show_text = ""
            if last_text and (now - last_text_t) <= float(args.show_text_s):
                show_text = last_text

            draw_counter(frame, accepted_count, decoded_preview=show_text)

            if args.debug and metrics is not None:
                draw_debug(frame, metrics, args)

            # encode for web
            ok, buffer = cv2.imencode(".jpg", frame)
            if ok:
                latest_jpeg = buffer.tobytes()

            # show on Pi screen
            cv2.imshow("QR calibration exercise (press q to quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                running = False
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        running = False
        print("Stopped.")


if __name__ == "__main__":
    main()
