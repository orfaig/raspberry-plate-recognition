#!/usr/bin/env python
import cv2
import time

CANDIDATE_INDICES = [27, 33, 34, 35, 36]

def test_index(idx):
    print(f"\n[TEST] Trying index {idx} with CAP_ANY ...")
    cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
    if not cap.isOpened():
        print(f"[FAIL] Could not open index {idx} (CAP_ANY).")
        cap.release()
        return False

    # Try to set some reasonable parameters (they may or may not be applied)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    got_frame = False
    for i in range(10):
        ret, frame = cap.read()
        if ret and frame is not None:
            got_frame = True
            print(f"[ OK ] index {idx}: got a valid frame at attempt {i+1}.")
            break
        else:
            print(f"[WARN] index {idx}: failed to read frame (attempt {i+1}).")
            time.sleep(0.1)

    if not got_frame:
        print(f"[FAIL] index {idx}: opened but did not deliver frames.")
        cap.release()
        return False

    # If we got here, index works -> show preview
    print(f"[INFO] Showing live preview from index {idx}. Press 'q' to quit preview.")
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARN] Lost frame during preview.")
            time.sleep(0.1)
            continue

        cv2.putText(
            frame,
            f"idx={idx}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(f"camera idx={idx} (q to quit)", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Closed preview for index {idx}.")
    return True

def main():
    print("[INFO] Candidate camera indices:", CANDIDATE_INDICES)
    any_working = False

    for idx in CANDIDATE_INDICES:
        ok = test_index(idx)
        if ok:
            any_working = True
            # Stop after the first working index; remove this break if you want to test all
            break

    if not any_working:
        print("\n[RESULT] None of the candidate indices produced frames.")
    else:
        print("\n[RESULT] At least one index produced frames (see logs above).")

if __name__ == "__main__":
    main()
