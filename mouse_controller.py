
import cv2
import mediapipe as mp
import pyautogui
import math
import time
import os
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

screen_w, screen_h = pyautogui.size()
frame_count = 0

PINCH_THRESHOLD = 40
CLICK_COOLDOWN = 0.5
SMOOTHING_FRAMES = 5
SCROLL_SENSITIVITY = 2

last_click_time = 0
position_history = deque(maxlen=SMOOTHING_FRAMES)
prev_scroll_y = None

# Avoid PyAutoGUI's corner fail-safe triggering accidentally
pyautogui.PAUSE = 0.01


def finger_is_up(landmarks, tip_idx, pip_idx):
    return landmarks[tip_idx].y < landmarks[pip_idx].y


try:
    while True:
        success, frame = cap.read()
        if not success:
            print("Could not read webcam frame")
            break

        frame = cv2.flip(frame, 1)
        frame_h, frame_w = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Monotonically increasing timestamps for VIDEO mode
        timestamp_ms = int(time.monotonic() * 1000)
        result = detector.detect_for_video(mp_image, timestamp_ms)

        mode = "NO HAND"

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]

            index_up = finger_is_up(hand, 8, 6)
            middle_up = finger_is_up(hand, 12, 10)
            ring_up = finger_is_up(hand, 16, 14)
            pinky_up = finger_is_up(hand, 20, 18)

            index_tip = hand[8]
            thumb_tip = hand[4]
            middle_tip = hand[12]

            ix = int(index_tip.x * frame_w)
            iy = int(index_tip.y * frame_h)
            tx = int(thumb_tip.x * frame_w)
            ty = int(thumb_tip.y * frame_h)
            mx = int(middle_tip.x * frame_w)
            my = int(middle_tip.y * frame_h)

            is_fist = not any([
                index_up, middle_up, ring_up, pinky_up
            ])
            is_scroll = (
                index_up and middle_up
                and not ring_up and not pinky_up
            )

            if is_fist:
                mode = "PAUSED"
                prev_scroll_y = None
                position_history.clear()

            elif is_scroll:
                mode = "SCROLL"
                position_history.clear()

                avg_y = (iy + my) / 2

                if prev_scroll_y is not None:
                    delta = prev_scroll_y - avg_y
                    if abs(delta) > 2:
                        scroll_amount = int(delta * SCROLL_SENSITIVITY)
                        if scroll_amount:
                            pyautogui.scroll(scroll_amount)

                prev_scroll_y = avg_y

                cv2.circle(frame, (ix, iy), 8, (255, 255, 0), -1)
                cv2.circle(frame, (mx, my), 8, (255, 255, 0), -1)

            else:
                mode = "MOVE"
                prev_scroll_y = None

                position_history.append((ix, iy))
                avg_x = sum(p[0] for p in position_history) / len(position_history)
                avg_y = sum(p[1] for p in position_history) / len(position_history)

                # Map camera coordinates to screen coordinates
                screen_x = int(avg_x * screen_w / frame_w)
                screen_y = int(avg_y * screen_h / frame_h)

                screen_x = max(0, min(screen_x, screen_w - 1))
                screen_y = max(0, min(screen_y, screen_h - 1))

                pyautogui.moveTo(screen_x, screen_y, duration=0)

                distance = math.hypot(tx - ix, ty - iy)

                cv2.circle(frame, (ix, iy), 8, (0, 255, 0), -1)
                cv2.circle(frame, (tx, ty), 8, (255, 0, 0), -1)
                cv2.line(frame, (ix, iy), (tx, ty), (255, 255, 0), 2)

                if distance < PINCH_THRESHOLD:
                    current_time = time.time()
                    if current_time - last_click_time > CLICK_COOLDOWN:
                        pyautogui.click()
                        last_click_time = current_time
                        cv2.putText(
                            frame, "CLICK", (50, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2
                        )

            cv2.putText(
                frame, f"Mode: {mode}", (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0) if mode == "MOVE" else (0, 165, 255), 2
            )

            cv2.putText(
                frame,
                f"Index: {index_up} Middle: {middle_up} "
                f"Ring: {ring_up} Pinky: {pinky_up}",
                (30, frame_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2
            )

        else:
            prev_scroll_y = None
            position_history.clear()
            cv2.putText(
                frame, "NO HAND DETECTED", (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )

        cv2.imshow("Hand Mouse - press Q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
