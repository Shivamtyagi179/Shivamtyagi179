"""
Virtual Keyboard with Text Storage + Display
- Stores all pressed keys in typed_text
- Shows typed text live on OpenCV window
"""

import cv2
import numpy as np
import mediapipe as mp
from pynput.keyboard import Controller, Key
import time

# --- Configuration ---
CAM_W, CAM_H = 1280, 720
KEY_WIDTH = 90
KEY_HEIGHT = 90
KEY_GAP = 10
TOP_MARGIN = 160
LEFT_MARGIN = 40
CLICK_DISTANCE_THRESH = 35
COOLDOWN = 0.45

# Keyboard layout
LAYOUT = [
    list("QWERTYUIOP"),
    list("ASDFGHJKL"),
    list("ZXCVBNM")
]

SPACE_WIDTH = KEY_WIDTH * 5 + KEY_GAP * 4

# --- Mediapipe Hands ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)

# Keyboard controller
kb = Controller()

# Build key list
keys = []
y = TOP_MARGIN
for row in LAYOUT:
    x = LEFT_MARGIN
    for k in row:
        keys.append({'label': k, 'x': x, 'y': y, 'w': KEY_WIDTH, 'h': KEY_HEIGHT, 'last': 0})
        x += KEY_WIDTH + KEY_GAP
    y += KEY_HEIGHT + KEY_GAP

# Space key
space_x = LEFT_MARGIN + (KEY_WIDTH + KEY_GAP) * 2
space_y = y
keys.append({'label': 'SPACE', 'x': space_x, 'y': space_y, 'w': SPACE_WIDTH, 'h': KEY_HEIGHT, 'last': 0})

# Enter key
enter_x = space_x + SPACE_WIDTH + KEY_GAP
enter_y = y
keys.append({'label': 'ENTER', 'x': enter_x, 'y': enter_y, 'w': KEY_WIDTH * 2, 'h': KEY_HEIGHT, 'last': 0})

# Store typed text
typed_text = ""


def draw_keyboard(img, keys, hover_label=None):
    """Draw all keys."""
    for k in keys:
        x, y, w, h = k['x'], k['y'], k['w'], k['h']
        label = k['label']
        if label == hover_label:
            cv2.rectangle(img, (x, y), (x + w, y + h), (80, 200, 120), -1)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 50), 2)

        text = label if label != 'SPACE' else 'SPACE'
        font = cv2.FONT_HERSHEY_SIMPLEX
        size = cv2.getTextSize(text, font, 0.8, 2)[0]
        tx = x + (w - size[0]) // 2
        ty = y + (h + size[1]) // 2
        cv2.putText(img, text, (tx, ty), font, 0.8, (10, 10, 10), 2, cv2.LINE_AA)


def find_hover_key(x, y, keys):
    for k in keys:
        if k['x'] <= x <= k['x'] + k['w'] and k['y'] <= y <= k['y'] + k['h']:
            return k
    return None


def press_key(k):
    """Simulate press + store in typed_text."""
    global typed_text
    label = k['label']
    if label == 'SPACE':
        kb.press(Key.space); kb.release(Key.space)
        typed_text += " "
    elif label == 'ENTER':
        kb.press(Key.enter); kb.release(Key.enter)
        typed_text += "\n"
    else:
        ch = label.lower()
        kb.press(ch); kb.release(ch)
        typed_text += ch
    print("Typed so far:", typed_text)


def distance(p1, p2):
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        hover_key = None

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape
            lm = hand.landmark
            index_tip = (int(lm[8].x * w), int(lm[8].y * h))
            middle_tip = (int(lm[12].x * w), int(lm[12].y * h))

            cv2.circle(frame, index_tip, 8, (0, 0, 255), -1)
            cv2.circle(frame, middle_tip, 6, (255, 0, 0), -1)

            hover = find_hover_key(index_tip[0], index_tip[1], keys)
            if hover:
                hover_key = hover['label']

            d = distance(index_tip, middle_tip)
            if hover and d < CLICK_DISTANCE_THRESH:
                now = time.time()
                if now - hover['last'] > COOLDOWN:
                    press_key(hover)
                    hover['last'] = now
                    cx = hover['x'] + hover['w'] // 2
                    cy = hover['y'] + hover['h'] // 2
                    cv2.circle(frame, (cx, cy), 30, (0, 255, 0), 4)

        # Draw keyboard
        draw_keyboard(frame, keys, hover_label=hover_key)

        # Draw typed text at top
        cv2.rectangle(frame, (20, 40), (CAM_W - 20, 100), (255, 255, 255), -1)
        cv2.putText(frame, typed_text[-70:], (30, 85), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.putText(frame, "Press 'q' to quit", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Virtual Keyboard", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
