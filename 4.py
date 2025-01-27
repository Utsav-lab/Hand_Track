import cv2
import mediapipe as mp
import time
from collections import deque
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
import math

# Initialize Mediapipe Hand Module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize Audio Control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

# Gesture tracking variables
gesture_history = deque(maxlen=10)  # History for gesture tracking
smooth_x = deque(maxlen=5)  # Smoothing for X-coordinate
smooth_y = deque(maxlen=5)  # Smoothing for Y-coordinate
gesture_cooldown = 2  # Cooldown time in seconds
last_gesture_time = time.time()

# Helper Functions
def smooth_coordinates(x, y):
    """Smooth the x and y coordinates using a moving average."""
    smooth_x.append(x)
    smooth_y.append(y)
    return sum(smooth_x) / len(smooth_x), sum(smooth_y) / len(smooth_y)

def detect_swipe_direction():
    """Detect the swipe direction based on movement history."""
    if len(gesture_history) < 2:
        return None
    dx = gesture_history[-1][0] - gesture_history[0][0]
    dy = gesture_history[-1][1] - gesture_history[0][1]
    if abs(dx) > 75 and abs(dx) > abs(dy):  # Horizontal swipe
        return "right" if dx > 0 else "left"
    elif abs(dy) > 75 and abs(dy) > abs(dx):  # Vertical swipe
        return "up" if dy < 0 else "down"
    return None

def increase_volume():
    """Increase system volume."""
    current_volume = volume.GetMasterVolumeLevelScalar()
    volume.SetMasterVolumeLevelScalar(min(current_volume + 0.1, 1.0), None)
    print(f"Volume increased to: {int(current_volume * 100)}%")

def lower_volume():
    """Lower system volume."""
    current_volume = volume.GetMasterVolumeLevelScalar()
    volume.SetMasterVolumeLevelScalar(max(current_volume - 0.1, 0.0), None)
    print(f"Volume decreased to: {int(current_volume * 100)}%")

# Main Gesture Detection
def detect_gestures(hand_landmarks, frame):
    global last_gesture_time
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    x, y = index_tip.x * frame.shape[1], index_tip.y * frame.shape[0]
    thumb_x, thumb_y = thumb_tip.x * frame.shape[1], thumb_tip.y * frame.shape[0]

    # Smooth the coordinates
    x, y = smooth_coordinates(x, y)
    gesture_history.append((x, y))

    # Detect swipe gestures
    direction = detect_swipe_direction()
    if direction and time.time() - last_gesture_time > gesture_cooldown:
        if direction == "right":
            print("Swipe Right Gesture Detected!")
        elif direction == "left":
            print("Swipe Left Gesture Detected!")
        elif direction == "up":
            print("Swipe Up Gesture Detected! Increasing volume.")
            increase_volume()
        elif direction == "down":
            print("Swipe Down Gesture Detected! Lowering volume.")
            lower_volume()
        last_gesture_time = time.time()

    # Detect pinch gesture
    distance = math.sqrt((thumb_x - x) ** 2 + (thumb_y - y) ** 2)
    if distance < 20 and time.time() - last_gesture_time > gesture_cooldown:
        print("Pinch Gesture Detected!")
        last_gesture_time = time.time()

    # Display gesture on frame
    cv2.putText(frame, f"Gesture: {direction if direction else 'None'}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# OpenCV Video Capture
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB for Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand landmarks
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Detect gestures
            detect_gestures(hand_landmarks, frame)

    # Display the frame
    cv2.imshow("Gesture Recognition", frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
