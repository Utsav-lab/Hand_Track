import cv2
import mediapipe as mp
import time
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

# Gesture variables
prev_x = 0
prev_y = 0
gesture_cooldown = 2  # Cooldown time in seconds
last_gesture_time = time.time()

def detect_gestures(hand_landmarks):
    global prev_x, prev_y, last_gesture_time

    # Get landmarks for the index finger tip and thumb tip
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    x, y = index_tip.x, index_tip.y
    thumb_x, thumb_y = thumb_tip.x, thumb_tip.y

    # Map normalized coordinates to screen size (optional)
    screen_width = 640  # Adjust based on your frame size
    screen_height = 480
    x, y = x * screen_width, y * screen_height
    thumb_x, thumb_y = thumb_x * screen_width, thumb_y * screen_height

    # Detect Swipe Gestures
    if prev_x != 0 and prev_y != 0:  # Ensure previous frame data exists
        if time.time() - last_gesture_time > gesture_cooldown:  # Cooldown check
            if x - prev_x > 30:  # Swipe Right
                print("Swipe Right Gesture Detected!")
                last_gesture_time = time.time()
            elif prev_x - x > 30:  # Swipe Left
                print("Swipe Left Gesture Detected!")
                last_gesture_time = time.time()
            elif prev_y - y > 30:  # Swipe Down
                print("Swipe Down Gesture Detected! Lowering volume.")
                lower_volume()
                last_gesture_time = time.time()
            elif y - prev_y > 30:  # Swipe Up
                print("Swipe Up qqGesture Detected! Increasing volume.")
                increase_volume()
                last_gesture_time = time.time()

    # Detect Pinch Gesture
    distance = math.sqrt((thumb_x - x) ** 2 + (thumb_y - y) ** 2)
    if distance < 20:  # Distance threshold for pinch
        print("Pinch Gesture Detected!")
        last_gesture_time = time.time()

    # Update previous positions
    prev_x, prev_y = x, y

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
            detect_gestures(hand_landmarks)

    # Display the frame
    cv2.imshow("Gesture Recognition", frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
