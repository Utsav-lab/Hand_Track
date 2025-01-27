import cv2
import mediapipe as mp
import time

# Initialize Mediapipe Hand Module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Gesture variables
prev_x = 0
prev_y = 0
gesture_cooldown = 2  # Seconds to avoid multiple triggers
last_gesture_time = time.time()

def detect_swipe_or_tap(hand_landmarks):
    global prev_x, prev_y, last_gesture_time

    # Get coordinates of the index finger tip
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    x, y = index_finger_tip.x, index_finger_tip.y  # Normalized coordinates (0 to 1)

    # Map normalized coordinates to screen size (optional)
    screen_width = 640  # Adjust based on your frame size
    screen_height = 480
    x *= screen_width
    y *= screen_height

    # Detect swipe gesture
    if prev_x != 0 and prev_y != 0:  # Ensure we have a previous frame
        if time.time() - last_gesture_time > gesture_cooldown:  # Cooldown check
            if x - prev_x > 50:  # Swipe Right
                print("Swipe Right Gesture Detected!")
                last_gesture_time = time.time()
            elif prev_x - x > 50:  # Swipe Left
                print("Swipe Left Gesture Detected!")
                last_gesture_time = time.time()
    
    # Detect tap gesture (vertical movement of index finger)
    if time.time() - last_gesture_time > gesture_cooldown:  # Cooldown check
        if prev_y - y > 20:  # If finger moved downward
            print("Tap Gesture Detected!")
            last_gesture_time = time.time()

    # Update previous positions
    prev_x, prev_y = x, y

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
            detect_swipe_or_tap(hand_landmarks)

    # Display the frame
    cv2.imshow("Gesture Recognition", frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
