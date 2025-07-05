import cv2
import mediapipe as mp

# Initialize MediaPipe and drawing utils
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Use MediaPipe Hands with detection + tracking
with mp_hands.Hands(static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.5) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip + convert to RGB
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame
        results = hands.process(rgb)
        gesture = "No hand detected"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                tips_ids = [8, 12, 16, 20]  # Index to pinky
                finger_states = []  # 1 if extended, 0 if folded

                for tip_id in tips_ids:
                    is_extended = hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y
                    finger_states.append(1 if is_extended else 0)

                # Define gestures by matching finger_states
                if finger_states == [1, 1, 1, 1]:
                    gesture = "STOP ✋"
                elif finger_states == [0, 0, 0, 0]:
                    gesture = "GO 👊"
                elif finger_states == [1, 0, 0, 1]:
                    gesture = "ROCK 🤟"
                elif finger_states == [1, 0, 0, 0]:
                    gesture = "POINT 👉"
                elif finger_states == [1, 1, 0, 0]:
                    gesture = "PEACE ✌️"
                else:
                    gesture = "Unknown Gesture"

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show result
        cv2.putText(frame, gesture, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("Hand Gesture Control", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

cap.release()
cv2.destroyAllWindows()
# Cleanup   
mp_hands.close()
