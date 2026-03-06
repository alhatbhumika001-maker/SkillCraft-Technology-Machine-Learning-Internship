import cv2
import mediapipe as mp

# New MediaPipe API
mp_hands = mp.tasks.vision.HandLandmarker
mp_base = mp.tasks.BaseOptions
mp_vision = mp.tasks.vision

# Load hand landmarker model
base_options = mp_base(model_asset_path="hand_landmarker.task")

options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)

detector = mp_vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()