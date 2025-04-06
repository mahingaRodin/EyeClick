import cv2
import mediapipe as mp
import pyautogui

# Initialize camera and screen size
cam = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()

# Set up MediaPipe FaceMesh with refined landmarks
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Drawing utils (optional, for debug)
mp_drawing = mp.solutions.drawing_utils

while True:
    success, frame = cam.read()
    if not success:
        break

    # Flip the frame horizontally for natural interaction
    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)

    # Extract facial landmarks
    if output.multi_face_landmarks:
        landmarks = output.multi_face_landmarks[0].landmark

        # Eye landmarks used for cursor control (right iris area)
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            # Use landmark 475 to control the mouse
            if id == 1:
                screen_x = screen_w / frame_w * x
                screen_y = screen_h / frame_h * y
                pyautogui.moveTo(screen_x, screen_y)

        # Eye blink detection (left eye)
        left_eye = [landmarks[145], landmarks[159]]
        for landmark in left_eye:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

        # Click if eye is closed
        if abs(left_eye[0].y - left_eye[1].y) < 0.004:
            pyautogui.click()
            pyautogui.sleep(1)

    # Show the video feed
    cv2.imshow('Eye Controlled Mouse', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cam.release()
cv2.destroyAllWindows()
