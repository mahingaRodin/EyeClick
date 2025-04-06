import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize camera and screen size
cam = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()

# Setup FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Click cooldown timer
last_click_time = 0
click_cooldown = 1  # seconds

while True:
    success, frame = cam.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)

    if output.multi_face_landmarks:
        landmarks = output.multi_face_landmarks[0].landmark

        # Cursor control using landmark 475 (right eye area)
        eye_landmark = landmarks[475]
        x = int(eye_landmark.x * frame_w)
        y = int(eye_landmark.y * frame_h)
        screen_x = screen_w / frame_w * x
        screen_y = screen_h / frame_h * y
        pyautogui.moveTo(screen_x, screen_y)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Left eye blink detection
        left_eye_top = landmarks[159]
        left_eye_bottom = landmarks[145]
        blink_distance = abs(left_eye_top.y - left_eye_bottom.y)

        # Visualize blink
        x1 = int(left_eye_top.x * frame_w)
        y1 = int(left_eye_top.y * frame_h)
        x2 = int(left_eye_bottom.x * frame_w)
        y2 = int(left_eye_bottom.y * frame_h)
        cv2.circle(frame, (x1, y1), 3, (255, 0, 255), -1)
        cv2.circle(frame, (x2, y2), 3, (255, 0, 255), -1)

        # If blink detected and cooldown passed, click
        if blink_distance < 0.004:
            current_time = time.time()
            if current_time - last_click_time > click_cooldown:
                pyautogui.click()
                print("Click triggered!")
                last_click_time = current_time

    # Show frame
    cv2.imshow('Eye Controlled Mouse', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
