import cv2
import mediapipe as mp
import numpy as np
import time
import subprocess

def scroll_once():
    try:
        # Swipe up = Scroll down
        subprocess.run(["adb", "shell", "input", "swipe", "500", "1200", "500", "800", "100"], check=True)
        print("✅ Scrolled down on mobile via ADB")
    except subprocess.CalledProcessError:
        print("❌ ADB swipe failed — is your device connected with USB debugging?")
    except FileNotFoundError:
        print("❌ 'adb' not found — make sure Android SDK platform-tools is installed and adb is in your PATH.")

def calculate_ear(landmarks, eye_indices):
    p2 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
    p6 = np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])
    p3 = np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])
    p5 = np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])
    p1 = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
    p4 = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = 2 * np.linalg.norm(p1 - p4)
    return vertical / horizontal

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Eye indices
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

# Blink parameters
EAR_THRESH = 0.27
CONSEC_FRAMES = 1
DOUBLE_BLINK_GAP = 0.5
COOLDOWN = 1.0

blink_frame_counter = 0
last_blink_time = 0
blink_times = []
last_scroll_time = 0

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        left_ear = calculate_ear(landmarks, LEFT_EYE_IDX)
        right_ear = calculate_ear(landmarks, RIGHT_EYE_IDX)
        avg_ear = (left_ear + right_ear) / 2

        if avg_ear < EAR_THRESH:
            blink_frame_counter += 1
        else:
            if blink_frame_counter >= CONSEC_FRAMES:
                current_time = time.time()

                if current_time - last_blink_time <= DOUBLE_BLINK_GAP:
                    blink_times.append(current_time)
                    if len(blink_times) >= 2:
                        if blink_times[-1] - blink_times[-2] <= DOUBLE_BLINK_GAP:
                            if current_time - last_scroll_time > COOLDOWN:
                                scroll_once()
                                last_scroll_time = current_time
                                blink_times = []
                else:
                    blink_times = [current_time]

                last_blink_time = current_time
            blink_frame_counter = 0

        # Draw landmarks
        for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX:
            cx, cy = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
            cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)

        cv2.putText(frame, f"EAR: {avg_ear:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Blink Frames: {blink_frame_counter}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Cooldown: {'ON' if time.time() - last_scroll_time < COOLDOWN else 'OFF'}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 100, 255), 2)

    # ✅ Add Watermark
    cv2.putText(frame, "Built by Manju", (w - 200, h - 20),
                cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Double Blink Scroll", frame)
    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
        break

cap.release()
cv2.destroyAllWindows()
