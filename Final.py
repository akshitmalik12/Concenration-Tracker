import cv2
import mediapipe as mp
import time
import matplotlib.pyplot as plt
from playsound import playsound
import threading
import numpy as np

# Constants
EAR_THRESHOLD = 0.23  # Adjusted for reliability
EYE_CLOSED_TIME_LIMIT = 3  # seconds
FACE_TILT_THRESHOLD = 0.05
CONCENTRATION_DROP_RATE = 5  # per second
CONCENTRATION_RECOVERY_RATE = 2  # per second when focused

# Mediapipe Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def compute_EAR(landmarks, eye_indices, w, h):
    # 2 vertical and 1 horizontal
    A = np.linalg.norm(np.array([landmarks[eye_indices[1]].x * w, landmarks[eye_indices[1]].y * h]) -
                       np.array([landmarks[eye_indices[5]].x * w, landmarks[eye_indices[5]].y * h]))
    B = np.linalg.norm(np.array([landmarks[eye_indices[2]].x * w, landmarks[eye_indices[2]].y * h]) -
                       np.array([landmarks[eye_indices[4]].x * w, landmarks[eye_indices[4]].y * h]))
    C = np.linalg.norm(np.array([landmarks[eye_indices[0]].x * w, landmarks[eye_indices[0]].y * h]) -
                       np.array([landmarks[eye_indices[3]].x * w, landmarks[eye_indices[3]].y * h]))
    ear = (A + B) / (2.0 * C)
    return ear

def play_alert():
    try:
        playsound('alert.wav')
    except:
        pass

# Init
cap = cv2.VideoCapture(0)
concentration = 100
concentration_history = []
timestamp_history = []
neutral_nose_x = None
prev_time = time.time()
eyes_closed_start = None
alert_played = False
calibrating = True
calibration_start = time.time()
ear_baseline = None
ear_values = []

print("Calibrating... Face the camera for 5 seconds")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    h, w, _ = frame.shape
    current_time = time.time()
    now_str = time.strftime("%H:%M:%S")

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        nose_x = landmarks[1].x

        # Calibration phase
        if calibrating:
            left_ear = compute_EAR(landmarks, LEFT_EYE, w, h)
            right_ear = compute_EAR(landmarks, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2
            ear_values.append(avg_ear)
            if neutral_nose_x is None:
                neutral_nose_x = nose_x

            cv2.putText(frame, "Calibrating... Please face the camera", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            if current_time - calibration_start > 5:
                ear_baseline = np.mean(ear_values)
                calibrating = False
                print(f"Calibration Complete: Baseline EAR = {ear_baseline:.3f}")
            cv2.imshow("Concentration Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('z'):
                break
            continue

        # EAR Detection
        left_ear = compute_EAR(landmarks, LEFT_EYE, w, h)
        right_ear = compute_EAR(landmarks, RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2

        # Eyes Closed Handling
        eyes_closed = avg_ear < EAR_THRESHOLD
        if eyes_closed:
            if eyes_closed_start is None:
                eyes_closed_start = current_time
            elif current_time - eyes_closed_start > EYE_CLOSED_TIME_LIMIT:
                drop = (current_time - eyes_closed_start) * CONCENTRATION_DROP_RATE
                concentration -= drop
                eyes_closed_start = current_time
        else:
            eyes_closed_start = None
            concentration = min(100, concentration + CONCENTRATION_RECOVERY_RATE)

        # Face tilt handling
        deviation = abs(nose_x - neutral_nose_x)
        if deviation > FACE_TILT_THRESHOLD:
            drop = (deviation - FACE_TILT_THRESHOLD) * 100
            concentration -= drop

        # Clamp
        concentration = max(0, min(100, concentration))

        # Alert
        if concentration < 50 and not alert_played:
            threading.Thread(target=play_alert).start()
            alert_played = True
        elif concentration >= 50:
            alert_played = False

        # Logging
        if current_time - prev_time >= 1:
            concentration_history.append(round(concentration, 2))
            timestamp_history.append(now_str)
            prev_time = current_time

        # Draw landmarks
        mp.solutions.drawing_utils.draw_landmarks(
            image=frame,
            landmark_list=results.multi_face_landmarks[0],
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec,
        )

        # Color-coded concentration bar
        bar_color = (0, 255, 0) if concentration >= 70 else (0, 255, 255) if concentration >= 50 else (0, 0, 255)
        cv2.rectangle(frame, (30, 60), (int(30 + 3 * concentration), 90), bar_color, -1)
        cv2.putText(frame, f"Concentration: {int(concentration)}%", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, bar_color, 2)

        # Eyes Closed Label
        if eyes_closed:
            cv2.putText(frame, "Eyes Closed", (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Blinking PLEASE FOCUS message
        if concentration < 50 and int(current_time) % 2 == 0:
            cv2.putText(frame, "PLEASE FOCUS", (int(w / 2) - 200, int(h / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

    cv2.imshow("Concentration Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('z'):
        break

cap.release()
cv2.destroyAllWindows()

# Plotting Concentration Graph
if concentration_history:
    avg_c = sum(concentration_history) / len(concentration_history)
    max_c = max(concentration_history)
    min_c = min(concentration_history)

    plt.figure(figsize=(12, 6))
    plt.plot(timestamp_history, concentration_history, marker='o', color='blue', label='Concentration')
    plt.axhline(avg_c, color='red', linestyle='--', label=f'Average: {avg_c:.2f}%')
    plt.axhline(max_c, color='green', linestyle='--', label=f'Highest: {max_c:.2f}%')
    plt.axhline(min_c, color='orange', linestyle='--', label=f'Lowest: {min_c:.2f}%')
    plt.xlabel("Time")
    plt.ylabel("Concentration (%)")
    plt.title("Concentration Over Time")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("concentration_graph.png")
    plt.show()
