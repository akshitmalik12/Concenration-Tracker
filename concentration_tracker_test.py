import cv2
import mediapipe as mp
import numpy as np
import os
import time
from collections import deque
import threading
import matplotlib.pyplot as plt
from mediapipe.python.solutions.drawing_utils import DrawingSpec
import pyttsx3
from playsound import playsound

# Initialize Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Initialize voice engine (not used but left for future use)
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Key landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
NOSE = 1

# Eye aspect ratio calculation
def eye_aspect_ratio(landmarks, eye_points, image_w, image_h):
    p = []
    for idx in eye_points:
        lm = landmarks[idx]
        x, y = int(lm.x * image_w), int(lm.y * image_h)
        p.append((x, y))
    A = np.linalg.norm(np.array(p[1]) - np.array(p[5]))
    B = np.linalg.norm(np.array(p[2]) - np.array(p[4]))
    C = np.linalg.norm(np.array(p[0]) - np.array(p[3]))
    return (A + B) / (2.0 * C)

# Calibration values
calibrated_ear_threshold = 0.22
calibrated_gaze_center = 0.6
calibrated_gaze_tolerance = 0.1

# Run calibration
cap = cv2.VideoCapture(0)
print("\n[CALIBRATION] Please sit straight and blink normally for 5 seconds...")
calibration_ears = []
calibration_gaze = []
start_time = time.time()

while time.time() - start_time < 5:
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_h, image_w, _ = frame.shape
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, image_w, image_h)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, image_w, image_h)
        calibration_ears.append((left_ear + right_ear) / 2)

        left_iris = landmarks[468]
        right_iris = landmarks[473]
        avg_x = (left_iris.x + right_iris.x) / 2.0
        calibration_gaze.append(avg_x)

calibrated_ear_threshold = np.mean(calibration_ears) * 0.85
calibrated_gaze_center = np.mean(calibration_gaze)
calibrated_gaze_tolerance = 0.1
print("[CALIBRATION DONE] Blink threshold:", round(calibrated_ear_threshold, 3), ", Gaze center:", round(calibrated_gaze_center, 3))

# State variables
score_history = deque(maxlen=10)
trend_history = []
distraction = 0
alert_last_played = 0
eye_closed_start_time = None
head_turn_start_time = None
focus_warning_blink = True
last_focus_blink_time = time.time()
first_frame_time = time.time()

# Blink detection with calibrated sensitivity
def is_blinking(ear):
    return ear < calibrated_ear_threshold

# Head pose score
def get_head_pose_score(landmarks):
    left_x = landmarks[LEFT_CHEEK].x
    right_x = landmarks[RIGHT_CHEEK].x
    nose_x = landmarks[NOSE].x
    center_x = (left_x + right_x) / 2
    diff = abs(nose_x - center_x)
    return 1.0 if diff < 0.1 else 0.0

# Gaze score with calibration
def get_gaze_score(landmarks):
    left_iris = landmarks[468]
    right_iris = landmarks[473]
    avg_x = (left_iris.x + right_iris.x) / 2.0
    return 1.0 if abs(avg_x - calibrated_gaze_center) < calibrated_gaze_tolerance else 0.0

# Concentration score computation
def compute_concentration_score(gaze, head_pose, eyes_closed_duration, head_tilt_duration):
    base_score = 0.4 * gaze + 0.4 * head_pose + 0.2
    if eyes_closed_duration > 3:
        decay = 0.05 * (eyes_closed_duration - 3)
        base_score -= decay
    if head_tilt_duration > 0:
        decay = 0.07 * head_tilt_duration
        base_score -= decay
    return round(max(0.0, min(base_score, 1.0)) * 100, 2)

# Draw concentration bar
def draw_bar(score, frame):
    bar_width = 200
    bar_height = 30
    bar_x, bar_y = 30, 100
    fill_width = int(score * bar_width / 100)
    color = (0, 255, 0) if score >= 70 else (0, 100, 255)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
    cv2.putText(frame, f"{score}%", (bar_x + bar_width + 10, bar_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# Trend graph with statistics
def show_trend_graph(data):
    total_time = len(data) / 30  # assuming 30 FPS
    below_50 = sum([1 for x in data if x < 50]) / 30
    plt.figure(figsize=(10, 5))
    plt.plot(data, label="Concentration Score", color="blue")
    plt.axhline(y=np.mean(data), color='orange', linestyle='--', label=f"Average: {np.mean(data):.2f}%")
    plt.axhline(y=np.min(data), color='red', linestyle=':', label=f"Min: {np.min(data):.2f}%")
    plt.axhline(y=np.max(data), color='green', linestyle=':', label=f"Max: {np.max(data):.2f}%")
    plt.title(f"Concentration Trend | Duration: {total_time:.1f}s | Time Below 50%: {below_50:.1f}s")
    plt.xlabel("Time (frames)")
    plt.ylabel("Score")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_h, image_w, _ = frame.shape
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                connection_drawing_spec=DrawingSpec(color=(255, 0, 0), thickness=1)
            )
            landmarks = face_landmarks.landmark
            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, image_w, image_h)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, image_w, image_h)
            avg_ear = (left_ear + right_ear) / 2
            is_closed = is_blinking(avg_ear)

            if is_closed:
                if eye_closed_start_time is None:
                    eye_closed_start_time = time.time()
                eyes_closed_duration = time.time() - eye_closed_start_time
            else:
                eyes_closed_duration = 0
                eye_closed_start_time = None

            gaze_score = get_gaze_score(landmarks)
            head_score = get_head_pose_score(landmarks)

            if head_score == 0:
                if head_turn_start_time is None:
                    head_turn_start_time = time.time()
                head_tilt_duration = time.time() - head_turn_start_time
            else:
                head_tilt_duration = 0
                head_turn_start_time = None

            concentration = compute_concentration_score(gaze_score, head_score, eyes_closed_duration, head_tilt_duration)

            score_history.append(concentration)
            trend_history.append(concentration)
            smooth_score = int(np.mean(score_history))
            draw_bar(smooth_score, frame)
            cv2.putText(frame, f"Concentration: {smooth_score}%", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            avg_concentration = int(np.mean(trend_history))
            status_text = "ACTIVE" if avg_concentration >= 80 else "DISTRACTED"
            cv2.putText(frame, f"Avg: {avg_concentration}% - {status_text}", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if smooth_score < 50:
                current_time = time.time()
                if current_time - last_focus_blink_time > 0.6:
                    focus_warning_blink = not focus_warning_blink
                    last_focus_blink_time = current_time
                if focus_warning_blink:
                    cv2.putText(frame, "PLEASE FOCUS", (int(image_w/2) - 150, int(image_h/2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

            if is_closed:
                cv2.putText(frame, f"Eyes Closed ({int(eyes_closed_duration)}s)", (30, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 255), 2)

            current_time = time.time()
            if smooth_score < 70:
                distraction += 1
                cv2.putText(frame, f"Distraction: {distraction}", (30, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
                beep_interval = 1.5 if smooth_score >= 50 else 0.7
                if current_time - alert_last_played > beep_interval:
                    threading.Thread(target=lambda: playsound("alert.wav")).start()
                    alert_last_played = current_time
            else:
                alert_last_played = 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.putText(frame, f"FPS: {fps:.1f}", (image_w - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

    status_color = (0, 255, 0) if smooth_score >= 80 else (0, 100, 255)
    cv2.circle(frame, (image_w - 30, 70), 15, status_color, -1)

    cv2.imshow("Concentration Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
show_trend_graph(trend_history)
