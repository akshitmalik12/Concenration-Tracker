import cv2
import mediapipe as mp
import time
import matplotlib.pyplot as plt
import threading
import numpy as np
from collections import deque
import subprocess
import os
import platform

# Constants
CALIBRATION_DURATION = 8  # seconds - increased for better calibration
EYE_CLOSED_TIME_LIMIT = 2.5  # seconds
FACE_TILT_THRESHOLD_X = 0.04  # horizontal tilt threshold
FACE_TILT_THRESHOLD_Y = 0.05  # vertical tilt threshold
CONCENTRATION_DROP_RATE = 3  # per second when distracted
CONCENTRATION_RECOVERY_RATE = 1.5  # per second when focused
SMOOTHING_WINDOW = 8  # frames for moving average - increased for smoother, slower changes
EAR_DROP_THRESHOLD = 0.7  # percentage drop from baseline to consider closed

# Mediapipe Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

# Eye landmark indices (MediaPipe Face Mesh)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Face orientation landmarks (nose tip, chin, forehead center, left/right face edge)
NOSE_TIP = 1
CHIN = 152
FOREHEAD = 10
LEFT_FACE = 234
RIGHT_FACE = 454

class SmoothingFilter:
    """Moving average filter for smoothing values"""
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
    
    def update(self, value):
        self.values.append(value)
        return np.mean(self.values) if len(self.values) > 0 else value
    
    def reset(self):
        self.values.clear()

def compute_EAR(landmarks, eye_indices, w, h):
    """Calculate Eye Aspect Ratio (EAR) - more accurate version"""
    # Get eye landmark points
    eye_points = []
    for idx in eye_indices:
        x = landmarks[idx].x * w
        y = landmarks[idx].y * h
        eye_points.append([x, y])
    eye_points = np.array(eye_points)
    
    # Calculate vertical distances
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    # Calculate horizontal distance
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    
    # EAR formula
    ear = (A + B) / (2.0 * C + 1e-6)  # Add small epsilon to avoid division by zero
    return ear

def compute_face_orientation(landmarks, w, h):
    """Calculate face orientation using multiple landmarks for better accuracy"""
    # Get key points
    nose = np.array([landmarks[NOSE_TIP].x, landmarks[NOSE_TIP].y])
    chin = np.array([landmarks[CHIN].x, landmarks[CHIN].y])
    forehead = np.array([landmarks[FOREHEAD].x, landmarks[FOREHEAD].y])
    left_face = np.array([landmarks[LEFT_FACE].x, landmarks[LEFT_FACE].y])
    right_face = np.array([landmarks[RIGHT_FACE].x, landmarks[RIGHT_FACE].y])
    
    # Calculate face center
    face_center_x = (left_face[0] + right_face[0]) / 2
    face_center_y = (forehead[1] + chin[1]) / 2
    
    # Calculate deviations from center
    horizontal_deviation = abs(nose[0] - face_center_x)
    vertical_deviation = abs(nose[1] - face_center_y)
    
    # Calculate face width and height for normalization
    face_width = abs(right_face[0] - left_face[0])
    face_height = abs(chin[1] - forehead[1])
    
    # Normalized deviations
    norm_horizontal = horizontal_deviation / (face_width + 1e-6)
    norm_vertical = vertical_deviation / (face_height + 1e-6)
    
    return norm_horizontal, norm_vertical

def draw_calibration_ui(frame, progress, time_remaining, status_text, w, h):
    """Draw improved calibration UI with progress indicator"""
    # Semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    # Calibration box
    box_width = 600
    box_height = 300
    box_x = (w - box_width) // 2
    box_y = (h - box_height) // 2
    
    # Draw box with rounded corners effect
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (50, 50, 50), -1)
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (100, 200, 255), 3)
    
    # Title
    title = "CALIBRATION IN PROGRESS"
    title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
    title_x = box_x + (box_width - title_size[0]) // 2
    cv2.putText(frame, title, (title_x, box_y + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 200, 255), 3)
    
    # Instructions
    instruction = "Please face the camera directly"
    inst_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    inst_x = box_x + (box_width - inst_size[0]) // 2
    cv2.putText(frame, instruction, (inst_x, box_y + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Progress bar background
    bar_x = box_x + 50
    bar_y = box_y + 150
    bar_width = box_width - 100
    bar_height = 30
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (40, 40, 40), -1)
    
    # Progress bar fill
    fill_width = int(bar_width * progress)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), (100, 200, 255), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
    
    # Progress percentage
    progress_text = f"{int(progress * 100)}%"
    progress_size = cv2.getTextSize(progress_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    progress_text_x = bar_x + (bar_width - progress_size[0]) // 2
    cv2.putText(frame, progress_text, (progress_text_x, bar_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    # Time remaining
    time_text = f"Time remaining: {int(time_remaining)}s"
    time_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    time_x = box_x + (box_width - time_size[0]) // 2
    cv2.putText(frame, time_text, (time_x, box_y + 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    # Status text
    if status_text:
        status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        status_x = box_x + (box_width - status_size[0]) // 2
        cv2.putText(frame, status_text, (status_x, box_y + 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 255, 150), 1)

def draw_main_ui(frame, concentration, avg_ear, face_tilt_x, face_tilt_y, 
                 eyes_closed, session_time, w, h):
    """Draw improved main UI with better layout and visual feedback"""
    # Top bar with concentration
    bar_height = 80
    cv2.rectangle(frame, (0, 0), (w, bar_height), (30, 30, 30), -1)
    cv2.rectangle(frame, (0, 0), (w, bar_height), (100, 100, 100), 2)
    
    # Concentration percentage (large, prominent)
    concentration_text = f"{int(concentration)}%"
    conc_size = cv2.getTextSize(concentration_text, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 4)[0]
    conc_x = 30
    conc_y = 55
    
    # Color based on concentration
    if concentration >= 70:
        color = (0, 255, 100)  # Green
        status = "FOCUSED"
    elif concentration >= 50:
        color = (0, 200, 255)  # Yellow/Orange
        status = "MODERATE"
    else:
        color = (0, 100, 255)  # Red
        status = "DISTRACTED"
    
    cv2.putText(frame, concentration_text, (conc_x, conc_y),
                cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 4)
    
    # Status text
    status_size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv2.putText(frame, status, (conc_x + conc_size[0] + 30, conc_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Concentration bar (horizontal, below percentage)
    bar_x = 30
    bar_y = 70
    bar_width = w - 60
    bar_height_small = 8
    
    # Background
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height_small), (50, 50, 50), -1)
    
    # Fill
    fill_width = int(bar_width * (concentration / 100))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height_small), color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height_small), (200, 200, 200), 1)
    
    # Session time (top right)
    time_text = f"Time: {int(session_time // 60):02d}:{int(session_time % 60):02d}"
    time_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.putText(frame, time_text, (w - time_size[0] - 30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    # Info panel (bottom left)
    info_y = h - 120
    info_bg_height = 110
    cv2.rectangle(frame, (20, info_y), (350, h - 10), (20, 20, 20), -1)
    cv2.rectangle(frame, (20, info_y), (350, h - 10), (100, 100, 100), 2)
    
    # EAR value
    ear_text = f"EAR: {avg_ear:.3f}"
    cv2.putText(frame, ear_text, (30, info_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Face orientation
    tilt_x_text = f"Tilt X: {face_tilt_x:.3f}"
    tilt_y_text = f"Tilt Y: {face_tilt_y:.3f}"
    cv2.putText(frame, tilt_x_text, (30, info_y + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, tilt_y_text, (30, info_y + 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Eyes status
    eyes_status = "Eyes: CLOSED" if eyes_closed else "Eyes: OPEN"
    eyes_color = (0, 0, 255) if eyes_closed else (0, 255, 0)
    cv2.putText(frame, eyes_status, (30, info_y + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, eyes_color, 1)
    
    # Warning overlay if concentration is low - ambulance-like red flashing
    if concentration < 50:
        # Ambulance-like flashing effect - faster pulsing
        flash_speed = 6  # Faster flashing
        alpha = 0.4 + 0.4 * np.sin(time.time() * flash_speed)  # Stronger red overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)  # Red background
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Warning text with stronger visibility
        warning_text = "PLEASE FOCUS!"
        warning_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 6)[0]
        warning_x = (w - warning_size[0]) // 2
        warning_y = h // 2
        
        # Text shadow for better visibility
        cv2.putText(frame, warning_text, (warning_x + 4, warning_y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 8)
        cv2.putText(frame, warning_text, (warning_x, warning_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 6)

def play_alert():
    """Play alert sound in separate thread using subprocess"""
    try:
        sound_path = os.path.join(os.path.dirname(__file__), 'alert.wav')
        if not os.path.exists(sound_path):
            sound_path = 'alert.wav'  # Try current directory
        
        if os.path.exists(sound_path):
            system = platform.system()
            if system == 'Darwin':  # macOS
                subprocess.Popen(['afplay', sound_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif system == 'Linux':
                subprocess.Popen(['aplay', sound_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif system == 'Windows':
                subprocess.Popen(['powershell', '-c', f'(New-Object Media.SoundPlayer "{sound_path}").PlaySync()'], 
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                # Fallback: try to use system default player
                subprocess.Popen(['open', sound_path] if system == 'Darwin' else ['xdg-open', sound_path],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"🔊 Alert sound playing...")
        else:
            print(f"⚠️ Warning: alert.wav not found at {sound_path}")
    except Exception as e:
        print(f"❌ Error playing alert sound: {e}")

# Initialize
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit(1)

# Set camera properties for better quality
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

concentration = 100.0
concentration_history = []
timestamp_history = []
prev_time = time.time()
last_log_time = time.time()
eyes_closed_start = None
alert_played = False
calibrating = True
calibration_start = time.time()
ear_baseline = None
ear_std = None
ear_values = []
neutral_face_x = None
neutral_face_y = None
face_orientations_x = []
face_orientations_y = []
session_start_time = None

# Smoothing filters
ear_filter = SmoothingFilter(SMOOTHING_WINDOW)
concentration_filter = SmoothingFilter(SMOOTHING_WINDOW)
face_tilt_x_filter = SmoothingFilter(SMOOTHING_WINDOW)
face_tilt_y_filter = SmoothingFilter(SMOOTHING_WINDOW)

print("Starting Concentration Tracker...")
print("Calibration will begin when your face is detected")

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

        # Calibration phase
        if calibrating:
            left_ear = compute_EAR(landmarks, LEFT_EYE, w, h)
            right_ear = compute_EAR(landmarks, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2
            ear_values.append(avg_ear)
            
            # Calculate face orientation
            face_tilt_x, face_tilt_y = compute_face_orientation(landmarks, w, h)
            face_orientations_x.append(face_tilt_x)
            face_orientations_y.append(face_tilt_y)
            
            if neutral_face_x is None:
                neutral_face_x = face_tilt_x
                neutral_face_y = face_tilt_y
            
            elapsed = current_time - calibration_start
            progress = min(1.0, elapsed / CALIBRATION_DURATION)
            time_remaining = max(0, CALIBRATION_DURATION - elapsed)
            
            # Status text
            if len(ear_values) < 10:
                status = "Collecting baseline data..."
            elif progress < 0.5:
                status = "Keep facing the camera..."
            else:
                status = "Almost done..."
            
            # Draw calibration UI
            draw_calibration_ui(frame, progress, time_remaining, status, w, h)
            
            # Draw face landmarks during calibration
            mp.solutions.drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=results.multi_face_landmarks[0],
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec,
            )
            
            if elapsed >= CALIBRATION_DURATION:
                # Calculate baseline and adaptive threshold
                ear_baseline = np.mean(ear_values)
                ear_std = np.std(ear_values)
                # Adaptive threshold: baseline - 2*std, but not less than 70% of baseline
                adaptive_threshold = max(ear_baseline * EAR_DROP_THRESHOLD, ear_baseline - 2 * ear_std)
                
                neutral_face_x = np.mean(face_orientations_x)
                neutral_face_y = np.mean(face_orientations_y)
                
                calibrating = False
                session_start_time = current_time
                print(f"Calibration Complete!")
                print(f"  Baseline EAR: {ear_baseline:.4f} ± {ear_std:.4f}")
                print(f"  Adaptive Threshold: {adaptive_threshold:.4f}")
                print(f"  Neutral Face Position: X={neutral_face_x:.4f}, Y={neutral_face_y:.4f}")
            
            cv2.imshow("Concentration Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Main tracking phase
        session_time = current_time - session_start_time
        
        # Calculate EAR
        left_ear = compute_EAR(landmarks, LEFT_EYE, w, h)
        right_ear = compute_EAR(landmarks, RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2
        avg_ear = ear_filter.update(avg_ear)
        
        # Calculate face orientation
        face_tilt_x, face_tilt_y = compute_face_orientation(landmarks, w, h)
        face_tilt_x = face_tilt_x_filter.update(face_tilt_x)
        face_tilt_y = face_tilt_y_filter.update(face_tilt_y)
        
        # Calculate deviations from neutral position
        deviation_x = abs(face_tilt_x - neutral_face_x)
        deviation_y = abs(face_tilt_y - neutral_face_y)
        
        # Eyes closed detection using adaptive threshold
        eyes_closed = avg_ear < adaptive_threshold
        
        # Calculate time delta
        dt = current_time - prev_time
        prev_time = current_time
        
        # Factor 2: Face orientation - no penalty for minimal tilts, drop to 0% for large tilts
        # When head is tilted away from screen, concentration should drop to 0%
        orientation_factor = 100.0
        if deviation_x > FACE_TILT_THRESHOLD_X or deviation_y > FACE_TILT_THRESHOLD_Y:
            # Calculate normalized deviation (how much above threshold)
            max_deviation_ratio = max(deviation_x / FACE_TILT_THRESHOLD_X, deviation_y / FACE_TILT_THRESHOLD_Y)
            
            # Dead zone for minimal tilts: no penalty until 1.5x threshold
            # Then quick ramp-up for larger tilts, allowing drop to 0%
            excess_deviation = max_deviation_ratio - 1.0  # How much above threshold
            
            if excess_deviation > 0.5:  # Only apply penalty if tilt is more than 1.5x threshold (minimal tilts ignored)
                # Quick drop for larger tilts - can go all the way to 0%
                # 1.5x threshold (excess=0.5) = 0% penalty (dead zone)
                # 2x threshold (excess=1.0) = 30% penalty
                # 3x threshold (excess=2.0) = 70% penalty
                # 4x threshold (excess=3.0) = 95% penalty
                # 5x+ threshold (excess=4.0+) = 100% penalty (drops to 0%)
                
                # Use exponential curve for quick drop: penalty increases rapidly
                effective_excess = excess_deviation - 0.5  # Subtract dead zone
                
                if effective_excess < 0.5:  # 1.5x to 2x threshold - small penalty
                    penalty = effective_excess * 60  # 0% to 30% penalty
                elif effective_excess < 1.5:  # 2x to 3x threshold - medium penalty
                    penalty = 30 + (effective_excess - 0.5) * 40  # 30% to 70% penalty
                elif effective_excess < 2.5:  # 3x to 4x threshold - large penalty
                    penalty = 70 + (effective_excess - 1.5) * 25  # 70% to 95% penalty
                else:  # 4x+ threshold - maximum penalty (100% = 0% concentration)
                    penalty = 95 + min(5, (effective_excess - 2.5) * 1.0)  # 95% to 100% penalty
                
                orientation_factor = max(0, 100 - penalty)  # Can drop to 0% (not just 2%)
            else:
                # Minimal tilt (within dead zone) - no penalty, keep concentration stable
                orientation_factor = 100.0
        else:
            # Head is stable (below threshold) - keep concentration stable
            orientation_factor = 100.0
        
        # Factor 1: Eye closure - stricter for exam/class use
        eye_factor = 100.0
        if eyes_closed:
            if eyes_closed_start is None:
                eyes_closed_start = current_time
            closed_duration = current_time - eyes_closed_start
            if closed_duration > EYE_CLOSED_TIME_LIMIT:
                # Stricter penalty: eyes closed for long = major distraction
                penalty = min(80, closed_duration * CONCENTRATION_DROP_RATE * 1.5)  # 1.5x faster drop
                eye_factor = max(0, 100 - penalty)
        else:
            eyes_closed_start = None
            # Recovery when eyes are open - but don't recover if head is still tilted
            if orientation_factor >= 90:  # Only recover if head is mostly stable
                eye_factor = min(100, concentration + CONCENTRATION_RECOVERY_RATE * dt)
            else:
                eye_factor = 100.0  # Eyes open but head tilted = still distracted
        
        # Factor 3: EAR deviation (drowsiness indicator) - slower response
        ear_factor = 100.0
        if avg_ear < ear_baseline:
            # If EAR is significantly below baseline, indicate drowsiness
            ear_ratio = avg_ear / (ear_baseline + 1e-6)
            if ear_ratio < 0.85:  # Changed to 0.85 (15% drop triggers penalty) for less sensitivity
                penalty = (0.85 - ear_ratio) * 80  # Reduced multiplier from 150 to 80 for slower drop
                ear_factor = max(0, 100 - penalty)
        
        # Combined concentration score - orientation is dominant for exam/class use
        # When head is tilted away from screen, concentration should drop to 0%
        # Very high weight on orientation (80%) so it can drive concentration to 0% when head is away
        concentration = 0.15 * eye_factor + 0.80 * orientation_factor + 0.05 * ear_factor
        concentration = concentration_filter.update(concentration)
        concentration = max(0, min(100, concentration))

        # Alert system - play sound at 50% concentration
        if concentration < 50 and not alert_played:
            threading.Thread(target=play_alert, daemon=True).start()
            alert_played = True
        elif concentration >= 50:
            alert_played = False

        # Logging (every second)
        if current_time - last_log_time >= 1.0:
            concentration_history.append(round(concentration, 2))
            timestamp_history.append(now_str)
            last_log_time = current_time

        # Draw face landmarks (optional, can be toggled)
        mp.solutions.drawing_utils.draw_landmarks(
            image=frame,
            landmark_list=results.multi_face_landmarks[0],
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec,
        )

        # Draw main UI
        draw_main_ui(frame, concentration, avg_ear, deviation_x, deviation_y,
                    eyes_closed, session_time, w, h)
        
    else:
        # No face detected - treat as major distraction
        if not calibrating:
            # Drop concentration to near 0 when face is not detected
            concentration = max(0, concentration - 5)  # Drop by 5% per frame
            concentration = concentration_filter.update(concentration)
            concentration = max(0, min(100, concentration))
            
            # Play alert sound if not already playing
            if concentration < 50 and not alert_played:
                threading.Thread(target=play_alert, daemon=True).start()
                alert_played = True
            elif concentration >= 50:
                alert_played = False
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            no_face_text = "NO FACE DETECTED"
            text_size = cv2.getTextSize(no_face_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h // 2
            cv2.putText(frame, no_face_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            # Show concentration even when no face detected
            draw_main_ui(frame, concentration, 0, 0, 0, False, 
                        (time.time() - session_start_time) if session_start_time else 0, w, h)
    
    # Instructions
    instruction_text = "Press 'q' to quit"
    cv2.putText(frame, instruction_text, (w - 200, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

    cv2.imshow("Concentration Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Generate and display statistics graph
if concentration_history:
    avg_c = np.mean(concentration_history)
    max_c = np.max(concentration_history)
    min_c = np.min(concentration_history)
    std_c = np.std(concentration_history)
    
    # Count time below thresholds
    below_50 = sum(1 for c in concentration_history if c < 50)
    below_70 = sum(1 for c in concentration_history if c < 70)
    
    plt.figure(figsize=(14, 8))
    
    # Main plot
    plt.subplot(2, 1, 1)
    plt.plot(timestamp_history, concentration_history, marker='o', color='#4A90E2', 
             linewidth=2, markersize=4, label='Concentration', alpha=0.8)
    plt.axhline(avg_c, color='red', linestyle='--', linewidth=2, 
                label=f'Average: {avg_c:.2f}%')
    plt.axhline(max_c, color='green', linestyle='--', linewidth=2, 
                label=f'Highest: {max_c:.2f}%')
    plt.axhline(min_c, color='orange', linestyle='--', linewidth=2, 
                label=f'Lowest: {min_c:.2f}%')
    plt.axhline(50, color='red', linestyle=':', linewidth=1, alpha=0.5, 
                label='Alert Threshold (50%)')
    plt.axhline(70, color='yellow', linestyle=':', linewidth=1, alpha=0.5, 
                label='Good Threshold (70%)')
    plt.fill_between(timestamp_history, 0, concentration_history, 
                     where=[c < 50 for c in concentration_history], 
                     color='red', alpha=0.2, label='Low Concentration')
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Concentration (%)", fontsize=12)
    plt.title("Concentration Over Time", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=9)
    plt.ylim(0, 100)
    
    # Statistics subplot
    plt.subplot(2, 1, 2)
    stats_data = {
        'Average': avg_c,
        'Maximum': max_c,
        'Minimum': min_c,
        'Std Dev': std_c
    }
    colors = ['#4A90E2', '#50C878', '#FF6B6B', '#FFA500']
    bars = plt.bar(stats_data.keys(), stats_data.values(), color=colors, alpha=0.7)
    plt.ylabel("Concentration (%)", fontsize=12)
    plt.title("Statistics Summary", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=10)
    
    # Add text statistics
    stats_text = f"Time below 50%: {below_50}s ({below_50/len(concentration_history)*100:.1f}%)\n"
    stats_text += f"Time below 70%: {below_70}s ({below_70/len(concentration_history)*100:.1f}%)"
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig("concentration_graph.png", dpi=150, bbox_inches='tight')
    print(f"\nStatistics saved to concentration_graph.png")
    print(f"Average Concentration: {avg_c:.2f}%")
    print(f"Maximum: {max_c:.2f}% | Minimum: {min_c:.2f}%")
    print(f"Standard Deviation: {std_c:.2f}%")
    print(f"Time below 50%: {below_50}s ({below_50/len(concentration_history)*100:.1f}%)")
    print(f"Time below 70%: {below_70}s ({below_70/len(concentration_history)*100:.1f}%)")
    plt.show()
else:
    print("No concentration data collected.")
