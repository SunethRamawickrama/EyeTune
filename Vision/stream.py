import numpy as np
import cv2
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks.python.vision import FaceLandmarkerResult
from concurrent.futures import ThreadPoolExecutor

from utils import calculate_eye_aspect_ratio
from config import (
    BRIGHTNESS_THRESHOLD,
    DISTANCE_CLOSE_THRESHOLD,
    RIGHT_EYE_EAR_POINTS,
    LEFT_EYE_EAR_POINTS,
)
from trackers import (
    BlinkTracker, AmbientLightTracker, DistanceTracker, DirectionTracker, ZoomController
)
from notifier import show_notification

# Initializing mediapipe options
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
mp_image_module = mp.Image

# Frame dimensions will be managed locally in the main loop

# Initializing the model path
model_path = Path(__file__).parent.parent/"models"/"face_landmarker.task"
model_path = str(model_path.resolve()) 


# Initializing mediapipe configs
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

# Face mesh
def face_mesh(img, detection_result, frame_width, frame_height):
    if detection_result.face_landmarks:
        for index in detection_result.face_landmarks:
            for landmarks in index:
                pixel_x = int (landmarks.x * frame_width)
                pixel_y = int (landmarks.y * frame_height)
                cv2.circle(img, (pixel_x, pixel_y), 1, (0, 255, 0), -1)

# eye
def eye(img, all_the_normalized_landmarks, point_list, frame_width, frame_height):
    eye_landmarks = point_list
    for index, i in enumerate(eye_landmarks):
        pixel_x = int (all_the_normalized_landmarks[i].x * frame_width)
        pixel_y = int (all_the_normalized_landmarks[i].y * frame_height)
        cv2.circle(img, (pixel_x, pixel_y), 1, (255, 255, 0), -1)
        cv2.putText(img, str(index), (pixel_x, pixel_y), cv2.FONT_HERSHEY_COMPLEX, 0.25,  (255, 255, 0), 1, cv2.LINE_AA)

def display_stats(img, blink_stats, ambient_stats, distance_stats, direction_stats, brightness, distance_cm, direction):
    """Display statistics on the frame"""
    y_offset = 30
    line_height = 25
    
    # Blink statistics
    cv2.putText(img, f"Recent Blinks: {blink_stats['total_blinks']}", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    y_offset += line_height
    
    cv2.putText(img, f"Blink Rate: {blink_stats['recent_blink_rate']:.1f}/min", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    y_offset += line_height
    
    cv2.putText(img, f"Currently Blinking: {'Yes' if blink_stats['is_currently_blinking'] else 'No'}", 
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    y_offset += line_height
    
    # Direction statistics
    cv2.putText(img, f"Direction: {direction}", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    y_offset += line_height
    
    cv2.putText(img, f"Look Away Time: {direction_stats['total_look_away_time']:.1f}s", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    y_offset += line_height
    
    # Ambient light statistics
    cv2.putText(img, f"Brightness: {brightness:.1f}", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    y_offset += line_height
    
    cv2.putText(img, f"Light State: {ambient_stats['current_state']}", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    y_offset += line_height
    
    # Distance statistics
    if distance_cm:
        cv2.putText(img, f"Distance: {distance_cm:.1f} cm", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        y_offset += line_height
        
        cv2.putText(img, f"Distance State: {distance_stats['current_state']}", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    else:
        cv2.putText(img, "Distance: N/A", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

# Initialize state management classes
blink_tracker = BlinkTracker()
ambient_tracker = AmbientLightTracker()
distance_tracker = DistanceTracker()
direction_tracker = DirectionTracker()
zoom_controller = ZoomController()

cap = cv2.VideoCapture(1)
with FaceLandmarker.create_from_options(options) as landmarker:
   MAX_CONTINUOUS_FOCUS = 60 
   while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    # Get frame dimensions
    frame_height, frame_width = img.shape[:2]

    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    if 'prev_timestamp' not in locals() or timestamp <= prev_timestamp:
        timestamp = (prev_timestamp if 'prev_timestamp' in locals() else 0) + 1
    prev_timestamp = timestamp

    detection_result = landmarker.detect_for_video(mp_image, timestamp)

    brightness = ambient_tracker.process(img)
    
    # Initialize variables
    blink_result = {
        "is_blinking": False,
        "blink_timestamps": [],
        "blink_counter": 0,
        "avg_ear": 0
    }
    distance_cm = None
    direction = "unknown"

    if detection_result.face_landmarks:
        all_the_normalized_landmarks = detection_result.face_landmarks[0]

        right_eye_landmarks = RIGHT_EYE_EAR_POINTS
        left_eye_landmarks = LEFT_EYE_EAR_POINTS

        # Calculate original EAR for zoom functionality
        left_ear = calculate_eye_aspect_ratio(all_the_normalized_landmarks, left_eye_landmarks, frame_width, frame_height)
        right_ear = calculate_eye_aspect_ratio(all_the_normalized_landmarks, right_eye_landmarks, frame_width, frame_height)
        ear_avg = (left_ear + right_ear) / 2
        zoom_controller.apply(ear_avg)
        
        # Advanced blink detection
        blink_result = blink_tracker.detect(all_the_normalized_landmarks, frame_width, frame_height)
        
        # Distance calculation
        distance_cm = distance_tracker.measure(img, all_the_normalized_landmarks)
        
        # Direction detection
        direction = direction_tracker.detect(all_the_normalized_landmarks, frame_width, frame_height)

        with ThreadPoolExecutor() as executor:
            # executor.submit(face_mesh, img, detection_result, frame_width, frame_height) # draws the face mesh
            executor.submit(eye, img, all_the_normalized_landmarks, right_eye_landmarks, frame_width, frame_height)
            executor.submit(eye, img, all_the_normalized_landmarks, left_eye_landmarks, frame_width, frame_height)

    # Get current statistics
    blink_stats = blink_tracker.snap()
    ambient_stats = ambient_tracker.snap()
    distance_stats = distance_tracker.snap()
    direction_stats = direction_tracker.snap()
    
    # Display statistics on frame
    display_stats(img, blink_stats, ambient_stats, distance_stats, direction_stats, brightness, distance_cm, direction)
    
    # Display warnings
    warning_y = frame_height - 30
    
    if brightness < BRIGHTNESS_THRESHOLD:
        cv2.putText(img, "WARNING: Low ambient light!", (10, warning_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        show_notification("Eye Tune Warning","WARNING: Low ambient lighting")
        warning_y -= 30
    
    if distance_cm and distance_cm < DISTANCE_CLOSE_THRESHOLD:
        cv2.putText(img, "WARNING: Too close to screen!", (10, warning_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        show_notification("Eye Tune Warning","WARNING: Too close to screen!")
        warning_y -= 30
    
    if direction_stats['continuous_look_time'] > MAX_CONTINUOUS_FOCUS:
        cv2.putText(img, 
                "Time for an eye break! Look away from the screen!", 
                (10, warning_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        show_notification("Eye Tune Warning", "Time for an eye break! Look away from the screen!")
    warning_y -= 30
    
    cv2.imshow("EyeTune", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()