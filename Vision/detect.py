import numpy as np
import cv2
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks.python.vision import FaceLandmarkerResult
from concurrent.futures import ThreadPoolExecutor
from utils import (
    core, detect_blink, process_ambient_light, check_distance, detect_eye_direction,
    get_blink_stats, get_ambient_light_stats, get_distance_stats, get_direction_stats,
    BRIGHTNESS_THRESHOLD, DISTANCE_CLOSE_THRESHOLD
)
from notifier import show_notification

# Initializing mediapipe options
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
mp_image_module = mp.Image

# Global varibales for drawing utils
frame_width = None
frame_height = None
frame_to_draw = None

# Initializing the model path
model_path = Path(__file__).parent.parent/"models"/"face_landmarker.task"
model_path = str(model_path.resolve()) 


# Initializing mediapipe configs
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_faces = 1
)

# Face mesh
def face_mesh(img, detection_result):
    if detection_result.face_landmarks:
        for index in detection_result.face_landmarks:
            for landmarks in index:
                pixel_x = int (landmarks.x * frame_width)
                pixel_y = int (landmarks.y * frame_height)
                cv2.circle(img, (pixel_x, pixel_y), 1, (0, 255, 0), -1)

# eye
def eye(img, all_the_normalized_landmarks, point_list):
    eye_landmarks = point_list
    for index, i in enumerate(eye_landmarks):
        pixel_x = int (all_the_normalized_landmarks[i].x * frame_width)
        pixel_y = int (all_the_normalized_landmarks[i].y * frame_height)
        cv2.circle(img, (pixel_x, pixel_y), 1, (255, 255, 0), -1)
        cv2.putText(img, str(index), (pixel_x, pixel_y), cv2.FONT_HERSHEY_COMPLEX, 0.25,  (255, 255, 0), 1, cv2.LINE_AA)

# Eye aspect ratio
def calc_ear(all_the_normalized_landmarks, right_eye_landmarks, left_eye_landmarks):

    def ear(landmark_list):
        verticle_dist_1 = np.linalg.norm( ( all_the_normalized_landmarks[landmark_list[5]].y * frame_height) - (all_the_normalized_landmarks[landmark_list[1]].y * frame_height) )
        verticle_dist_2 = np.linalg.norm( ( all_the_normalized_landmarks[landmark_list[4]].y * frame_height) - (all_the_normalized_landmarks[landmark_list[2]].y * frame_height) )
        horizontal_dist = np.linalg.norm( ( all_the_normalized_landmarks[landmark_list[3]].x * frame_width) - ( all_the_normalized_landmarks[landmark_list[0]].x * frame_width) )

        return (verticle_dist_1 + verticle_dist_2) / (2 * horizontal_dist)
    
    ear_avg = (ear(right_eye_landmarks) + ear(left_eye_landmarks)) /2
    core(ear_avg)
    return ear_avg


def display_stats(img, blink_stats, ambient_stats, distance_stats, direction_stats, brightness, distance_cm, direction):
    """Display statistics on the frame"""
    y_offset = 30
    line_height = 25
    
    # Blink statistics
    cv2.putText(img, f"Blinks: {blink_stats['total_blinks']}", (10, y_offset), 
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

cap = cv2.VideoCapture(0)
with FaceLandmarker.create_from_options(options) as landmarker:
   MAX_CONTINUOUS_FOCUS = 60 
   while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    frame_to_draw = img.copy()
    if frame_width is None:
        frame_height, frame_width = img.shape[:2]

    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    if 'prev_timestamp' not in locals() or timestamp <= prev_timestamp:
        timestamp = (prev_timestamp if 'prev_timestamp' in locals() else 0) + 1
    prev_timestamp = timestamp

    detection_result = landmarker.detect_for_video(mp_image, timestamp)

    brightness = process_ambient_light(img)
    
    # Initialize variables
    blink_result = {"is_blinking": False, "blink_timestamps": [], "blink_counter": 0, "avg_ear": 0}
    distance_cm = None
    direction = "unknown"

    if detection_result.face_landmarks:
        all_the_normalized_landmarks = detection_result.face_landmarks[0]

        right_eye_landmarks = [33, 159, 158, 133, 153, 145]
        left_eye_landmarks = [362, 385, 386, 263, 374, 380]

        # Calculate original EAR for zoom functionality
        ear_avg = calc_ear(all_the_normalized_landmarks, left_eye_landmarks, right_eye_landmarks)
        
        # Advanced blink detection
        blink_result = detect_blink(all_the_normalized_landmarks, frame_width, frame_height)
        
        # Distance calculation
        distance_cm = check_distance(img, all_the_normalized_landmarks)
        
        # Direction detection
        direction = detect_eye_direction(all_the_normalized_landmarks, frame_width, frame_height)

        with ThreadPoolExecutor() as executor:
            executor.submit(face_mesh, img, detection_result) # draws the face mesh
            executor.submit(eye, img, all_the_normalized_landmarks, right_eye_landmarks)
            executor.submit(eye, img, all_the_normalized_landmarks, left_eye_landmarks)

    # Get current statistics
    blink_stats = get_blink_stats()
    ambient_stats = get_ambient_light_stats()
    distance_stats = get_distance_stats()
    direction_stats = get_direction_stats()
    
    # Display statistics on frame
    display_stats(img, blink_stats, ambient_stats, distance_stats, direction_stats, brightness, distance_cm, direction)
    
    # Display warnings
    warning_y = frame_height - 30
    
    if brightness < BRIGHTNESS_THRESHOLD:
        cv2.putText(img, "WARNING: Low ambient light!", (10, warning_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        warning_y -= 30
    
    if distance_cm and distance_cm < DISTANCE_CLOSE_THRESHOLD:
        cv2.putText(img, "WARNING: Too close to screen!", (10, warning_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        warning_y -= 30
    
    if direction_stats['continuous_look_time'] > MAX_CONTINUOUS_FOCUS:
        cv2.putText(img, 
                "⚠️ Time for an eye break! Look away from the screen!", 
                (10, warning_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    warning_y -= 30
    
    cv2.imshow("EyeTune - Enhanced Detection", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()






