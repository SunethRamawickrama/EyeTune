import screen_controller
import time
import numpy as np
import cv2

adjusted = False

# Eye Aspect Ratio thresholds
ZOOM_IN_THRESHOLD = 0.19
ZOOM_OUT_THRESHOLD = 0.25
EAR_THRESHOLD = 0.3  # For blink detection (increased for better sensitivity)

# Ambient light thresholds
BRIGHTNESS_THRESHOLD = 90  # Below this is considered dark

# Distance thresholds (in cm)
DISTANCE_CLOSE_THRESHOLD = 35
DISTANCE_FAR_THRESHOLD = 40

# Distance calculation constants
REAL_VERTICAL_DISTANCE = 8.0  # Actual vertical distance from forehead to nose tip (cm)
FOCAL_LENGTH = 700  # Focal length for distance calculation

# Blink tracking variables
blink_timestamps = []
blink_counter = 0
is_currently_blinking = False
last_blink_time = time.time()

# Ambient light tracking
ambient_light_data = {"ambient_light": "light", "timestamp": None}
last_known_light_state = None
light_state_changes = []
dark_environment_start_time = None

# Distance tracking
distance_changes = []
last_known_distance_state = None
distance_state_start_time = time.time()

# Direction tracking
direction_changes = []
last_known_direction = None
direction_state_start_time = time.time()
DEBOUNCE_TIME = 0.5  # Seconds to debounce direction changes
last_direction_change_time = time.time()

def core(ear):
    """Original zoom functionality based on EAR"""
    print(f"EAR: {ear}")
    global adjusted
    global squint_start_time, squint_hold_start
    hold_duration = 30  # seconds (how long to keep scaled)
    squint_required_duration = 2  # seconds (how long EAR must be low)

    if 'squint_start_time' not in globals():
        squint_start_time = None
    if 'squint_hold_start' not in globals():
        squint_hold_start = None

    if ear < ZOOM_IN_THRESHOLD:
        if squint_start_time is None:
            squint_start_time = time.time()
        elif not adjusted and (time.time() - squint_start_time >= squint_required_duration):
            screen_controller.scale()
            adjusted = True
            squint_hold_start = time.time()
    else:
        squint_start_time = None

    if adjusted:
        # Hold scaled state for at least hold_duration
        if squint_hold_start is not None and (time.time() - squint_hold_start >= hold_duration):
            if ear > ZOOM_OUT_THRESHOLD:
                screen_controller.reset()
                adjusted = False
                squint_hold_start = None

def detect_blink(face_landmarks, img_w, img_h):
    """
    Detect if the person is blinking by calculating the eye aspect ratio (EAR)
    Returns: Object with blinks and blink_timestamps
    """
    global blink_timestamps, blink_counter, is_currently_blinking, last_blink_time

    # Check if we have enough landmarks
    if len(face_landmarks) < 400:  # Face Landmarker should have 468 landmarks
        print(f"Warning: Only {len(face_landmarks)} landmarks detected, expected 468")
        return {
            "is_blinking": False,
            "blink_timestamps": blink_timestamps,
            "blink_counter": blink_counter,
            "avg_ear": 0.0
        }

    # Standard MediaPipe Face Landmarker eye landmark indices
    LEFT_EYE = [33, 160, 158, 133, 153, 144]  # Left eye landmarks
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]  # Right eye landmarks
    
    def calculate_ear(eye_points):
        try:
            # Convert normalized coordinates to pixel coordinates
            points = []
            for idx in eye_points:
                if idx < len(face_landmarks):
                    landmark = face_landmarks[idx]
                    x = int(landmark.x * img_w)
                    y = int(landmark.y * img_h)
                    points.append((x, y))
                else:
                    print(f"Warning: Landmark index {idx} out of range (max: {len(face_landmarks)-1})")
                    return 0.0
            
            if len(points) != 6:
                print(f"Warning: Expected 6 points, got {len(points)}")
                return 0.0
            
            # Calculate the eye aspect ratio
            # EAR = (||p2-p6|| + ||p3-p5||) / (2||p1-p4||)
            vertical_dist1 = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
            vertical_dist2 = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
            horizontal_dist = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
            
            # EAR should not be 0 (if points are too close)
            ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist) if horizontal_dist > 0 else 0
            return ear
        except Exception as e:
            print(f"Error calculating EAR: {str(e)}")
            return 0.0
    
    # Calculate EAR for both eyes
    left_ear = calculate_ear(LEFT_EYE)
    right_ear = calculate_ear(RIGHT_EYE)
    
    # Average EAR of both eyes
    avg_ear = (left_ear + right_ear) / 2.0
    
    # Debug information (reduced frequency for cleaner output)
    if len(blink_timestamps) % 50 == 0:  # Print every 50th frame to reduce spam
        print(f"EAR Debug - Left: {left_ear:.3f}, Right: {right_ear:.3f}, Avg: {avg_ear:.3f}, Threshold: {EAR_THRESHOLD}")
    
    # Convert numpy.bool_ to Python bool before returning
    is_blinking = bool(avg_ear < EAR_THRESHOLD)

    current_time = time.time()

    # Check if this is the start of a new blink
    if is_blinking and not is_currently_blinking:
        # Only count this as a new blink if enough time has passed since the last blink
        if not blink_timestamps or current_time - blink_timestamps[-1] >= 0.25:
            blink_counter += 1
            blink_timestamps.append(current_time)
            last_blink_time = current_time
            print(f"Blink detected! EAR: {avg_ear:.3f}, Count: {blink_counter}")
        
        # Update the blinking state
        is_currently_blinking = True
    
    # If the person is not blinking anymore, update the state
    elif not is_blinking and is_currently_blinking:
        is_currently_blinking = False
        print("Blink ended")
    
    return {
        "is_blinking": is_blinking,
        "blink_timestamps": blink_timestamps,
        "blink_counter": blink_counter,
        "avg_ear": avg_ear
    }

import color_theme
last_color_adjust_time = 0
COLOR_ADJUST_INTERVAL = 30 

# def process_ambient_light(frame):
#     """
#     Process ambient light from frame and return brightness value
#     Also tracks ambient light state changes
#     """
#     global last_known_light_state, light_state_changes, dark_environment_start_time

#     global last_color_adjust_time
#     current_time = time.time()

#     # if current_time - last_color_adjust_time > COLOR_ADJUST_INTERVAL:
#     #     try:
#     #         color_theme.auto_adjust(frame)
#     #         last_color_adjust_time = current_time
#     #     except Exception as e:
#     #         print(f"Screen tint adjustment failed: {e}")
    
#     try:
#         # Convert the BGR image to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#         # Calculate average brightness
#         brightness = np.mean(gray)
#         current_time = time.time()
        
#         # Determine the current state based on brightness
#         current_state = "light" if brightness >= BRIGHTNESS_THRESHOLD else "dark"
        
#         # Track time spent in dark environment
#         if current_state == "dark":
#             if dark_environment_start_time is None:
#                 dark_environment_start_time = current_time
#         else:  # Reset dark environment timer when it's bright
#             dark_environment_start_time = None
        
#         # Check if the state has changed
#         if last_known_light_state is None:
#             # Initialize the last known state
#             last_known_light_state = current_state
#             ambient_light_data["timestamp"] = current_time
#             light_state_changes.append(ambient_light_data.copy())
#         elif current_state != last_known_light_state:
#             # State has changed, update the timestamp and state
#             ambient_light_data["ambient_light"] = current_state
#             ambient_light_data["timestamp"] = current_time
#             last_known_light_state = current_state
#             light_state_changes.append(ambient_light_data.copy())
        
#         return float(brightness)
    
#     except Exception as e:
#         print(f"Error processing ambient light: {str(e)}")
#         return 0.0

def process_ambient_light(frame):
    """
    Process ambient light from frame and return brightness value
    Also tracks ambient light state changes and adjusts screen tint.
    Non-blocking for other processes.
    """
    global last_known_light_state, light_state_changes, dark_environment_start_time
    global last_color_adjust_time

    current_time = time.time()

    try:
        # Convert to grayscale and calculate brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)

        # Determine light state
        current_state = "light" if brightness >= BRIGHTNESS_THRESHOLD else "dark"

        # Track time spent in dark environment
        if current_state == "dark" and dark_environment_start_time is None:
            dark_environment_start_time = current_time
        elif current_state == "light":
            dark_environment_start_time = None

        # Track state changes
        if last_known_light_state is None:
            last_known_light_state = current_state
            ambient_light_data["ambient_light"] = current_state
            ambient_light_data["timestamp"] = current_time
            light_state_changes.append(ambient_light_data.copy())
        elif current_state != last_known_light_state:
            ambient_light_data["ambient_light"] = current_state
            ambient_light_data["timestamp"] = current_time
            last_known_light_state = current_state
            light_state_changes.append(ambient_light_data.copy())

        # --- Screen tint adjustment (non-blocking, infrequent) ---
        if current_time - last_color_adjust_time > COLOR_ADJUST_INTERVAL:
            try:
                # Run in a separate thread to prevent blocking
                import threading
                threading.Thread(target=color_theme.auto_adjust, args=(frame,)).start()
                last_color_adjust_time = current_time
            except Exception as e:
                print(f"Screen tint adjustment failed: {e}")

        return float(brightness)

    except Exception as e:
        print(f"Error processing ambient light: {str(e)}")
        return 0.0


def check_distance(frame, face_landmarks):
    """
    Calculate the distance between user and screen using facial landmarks.
    Returns distance in centimeters
    """
    global last_known_distance_state, distance_changes, distance_state_start_time

    # Define key point indices for MediaPipe Face Landmarker
    FOREHEAD_TOP = 10    # Forehead top key point
    NOSE_TIP = 1         # Nose tip key point (different index in Face Landmarker)

    distance = None
    current_time = time.time()

    if face_landmarks:
        # Get key point coordinates
        forehead = face_landmarks[FOREHEAD_TOP]
        nose_tip = face_landmarks[NOSE_TIP]

        # Calculate vertical pixel distance (forehead to nose tip)
        ih, iw = frame.shape[:2]
        y1 = int(forehead.y * ih)  # Forehead Y coordinate
        y2 = int(nose_tip.y * ih)   # Nose tip Y coordinate
        pixel_distance = abs(y2 - y1)

        # Calculate actual distance
        if pixel_distance > 0:
            distance = (REAL_VERTICAL_DISTANCE * FOCAL_LENGTH) / pixel_distance

        # Determine the current distance state
        if distance and distance < DISTANCE_CLOSE_THRESHOLD:
            current_distance_state = "close"
        elif distance and DISTANCE_CLOSE_THRESHOLD <= distance <= DISTANCE_FAR_THRESHOLD:
            current_distance_state = "med"
        else:
            current_distance_state = "far"

        # Check if the distance state has changed
        if last_known_distance_state is None:
            # Initialize the last known distance state
            last_known_distance_state = current_distance_state
            distance_state_start_time = current_time
        elif current_distance_state != last_known_distance_state:
            # State has changed, log the time spent in the previous state
            distance_changes.append({
                "distance": last_known_distance_state,
                "start_time": distance_state_start_time,
                "end_time": current_time,
            })
            # Update the last known state and start time
            last_known_distance_state = current_distance_state
            distance_state_start_time = current_time

    return distance

def get_blink_stats():
    """Get current blink statistics"""
    return {
        "total_blinks": len(blink_timestamps),
        "blink_counter": blink_counter,
        "is_currently_blinking": is_currently_blinking,
        "last_blink_time": last_blink_time
    }

def get_ambient_light_stats():
    """Get current ambient light statistics"""
    return {
        "current_state": ambient_light_data["ambient_light"],
        "state_changes": light_state_changes,
        "dark_start_time": dark_environment_start_time
    }

def get_distance_stats():
    """Get current distance statistics"""
    return {
        "current_state": last_known_distance_state,
        "state_changes": distance_changes
    }

def detect_eye_direction(face_landmarks, img_w, img_h):
    """
    Detect eye gaze direction by tracking pupil positions relative to eye corners.
    Returns: "left", "right", "center", or "unknown".
    """
    global last_known_direction, direction_changes, last_direction_change_time

    # Check if we have enough landmarks
    if len(face_landmarks) < 400:
        return "unknown"

    # MediaPipe Face Landmarker indices for eye landmarks
    # Using eye corner landmarks for direction detection (more reliable than pupil detection)
    left_eye_landmarks = [33, 133, 159, 145, 144, 153]  # Left eye landmarks
    right_eye_landmarks = [362, 263, 386, 374, 380, 373]  # Right eye landmarks
    
    def get_landmark_coords(landmark_index):
        if landmark_index < len(face_landmarks):
            landmark = face_landmarks[landmark_index]
            x, y = int(landmark.x * img_w), int(landmark.y * img_h)
            return (x, y)
        return None
    
    # Get eye corner coordinates
    left_eye_left = get_landmark_coords(33)
    left_eye_right = get_landmark_coords(133)
    right_eye_left = get_landmark_coords(362)
    right_eye_right = get_landmark_coords(263)
    
    # Check if we have valid coordinates
    if not all([left_eye_left, left_eye_right, right_eye_left, right_eye_right]):
        return "unknown"
    
    # Calculate eye center positions
    left_eye_center_x = (left_eye_left[0] + left_eye_right[0]) / 2
    right_eye_center_x = (right_eye_left[0] + right_eye_right[0]) / 2
    
    # Calculate the average eye center position
    avg_eye_center_x = (left_eye_center_x + right_eye_center_x) / 2
    
    # Calculate face center (approximate)
    face_center_x = img_w / 2
    
    # Calculate offset from face center
    eye_offset = avg_eye_center_x - face_center_x
    
    # Define direction based on eye position relative to face center
    # Using a threshold of 20 pixels for direction detection
    if eye_offset < -20:
        current_direction = "left"
    elif eye_offset > 20:
        current_direction = "right"
    else:
        current_direction = "center"

    # Check if the direction has changed
    current_time = time.time()
    if last_known_direction is None:
        # Initialize the last known direction
        last_known_direction = current_direction
        direction_changes.append({
            "looking_away": 0 if current_direction == "center" else 1,
            "timestamp": current_time
        })
        print(f"Direction initialized: {current_direction}")
    elif current_direction != last_known_direction and (current_time - last_direction_change_time) > DEBOUNCE_TIME:
        # Direction has changed and debounce time has passed
        direction_changes.append({
            "looking_away": 0 if current_direction == "center" else 1,
            "timestamp": current_time
        })
        print(f"Direction changed: {last_known_direction} -> {current_direction}")
        last_known_direction = current_direction  # Update the last known direction
        last_direction_change_time = current_time  # Update the last change time

    return current_direction

def get_direction_stats():
    """Get current direction statistics"""
    total_look_away_time = calculate_look_away_time(direction_changes)
    
    # Print periodic summary (every 10 direction changes)
    if len(direction_changes) > 0 and len(direction_changes) % 10 == 0:
        print(f"Direction Summary - Changes: {len(direction_changes)}, Look Away Time: {total_look_away_time:.1f}s")
    
    return {
        "current_direction": last_known_direction,
        "direction_changes": direction_changes,
        "total_look_away_time": total_look_away_time
    }

def calculate_look_away_time(direction_changes):
    """Calculate total time spent looking away from center"""
    total_time = 0
    for i in range(len(direction_changes)-1):
        if direction_changes[i]["looking_away"] == 1:
            total_time += direction_changes[i+1]["timestamp"] - direction_changes[i]["timestamp"]
    return total_time

def reset_direction_tracking():
    """Reset direction tracking for testing purposes"""
    global direction_changes, last_known_direction, direction_state_start_time, last_direction_change_time
    direction_changes.clear()
    last_known_direction = None
    direction_state_start_time = time.time()
    last_direction_change_time = time.time()
    print("Direction tracking reset")

