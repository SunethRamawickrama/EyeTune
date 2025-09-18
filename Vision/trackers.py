import time
import threading
import color_theme
import numpy as np
import screen_controller
from utils import calculate_eye_aspect_ratio

from config import (
    EAR_ZOOM_IN_THRES,
    EAR_ZOOM_OUT_THRES,
    EAR_BLINK_THRES,
    BRIGHTNESS_THRESHOLD,
    MAX_BLINK_HISTORY,
    FOCAL_LENGTH,
    IRIS_DIAMETER_CM,
    DISTANCE_CLOSE_THRESHOLD,
    DISTANCE_FAR_THRESHOLD,
    RIGHT_EYE_EAR_POINTS,
    LEFT_EYE_EAR_POINTS,
)

class BlinkTracker:
    """Manages blink detection state"""
    def __init__(self):
        self.counter = 0
        self.is_currently_blinking = False
        self.last_blink_time = time.time()
        self.recent_times = [] # track recent blink timestamps for rate calculation
        self.REFRACTORY_SEC = 0.27 # refractory window to count distinct blinks

    def snap(self):
        """Get current blink statistics"""
        return {
            "total_blinks": self.counter,
            "blink_counter": self.counter,
            "is_currently_blinking": self.is_currently_blinking,
            "last_blink_time": self.last_blink_time,
            "recent_blink_rate": self.calculate_blink_rate()
        }

    def calculate_blink_rate(self):
        """Calculate blink rate (blinks per minute) from recent timestamps"""
        if len(self.recent_times) < 2:
            return 0.0
        time_span = self.recent_times[-1] - self.recent_times[0]
        if time_span > 0:
            minutes = time_span / 60.0
            return (len(self.recent_times) - 1) / minutes
        return 0.0

    def detect(self, face_landmarks, img_w, img_h):
        """
        Detect if the person is blinking by calculating EAR.
        Returns dict with is_blinking, avg_ear, blink_counter
        """
        if len(face_landmarks) < 400:
            return {"is_blinking": False, "avg_ear": 0.0, "blink_counter": self.counter}

        left_ear = calculate_eye_aspect_ratio(face_landmarks, LEFT_EYE_EAR_POINTS, img_w, img_h)
        right_ear = calculate_eye_aspect_ratio(face_landmarks, RIGHT_EYE_EAR_POINTS, img_w, img_h)
        avg_ear = (left_ear + right_ear) / 2.0

        is_blinking = avg_ear < EAR_BLINK_THRES
        current_time = time.time()
        if is_blinking and not self.is_currently_blinking:
            if current_time - self.last_blink_time >= self.REFRACTORY_SEC:
                self.counter += 1
                self.recent_times.append(current_time)
                if len(self.recent_times) > MAX_BLINK_HISTORY:
                    self.recent_times.pop(0)
                self.last_blink_time = current_time
            self.is_currently_blinking = True
        elif not is_blinking and self.is_currently_blinking:
            self.is_currently_blinking = False

        return {"is_blinking": is_blinking, "avg_ear": avg_ear, "blink_counter": self.counter}


class AmbientLightTracker:
    """Manages ambient light detection state"""
    def __init__(self):
        self.data = {"ambient_light": "light", "timestamp": None}
        self.last_known_state = None
        self.state_changes = []
        self.dark_start_time = None
        self.last_color_adjust_time = 0
        self.color_adjust_interval = 30

    def snap(self):
        """Get current ambient light statistics"""
        return {
            "current_state": self.data["ambient_light"],
            "state_changes": self.state_changes,
            "dark_start_time": self.dark_start_time
        }

    def process(self, frame):
        """Calculate perceived luminance and update ambient state."""
        current_time = time.time()
        try:
            r = frame[:, :, 2].astype(np.float32)
            g = frame[:, :, 1].astype(np.float32)
            b = frame[:, :, 0].astype(np.float32)
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            brightness = float(np.mean(luminance))

            prev_state = self.last_known_state or "light"
            threshold_up = BRIGHTNESS_THRESHOLD + 5
            threshold_down = BRIGHTNESS_THRESHOLD - 5
            if prev_state == "light":
                current_state = "light" if brightness >= threshold_down else "dark"
            else:
                current_state = "light" if brightness >= threshold_up else "dark"

            if current_state == "dark" and self.dark_start_time is None:
                self.dark_start_time = current_time
            elif current_state == "light":
                self.dark_start_time = None

            if self.last_known_state is None:
                self.last_known_state = current_state
                self.data["ambient_light"] = current_state
                self.data["timestamp"] = current_time
                self.state_changes.append(self.data.copy())
            elif current_state != self.last_known_state:
                self.data["ambient_light"] = current_state
                self.data["timestamp"] = current_time
                self.last_known_state = current_state
                self.state_changes.append(self.data.copy())

            if current_time - self.last_color_adjust_time > self.color_adjust_interval:
                try:
                    threading.Thread(target=color_theme.auto_adjust, args=(frame,)).start()
                    self.last_color_adjust_time = current_time
                except Exception as e:
                    print(f"Screen tint adjustment failed: {e}")

            return brightness
        except Exception as e:
            print(f"Error processing ambient light: {str(e)}")
            return 0.0


class DistanceTracker:
    """Manages distance detection state"""
    def __init__(self):
        self.changes = []
        self.last_known_state = None
        self.state_start_time = time.time()

    def snap(self):
        """Get current distance statistics"""
        return {
            "current_state": self.last_known_state,
            "state_changes": self.changes
        }

    def measure(self, frame, face_landmarks):
        """Calculate distance to screen in cm using iris width."""
        distance = None
        current_time = time.time()
        if face_landmarks and len(face_landmarks) >= 468:
            try:
                LEFT_IRIS_LEFT = 469
                LEFT_IRIS_RIGHT = 470
                RIGHT_IRIS_LEFT = 474
                RIGHT_IRIS_RIGHT = 475
                _, iw = frame.shape[:2]
                left_iris_left = face_landmarks[LEFT_IRIS_LEFT]
                left_iris_right = face_landmarks[LEFT_IRIS_RIGHT]
                left_iris_width_px = abs(left_iris_right.x - left_iris_left.x) * iw
                right_iris_left = face_landmarks[RIGHT_IRIS_LEFT]
                right_iris_right = face_landmarks[RIGHT_IRIS_RIGHT]
                right_iris_width_px = abs(right_iris_right.x - right_iris_left.x) * iw
                avg_iris_width_px = (left_iris_width_px + right_iris_width_px) / 2.0
                if avg_iris_width_px > 0:
                    distance = (IRIS_DIAMETER_CM * FOCAL_LENGTH) / avg_iris_width_px

                if distance and distance < DISTANCE_CLOSE_THRESHOLD:
                    current_distance_state = "close"
                elif distance and DISTANCE_CLOSE_THRESHOLD <= distance <= DISTANCE_FAR_THRESHOLD:
                    current_distance_state = "med"
                else:
                    current_distance_state = "far"

                if self.last_known_state is None:
                    self.last_known_state = current_distance_state
                    self.state_start_time = current_time
                elif current_distance_state != self.last_known_state:
                    self.changes.append({
                        "state": self.last_known_state,
                        "start_time": self.state_start_time,
                        "end_time": current_time,
                    })
                    self.last_known_state = current_distance_state
                    self.state_start_time = current_time
            except (IndexError, AttributeError) as e:
                print(f"Error accessing iris landmarks: {e}")
                return None
        return distance


class DirectionTracker:
    """Manages eye direction detection state"""
    def __init__(self):
        self.changes = []
        self.last_known_direction = None
        self.state_start_time = time.time()
        # Stability buffer (seconds) to confirm direction changes
        self.buffer_time = 0.5
        self.last_change_time = time.time()

    def snap(self):
        """Get current direction statistics"""
        total_look_away_time = self.get_look_away_time()
        continuous_look_time = self.calculate_continuous_look_time()
        if len(self.changes) > 0 and len(self.changes) % 10 == 0:
            print(
                f"Direction Summary - Changes: {len(self.changes)}, "
                f"Look Away Time: {total_look_away_time:.1f}s, "
                f"Continuous Look: {continuous_look_time:.1f}s"
            )
        return {
            "current_direction": self.last_known_direction,
            "direction_changes": self.changes,
            "total_look_away_time": total_look_away_time,
            "continuous_look_time": continuous_look_time
        }

    def get_look_away_time(self):
        """Calculate total time spent looking away from center"""
        if len(self.changes) < 2:
            return 0
        
        total_time = 0
        for current, next_change in zip(self.changes[:-1], self.changes[1:]):
            if current["away"] == 1:
                total_time += next_change["timestamp"] - current["timestamp"]
        return total_time

    def calculate_continuous_look_time(self):
        """
        Returns time in seconds that the user has continuously been looking at the screen.
        """
        if not self.changes:
            return 0
        current_time = time.time()
        last_look_away_time = None
        for entry in reversed(self.changes):
            if entry["away"] == 1:
                last_look_away_time = entry["timestamp"]
                break
        if last_look_away_time is None:
            last_look_away_time = self.changes[0]["timestamp"]
        return current_time - last_look_away_time

    def reset_tracking(self):
        """Reset direction tracking"""
        self.changes.clear()
        self.last_known_direction = None
        self.state_start_time = time.time()
        self.last_change_time = time.time()
        print("Direction tracking reset")

    def detect(self, face_landmarks, img_w, img_h):
        """Detect eye gaze direction based on eye corner centers offset."""
        if len(face_landmarks) < 400:
            return "unknown"

        def get_landmark_coords(landmark_index):
            if landmark_index < len(face_landmarks):
                landmark = face_landmarks[landmark_index]
                x, y = int(landmark.x * img_w), int(landmark.y * img_h)
                return (x, y)
            return None

        left_eye_left = get_landmark_coords(33)
        left_eye_right = get_landmark_coords(133)
        right_eye_left = get_landmark_coords(362)
        right_eye_right = get_landmark_coords(263)
        if not all([left_eye_left, left_eye_right, right_eye_left, right_eye_right]):
            return "unknown"

        left_eye_center_x = (left_eye_left[0] + left_eye_right[0]) / 2
        right_eye_center_x = (right_eye_left[0] + right_eye_right[0]) / 2
        avg_eye_center_x = (left_eye_center_x + right_eye_center_x) / 2
        face_center_x = img_w / 2

        # relative offset threshold
        offset_fraction = (avg_eye_center_x - face_center_x) / max(1, img_w)
        if offset_fraction < -0.03:
            current_direction = "left"
        elif offset_fraction > 0.03:
            current_direction = "right"
        else:
            current_direction = "center"

        current_time = time.time()
        if self.last_known_direction is None:
            self.last_known_direction = current_direction
            self.changes.append({
                "away": 0 if current_direction == "center" else 1,
                "timestamp": current_time
            })
            print(f"Direction initialized: {current_direction}")
        elif current_direction != self.last_known_direction and (current_time - self.last_change_time) > self.buffer_time:
            self.changes.append({
                "away": 0 if current_direction == "center" else 1,
                "timestamp": current_time
            })
            print(f"Direction changed: {self.last_known_direction} -> {current_direction}")
            self.last_known_direction = current_direction
            self.last_change_time = current_time

        return current_direction


class ZoomController:
    """Manages zoom functionality state"""
    def __init__(self):
        self.adjusted = False
        self.squint_start_time = None
        self.squint_hold_start = None
        self.hold_duration = 30  # seconds (how long to keep scaled)
        self.squint_required_duration = 2  # seconds (how long EAR must be low)

    def apply(self, ear):
        """Apply zoom logic based on current EAR."""
        print(f"EAR: {ear}")
        if ear < EAR_ZOOM_IN_THRES:
            if self.squint_start_time is None:
                self.squint_start_time = time.time()
            elif not self.adjusted and (time.time() - self.squint_start_time >= self.squint_required_duration):
                screen_controller.scale()
                self.adjusted = True
                self.squint_hold_start = time.time()
        else:
            self.squint_start_time = None

        if self.adjusted:
            if self.squint_hold_start is not None and (time.time() - self.squint_hold_start >= self.hold_duration):
                if ear > EAR_ZOOM_OUT_THRES:
                    screen_controller.reset()
                    self.adjusted = False
                    self.squint_hold_start = None