"""
EyeTune System Tray Application
A desktop app that runs in the system tray with a dropdown dashboard
Integrates with existing Vision detection system
"""

import sys
import threading
import time
from datetime import datetime, timedelta
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFrame, QSystemTrayIcon, 
                             QMenu, QAction, QProgressBar, QGridLayout, QCheckBox, QGraphicsDropShadowEffect)
from PyQt5.QtCore import QTimer, pyqtSignal, QObject, Qt, QPropertyAnimation, QRect, QEasingCurve, QSequentialAnimationGroup, QParallelAnimationGroup
from PyQt5.QtGui import QIcon, QPixmap, QFont, QPainter, QColor, QBrush, QLinearGradient
import numpy as np
import cv2
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks.python.vision import FaceLandmarkerResult
from concurrent.futures import ThreadPoolExecutor

# Import your existing modules
from vision.utils import (
    core, detect_blink, process_ambient_light, check_distance, detect_eye_direction,
    get_blink_stats, get_ambient_light_stats, get_distance_stats, get_direction_stats,
    BRIGHTNESS_THRESHOLD, DISTANCE_CLOSE_THRESHOLD, DISTANCE_FAR_THRESHOLD,
    ZOOM_IN_THRESHOLD, ZOOM_OUT_THRESHOLD
)

class EyeMetrics:
    """Data class to store eye tracking metrics"""
    def __init__(self):
        self.blink_count = 0
        self.blink_rate = 0  # blinks per minute
        self.distance_cm = 0
        self.brightness = 0
        self.gaze_direction = "center"
        self.look_away_time = 0
        self.ear_left = 0.0
        self.ear_right = 0.0
        self.is_squinting = False
        self.last_blink_time = time.time()
        self.session_start = time.time()
        self.warnings = []
        self.is_currently_blinking = False
        self.light_state = "light"
        self.distance_state = "medium"

class NotificationWidget(QWidget):
    """On-screen notification widget for eye health warnings"""
    
    def __init__(self, message, notification_type="warning", duration=5000):
        super().__init__()
        self.notification_type = notification_type
        self.duration = duration
        self.setup_ui(message)
        self.setup_animations()
        
    def setup_ui(self, message):
        """Setup the notification UI with modern design"""
        # Modern notification size
        self.setFixedSize(380, 80)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Modern color scheme based on notification type
        if self.notification_type == "warning":
            bg_color = "#FFF3CD"  # Light amber background
            border_color = "#FFC107"  # Amber border
            text_color = "#856404"    # Dark amber text
            icon_color = "#FFC107"    # Amber icon
            icon = "⚠"
        elif self.notification_type == "error":
            bg_color = "#F8D7DA"  # Light red background
            border_color = "#DC3545"  # Red border
            text_color = "#721C24"    # Dark red text
            icon_color = "#DC3545"    # Red icon
            icon = "●"
        elif self.notification_type == "info":
            bg_color = "#D1ECF1"  # Light blue background
            border_color = "#17A2B8"  # Blue border
            text_color = "#0C5460"    # Dark blue text
            icon_color = "#17A2B8"    # Blue icon
            icon = "ℹ"
        else:  # success
            bg_color = "#D4EDDA"  # Light green background
            border_color = "#28A745"  # Green border
            text_color = "#155724"    # Dark green text
            icon_color = "#28A745"    # Green icon
            icon = "✓"
        
        # Modern styling with subtle borders and shadows
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {bg_color};
                border: 2px solid {border_color};
                border-radius: 16px;
                color: {text_color};
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }}
            QLabel {{
                color: {text_color};
                font-size: 14px;
                font-weight: 500;
                background: transparent;
                border: none;
            }}
        """)
        
        # Enhanced shadow effect for depth
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(25)
        shadow.setXOffset(0)
        shadow.setYOffset(8)
        shadow.setColor(QColor(0, 0, 0, 80))
        self.setGraphicsEffect(shadow)
        
        # Modern layout with better spacing
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(16)
        
        # Icon with modern styling
        icon_label = QLabel(icon)
        icon_label.setFont(QFont("SF Pro Display", 18, QFont.Bold))
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setStyleSheet(f"""
            QLabel {{
                color: {icon_color};
                font-size: 18px;
                font-weight: bold;
                background: transparent;
                border: none;
                min-width: 24px;
            }}
        """)
        layout.addWidget(icon_label)
        
        # Message with better typography and HTML support
        self.message_label = QLabel(message)
        self.message_label.setWordWrap(True)
        self.message_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.message_label.setTextFormat(Qt.RichText)  # Enable HTML formatting
        self.message_label.setStyleSheet(f"""
            QLabel {{
                color: {text_color};
                font-size: 14px;
                font-weight: 500;
                line-height: 1.4;
                background: transparent;
                border: none;
            }}
        """)
        layout.addWidget(self.message_label)
        
        # Modern close button
        close_btn = QPushButton("×")
        close_btn.setFixedSize(28, 28)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                border: 2px solid {border_color};
                border-radius: 14px;
                color: {text_color};
                font-size: 16px;
                font-weight: bold;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }}
            QPushButton:hover {{
                background-color: {border_color};
                color: {bg_color};
            }}
            QPushButton:pressed {{
                background-color: {text_color};
                color: {bg_color};
            }}
        """)
        close_btn.clicked.connect(self.close_notification)
        layout.addWidget(close_btn)
        
    def setup_animations(self):
        """Setup modern slide-in and fade animations"""
        # Smooth slide in from right
        self.slide_animation = QPropertyAnimation(self, b"geometry")
        self.slide_animation.setDuration(400)  # Slightly longer for smoother effect
        self.slide_animation.setEasingCurve(QEasingCurve.OutQuart)  # More modern easing
        
        # Smooth fade out
        self.fade_animation = QPropertyAnimation(self, b"windowOpacity")
        self.fade_animation.setDuration(300)  # Longer fade for smoother effect
        self.fade_animation.setStartValue(1.0)
        self.fade_animation.setEndValue(0.0)
        self.fade_animation.setEasingCurve(QEasingCurve.InQuart)
        self.fade_animation.finished.connect(self.close)
        
        # Auto-hide timer
        self.auto_hide_timer = QTimer()
        self.auto_hide_timer.timeout.connect(self.fade_out)
        self.auto_hide_timer.setSingleShot(True)
        
        # Subtle pulse animation for attention
        self.pulse_animation = QPropertyAnimation(self, b"windowOpacity")
        self.pulse_animation.setDuration(1000)
        self.pulse_animation.setStartValue(1.0)
        self.pulse_animation.setKeyValueAt(0.5, 0.7)
        self.pulse_animation.setEndValue(1.0)
        self.pulse_animation.setEasingCurve(QEasingCurve.InOutSine)
        self.pulse_animation.setLoopCount(2)  # Pulse twice then stop
        
    def show_notification(self, x, y):
        """Show notification with slide-in animation"""
        # Get screen geometry and ensure we're within bounds
        screen = QApplication.desktop().screenGeometry()
        
        # Ensure x and y are within screen bounds
        x = max(0, min(x, screen.width() - self.width()))
        y = max(0, min(y, screen.height() - self.height()))
        
        # Set initial position (off-screen to the right)
        initial_x = screen.width()
        initial_y = y
        
        # Ensure initial position is also within bounds
        initial_x = max(0, min(initial_x, screen.width()))
        initial_y = max(0, min(initial_y, screen.height() - self.height()))
        
        # Make sure the widget is visible and on top
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        
        self.move(initial_x, initial_y)
        self.show()
        self.raise_()
        self.activateWindow()
        
        # Animate to final position
        final_rect = QRect(x, y, self.width(), self.height())
        self.slide_animation.setStartValue(QRect(initial_x, initial_y, self.width(), self.height()))
        self.slide_animation.setEndValue(final_rect)
        self.slide_animation.start()
        
        # Start subtle pulse animation for attention
        self.pulse_animation.start()
        
        # Start auto-hide timer
        self.auto_hide_timer.start(self.duration)
        
    def fade_out(self):
        """Fade out the notification"""
        self.fade_animation.start()
        
    def close_notification(self):
        """Close notification immediately"""
        self.auto_hide_timer.stop()
        self.fade_out()

class NotificationManager(QObject):
    """Manages on-screen notifications for eye health warnings"""
    
    def __init__(self):
        super().__init__()
        self.active_notifications = []
        self.notification_cooldowns = {}  # Prevent spam
        self.cooldown_duration = 3  # seconds - reduced to 3 seconds as requested
        
    def show_notification(self, message, notification_type="warning", duration=5000):
        """Show a notification on screen"""
        current_time = time.time()
        
        # Check cooldown to prevent spam - use a hash of the message for better cooldown
        message_hash = hash(message)
        if message_hash in self.notification_cooldowns:
            if current_time - self.notification_cooldowns[message_hash] < self.cooldown_duration:
                print(f"Notification blocked by cooldown: {message[:50]}...")
                return
        
        # Create and show notification
        notification = NotificationWidget(message, notification_type, duration)
        
        # Position notification (top-right, stacked)
        screen = QApplication.desktop().screenGeometry()
        
        # Calculate position with proper bounds checking
        x = max(20, screen.width() - notification.width() - 20)
        y = max(50, 50 + (len(self.active_notifications) * 90))  # Spacing for larger notifications
        
        # Ensure we don't go off-screen vertically
        if y + notification.height() > screen.height():
            y = max(50, screen.height() - notification.height() - 20)
        
        # Debug output
        print(f"Showing notification: {message[:50]}... at position ({x},{y})")
        
        notification.show_notification(x, y)
        
        # Track active notification
        self.active_notifications.append(notification)
        
        # Set cooldown using message hash
        self.notification_cooldowns[message_hash] = current_time
        
        # Clean up when notification closes
        notification.destroyed.connect(lambda: self.active_notifications.remove(notification))
        
    def show_eye_health_warning(self, warning_type, details=""):
        """Show specific eye health warnings"""
        current_time = time.time()
        
        # Check cooldown per warning type to prevent spam
        if warning_type in self.notification_cooldowns:
            if current_time - self.notification_cooldowns[warning_type] < self.cooldown_duration:
                print(f"Warning {warning_type} blocked by cooldown")
                return
        
        warnings = {
            "low_light": ("Low Light Alert", "Increase room brightness for better eye comfort", "warning"),
            "too_close": ("Too Close to Screen", "Move back to maintain healthy viewing distance", "error"),
            "too_far": ("Too Far from Screen", "Move closer for better viewing experience", "info"),
            "look_away": ("Look Away Break", "Take a 20-second break and look at something 20 feet away", "warning"),
            "low_blink": ("Low Blink Rate", "Remember to blink regularly to prevent dry eyes", "warning"),
            "squinting": ("Squinting Detected", "Adjust screen brightness or increase font size", "warning")
        }
        
        if warning_type in warnings:
            title, description, notif_type = warnings[warning_type]
            
            # Create a formatted message with title and description
            if details:
                message = f"<b>{title}</b><br/>{description}<br/><i>{details}</i>"
            else:
                message = f"<b>{title}</b><br/>{description}"
            
            # Show the notification
            self.show_notification(message, notif_type)
            
            # Set cooldown for this warning type
            self.notification_cooldowns[warning_type] = current_time
            
            # Force the notification to be visible
            QApplication.processEvents()
    
    def clear_all_notifications(self):
        """Clear all active notifications"""
        for notification in self.active_notifications[:]:
            notification.close_notification()
        self.active_notifications.clear()

class EyeTracker(QObject):
    """Eye tracking worker that runs in background thread using your existing detection system"""
    
    metrics_updated = pyqtSignal(object)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.metrics = EyeMetrics()
        self.setup_mediapipe()
        
        # Global variables for drawing utils (from your detect.py)
        self.frame_width = None
        self.frame_height = None
        self.frame_to_draw = None
        
    def setup_mediapipe(self):
        """Initialize MediaPipe using your existing configuration"""
        # Initialize mediapipe options (from your detect.py)
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        
        # Model path (from your detect.py)
        model_path = Path(__file__).parent / "models" / "face_landmarker.task"
        if not model_path.exists():
            # Try relative path
            model_path = Path("models") / "face_landmarker.task"
        model_path = str(model_path.resolve())
        
        # MediaPipe configs (from your detect.py)
        self.options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1
        )
        
        self.FaceLandmarker = FaceLandmarker
        
    def calculate_ear_for_display(self, all_landmarks, frame_width, frame_height):
        """Calculate EAR for display purposes using your existing method"""
        if len(all_landmarks) < 400:
            return 0.0, 0.0
            
        def ear(landmark_list):
            try:
                verticle_dist_1 = np.linalg.norm((all_landmarks[landmark_list[5]].y * frame_height) - (all_landmarks[landmark_list[1]].y * frame_height))
                verticle_dist_2 = np.linalg.norm((all_landmarks[landmark_list[4]].y * frame_height) - (all_landmarks[landmark_list[2]].y * frame_height))
                horizontal_dist = np.linalg.norm((all_landmarks[landmark_list[3]].x * frame_width) - (all_landmarks[landmark_list[0]].x * frame_width))
                
                return (verticle_dist_1 + verticle_dist_2) / (2 * horizontal_dist) if horizontal_dist > 0 else 0
            except:
                return 0.0
        
        # Using your landmark indices
        right_eye_landmarks = [33, 159, 158, 133, 153, 145]
        left_eye_landmarks = [362, 385, 386, 263, 374, 380]
        
        left_ear = ear(left_eye_landmarks)
        right_ear = ear(right_eye_landmarks)
        
        return left_ear, right_ear
    
    def start_tracking(self):
        """Start the eye tracking in a separate thread"""
        self.running = True
        self.thread = threading.Thread(target=self._track_loop)
        self.thread.daemon = True
        self.thread.start()
    
    def stop_tracking(self):
        """Stop the eye tracking"""
        self.running = False
        if hasattr(self, 'cap'):
            self.cap.release()
    
    def _track_loop(self):
        """Main tracking loop using your existing detection system"""
        try:
            self.cap = cv2.VideoCapture(1)  # Try camera 0 first
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(1)  # Fall back to camera 1 like in your detect.py
                
            with self.FaceLandmarker.create_from_options(self.options) as landmarker:
                prev_timestamp = 0
                
                while self.running:
                    ret, img = self.cap.read()
                    if not ret:
                        continue
                    
                    # Initialize frame dimensions (from your detect.py)
                    if self.frame_width is None:
                        self.frame_height, self.frame_width = img.shape[:2]
                    
                    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    
                    # Timestamp handling (from your detect.py)
                    timestamp = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))
                    if timestamp <= prev_timestamp:
                        timestamp = prev_timestamp + 1
                    prev_timestamp = timestamp
                    
                    detection_result = landmarker.detect_for_video(mp_image, timestamp)
                    
                    # Process ambient light (using your function)
                    self.metrics.brightness = process_ambient_light(img)
                    
                    # Initialize variables
                    blink_result = {"is_blinking": False, "blink_timestamps": [], "blink_counter": 0, "avg_ear": 0}
                    distance_cm = None
                    direction = "unknown"
                    
                    if detection_result.face_landmarks:
                        all_landmarks = detection_result.face_landmarks[0]
                        
                        # Calculate EAR for display
                        left_ear, right_ear = self.calculate_ear_for_display(all_landmarks, self.frame_width, self.frame_height)
                        self.metrics.ear_left = left_ear
                        self.metrics.ear_right = right_ear
                        avg_ear = (left_ear + right_ear) / 2.0
                        
                        # Use your core function for zoom functionality
                        core(avg_ear)
                        
                        # Detect squinting
                        self.metrics.is_squinting = avg_ear < ZOOM_IN_THRESHOLD
                        
                        # Advanced blink detection (using your function)
                        blink_result = detect_blink(all_landmarks, self.frame_width, self.frame_height)
                        
                        # Distance calculation (using your function)
                        distance_cm = check_distance(img, all_landmarks)
                        if distance_cm:
                            self.metrics.distance_cm = distance_cm
                        
                        # Direction detection (using your function)
                        direction = detect_eye_direction(all_landmarks, self.frame_width, self.frame_height)
                        self.metrics.gaze_direction = direction
                    
                    # Get current statistics (using your functions)
                    blink_stats = get_blink_stats()
                    ambient_stats = get_ambient_light_stats()
                    distance_stats = get_distance_stats()
                    direction_stats = get_direction_stats()
                    
                    # Update metrics from your stats
                    self.metrics.blink_count = blink_stats['total_blinks']
                    self.metrics.is_currently_blinking = blink_stats['is_currently_blinking']
                    self.metrics.light_state = ambient_stats['current_state']
                    self.metrics.distance_state = distance_stats.get('current_state', 'medium')
                    self.metrics.look_away_time = direction_stats['total_look_away_time']
                    
                    # Calculate blink rate (per minute)
                    session_time = time.time() - self.metrics.session_start
                    if session_time > 60:  # Only calculate after 1 minute
                        self.metrics.blink_rate = (self.metrics.blink_count / session_time) * 60
                    
                    # Update warnings based on your logic
                    self.metrics.warnings = []
                    if self.metrics.brightness < BRIGHTNESS_THRESHOLD:
                        self.metrics.warnings.append("Low ambient light")
                    if self.metrics.distance_cm and self.metrics.distance_cm < DISTANCE_CLOSE_THRESHOLD:
                        self.metrics.warnings.append("Too close to screen")
                    if self.metrics.distance_cm and self.metrics.distance_cm > DISTANCE_FAR_THRESHOLD:
                        self.metrics.warnings.append("Too far from screen")
                    if direction != "center" and self.metrics.look_away_time > 10:
                        self.metrics.warnings.append("Looking away too long")
                    if self.metrics.blink_rate < 15 and session_time > 60:
                        self.metrics.warnings.append("Low blink rate - dry eyes risk")
                    
                    # Emit updated metrics
                    self.metrics_updated.emit(self.metrics)
                    
                    time.sleep(0.1)  # 10 FPS
                    
        except Exception as e:
            print(f"Error in tracking loop: {e}")
        finally:
            if hasattr(self, 'cap'):
                self.cap.release()

class DashboardWidget(QWidget):
    """Mini dashboard widget that shows as dropdown"""
    
    def __init__(self):
        super().__init__()
        self.setFixedSize(340, 420)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("""
            QWidget {
                background-color: rgba(40, 44, 52, 240);
                border-radius: 12px;
                color: white;
                font-family: 'SF Pro Display', 'Helvetica Neue', sans-serif;
            }
            QLabel {
                color: white;
                font-size: 12px;
                padding: 2px;
            }
            QPushButton {
                background-color: rgba(70, 130, 180, 180);
                border: none;
                border-radius: 6px;
                padding: 8px;
                color: white;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: rgba(100, 150, 200, 200);
            }
            QPushButton:pressed {
                background-color: rgba(50, 100, 150, 200);
            }
            QProgressBar {
                border: 2px solid rgba(70, 130, 180, 100);
                border-radius: 4px;
                text-align: center;
                color: white;
                height: 16px;
            }
            QProgressBar::chunk {
                background-color: rgba(70, 130, 180, 200);
                border-radius: 2px;
            }
            QCheckBox {
                color: white;
                font-size: 11px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 2px solid rgba(70, 130, 180, 180);
            }
            QCheckBox::indicator:checked {
                background-color: rgba(70, 130, 180, 200);
            }
        """)
        
        self.setup_ui()
        self.hide()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(8)
        
        # Title with status indicator
        title_layout = QHBoxLayout()
        title = QLabel("EyeTune Dashboard")
        title.setFont(QFont("SF Pro Display", 14, QFont.Bold))
        self.status_indicator = QLabel("●")
        self.status_indicator.setFont(QFont("Arial", 12))
        self.status_indicator.setStyleSheet("color: #90EE90;")
        title_layout.addWidget(title)
        title_layout.addStretch()
        title_layout.addWidget(self.status_indicator)
        layout.addLayout(title_layout)
        
        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: rgba(255, 255, 255, 60);")
        layout.addWidget(line)
        
        # Real-time metrics grid
        grid = QGridLayout()
        grid.setSpacing(5)
        
        # Blink metrics
        self.blink_count_label = QLabel("Count: 0")
        self.blink_rate_label = QLabel("Rate: 0.0/min")
        self.blink_status_label = QLabel("Status: Normal")
        grid.addWidget(QLabel("👁️ Blinking:"), 0, 0)
        grid.addWidget(self.blink_count_label, 0, 1)
        grid.addWidget(self.blink_rate_label, 0, 2)
        grid.addWidget(self.blink_status_label, 1, 1, 1, 2)
        
        # Distance metrics
        self.distance_label = QLabel("Distance: --")
        self.distance_state_label = QLabel("State: --")
        grid.addWidget(QLabel("📏 Distance:"), 2, 0)
        grid.addWidget(self.distance_label, 2, 1)
        grid.addWidget(self.distance_state_label, 2, 2)
        
        # Eye tracking
        self.ear_label = QLabel("EAR: L:0.00 R:0.00")
        self.squint_label = QLabel("Squinting: No")
        grid.addWidget(QLabel("👀 Eye State:"), 3, 0)
        grid.addWidget(self.ear_label, 3, 1, 1, 2)
        grid.addWidget(self.squint_label, 4, 1, 1, 2)
        
        # Gaze direction
        self.gaze_label = QLabel("Direction: Center")
        self.look_away_label = QLabel("Away: 0.0s")
        grid.addWidget(QLabel("👁️‍🗨️ Gaze:"), 5, 0)
        grid.addWidget(self.gaze_label, 5, 1)
        grid.addWidget(self.look_away_label, 5, 2)
        
        layout.addLayout(grid)
        
        # Brightness section
        brightness_layout = QHBoxLayout()
        brightness_layout.addWidget(QLabel("💡 Brightness:"))
        self.brightness_progress = QProgressBar()
        self.brightness_progress.setRange(0, 255)
        self.brightness_value_label = QLabel("0")
        brightness_layout.addWidget(self.brightness_progress)
        brightness_layout.addWidget(self.brightness_value_label)
        layout.addLayout(brightness_layout)
        
        # Warnings section
        self.warnings_label = QLabel("✅ All systems normal")
        self.warnings_label.setWordWrap(True)
        self.warnings_label.setStyleSheet("color: #90EE90; font-weight: bold; padding: 8px; background-color: rgba(0,0,0,30); border-radius: 6px;")
        layout.addWidget(self.warnings_label)
        
        # Session info
        self.session_label = QLabel("Session: 00:00:00")
        self.session_label.setAlignment(Qt.AlignCenter)
        self.session_label.setFont(QFont("SF Pro Display", 11, QFont.Bold))
        layout.addWidget(self.session_label)
        
        # Settings
        settings_layout = QHBoxLayout()
        self.auto_zoom_checkbox = QCheckBox("Auto-zoom")
        self.auto_zoom_checkbox.setChecked(True)
        self.show_camera_checkbox = QCheckBox("Show camera")
        settings_layout.addWidget(self.auto_zoom_checkbox)
        settings_layout.addWidget(self.show_camera_checkbox)
        layout.addLayout(settings_layout)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.toggle_button = QPushButton("⏸️ Pause")
        self.settings_button = QPushButton("⚙️ Settings")
        self.close_button = QPushButton("✖️ Hide")
        
        button_layout.addWidget(self.toggle_button)
        button_layout.addWidget(self.settings_button)
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
        
    def update_metrics(self, metrics):
        """Update dashboard with new metrics"""
        # Update status indicator
        self.status_indicator.setStyleSheet("color: #90EE90;" if not metrics.warnings else "color: #ff6b6b;")
        
        # Update blink metrics
        self.blink_count_label.setText(f"Count: {metrics.blink_count}")
        self.blink_rate_label.setText(f"Rate: {metrics.blink_rate:.1f}/min")
        blink_status = "Blinking" if metrics.is_currently_blinking else "Normal"
        self.blink_status_label.setText(f"Status: {blink_status}")
        
        # Update distance
        if metrics.distance_cm > 0:
            self.distance_label.setText(f"Distance: {metrics.distance_cm:.0f}cm")
        else:
            self.distance_label.setText("Distance: --")
        
        # Fix for NoneType error
        distance_state = metrics.distance_state or "unknown"
        self.distance_state_label.setText(f"State: {distance_state.title()}")
        
        # Update EAR and squinting
        self.ear_label.setText(f"EAR: L:{metrics.ear_left:.2f} R:{metrics.ear_right:.2f}")
        squint_text = "Yes" if metrics.is_squinting else "No"
        squint_color = "#ffa500" if metrics.is_squinting else "#90EE90"
        self.squint_label.setText(f"Squinting: {squint_text}")
        self.squint_label.setStyleSheet(f"color: {squint_color};")
        
        # Update gaze direction
        gaze_direction = metrics.gaze_direction or "unknown"
        self.gaze_label.setText(f"Direction: {gaze_direction.title()}")
        self.look_away_label.setText(f"Away: {metrics.look_away_time:.1f}s")
        
        # Update brightness
        brightness_val = int(metrics.brightness)
        self.brightness_progress.setValue(brightness_val)
        self.brightness_value_label.setText(str(brightness_val))
        
        # Update warnings
        if metrics.warnings:
            warning_text = "⚠️ " + " | ".join(metrics.warnings)
            self.warnings_label.setText(warning_text)
            self.warnings_label.setStyleSheet("color: #ff6b6b; font-weight: bold; padding: 8px; background-color: rgba(255,107,107,20); border-radius: 6px;")
        else:
            self.warnings_label.setText("✅ All systems normal")
            self.warnings_label.setStyleSheet("color: #90EE90; font-weight: bold; padding: 8px; background-color: rgba(144,238,144,20); border-radius: 6px;")
        
        # Update session time
        session_time = time.time() - metrics.session_start
        hours = int(session_time // 3600)
        minutes = int((session_time % 3600) // 60)
        seconds = int(session_time % 60)
        self.session_label.setText(f"Session: {hours:02d}:{minutes:02d}:{seconds:02d}")

class EyeTuneApp(QMainWindow):
    """Main application class"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EyeTune")
        self.setFixedSize(400, 300)
        
        # Initialize components
        self.eye_tracker = EyeTracker()
        self.dashboard = DashboardWidget()
        self.notification_manager = NotificationManager()
        self.is_tracking = False
        
        # Setup system tray
        self.setup_system_tray()
        
        # Connect signals
        self.eye_tracker.metrics_updated.connect(self.dashboard.update_metrics)
        self.eye_tracker.metrics_updated.connect(self.handle_eye_health_notifications)
        self.dashboard.toggle_button.clicked.connect(self.toggle_tracking)
        self.dashboard.close_button.clicked.connect(self.hide_dashboard)
        self.dashboard.settings_button.clicked.connect(self.show_settings)
        self.dashboard.show_camera_checkbox.toggled.connect(self.toggle_camera_window)
        
        # Timer for hiding dashboard
        self.hide_timer = QTimer()
        self.hide_timer.timeout.connect(self.hide_dashboard)
        self.hide_timer.setSingleShot(True)
        
        # Auto-start tracking
        self.toggle_tracking()
        
        # Test notification after a short delay to ensure app is running
        QTimer.singleShot(2000, self.test_startup_notification)
    
    def handle_eye_health_notifications(self, metrics):
        """Handle eye health notifications based on metrics"""
        # Only show notifications if tracking is active
        if not self.is_tracking:
            return
        
        # Add some stability checks to prevent flickering notifications
        current_time = time.time()
        
        # Check for low ambient light (only if consistently low for 3 seconds)
        if not hasattr(self, 'low_light_start_time'):
            self.low_light_start_time = None
        if metrics.brightness < BRIGHTNESS_THRESHOLD:
            if self.low_light_start_time is None:
                self.low_light_start_time = current_time
            elif current_time - self.low_light_start_time > 3:  # 3 seconds of low light
                self.notification_manager.show_eye_health_warning("low_light", 
                    f"Current brightness: {metrics.brightness:.0f}")
        else:
            self.low_light_start_time = None
        
        # Check for distance issues (only if consistently close/far for 2 seconds)
        if not hasattr(self, 'distance_warning_start_time'):
            self.distance_warning_start_time = None
        if metrics.distance_cm > 0:
            if metrics.distance_cm < DISTANCE_CLOSE_THRESHOLD:
                if self.distance_warning_start_time is None:
                    self.distance_warning_start_time = current_time
                elif current_time - self.distance_warning_start_time > 2:
                    self.notification_manager.show_eye_health_warning("too_close", 
                        f"Distance: {metrics.distance_cm:.0f}cm")
            elif metrics.distance_cm > DISTANCE_FAR_THRESHOLD:
                if self.distance_warning_start_time is None:
                    self.distance_warning_start_time = current_time
                elif current_time - self.distance_warning_start_time > 2:
                    self.notification_manager.show_eye_health_warning("too_far", 
                        f"Distance: {metrics.distance_cm:.0f}cm")
            else:
                self.distance_warning_start_time = None
        
        # Check for looking away too long (only if consistently looking away for 15 seconds)
        if metrics.gaze_direction != "center" and metrics.look_away_time > 15:
            self.notification_manager.show_eye_health_warning("look_away", 
                f"Looking away for {metrics.look_away_time:.1f}s")
        
        # Check for low blink rate (after 2 minutes of tracking, and only if consistently low)
        session_time = time.time() - metrics.session_start
        if session_time > 120 and metrics.blink_rate < 12:  # Increased threshold and time
            self.notification_manager.show_eye_health_warning("low_blink", 
                f"Current rate: {metrics.blink_rate:.1f} blinks/min")
        
        # Check for squinting (only if consistently squinting for 2 seconds)
        if not hasattr(self, 'squinting_start_time'):
            self.squinting_start_time = None
        if metrics.is_squinting:
            if self.squinting_start_time is None:
                self.squinting_start_time = current_time
            elif current_time - self.squinting_start_time > 2:
                self.notification_manager.show_eye_health_warning("squinting", 
                    f"EAR: {metrics.ear_left:.2f}/{metrics.ear_right:.2f}")
        else:
            self.squinting_start_time = None
        
    def setup_system_tray(self):
        """Setup system tray icon and menu"""
        # Create tray icon
        pixmap = QPixmap(32, 32)
        pixmap.fill(QColor(70, 130, 180))
        painter = QPainter(pixmap)
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Arial", 18, QFont.Bold))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "👁️")
        painter.end()
        
        icon = QIcon(pixmap)
        
        self.tray_icon = QSystemTrayIcon(icon, self)
        
        # Create context menu
        self.tray_menu = QMenu()
        
        # Actions
        self.show_dashboard_action = QAction("📊 Show Dashboard", self)
        self.toggle_action = QAction("▶️ Start Tracking", self)
        self.test_notification_action = QAction("🔔 Test Notification", self)
        self.clear_notifications_action = QAction("🧹 Clear Notifications", self)
        self.settings_action = QAction("⚙️ Settings", self)
        self.about_action = QAction("ℹ️ About EyeTune", self)
        self.quit_action = QAction("🚪 Quit", self)
        
        # Connect actions
        self.show_dashboard_action.triggered.connect(self.show_dashboard)
        self.toggle_action.triggered.connect(self.toggle_tracking)
        self.test_notification_action.triggered.connect(self.test_notification)
        self.clear_notifications_action.triggered.connect(self.clear_notifications)
        self.settings_action.triggered.connect(self.show_settings)
        self.about_action.triggered.connect(self.show_about)
        self.quit_action.triggered.connect(self.quit_app)
        
        # Add actions to menu
        self.tray_menu.addAction(self.show_dashboard_action)
        self.tray_menu.addSeparator()
        self.tray_menu.addAction(self.toggle_action)
        self.tray_menu.addAction(self.test_notification_action)
        self.tray_menu.addAction(self.clear_notifications_action)
        self.tray_menu.addSeparator()
        self.tray_menu.addAction(self.settings_action)
        self.tray_menu.addAction(self.about_action)
        self.tray_menu.addSeparator()
        self.tray_menu.addAction(self.quit_action)
        
        self.tray_icon.setContextMenu(self.tray_menu)
        self.tray_icon.activated.connect(self.tray_icon_activated)
        self.tray_icon.show()
        
        # Hide main window and show notification
        self.hide()
        self.tray_icon.showMessage("EyeTune Started", 
                                  "Eye tracking application is now running.\nClick the tray icon to view dashboard.", 
                                  QSystemTrayIcon.Information, 3000)
    
    def tray_icon_activated(self, reason):
        """Handle tray icon activation"""
        if reason == QSystemTrayIcon.Trigger:  # Single click
            self.show_dashboard()
    
    def show_dashboard(self):
        """Show the dashboard widget"""
        # Position dashboard near system tray (top-right for macOS)
        screen = QApplication.desktop().availableGeometry()
        x = screen.width() - self.dashboard.width() - 20
        y = 50
        self.dashboard.move(x, y)
        self.dashboard.show()
        self.dashboard.raise_()
        self.dashboard.activateWindow()
        
        # Auto-hide after 15 seconds of no interaction
        self.hide_timer.start(15000)
    
    def hide_dashboard(self):
        """Hide the dashboard widget"""
        self.dashboard.hide()
        self.hide_timer.stop()
    
    def toggle_tracking(self):
        """Toggle eye tracking on/off"""
        if self.is_tracking:
            self.eye_tracker.stop_tracking()
            self.toggle_action.setText("▶️ Start Tracking")
            self.dashboard.toggle_button.setText("▶️ Start")
            self.is_tracking = False
        else:
            self.eye_tracker.start_tracking()
            self.toggle_action.setText("⏸️ Pause Tracking")
            self.dashboard.toggle_button.setText("⏸️ Pause")
            self.is_tracking = True
    
    def toggle_camera_window(self, checked):
        """Toggle camera preview window (placeholder)"""
        if checked:
            self.tray_icon.showMessage("Camera View", "Camera window would open here (integration needed)", 
                                      QSystemTrayIcon.Information, 2000)
        # TODO: Integrate with your detect.py camera window
    
    def show_settings(self):
        """Show settings dialog (placeholder)"""
        self.tray_icon.showMessage("Settings", "Settings: Brightness threshold, distance thresholds, etc.", 
                                  QSystemTrayIcon.Information, 2000)
    
    def test_startup_notification(self):
        """Test notification on startup"""
        self.notification_manager.show_notification(
            "EyeTune started! Notifications are now active.", 
            "info", 3000
        )
    
    def test_notification(self):
        """Test notification system"""
        # Get screen info for debugging
        screen = QApplication.desktop().screenGeometry()
        screen_info = f"Screen: {screen.width()}x{screen.height()}"
        
        self.notification_manager.show_notification(
            f"This is a test notification! EyeTune is working properly.\n{screen_info}", 
            "info", 5000
        )
    
    def clear_notifications(self):
        """Clear all active notifications"""
        self.notification_manager.clear_all_notifications()
        self.tray_icon.showMessage("Notifications Cleared", 
                                  "All active notifications have been cleared.", 
                                  QSystemTrayIcon.Information, 2000)
    
    def show_about(self):
        """Show about dialog"""
        self.tray_icon.showMessage("About EyeTune", 
                                  "EyeTune v1.0 - Eye Health Monitoring\nBuilt with Computer Vision & MediaPipe\n\nNow with Push Notifications!", 
                                  QSystemTrayIcon.Information, 4000)
    
    def quit_app(self):
        """Quit the application"""
        if self.is_tracking:
            self.eye_tracker.stop_tracking()
        QApplication.quit()
    
    def closeEvent(self, event):
        """Handle close event - minimize to tray instead of closing"""
        event.ignore()
        self.hide()
        self.tray_icon.showMessage("EyeTune Minimized", 
                                  "Application is still running in the system tray.", 
                                  QSystemTrayIcon.Information, 2000)

def main():
    app = QApplication(sys.argv)
    
    # Check if system tray is available
    if not QSystemTrayIcon.isSystemTrayAvailable():
        print("System tray not available on this system.")
        sys.exit(1)
    
    # Don't quit when last window is closed (since we're using system tray)
    app.setQuitOnLastWindowClosed(False)
    
    # Create and run application
    eyetune_app = EyeTuneApp()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()