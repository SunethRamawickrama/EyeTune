#!/usr/bin/env python3
"""
Test script for EyeTune notification system
This script tests the notification functionality without running the full eye tracking app
"""

import sys
import time
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QSystemTrayIcon
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont

# Import the notification classes from the main app
from eyetune_app import NotificationWidget, NotificationManager

class NotificationTester(QWidget):
    """Simple test interface for notifications"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EyeTune Notification Tester")
        self.setFixedSize(400, 300)
        
        # Initialize notification manager
        self.notification_manager = NotificationManager()
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the test interface"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("EyeTune Notification Tester")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Test buttons
        test_info_btn = QPushButton("Test Info Notification")
        test_info_btn.clicked.connect(lambda: self.test_notification("info"))
        layout.addWidget(test_info_btn)
        
        test_warning_btn = QPushButton("Test Warning Notification")
        test_warning_btn.clicked.connect(lambda: self.test_notification("warning"))
        layout.addWidget(test_warning_btn)
        
        test_error_btn = QPushButton("Test Error Notification")
        test_error_btn.clicked.connect(lambda: self.test_notification("error"))
        layout.addWidget(test_error_btn)
        
        test_success_btn = QPushButton("Test Success Notification")
        test_success_btn.clicked.connect(lambda: self.test_notification("success"))
        layout.addWidget(test_success_btn)
        
        # Eye health warning tests
        layout.addWidget(QLabel("Eye Health Warnings:"))
        
        low_light_btn = QPushButton("Test Low Light Warning")
        low_light_btn.clicked.connect(lambda: self.test_eye_warning("low_light"))
        layout.addWidget(low_light_btn)
        
        too_close_btn = QPushButton("Test Too Close Warning")
        too_close_btn.clicked.connect(lambda: self.test_eye_warning("too_close"))
        layout.addWidget(too_close_btn)
        
        look_away_btn = QPushButton("Test Look Away Warning")
        look_away_btn.clicked.connect(lambda: self.test_eye_warning("look_away"))
        layout.addWidget(look_away_btn)
        
        low_blink_btn = QPushButton("Test Low Blink Warning")
        low_blink_btn.clicked.connect(lambda: self.test_eye_warning("low_blink"))
        layout.addWidget(low_blink_btn)
        
        squinting_btn = QPushButton("Test Squinting Warning")
        squinting_btn.clicked.connect(lambda: self.test_eye_warning("squinting"))
        layout.addWidget(squinting_btn)
        
        # Control buttons
        clear_btn = QPushButton("Clear All Notifications")
        clear_btn.clicked.connect(self.notification_manager.clear_all_notifications)
        layout.addWidget(clear_btn)
        
        # Auto-test button
        auto_test_btn = QPushButton("Run Auto Test (5 notifications)")
        auto_test_btn.clicked.connect(self.run_auto_test)
        layout.addWidget(auto_test_btn)
        
    def test_notification(self, notification_type):
        """Test basic notification"""
        messages = {
            "info": "This is an informational notification",
            "warning": "This is a warning notification",
            "error": "This is an error notification",
            "success": "This is a success notification"
        }
        
        self.notification_manager.show_notification(
            messages[notification_type], 
            notification_type, 
            3000
        )
    
    def test_eye_warning(self, warning_type):
        """Test eye health warning"""
        self.notification_manager.show_eye_health_warning(warning_type, "Test details")
    
    def run_auto_test(self):
        """Run automatic test with multiple notifications"""
        warnings = ["low_light", "too_close", "look_away", "low_blink", "squinting"]
        
        for i, warning in enumerate(warnings):
            QTimer.singleShot(i * 1000, lambda w=warning: self.test_eye_warning(w))

def main():
    app = QApplication(sys.argv)
    
    # Check if system tray is available
    if not QSystemTrayIcon.isSystemTrayAvailable():
        print("System tray not available on this system.")
        sys.exit(1)
    
    tester = NotificationTester()
    tester.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
