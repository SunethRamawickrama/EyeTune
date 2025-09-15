# EyeTune

**HopHacks 2025 - Healthcare Track: Best Healthcare Hack by Commure, 1st place**

EyeTune is a computer vision application that monitors eye health and screen usage patterns using real-time facial landmark detection.

## Features

- **Eye Aspect Ratio (EAR) Detection**: Monitors eye openness for automatic screen zoom adjustment
- **Real-time Face Tracking**: Uses MediaPipe for facial landmark detection
- **Cross-platform Screen Control**: Automatic zoom in/out based on eye squinting detection
- **Blink Detection**: 
  - Precise blink counting with debouncing
  - Real-time blink rate monitoring
  - Blink state tracking and statistics
  
- **Ambient Light Detection**:
  - Real-time brightness analysis
  - Light state change tracking
  - Dark environment warnings
  
- **Distance Calculation**:
  - User-to-screen distance estimation using facial landmarks
  - Distance state monitoring (close/medium/far)
  - Posture warnings for optimal viewing distance
  
- **Eye Direction Tracking**:
  - Real-time gaze direction detection (left/right/center)
  - Look away time tracking and monitoring
  - Direction change state management with debouncing

### 📊 Real-time Statistics
- Total blink count and current blinking state
- Eye gaze direction (left/right/center)
- Total look away time tracking
- Ambient light brightness and state
- Distance from screen in centimeters
- Visual warnings for suboptimal conditions

## Installation

1. Clone the repository:
```bash
git clone <repo-url>
cd EyeTune
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the MediaPipe face landmark model:
```bash
# The model file should be placed in models/face_landmarker.task
```

## Usage

### Basic Usage
```bash
python3 Vision/detect.py
```

## Configuration

### Thresholds (in Vision/utils.py)
- ZOOM_IN_THRESHOLD = 0.19: EAR threshold for zoom in
- ZOOM_OUT_THRESHOLD = 0.25: EAR threshold for zoom out
- EAR_THRESHOLD = 0.25: Blink detection threshold
- BRIGHTNESS_THRESHOLD = 70: Ambient light threshold
- DISTANCE_CLOSE_THRESHOLD = 20: Close distance threshold (cm)
- DISTANCE_FAR_THRESHOLD = 40: Far distance threshold (cm)

### Distance Calculation Constants
- REAL_VERTICAL_DISTANCE = 8.0: Actual forehead-to-nose distance (cm)
- FOCAL_LENGTH = 700: Camera focal length for distance calculation

## Technical Details

### Blinking Detection
- Uses Eye Aspect Ratio (EAR) calculation
- Tracks both left and right eye landmarks
- Implements debouncing to prevent false positives
- Maintains blink timestamps and counters

### Ambient Light Detection
- Converts frames to grayscale for brightness analysis
- Tracks light state changes over time
- Monitors time spent in dark environments
- Provides visual warnings for low light conditions

### Distance Calculation
- Uses forehead and nose tip landmarks
- Calculates pixel distance between key points
- Converts to real-world distance using focal length
- Tracks distance state changes and duration

### Eye Direction Tracking
- Uses eye corner landmarks for gaze direction detection
- Calculates eye center position relative to face center
- Implements debouncing to prevent rapid direction changes
- Tracks total time spent looking away from center

## Dependencies

- numpy: Numerical computations
- opencv-python: Computer vision and image processing
- mediapipe: Facial landmark detection
- pyautogui: Cross-platform screen control
- pywinauto: Windows-specific screen control

## Controls

- ESC: Exit the application
- Automatic: Screen zoom adjusts based on eye squinting
- Visual Warnings: Displayed for low light and close distance
