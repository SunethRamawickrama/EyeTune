# EyeTune

**[HopHacks 2025 - Healthcare Track: Best Healthcare Hack by Commure, 1st place](https://devpost.com/software/eyetune)**

EyeTune is a computer vision application that monitors eye health and screen usage patterns using real-time facial landmark detection.

## Features

- **Eye Aspect Ratio (EAR) Detection**: Monitors eye openness for automatic screen zoom adjustment
- **Real-time Face Tracking**: Uses MediaPipe for facial landmark detection
- **Cross-platform Screen Control**: Automatic zoom in/out based on eye squinting detection
- **Advanced Blink Detection**: 
  - Precise blink counting with refractory period debouncing
  - Real-time blink rate monitoring (blinks per minute)
  - Blink state tracking and statistics
  
- **Ambient Light Detection**:
  - Real-time brightness analysis using luminance calculation
  - Light state change tracking with hysteresis
  - Dark environment warnings and notifications
  - Automatic screen color temperature adjustment
  
- **Distance Calculation**:
  - User-to-screen distance estimation using iris width measurement
  - Distance state monitoring (close/medium/far)
  - Posture warnings for optimal viewing distance
  
- **Eye Direction Tracking**:
  - Real-time gaze direction detection (left/right/center)
  - Look away time tracking and monitoring
  - Direction change state management with debouncing
  - Continuous focus time warnings

### 📊 Real-time Statistics
- Total blink count and current blinking state
- Blink rate calculation (blinks per minute)
- Eye gaze direction (left/right/center)
- Total look away time tracking
- Continuous focus time monitoring
- Ambient light brightness and state
- Distance from screen in centimeters
- Visual warnings for suboptimal conditions
- Cross-platform desktop notifications

## Installation

1. Clone the repository:
```bash
git clone https://github.com/SunethRamawickrama/EyeTune.git eyetune
cd eyetune
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the MediaPipe face landmark model:
```bash
# The model file should be placed in models/face_landmarker.task
# Download from: https://developers.google.com/mediapipe/solutions/vision/face_landmarker
```

## Usage

### Basic Usage
```bash
python3 src/stream.py
```

### Project Structure
```
eyetune-clean/
├── models/
│   └── face_landmarker.task    # MediaPipe face landmark model
├── src/
│   ├── stream.py              # Main application entry point
│   ├── config.py              # Configuration and thresholds
│   ├── trackers.py            # Eye health tracking modules
│   ├── utils.py               # Utility functions (EAR calculation)
│   ├── screen_controller.py   # Cross-platform screen control
│   ├── notifier.py            # Desktop notifications
│   └── color_theme.py         # Screen color temperature adjustment
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## Configuration

### Thresholds (in `src/config.py`)
- `EAR_ZOOM_IN_THRES = 0.23`: EAR threshold for zoom in
- `EAR_ZOOM_OUT_THRES = 0.28`: EAR threshold for zoom out  
- `EAR_BLINK_THRES = 0.20`: Blink detection threshold
- `BRIGHTNESS_THRESHOLD = 90`: Ambient light threshold (below this is considered dark)
- `DISTANCE_CLOSE_THRESHOLD = 35`: Close distance threshold (cm)
- `DISTANCE_FAR_THRESHOLD = 40`: Far distance threshold (cm)

### Distance Calculation Constants
- `FOCAL_LENGTH = 600`: Camera focal length for distance calculation (px)
- `IRIS_DIAMETER_CM = 1.17`: Average horizontal iris diameter (cm)

### Blink Detection Settings
- `MAX_BLINK_HISTORY = 60`: Number of recent blinks to keep for rate calculation
- `REFRACTORY_SEC = 0.27`: Minimum time between distinct blinks (seconds)

### MediaPipe Landmark Indices
- `RIGHT_EYE_EAR_POINTS = [33, 160, 158, 133, 153, 144]`: Right eye landmarks for EAR calculation
- `LEFT_EYE_EAR_POINTS = [362, 385, 387, 263, 373, 380]`: Left eye landmarks for EAR calculation

## Technical Details

### Modular Architecture
The application uses a modular design with specialized tracker classes:

- **BlinkTracker**: Manages blink detection with refractory period debouncing
- **AmbientLightTracker**: Handles light detection with hysteresis and color temperature adjustment
- **DistanceTracker**: Calculates screen distance using iris width measurement
- **DirectionTracker**: Tracks eye gaze direction with state management
- **ZoomController**: Manages automatic screen zoom functionality

### Blinking Detection
- Uses Eye Aspect Ratio (EAR) calculation with 6-point eye landmarks
- Tracks both left and right eye landmarks independently
- Implements refractory period (0.27s) to prevent false positives
- Maintains blink history for rate calculation (blinks per minute)
- Real-time blink state tracking and statistics

### Ambient Light Detection
- Uses luminance calculation (0.2126*R + 0.7152*G + 0.0722*B)
- Implements hysteresis to prevent rapid state changes
- Tracks light state changes and dark environment duration
- Automatic screen color temperature adjustment based on ambient lighting
- Cross-platform color temperature control (Windows/macOS/Linux)

### Distance Calculation
- Uses iris width measurement for accurate distance estimation
- Calculates pixel distance between iris landmarks (left/right iris edges)
- Converts to real-world distance using focal length and known iris diameter
- Tracks distance state changes (close/medium/far) and duration
- Provides posture warnings for optimal viewing distance

### Eye Direction Tracking
- Uses eye corner landmarks for gaze direction detection
- Calculates eye center position relative to face center
- Implements stability buffer (0.5s) to prevent rapid direction changes
- Tracks total time spent looking away from center
- Monitors continuous focus time for eye break reminders

### Screen Control
- Cross-platform zoom functionality (Ctrl/Cmd + Plus/Minus)
- Automatic zoom based on sustained eye squinting (2+ seconds)
- Zoom hold duration of 30 seconds before reset
- Platform-specific hotkey detection (Windows/macOS/Linux)

## Dependencies

- **numpy**: Numerical computations and array operations
- **opencv-python**: Computer vision and image processing
- **mediapipe**: Facial landmark detection and face mesh
- **pyautogui**: Cross-platform screen control and automation
- **pywinauto**: Windows-specific screen control utilities
- **plyer**: Cross-platform desktop notifications

## Controls

- **ESC**: Exit the application
- **Automatic**: Screen zoom adjusts based on sustained eye squinting (2+ seconds)
- **Visual Warnings**: Displayed for low light, close distance, and prolonged focus
- **Desktop Notifications**: Cross-platform notifications for health warnings
