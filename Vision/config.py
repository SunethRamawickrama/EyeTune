# Thresholds and constants shared across modules

# Eye Aspect Ratio (EAR) thresholds
EAR_ZOOM_IN_THRES = 0.23
EAR_ZOOM_OUT_THRES = 0.28
EAR_BLINK_THRES = 0.20

# Ambient light threshold
BRIGHTNESS_THRESHOLD = 90  # below this is considered dark

# Distance thresholds (in cm)
DISTANCE_CLOSE_THRESHOLD = 35
DISTANCE_FAR_THRESHOLD = 40

# Camera / biometric constants
FOCAL_LENGTH = 600  # camera focal length (in px), TODO: calibrate
IRIS_DIAMETER_CM = 1.17  # avg horizontal iris diameter (in cm)

# Blink history
MAX_BLINK_HISTORY = 60  # Keep last 20 blinks for rate calculation

# MediaPipe Face Mesh landmark indices for Eye Aspect Ratio (EAR)
# Order: [p1, p2, p3, p4, p5, p6] where
# p1-p4 are horizontal eye corners; (p2,p6) and (p3,p5) are vertical eyelid pairs
RIGHT_EYE_EAR_POINTS = [33, 160, 158, 133, 153, 144]
LEFT_EYE_EAR_POINTS = [362, 385, 387, 263, 373, 380]