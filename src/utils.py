import numpy as np

def calculate_eye_aspect_ratio(face_landmarks, eye_points, img_w, img_h):
    """
    Calculate Eye Aspect Ratio (EAR) using facial landmarks.
    https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf
    Args:
        face_landmarks: List of facial landmarks
        eye_points: List of 6 eye landmark indices
        img_w: Image width in pixels
        img_h: Image height in pixels
    
    Returns:
        EAR value (float)
    """
    try:
        # normalized coordinates -> pixel coordinates
        points = np.array([
            (face_landmarks[idx].x * img_w, face_landmarks[idx].y * img_h) 
            for idx in eye_points if idx < len(face_landmarks)
        ])
        
        right_vertical_dist = np.linalg.norm(points[1] - points[5])
        left_vertical_dist = np.linalg.norm(points[2] - points[4])
        horizontal_dist = np.linalg.norm(points[0] - points[3])
        
        return (
            (right_vertical_dist + left_vertical_dist) / (2.0 * horizontal_dist) 
            if horizontal_dist > 0 else 0.
        )
    except Exception as e:
        print(f"Error calculating EAR: {e}")
        return 0.