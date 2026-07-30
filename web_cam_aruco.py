import os

import cv2 as cv
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_FILE = os.path.join(SCRIPT_DIR, "calibration_results", "webcam_calibration.npz")

video_capture = cv.VideoCapture(0)

# Support both legacy and modern OpenCV ArUco APIs.
if hasattr(cv.aruco, 'getPredefinedDictionary'):
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
else:
    aruco_dict = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_250)

detector = cv.aruco.ArucoDetector(aruco_dict) if hasattr(cv.aruco, 'ArucoDetector') else None

camera_matrix = None
dist_coeff = None
new_camera_matrix = None

if os.path.exists(CALIBRATION_FILE):
    calibration_data = np.load(CALIBRATION_FILE)
    camera_matrix = calibration_data["camera_matrix"]
    dist_coeff = calibration_data["dist_coeff"]
    print(f"Loaded calibration from {CALIBRATION_FILE}")
else:
    print(f"Calibration file not found: {CALIBRATION_FILE}")
    print("ArUco detection will run without undistortion.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    if camera_matrix is not None and dist_coeff is not None:
        if new_camera_matrix is None:
            height, width = frame.shape[:2]
            new_camera_matrix, _ = cv.getOptimalNewCameraMatrix(
                camera_matrix, dist_coeff, (width, height), 1, (width, height)
            )
        frame = cv.undistort(frame, camera_matrix, dist_coeff, None, new_camera_matrix)

    cv.imshow('Video', frame)
    grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('Grey Video', grey_frame)
    if detector is not None:
        corners, ids, rejected = detector.detectMarkers(grey_frame)
    else:
        corners, ids, rejected = cv.aruco.detectMarkers(grey_frame, aruco_dict)
    print(corners, ids, rejected)
    if ids is not None:
        cv.aruco.drawDetectedMarkers(frame, corners, ids)
        cv.imshow('Detected Aruco Markers', frame)
        
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv.destroyAllWindows()