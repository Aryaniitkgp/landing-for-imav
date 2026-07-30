import os
import time
import numpy as np
import cv2

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Number of INTERNAL corners (columns, rows).
# A 9x7 square checkerboard has 8x6 internal corners.
CHECKERBOARD = (8, 6)

# Size of one square in meters (e.g., 0.025 = 25mm)
SQUARE_SIZE = 0.025  

# Minimum number of snapshots recommended before calibrating
TARGET_CAPTURES = 15  

# Web camera index (0 is usually default integrated/USB webcam)
WEBCAM_INDEX = 0

# Sub-pixel corner refinement criteria
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# ==========================================
# 2. SETUP WORLD COORDINATES & STORAGE
# ==========================================
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

objpoints = []  # 3D world points
imgpoints = []  # 2D pixel coordinates

cap = cv2.VideoCapture(WEBCAM_INDEX)

# Try setting standard HD resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    raise RuntimeError(f"Could not open webcam at index {WEBCAM_INDEX}.")

print("\n" + "="*60)
print(" WEBCAM CALIBRATION INTERACTIVE MODE")
print("="*60)
print(" Instructions:")
print("  - Hold the checkerboard at varied distances, tilts, and corners.")
print("  - Press [SPACE] to capture frame when grid highlights in GREEN.")
print("  - Press [C]     to compute calibration once you have >= 10 captures.")
print("  - Press [Q]     to exit anytime.")
print("="*60 + "\n")

captured_count = 0
flash_timer = 0  # Visual feedback timer on snapshot

# ==========================================
# 3. LIVE CAPTURE LOOP
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    display_frame = frame.copy()

    # Find pattern
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)

    # OpenCV's newer SB detector is often more reliable on phone displays.
    if not found and hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(gray, CHECKERBOARD)

    corners_subpix = None
    if found:
        # Refine corner locations
        corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA)
        # Draw detected pattern
        cv2.drawChessboardCorners(display_frame, CHECKERBOARD, corners_subpix, found)
        status_color = (0, 255, 0)  # Green when ready
        status_text = "Board Detected! Press [SPACE] to capture."
    else:
        status_color = (0, 0, 255)  # Red when not detected
        status_text = "Searching for pattern..."

    # Visual flash effect when frame is saved
    if time.time() - flash_timer < 0.2:
        cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], display_frame.shape[2]), (255, 255, 255), -1)

    # On-screen HUD
    cv2.putText(display_frame, f"Captures: {captured_count}/{TARGET_CAPTURES}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    cv2.putText(display_frame, status_text, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(display_frame, "[SPACE]: Capture  |  [C]: Calibrate  |  [Q]: Quit", 
                (20, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Webcam Calibration", display_frame)
    key = cv2.waitKey(1) & 0xFF

    # Handle Keypresses
    if key == ord(' '):  # SPACE to capture
        if found and corners_subpix is not None:
            objpoints.append(objp)
            imgpoints.append(corners_subpix)
            captured_count += 1
            flash_timer = time.time()
            print(f"[SAVED] Capture #{captured_count}")
        else:
            print("[WARNING] Checkerboard not detected. Move board into clear view.")

    elif key == ord('c') or key == ord('C'):  # Compute calibration
        if captured_count < 5:
            print(f"[ERROR] Only {captured_count} captures. Take at least 5-10 captures across different angles.")
        else:
            print("\nComputing calibration... Please wait...")
            break

    elif key == ord('q') or key == ord('Q') or key == 27:  # ESC / Q
        print("Exiting without calibration.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

# ==========================================
# 4. SOLVE INTRINSICS
# ==========================================
img_shape = gray.shape[::-1]
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_shape, None, None
)

print("\n" + "="*50)
print(f"CALIBRATION COMPLETE!")
print(f"Overall RMS Error : {ret:.4f} pixels (Target < 0.5 px)")
print("="*50)
print("\nCamera Matrix (K):\n", mtx)
print("\nDistortion Coefficients (D):\n", dist.ravel())

# Save to disk
os.makedirs("calibration_results", exist_ok=True)
np.savez("calibration_results/webcam_calibration.npz", camera_matrix=mtx, dist_coeff=dist, rms=ret)
print("\nSaved parameters to 'calibration_results/webcam_calibration.npz'")

# ==========================================
# 5. LIVE RECTIFICATION PREVIEW
# ==========================================
print("\nOpening live undistortion preview. Press [Q] to close.")
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, img_shape, 1, img_shape)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Rectify frame using calculated matrices
    undistorted = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # Stack original and corrected video side-by-side
    combined = np.hstack((frame, undistorted))
    
    cv2.putText(combined, "Raw Webcam Stream", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(combined, "Undistorted / Rectified", (frame.shape[1] + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Live Rectification (Left: Raw | Right: Corrected)", combined)

    if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
        break

cap.release()
cv2.destroyAllWindows()