#!/usr/bin/env python3
"""Webcam ArUco detection + pose estimation using saved calibration.

Usage:
  python3 "aruco _pose.py" --marker-length 0.05

Defaults assume a calibration file saved at
  calibration_results/webcam_calibration.npz
or
  calibration_results/camera_calibration.npz

The script undistorts frames (if calibration found), detects ArUco markers,
estimates pose for each marker using the loaded camera matrix and distortion
coefficients, and draws axes + prints poses.
"""

import os
import argparse
import time
import cv2 as cv
import numpy as np


def load_calibration(candidate_paths):
    for p in candidate_paths:
        if os.path.exists(p):
            try:
                d = np.load(p)
                # Accept different key names used in different scripts
                if "camera_matrix" in d and "dist_coeff" in d:
                    return d["camera_matrix"], d["dist_coeff"], p
                if "camera_matrix" in d and "dist" in d:
                    return d["camera_matrix"], d["dist"], p
                # common webcam cal file name
                if "mtx" in d and "dist" in d:
                    return d["mtx"], d["dist"], p
            except Exception:
                pass
    return None, None, None


def make_aruco_detector():
    # Create dictionary (compat across OpenCV versions)
    aruco = cv.aruco
    if hasattr(aruco, "getPredefinedDictionary"):
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    else:
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)

    # Prefer the newer ArucoDetector when available
    detector = aruco.ArucoDetector(aruco_dict) if hasattr(aruco, "ArucoDetector") else None
    return aruco, aruco_dict, detector


def estimate_pose_single_markers_fallback(aruco, corners, marker_length, camera_matrix, dist_coeff):
    """Fallback estimator using solvePnP when estimatePoseSingleMarkers is unavailable.

    corners: list/array of detected marker corner arrays (each Nx1x2 or 1x4x2)
    Returns rvecs, tvecs shaped like aruco.estimatePoseSingleMarkers: (N,1,3)
    """
    rvecs = []
    tvecs = []
    # define 3D object points for a square marker in marker coordinate frame
    s = float(marker_length) / 2.0
    # Order: top-left, top-right, bottom-right, bottom-left (OpenCV ArUco ordering)
    objp = np.array([
        [-s,  s, 0.0],
        [ s,  s, 0.0],
        [ s, -s, 0.0],
        [-s, -s, 0.0],
    ], dtype=np.float32)

    # Choose PnP method if available
    if hasattr(cv, 'SOLVEPNP_IPPE_SQUARE'):
        pnp_flag = cv.SOLVEPNP_IPPE_SQUARE
    elif hasattr(cv, 'SOLVEPNP_IPPE'):
        pnp_flag = cv.SOLVEPNP_IPPE
    else:
        pnp_flag = cv.SOLVEPNP_ITERATIVE

    for c in corners:
        pts = np.asarray(c).reshape(-1, 2).astype(np.float32)
        if pts.shape[0] < 4:
            # skip malformed detection
            continue
        # solvePnP expects matching order between objp and pts
        success, rvec, tvec = cv.solvePnP(objp, pts, camera_matrix, dist_coeff, flags=pnp_flag)
        if not success:
            # fallback to iterative without flags
            success, rvec, tvec = cv.solvePnP(objp, pts, camera_matrix, dist_coeff)
        rvecs.append(rvec.reshape(1, 3))
        tvecs.append(tvec.reshape(1, 3))

    if len(rvecs) == 0:
        return None, None, None

    rvecs = np.array(rvecs, dtype=np.float32).reshape(-1, 1, 3)
    tvecs = np.array(tvecs, dtype=np.float32).reshape(-1, 1, 3)
    return rvecs, tvecs, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--marker-length", type=float, default=0.05,
                        help="Marker side length in meters (same unit as calibration)")
    parser.add_argument("--no-undistort", action="store_true", help="Disable undistortion even if calibration found")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "calibration_results", "webcam_calibration.npz"),
        os.path.join(script_dir, "calibration_results", "camera_calibration.npz"),
        os.path.join(script_dir, "calibration_results", "camera_calibration.npz"),
    ]

    cam_mtx, dist_coeff, loaded_path = load_calibration(candidates)
    if loaded_path:
        print(f"Loaded calibration from: {loaded_path}")
    else:
        print("No calibration file found. Running without undistortion.")

    aruco, aruco_dict, detector = make_aruco_detector()

    cap = cv.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")

    map1 = map2 = None
    new_mtx = None

    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        # prepare undistort maps on first frame if calibration present
        if cam_mtx is not None and dist_coeff is not None and not args.no_undistort:
            if map1 is None or map2 is None:
                new_mtx, roi = cv.getOptimalNewCameraMatrix(cam_mtx, dist_coeff, (w, h), 1, (w, h))
                map1, map2 = cv.initUndistortRectifyMap(cam_mtx, dist_coeff, None, new_mtx, (w, h), cv.CV_16SC2)
            frame_undist = cv.remap(frame, map1, map2, cv.INTER_LINEAR)
            proc_frame = frame_undist
        else:
            proc_frame = frame

        gray = cv.cvtColor(proc_frame, cv.COLOR_BGR2GRAY)

        # Detect markers
        if detector is not None:
            corners, ids, rejected = detector.detectMarkers(gray)
        else:
            corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict)

        if ids is not None and len(ids) > 0:
            aruco.drawDetectedMarkers(proc_frame, corners, ids)

            # Estimate pose for each marker (requires camera params)
            if cam_mtx is not None and dist_coeff is not None:
                # corners -> list of N arrays
                if hasattr(aruco, 'estimatePoseSingleMarkers'):
                    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, args.marker_length, cam_mtx, dist_coeff)
                else:
                    rvecs, tvecs, _ = estimate_pose_single_markers_fallback(aruco, corners, args.marker_length, cam_mtx, dist_coeff)
                if rvecs is None or tvecs is None:
                    # nothing to do
                    continue
                for i, id_ in enumerate(ids.flatten()):
                    if i >= len(rvecs):
                        break
                    rvec = rvecs[i][0]
                    tvec = tvecs[i][0]
                    # Draw axis (length set to marker_length * 0.5)
                    axis_len = args.marker_length * 0.5
                    try:
                        cv.drawFrameAxes(proc_frame, cam_mtx, dist_coeff, rvec, tvec, axis_len)
                    except Exception:
                        # fallback to aruco.drawAxis if available
                        if hasattr(aruco, 'drawAxis'):
                            aruco.drawAxis(proc_frame, cam_mtx, dist_coeff, rvec, tvec, axis_len)

                    # Print id and translation in meters (or units used)
                    tv = tvec
                    rv = rvec
                    print(f"ID {int(id_)}: t = [{tv[0]:.3f}, {tv[1]:.3f}, {tv[2]:.3f}]  r = [{rv[0]:.3f}, {rv[1]:.3f}, {rv[2]:.3f}]")

        # Show rejected candidates lightly for debugging
        if rejected is not None and len(rejected) > 0:
            aruco.drawDetectedMarkers(proc_frame, rejected, borderColor=(100, 100, 100))

        # Display
        combined = np.hstack((frame, proc_frame)) if frame.shape == proc_frame.shape else proc_frame
        cv.imshow("ArUco Pose (Left: Raw | Right: Processed)", combined)

        key = cv.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
