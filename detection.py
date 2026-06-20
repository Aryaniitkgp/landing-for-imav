"""
Autonomous landing on a moving ArUco platform (PX4 SITL + Gazebo).

Pipeline (after Falanga/Vlahek-style moving-platform landing, adapted for ArUco):
  ArUco detect (down cam) -> marker world pose -> constant-velocity EKF
  -> OFFBOARD position + velocity-feedforward setpoints -> PX4 controller.

State machine:  TAKEOFF -> SEARCH -> TRACK -> DESCEND -> LAND

The EKF predicts every tick and only updates when the target marker is seen,
so short camera dropouts (target leaves FOV) are bridged by prediction.
"""

import math
import sys
import termios
import time
import tty
import threading

import numpy as np
import cv2
import cv2.aruco as aruco
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
)

# ============================================================================
# Configuration (all grounded in the imav2026_scaled world / x500_dual_cam)
# ============================================================================
# --- Cameras (bridge.py publishes 640x480) ---
FRONT_CAM_TOPIC = '/drone/camera/front_camera'
DOWN_CAM_TOPIC  = '/drone/camera/down_camera'
IMG_W, IMG_H    = 640, 480
DOWN_CAM_HFOV   = 1.74                      # rad, from model.sdf

# Pinhole intrinsics derived from HFOV (Gazebo camera, no distortion)
FX = 0.5 * IMG_W / math.tan(0.5 * DOWN_CAM_HFOV)
FY = FX                                     # square pixels
CX, CY = IMG_W / 2.0, IMG_H / 2.0

# --- ArUco ---
ARUCO_DICT      = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
TARGET_ID       = 1                         # moving landing_platform marker
PLATFORM_TOP_Z  = 0.28                      # marker height above ground (m)

# --- Camera-axis -> body-axis mapping. Flip these if the drone chases the
#     marker the WRONG way during a test run. ---
FWD_SIGN   = -1.0   # image-down (+v) maps to body-forward with this sign
RIGHT_SIGN = +1.0   # image-right (+u) maps to body-right with this sign

# --- Mission ---
SEARCH_ALT   = 4.0          # cruise/search altitude (m AGL)
TRACK_ALT    = 3.0          # altitude while locking onto the target
LAND_ALT     = 0.6          # altitude at which we commit to touchdown
DESCEND_RATE = 0.4          # m/s descent while aligned
ALIGN_TOL    = 0.30         # horizontal error (m) considered "aligned"
LOCK_TIME    = 1.5          # s of continuous alignment before descending
LOST_TIMEOUT = 3.0          # s without detection before reverting to SEARCH

# Search lawnmower (NED, relative to takeoff origin)
SEARCH_WPS = [
    (0.0, 0.0), (8.0, 0.0), (8.0, 3.0), (0.0, 3.0),
    (0.0, -3.0), (8.0, -3.0), (8.0, 6.0), (0.0, 6.0),
]
WP_REACH_TOL = 0.8

# --- Control gains ---
KP_XY = 1.0                 # not used directly; PX4 position controller handles it

# --- EKF (constant velocity: [N, E, vN, vE]) ---
EKF_Q_POS = 0.05
EKF_Q_VEL = 1.0
EKF_R_MEAS = 0.10

HELP = """
============== AUTONOMOUS MOVING-PLATFORM LANDING ==============
  Mission runs automatically: TAKEOFF -> SEARCH -> TRACK -> LAND
  Target = ArUco ID {tid} (moving landing platform)
  Keys:  P pause/hover   R resume   L force-land   X quit
===============================================================
""".format(tid=TARGET_ID)


# ============================================================================
# Constant-velocity EKF for the platform position in world NED
# ============================================================================
class PlatformEKF:
    def __init__(self):
        self.x = np.zeros(4)            # [N, E, vN, vE]
        self.P = np.eye(4) * 10.0
        self.initialized = False
        self.last_update = 0.0

    def predict(self, dt):
        if not self.initialized:
            return
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)
        q = np.diag([EKF_Q_POS, EKF_Q_POS, EKF_Q_VEL, EKF_Q_VEL]) * dt
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + q

    def update(self, n, e, t_now):
        z = np.array([n, e])
        if not self.initialized:
            self.x = np.array([n, e, 0.0, 0.0])
            self.P = np.eye(4) * 1.0
            self.initialized = True
            self.last_update = t_now
            return
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        R = np.eye(2) * EKF_R_MEAS
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        self.last_update = t_now

    def pos(self):
        return self.x[0], self.x[1]

    def vel(self):
        return self.x[2], self.x[3]

    def predict_ahead(self, horizon):
        """Platform position 'horizon' seconds in the future."""
        return self.x[0] + self.x[2] * horizon, self.x[1] + self.x[3] * horizon


# ============================================================================
# Main controller node
# ============================================================================
class LandingController(Node):
    # mission states
    INIT, TAKEOFF, SEARCH, TRACK, DESCEND, LAND, DONE = range(7)

    def __init__(self):
        super().__init__('landing_controller')

        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        cam_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        # PX4 I/O
        self.offboard_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', px4_qos)
        self.traj_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', px4_qos)
        self.cmd_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', px4_qos)
        self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1',
            self._pos_cb, px4_qos)
        self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status_v4', self._status_cb, px4_qos)

        # Vision
        self.cvbridge = CvBridge()
        self.detector = aruco.ArucoDetector(ARUCO_DICT, aruco.DetectorParameters())
        self.create_subscription(
            Image, DOWN_CAM_TOPIC, lambda m: self._img_cb(m, 'Down'), cam_qos)
        self.create_subscription(
            Image, FRONT_CAM_TOPIC, lambda m: self._img_cb(m, 'Front'), cam_qos)

        # State
        self.local_pos = None
        self.ekf = PlatformEKF()
        self.state = self.INIT
        self.origin_z = None                 # NED z at arming (ground level)
        self.takeoff_xy = (0.0, 0.0)
        self.target_z = 0.0                  # commanded NED z
        self.wp_idx = 0
        self.aligned_since = None
        self.last_target_seen = 0.0
        self.last_det_id = None
        self.paused = False
        self.force_land = False
        self.running = True
        self.tick = 0
        self.lock = threading.Lock()

        self.create_timer(0.02, self._control_loop)    # 50 Hz
        self.get_logger().info(
            f'Intrinsics fx={FX:.1f} fy={FY:.1f} cx={CX:.0f} cy={CY:.0f}')
        print(HELP)

    # -------------------------------------------------------------- PX4 callbacks
    def _pos_cb(self, msg):
        self.local_pos = msg

    def _status_cb(self, msg):
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state

    # -------------------------------------------------------------- vision
    def _img_cb(self, msg, cam_label):
        try:
            frame = self.cvbridge.imgmsg_to_cv2(msg, 'bgr8')
            frame = cv2.resize(frame, (IMG_W, IMG_H))
        except CvBridgeError as e:
            self.get_logger().error(str(e))
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is not None:
            for i, mid in enumerate(ids.flatten()):
                mid = int(mid)
                if mid != TARGET_ID:
                    continue                       # ignore every other marker
                c = corners[i][0]
                cx, cy = c.mean(axis=0)
                aruco.drawDetectedMarkers(frame, corners[i:i + 1])
                cv2.putText(frame, f'TARGET {mid}', (int(cx), int(cy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if mid != self.last_det_id:
                    print(f'[{cam_label} cam] Target ArUco detected -> ID: {mid}')
                    self.last_det_id = mid
                # Only the DOWN camera feeds the landing EKF.
                if cam_label == 'Down':
                    self._ingest_target(cx, cy)

        cv2.putText(frame, f'[{cam_label}] {self._state_name()}',
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        cv2.imshow(f'{cam_label} Camera', frame)
        cv2.waitKey(1)

    def _ingest_target(self, u, v):
        """Convert a down-cam marker pixel to a world-NED position, feed EKF."""
        if self.local_pos is None or self.origin_z is None:
            return
        lp = self.local_pos
        # Height of the camera above the marker plane.
        h = (lp.z - self.origin_z) * -1.0 - PLATFORM_TOP_Z   # NED z is down
        h = max(h, 0.3)
        # Normalised pixel offset from principal point.
        nx = (u - CX) / FX           # +right in image
        ny = (v - CY) / FY           # +down in image
        # Metric offset of marker from drone, in body FRD (forward, right).
        off_fwd   = FWD_SIGN   * ny * h
        off_right = RIGHT_SIGN * nx * h
        # Rotate body->world NED by drone heading.
        yaw = lp.heading
        north = lp.x + math.cos(yaw) * off_fwd - math.sin(yaw) * off_right
        east  = lp.y + math.sin(yaw) * off_fwd + math.cos(yaw) * off_right
        with self.lock:
            self.ekf.update(north, east, time.monotonic())
            self.last_target_seen = time.monotonic()

    # -------------------------------------------------------------- helpers
    def _state_name(self):
        return ['INIT', 'TAKEOFF', 'SEARCH', 'TRACK',
                'DESCEND', 'LAND', 'DONE'][self.state]

    def send_command(self, command, **p):
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.command = command
        for i in range(1, 8):
            setattr(msg, f'param{i}', float(p.get(f'param{i}', 0.0)))
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.cmd_pub.publish(msg)

    def _arm_and_offboard(self):
        self.send_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
                          param1=1.0, param2=6.0)
        self.send_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                          param1=1.0)

    def _publish_setpoint(self, n, e, d, vn=None, ve=None, vd=None, yaw=None):
        now_us = int(self.get_clock().now().nanoseconds / 1000)
        ob = OffboardControlMode()
        ob.timestamp = now_us
        ob.position = True
        ob.velocity = vn is not None
        self.offboard_pub.publish(ob)

        sp = TrajectorySetpoint()
        sp.timestamp = now_us
        sp.position = [float(n), float(e), float(d)]
        nan = float('nan')
        sp.velocity = [
            float(vn) if vn is not None else nan,
            float(ve) if ve is not None else nan,
            float(vd) if vd is not None else nan,
        ]
        sp.yaw = float(yaw) if yaw is not None else nan
        self.traj_pub.publish(sp)

    def _horizontal_err(self):
        """Distance from drone to current EKF platform estimate."""
        lp = self.local_pos
        pn, pe = self.ekf.pos()
        return math.hypot(lp.x - pn, lp.y - pe)

    # -------------------------------------------------------------- main loop
    def _control_loop(self):
        if self.local_pos is None:
            return
        lp = self.local_pos
        now = time.monotonic()

        # First valid position -> capture ground level, arm, start mission.
        if self.state == self.INIT:
            if lp.xy_valid and lp.z_valid:
                self.origin_z = lp.z
                self.takeoff_xy = (lp.x, lp.y)
                self.target_z = lp.z - SEARCH_ALT
                self.state = self.TAKEOFF
                self.get_logger().info('Arming + takeoff...')
                self._arm_and_offboard()
            else:
                self._publish_setpoint(lp.x, lp.y, lp.z)
                return

        # Keep re-asserting offboard/arm during the first second.
        if self.state == self.TAKEOFF and self.tick < 50 and self.tick % 10 == 0:
            self._arm_and_offboard()
        self.tick += 1

        # EKF time update.
        with self.lock:
            self.ekf.predict(0.02)
            have_target = self.ekf.initialized
            seen_recently = (now - self.last_target_seen) < LOST_TIMEOUT

        # Manual overrides.
        if self.force_land:
            self.send_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            self.state = self.DONE
        if self.paused:
            self._publish_setpoint(lp.x, lp.y, self.target_z)
            return

        # ---- state machine ----
        if self.state == self.TAKEOFF:
            tn, te = self.takeoff_xy
            self._publish_setpoint(tn, te, self.target_z)
            if abs((lp.z - self.origin_z) + SEARCH_ALT) < 0.4:
                self.state = self.SEARCH
                self.get_logger().info('Reached search altitude -> SEARCH')

        elif self.state == self.SEARCH:
            tn, te = self.takeoff_xy
            wn, we = SEARCH_WPS[self.wp_idx]
            gn, ge = tn + wn, te + we
            self._publish_setpoint(gn, ge, self.target_z)
            if math.hypot(lp.x - gn, lp.y - ge) < WP_REACH_TOL:
                self.wp_idx = (self.wp_idx + 1) % len(SEARCH_WPS)
            if have_target and seen_recently:
                self.state = self.TRACK
                self.target_z = lp.z - TRACK_ALT
                self.aligned_since = None
                self.get_logger().info('Target acquired -> TRACK')

        elif self.state == self.TRACK:
            self._track_platform(self.origin_z - TRACK_ALT)
            if not seen_recently:
                self.state = self.SEARCH
                self.get_logger().warn('Target lost -> back to SEARCH')
            elif self._horizontal_err() < ALIGN_TOL:
                if self.aligned_since is None:
                    self.aligned_since = now
                elif now - self.aligned_since > LOCK_TIME:
                    self.state = self.DESCEND
                    self.get_logger().info('Locked -> DESCEND')
            else:
                self.aligned_since = None

        elif self.state == self.DESCEND:
            # Descend only while staying aligned; EKF carries short dropouts.
            aligned = self._horizontal_err() < ALIGN_TOL * 2.0
            vd = DESCEND_RATE if aligned else -0.2     # ease up if drifting
            self._track_platform(None, vd=vd)
            agl = (lp.z - self.origin_z) * -1.0
            if not seen_recently and (now - self.last_target_seen) > LOST_TIMEOUT:
                self.state = self.SEARCH
                self.target_z = lp.z - SEARCH_ALT
                self.get_logger().warn('Lost during descent -> SEARCH')
            elif agl < LAND_ALT:
                self.state = self.LAND
                self.get_logger().info('Low + aligned -> LAND')

        elif self.state == self.LAND:
            # Final committed descent tracking platform horizontally.
            self._track_platform(None, vd=DESCEND_RATE)
            if self.local_pos and self._is_landed():
                self.send_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                                  param1=0.0)   # disarm
                self.state = self.DONE
                self.get_logger().info('Touchdown -> disarmed. DONE.')

        elif self.state == self.DONE:
            return

    def _track_platform(self, fixed_z, vd=None):
        """Command setpoint at EKF-predicted platform pose with velocity FF."""
        lp = self.local_pos
        # Lead the target slightly to compensate control lag.
        pn, pe = self.ekf.predict_ahead(0.3)
        vn, ve = self.ekf.vel()
        if vd is not None:
            d = lp.z + vd * 0.5            # gentle altitude target step
            self._publish_setpoint(pn, pe, d, vn=vn, ve=ve, vd=vd)
        else:
            self._publish_setpoint(pn, pe, fixed_z, vn=vn, ve=ve, vd=0.0)

    def _is_landed(self):
        # On the moving pad we approximate touchdown by low altitude + low climb.
        lp = self.local_pos
        agl = (lp.z - self.origin_z) * -1.0
        return agl < 0.15

    # -------------------------------------------------------------- keyboard
    def handle_key(self, key):
        k = key.lower()
        if k == 'p':
            self.paused = True
            self.get_logger().info('PAUSED (hover)')
        elif k == 'r':
            self.paused = False
            self.get_logger().info('RESUMED')
        elif k == 'l':
            self.force_land = True
        elif k == 'x':
            self.running = False


def keyboard_thread(node):
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while node.running and rclpy.ok():
            ch = sys.stdin.read(1)
            if ch:
                node.handle_key(ch)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main(args=None):
    rclpy.init(args=args)
    node = LandingController()
    kb = threading.Thread(target=keyboard_thread, args=(node,), daemon=True)
    kb.start()
    try:
        while rclpy.ok() and node.running:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
