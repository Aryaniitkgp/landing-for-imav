"""
Landing on a horizontally-oscillating platform (PX4 SITL, Gazebo, ROS2/XRCE-DDS).

Adapts Baca/Stepan/Saska (MBZIRC 2017, "Autonomous Landing On A Moving Car"):
  - car-circle detector      -> ArUco (DICT_4X4_50, ID 1)
  - constant-curvature model -> per-axis constant-ACCELERATION Kalman filter
                                + sinusoid predictor (sway-aware)
  - MPC tracker (Stage 2)    -> Stage 1: PD position control + pad-velocity FF
  - custom SO3 controller    -> PX4 mc_pos_control (we only send velocity setpoints)

Mission FSM:
  TAKEOFF -> SEARCH -> ACQUIRE -> APPROACH -> DESCEND -> FINAL -> LANDED
Touchdown is timed near a sway turnaround (pad velocity ~= 0).

All parameters are inherited from the earlier detection.py / world setup.
"""

import math
import sys
import termios
import time
import tty
import threading
from collections import deque

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
# Configuration (inherited from detection.py / imav2026_scaled world)
# ============================================================================
DOWN_CAM_TOPIC  = '/drone/camera/down_camera'
FRONT_CAM_TOPIC = '/drone/camera/front_camera'
IMG_W, IMG_H    = 640, 480
DOWN_CAM_HFOV   = 1.74
FX = 0.5 * IMG_W / math.tan(0.5 * DOWN_CAM_HFOV)
FY = FX
CX, CY = IMG_W / 2.0, IMG_H / 2.0

ARUCO_DICT     = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
TARGET_ID      = 1
MARKER_LEN     = 0.73                 # black-square side (m), for solvePnP option
PLATFORM_TOP_Z = 0.28                 # marker height above ground (m)

# Camera-axis -> body-axis signs (flip if the drone chases the wrong way).
FWD_SIGN   = -1.0
RIGHT_SIGN = +1.0

# --- Altitudes / mission ---
SEARCH_ALT       = 5.0                # takeoff + search + approach altitude
FINAL_ALT        = 1.0               # altitude (AGL) at which FINAL phase begins
TOUCHDOWN_HEIGHT = 0.25              # cut motors at this height above the marker
SEARCH_SPEED     = 4.5               # max search speed (m/s)
ALIGN_TOL        = 0.25             # horizontal error considered aligned (m)
N_STABLE         = 5                # consecutive detections before trusting
LOST_TIMEOUT     = 2.0            # s without detection before degrading

# --- Descent rates ---
DESCENT_SPEED    = 0.1            # m/s nominal descent while tracking
FINAL_DESCENT    = 0.35          # m/s during the committed touchdown drop

# --- Pad oscillation prior (spring pad: omega=sqrt(k/m)~1.58, period ~4s) ---
OSC_PERIOD_PRIOR = 4.0
V_TURNAROUND     = 0.25          # |pad vel| below this = near a sway turnaround

# --- Predictor ---
LOOKAHEAD        = 0.30          # s control lead (KF const-accel valid here)
HIST_SECONDS     = 6.0          # position history kept for sinusoid fit

# --- Stage-1 tracker: PD + pad-velocity feedforward ---
KP_XY = 1.2
KD_XY = 0.30
VEL_MAX_XY = 2.5

# --- Kalman filter (per axis, state [pos, vel, acc]) ---
KF_SIGMA_JERK = 2.0             # process noise (m/s^3)
KF_R_MEAS     = 0.01           # measurement variance (m^2) ~ (0.1 m)^2
CONF_POS_STD  = 0.5           # gate descent when position std below this (m)

# Search lawnmower (NED, relative to takeoff origin)
SEARCH_WPS = [
    (0.0, 0.0), (8.0, 0.0), (8.0, 3.0), (0.0, 3.0),
    (0.0, -3.0), (8.0, -3.0), (8.0, 6.0), (0.0, 6.0),
]

HELP = """
========= OSCILLATING-PAD LANDING (PD + feedforward) =========
  TAKEOFF -> SEARCH -> ACQUIRE -> APPROACH -> DESCEND -> FINAL
  Target = ArUco ID {tid}. Touchdown timed near pad v=0.
  Keys:  P pause   R resume   X quit
==============================================================
""".format(tid=TARGET_ID)


# ============================================================================
# Per-axis constant-acceleration Kalman filter
# ============================================================================
class AxisKF:
    def __init__(self):
        self.x = np.zeros(3)              # [pos, vel, acc]
        self.P = np.eye(3) * 100.0
        self.initialized = False

    def predict(self, dt):
        if not self.initialized:
            return
        F = np.array([[1, dt, 0.5 * dt * dt],
                      [0, 1, dt],
                      [0, 0, 1]], dtype=float)
        q = KF_SIGMA_JERK ** 2          # discrete white-noise jerk process noise
        Q = q * np.array([
            [dt**5 / 20, dt**4 / 8, dt**3 / 6],
            [dt**4 / 8,  dt**3 / 3, dt**2 / 2],
            [dt**3 / 6,  dt**2 / 2, dt],
        ], dtype=float)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z):
        if not self.initialized:
            self.x = np.array([z, 0.0, 0.0])
            self.P = np.diag([KF_R_MEAS, 1.0, 1.0])
            self.initialized = True
            return
        H = np.array([[1.0, 0.0, 0.0]])
        R = np.array([[KF_R_MEAS]])
        y = np.array([z]) - H @ self.x
        S = H @ self.P @ H.T + R
        K = (self.P @ H.T) @ np.linalg.inv(S)
        self.x = self.x + (K @ y).flatten()
        self.P = (np.eye(3) - K @ H) @ self.P

    def pos(self): return self.x[0]
    def vel(self): return self.x[1]
    def acc(self): return self.x[2]
    def pos_std(self): return math.sqrt(max(self.P[0, 0], 0.0))


# ============================================================================
# Pad estimator: two axis-KFs + sway-aware sinusoid predictor
# ============================================================================
class PadEstimator:
    def __init__(self):
        self.kf_n = AxisKF()
        self.kf_e = AxisKF()
        self.hist = deque()              # (t, n, e)
        self.last_meas_t = 0.0
        self.lock = threading.Lock()

    def predict(self, dt):
        with self.lock:
            self.kf_n.predict(dt)
            self.kf_e.predict(dt)

    def update(self, n, e, t):
        with self.lock:
            self.kf_n.update(n)
            self.kf_e.update(e)
            self.last_meas_t = t
            self.hist.append((t, n, e))
            while self.hist and t - self.hist[0][0] > HIST_SECONDS:
                self.hist.popleft()

    @property
    def ready(self):
        return self.kf_n.initialized and self.kf_e.initialized

    def state(self):
        with self.lock:
            return (self.kf_n.pos(), self.kf_e.pos(),
                    self.kf_n.vel(), self.kf_e.vel())

    def speed(self):
        with self.lock:
            return math.hypot(self.kf_n.vel(), self.kf_e.vel())

    def confident(self):
        with self.lock:
            return (self.kf_n.pos_std() < CONF_POS_STD and
                    self.kf_e.pos_std() < CONF_POS_STD)

    # -- sinusoid fit on one axis: p(t) ~= c + A sin(wt) + B cos(wt) -----------
    def _fit_axis(self, idx, omega):
        ts = np.array([h[0] for h in self.hist])
        ps = np.array([h[idx] for h in self.hist])
        if len(ts) < 10:
            return None
        t0 = ts[0]
        tt = ts - t0
        M = np.column_stack([np.ones_like(tt),
                             np.sin(omega * tt), np.cos(omega * tt)])
        coef, *_ = np.linalg.lstsq(M, ps, rcond=None)
        return coef, t0          # coef = [c, A, B]

    def _omega_from_hist(self, idx):
        """Angular frequency from zero-crossings; fall back to the prior."""
        if len(self.hist) < 12:
            return 2 * math.pi / OSC_PERIOD_PRIOR
        ts = np.array([h[0] for h in self.hist])
        ps = np.array([h[idx] for h in self.hist])
        ps = ps - ps.mean()
        crossings = np.where(np.diff(np.sign(ps)) != 0)[0]
        if len(crossings) >= 2:
            period = 2.0 * (ts[crossings[-1]] - ts[crossings[0]]) / (len(crossings) - 1)
            if 1.0 < period < 12.0:
                return 2 * math.pi / period
        return 2 * math.pi / OSC_PERIOD_PRIOR

    def predict_pos(self, horizon):
        """Pad (N, E) 'horizon' s ahead. KF for short, sinusoid for long."""
        with self.lock:
            if not self.ready:
                return None
            n0, e0 = self.kf_n.pos(), self.kf_e.pos()
            vn, ve = self.kf_n.vel(), self.kf_e.vel()
            an, ae = self.kf_n.acc(), self.kf_e.acc()
            hist_len = len(self.hist)

            if horizon <= 0.5 or hist_len < 16:
                pn = n0 + vn * horizon + 0.5 * an * horizon ** 2
                pe = e0 + ve * horizon + 0.5 * ae * horizon ** 2
                return pn, pe

            out = []
            for idx, p0 in ((1, n0), (2, e0)):
                omega = self._omega_from_hist(idx)
                fit = self._fit_axis(idx, omega)
                if fit is None:
                    out.append(p0)
                    continue
                (c, A, B), t0 = fit
                t_future = (self.hist[-1][0] - t0) + horizon
                out.append(c + A * math.sin(omega * t_future) +
                           B * math.cos(omega * t_future))
            return out[0], out[1]


# ============================================================================
# Controller node
# ============================================================================
class PadLandingController(Node):
    (INIT, TAKEOFF, SEARCH, ACQUIRE, APPROACH,
     DESCEND, FINAL, LANDED) = range(8)

    def __init__(self):
        super().__init__('pad_landing_controller')

        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        cam_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)

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

        self.cvbridge = CvBridge()
        self.detector = aruco.ArucoDetector(ARUCO_DICT, aruco.DetectorParameters())
        self.create_subscription(
            Image, DOWN_CAM_TOPIC, lambda m: self._img_cb(m, 'Down'), cam_qos)
        self.create_subscription(
            Image, FRONT_CAM_TOPIC, lambda m: self._img_cb(m, 'Front'), cam_qos)

        self.local_pos = None
        self.est = PadEstimator()
        self.state = self.INIT
        self.origin_z = None
        self.takeoff_xy = (0.0, 0.0)
        self.target_z = 0.0
        self.wp_idx = 0
        self.stable_count = 0
        self.last_seen = 0.0
        self.last_det_id = None
        self.paused = False
        self.running = True
        self.tick = 0

        self.create_timer(0.02, self._control_loop)        # 50 Hz
        self.get_logger().info(
            f'Intrinsics fx={FX:.1f} cx={CX:.0f} cy={CY:.0f}; '
            f'period prior {OSC_PERIOD_PRIOR}s')
        print(HELP)

    # ---------------------------------------------------------------- PX4 cb
    def _pos_cb(self, msg):
        self.local_pos = msg

    def _status_cb(self, msg):
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state

    # ---------------------------------------------------------------- vision
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
                if int(mid) != TARGET_ID:
                    continue
                c = corners[i][0]
                cx, cy = c.mean(axis=0)
                aruco.drawDetectedMarkers(frame, corners[i:i + 1])
                cv2.putText(frame, f'TARGET {TARGET_ID}', (int(cx), int(cy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if self.last_det_id != TARGET_ID:
                    print(f'[{cam_label}] target ID {TARGET_ID} acquired')
                    self.last_det_id = TARGET_ID
                if cam_label == 'Down':
                    self._ingest_target(cx, cy)

        cv2.putText(frame, f'[{cam_label}] {self._state_name()}', (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        cv2.imshow(f'{cam_label} Camera', frame)
        cv2.waitKey(1)

    def _ingest_target(self, u, v):
        """Down-cam marker pixel -> world-NED (N,E); feed both axis-KFs."""
        if self.local_pos is None or self.origin_z is None:
            return
        lp = self.local_pos
        h = (lp.z - self.origin_z) * -1.0 - PLATFORM_TOP_Z
        h = max(h, 0.3)
        nx = (u - CX) / FX
        ny = (v - CY) / FY
        off_fwd   = FWD_SIGN   * ny * h
        off_right = RIGHT_SIGN * nx * h
        yaw = lp.heading
        north = lp.x + math.cos(yaw) * off_fwd - math.sin(yaw) * off_right
        east  = lp.y + math.sin(yaw) * off_fwd + math.cos(yaw) * off_right
        self.est.update(north, east, time.monotonic())
        self.last_seen = time.monotonic()
        self.stable_count = min(self.stable_count + 1, N_STABLE + 5)

    # ---------------------------------------------------------------- helpers
    def _state_name(self):
        return ['INIT', 'TAKEOFF', 'SEARCH', 'ACQUIRE', 'APPROACH',
                'DESCEND', 'FINAL', 'LANDED'][self.state]

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

    def _publish_position(self, n, e, d):
        now_us = int(self.get_clock().now().nanoseconds / 1000)
        ob = OffboardControlMode(); ob.timestamp = now_us; ob.position = True
        self.offboard_pub.publish(ob)
        sp = TrajectorySetpoint(); sp.timestamp = now_us
        sp.position = [float(n), float(e), float(d)]
        sp.yaw = float('nan')
        self.traj_pub.publish(sp)

    def _publish_velocity(self, vn, ve, vd, hold_z=None):
        now_us = int(self.get_clock().now().nanoseconds / 1000)
        ob = OffboardControlMode(); ob.timestamp = now_us
        ob.velocity = True
        ob.position = hold_z is not None
        self.offboard_pub.publish(ob)
        nan = float('nan')
        sp = TrajectorySetpoint(); sp.timestamp = now_us
        sp.position = [nan, nan, float(hold_z) if hold_z is not None else nan]
        sp.velocity = [float(vn), float(ve),
                       nan if hold_z is not None else float(vd)]
        sp.yaw = nan
        self.traj_pub.publish(sp)

    @staticmethod
    def _clamp(v, lim):
        return max(-lim, min(lim, v))

    def _track(self, vd, hold_z=None):
        """Stage-1 PD + pad-velocity feedforward -> horizontal velocity setpoint.

        v_sp = Kp*(pred_pad - drone) + Kd*(pad_vel - drone_vel) + pad_vel
        Returns the current horizontal tracking error (m).
        """
        lp = self.local_pos
        pred = self.est.predict_pos(LOOKAHEAD)
        if pred is None:
            self._publish_velocity(0.0, 0.0, vd, hold_z)
            return 1e9
        pn, pe = pred
        _, _, vpn, vpe = self.est.state()
        en, ee = pn - lp.x, pe - lp.y
        vn = KP_XY * en + KD_XY * (vpn - lp.vx) + vpn
        ve = KP_XY * ee + KD_XY * (vpe - lp.vy) + vpe
        self._publish_velocity(self._clamp(vn, VEL_MAX_XY),
                               self._clamp(ve, VEL_MAX_XY), vd, hold_z)
        return math.hypot(en, ee)

    def _horizontal_err(self):
        lp = self.local_pos
        pn, pe, _, _ = self.est.state()
        return math.hypot(lp.x - pn, lp.y - pe)

    def _search_to(self, gn, ge):
        """Speed-limited velocity toward a search waypoint, altitude held."""
        lp = self.local_pos
        dn, de = gn - lp.x, ge - lp.y
        dist = math.hypot(dn, de)
        if dist < 1e-3:
            vn = ve = 0.0
        else:
            spd = min(SEARCH_SPEED, dist)
            vn, ve = spd * dn / dist, spd * de / dist
        self._publish_velocity(vn, ve, 0.0, hold_z=self.target_z)

    # ---------------------------------------------------------------- main loop
    def _control_loop(self):
        if self.local_pos is None:
            return
        lp = self.local_pos
        now = time.monotonic()
        self.est.predict(0.02)
        seen_recently = (now - self.last_seen) < LOST_TIMEOUT
        if not seen_recently:
            self.stable_count = 0

        if self.state == self.INIT:
            if lp.xy_valid and lp.z_valid:
                self.origin_z = lp.z
                self.takeoff_xy = (lp.x, lp.y)
                self.target_z = lp.z - SEARCH_ALT
                self.state = self.TAKEOFF
                self.get_logger().info('Arming + takeoff...')
                self._arm_and_offboard()
            else:
                self._publish_position(lp.x, lp.y, lp.z)
            return

        if self.state == self.TAKEOFF and self.tick < 50 and self.tick % 10 == 0:
            self._arm_and_offboard()
        self.tick += 1

        if self.paused:
            self._publish_position(lp.x, lp.y, self.target_z)
            return

        agl = (lp.z - self.origin_z) * -1.0

        # ----------------------------------------------------------- TAKEOFF
        if self.state == self.TAKEOFF:
            tn, te = self.takeoff_xy
            self._publish_position(tn, te, self.target_z)
            if abs(agl - SEARCH_ALT) < 0.4:
                self.state = self.SEARCH
                self.get_logger().info('At altitude -> SEARCH')

        # ----------------------------------------------------------- SEARCH
        elif self.state == self.SEARCH:
            tn, te = self.takeoff_xy
            wn, we = SEARCH_WPS[self.wp_idx]
            self._search_to(tn + wn, te + we)
            if math.hypot(lp.x - (tn + wn), lp.y - (te + we)) < 0.8:
                self.wp_idx = (self.wp_idx + 1) % len(SEARCH_WPS)
            if seen_recently and self.est.ready:
                self.state = self.ACQUIRE
                self.get_logger().info('Target seen -> ACQUIRE (stabilizing KF)')

        # ----------------------------------------------------------- ACQUIRE
        elif self.state == self.ACQUIRE:
            self._track(0.0, hold_z=self.target_z)
            if not seen_recently:
                self.state = self.SEARCH
                self.get_logger().warn('Lost during ACQUIRE -> SEARCH')
            elif self.stable_count >= N_STABLE and self.est.confident():
                self.state = self.APPROACH
                self.get_logger().info('Estimate trusted -> APPROACH')

        # ----------------------------------------------------------- APPROACH
        elif self.state == self.APPROACH:
            err = self._track(0.0, hold_z=self.target_z)
            if not seen_recently:
                self.state = self.ACQUIRE
                self.get_logger().warn('Lost -> step back to ACQUIRE')
            elif err < ALIGN_TOL:
                self.state = self.DESCEND
                self.get_logger().info('Aligned -> DESCEND')

        # ----------------------------------------------------------- DESCEND
        elif self.state == self.DESCEND:
            ok = (seen_recently and self.est.confident()
                  and self._horizontal_err() < ALIGN_TOL * 2.0)
            vd = DESCENT_SPEED if ok else -0.3        # climb back if unsure
            self._track(vd)
            if not seen_recently:
                self.state = self.APPROACH
                self.target_z = lp.z - SEARCH_ALT
                self.get_logger().warn('Lost in DESCEND -> APPROACH')
            elif agl < FINAL_ALT:
                self.state = self.FINAL
                self.get_logger().info('Low -> FINAL (await sway turnaround)')

        # ----------------------------------------------------------- FINAL
        elif self.state == self.FINAL:
            height_above_marker = agl - PLATFORM_TOP_Z
            if height_above_marker <= TOUCHDOWN_HEIGHT:
                self.send_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                                  param1=0.0, param2=21196.0)   # force disarm
                self.state = self.LANDED
                self.get_logger().info('TOUCHDOWN -> motors off. LANDED.')
                return
            near_turnaround = self.est.speed() < V_TURNAROUND
            aligned = self._horizontal_err() < ALIGN_TOL
            if seen_recently and near_turnaround and aligned:
                self._track(FINAL_DESCENT)            # commit the drop
            elif seen_recently:
                self._track(0.0)                      # hold altitude, stay centered
            else:
                self._publish_position(lp.x, lp.y, lp.z)
                self.state = self.APPROACH
                self.target_z = lp.z - SEARCH_ALT
                self.get_logger().warn('Lost in FINAL -> APPROACH')

        elif self.state == self.LANDED:
            return

    # ---------------------------------------------------------------- keyboard
    def handle_key(self, key):
        k = key.lower()
        if k == 'p':
            self.paused = True; self.get_logger().info('PAUSED')
        elif k == 'r':
            self.paused = False; self.get_logger().info('RESUMED')
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
    node = PadLandingController()
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
