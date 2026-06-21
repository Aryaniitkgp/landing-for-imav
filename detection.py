"""
Simple ArUco seek-and-hover (PX4 SITL + Gazebo).

Behaviour:  TAKEOFF -> SEARCH for ArUco ID 1 -> fly over it and HOVER.
No EKF, no descent, no landing. Every other marker is ignored.
"""

import math
import sys
import termios
import time
import tty
import threading

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
# Configuration
# ============================================================================
FRONT_CAM_TOPIC = '/drone/camera/front_camera'
DOWN_CAM_TOPIC  = '/drone/camera/down_camera'
IMG_W, IMG_H    = 640, 480
DOWN_CAM_HFOV   = 1.74                       # rad, from model.sdf

FX = 0.5 * IMG_W / math.tan(0.5 * DOWN_CAM_HFOV)
FY = FX
CX, CY = IMG_W / 2.0, IMG_H / 2.0

ARUCO_DICT     = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
TARGET_ID      = 1
PLATFORM_TOP_Z = 0.28                        # marker height above ground (m)

# Camera-axis -> body-axis signs. Flip if the drone moves the WRONG way.
FWD_SIGN   = -1.0
RIGHT_SIGN = +1.0

HOVER_ALT    = 5.0           # takeoff / search / hover altitude (m AGL)
SEARCH_SPEED = 4.5           # max horizontal speed while searching (m/s)
LOST_TIMEOUT = 3.0           # s without detection before resuming SEARCH

# --- Descent ---
STABILIZE_TIME = 4.0         # s to stabilize over the pad before descending
DESCENT_SPEED  = 0.1         # m/s average descent rate
TOUCHDOWN_HEIGHT = 0.25      # cut motors at this height above the marker (m)
# Pad next-position prediction = 75% current frame + 25% previous frame.
PRED_CURR_W = 0.80
PRED_PREV_W = 0.20

# Search lawnmower (NED, relative to takeoff origin)
SEARCH_WPS = [
    (0.0, 0.0), (8.0, 0.0), (8.0, 3.0), (0.0, 3.0),
    (0.0, -3.0), (8.0, -3.0), (8.0, 6.0), (0.0, 6.0),
]
WP_REACH_TOL = 0.8

HELP = """
================ ARUCO SEEK-AND-HOVER ================
  TAKEOFF -> SEARCH for ID {tid} -> HOVER over it.
  Keys:  P pause/hover   R resume   X quit
=====================================================
""".format(tid=TARGET_ID)


class HoverController(Node):
    INIT, TAKEOFF, SEARCH, HOVER, DESCEND, LANDED = range(6)

    def __init__(self):
        super().__init__('hover_controller')

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
        self.state = self.INIT
        self.origin_z = None
        self.takeoff_xy = (0.0, 0.0)
        self.target_z = 0.0
        self.wp_idx = 0
        self.marker_world = None             # (N, E) latest detection
        self.marker_prev = None              # (N, E) previous detection
        self.hover_enter = None              # time HOVER began (for stabilize)
        self.last_seen = 0.0
        self.last_det_id = None
        self.paused = False
        self.running = True
        self.tick = 0
        self.lock = threading.Lock()

        self.create_timer(0.02, self._control_loop)
        print(HELP)

    # ---------------------------------------------------------------- callbacks
    def _pos_cb(self, msg):
        self.local_pos = msg

    def _status_cb(self, msg):
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state

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
                if cam_label == 'Down':
                    self._ingest_target(cx, cy)

        cv2.putText(frame, f'[{cam_label}] {self._state_name()}',
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        cv2.imshow(f'{cam_label} Camera', frame)
        cv2.waitKey(1)

    def _ingest_target(self, u, v):
        """Down-cam marker pixel -> world-NED (N, E) of the marker."""
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
        with self.lock:
            self.marker_prev = self.marker_world      # shift current -> previous
            self.marker_world = (north, east)
            self.last_seen = time.monotonic()

    # ---------------------------------------------------------------- helpers
    def _state_name(self):
        return ['INIT', 'TAKEOFF', 'SEARCH', 'HOVER', 'DESCEND', 'LANDED'][self.state]

    def _predicted_marker(self):
        """Pad next position = 75% current frame + 25% previous frame."""
        if self.marker_world is None:
            return None
        if self.marker_prev is None:
            return self.marker_world
        cn, ce = self.marker_world
        pn, pe = self.marker_prev
        return (PRED_CURR_W * cn + PRED_PREV_W * pn,
                PRED_CURR_W * ce + PRED_PREV_W * pe)

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

    def _publish_setpoint(self, n, e, d):
        now_us = int(self.get_clock().now().nanoseconds / 1000)
        ob = OffboardControlMode()
        ob.timestamp = now_us
        ob.position = True
        self.offboard_pub.publish(ob)

        sp = TrajectorySetpoint()
        sp.timestamp = now_us
        sp.position = [float(n), float(e), float(d)]
        sp.yaw = float('nan')
        self.traj_pub.publish(sp)

    def _publish_velocity_xy(self, gn, ge, d):
        """Fly toward (gn, ge) at <= SEARCH_SPEED while holding altitude d.

        Horizontal axes use velocity control (speed-limited), altitude uses
        position control. Decelerates within the last SEARCH_SPEED metres.
        """
        lp = self.local_pos
        dn, de = gn - lp.x, ge - lp.y
        dist = math.hypot(dn, de)
        if dist < 1e-3:
            vn = ve = 0.0
        else:
            speed = min(SEARCH_SPEED, dist)      # ease off near the goal
            vn, ve = speed * dn / dist, speed * de / dist

        now_us = int(self.get_clock().now().nanoseconds / 1000)
        ob = OffboardControlMode()
        ob.timestamp = now_us
        ob.position = True
        ob.velocity = True
        self.offboard_pub.publish(ob)

        nan = float('nan')
        sp = TrajectorySetpoint()
        sp.timestamp = now_us
        sp.position = [nan, nan, float(d)]       # hold altitude only
        sp.velocity = [float(vn), float(ve), nan]
        sp.yaw = nan
        self.traj_pub.publish(sp)

    # ---------------------------------------------------------------- main loop
    def _control_loop(self):
        if self.local_pos is None:
            return
        lp = self.local_pos
        now = time.monotonic()

        if self.state == self.INIT:
            if lp.xy_valid and lp.z_valid:
                self.origin_z = lp.z
                self.takeoff_xy = (lp.x, lp.y)
                self.target_z = lp.z - HOVER_ALT
                self.state = self.TAKEOFF
                self.get_logger().info('Arming + takeoff...')
                self._arm_and_offboard()
            else:
                self._publish_setpoint(lp.x, lp.y, lp.z)
            return

        if self.state == self.TAKEOFF and self.tick < 50 and self.tick % 10 == 0:
            self._arm_and_offboard()
        self.tick += 1

        with self.lock:
            marker = self.marker_world
            seen_recently = (now - self.last_seen) < LOST_TIMEOUT

        if self.paused:
            self._publish_setpoint(lp.x, lp.y, self.target_z)
            return

        # First acquisition (from TAKEOFF/SEARCH) -> hover over the pad.
        if (marker is not None and seen_recently
                and self.state in (self.TAKEOFF, self.SEARCH)):
            self.state = self.HOVER
            self.hover_enter = now
            self.get_logger().info('Target found -> HOVER over it')

        if self.state == self.TAKEOFF:
            tn, te = self.takeoff_xy
            self._publish_setpoint(tn, te, self.target_z)
            if abs((lp.z - self.origin_z) + HOVER_ALT) < 0.4:
                self.state = self.SEARCH
                self.get_logger().info('At altitude -> SEARCH')

        elif self.state == self.SEARCH:
            tn, te = self.takeoff_xy
            wn, we = SEARCH_WPS[self.wp_idx]
            gn, ge = tn + wn, te + we
            self._publish_velocity_xy(gn, ge, self.target_z)   # speed-limited
            if math.hypot(lp.x - gn, lp.y - ge) < WP_REACH_TOL:
                self.wp_idx = (self.wp_idx + 1) % len(SEARCH_WPS)

        elif self.state == self.HOVER:
            pred = self._predicted_marker()
            if pred is not None and seen_recently:
                self._publish_setpoint(pred[0], pred[1], self.target_z)
                # Hold over the pad for STABILIZE_TIME, then start descending.
                if now - self.hover_enter > STABILIZE_TIME:
                    self.state = self.DESCEND
                    self.target_z = lp.z       # begin descent from current alt
                    self.get_logger().info('Stabilized -> DESCEND @ 0.1 m/s')
            else:
                self._publish_setpoint(lp.x, lp.y, self.target_z)
                if not seen_recently:
                    self.state = self.SEARCH
                    self.get_logger().warn('Target lost -> SEARCH')

        elif self.state == self.DESCEND:
            pred = self._predicted_marker()
            agl = (lp.z - self.origin_z) * -1.0
            height_above_marker = agl - PLATFORM_TOP_Z
            if height_above_marker <= TOUCHDOWN_HEIGHT:
                # ~25 cm above the marker: cut motors and drop onto the pad.
                self.send_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                                  param1=0.0, param2=21196.0)   # force disarm
                self.state = self.LANDED
                self.get_logger().info(
                    f'{TOUCHDOWN_HEIGHT*100:.0f} cm above marker -> MOTORS OFF, landed')
            elif pred is not None and seen_recently:
                # Step the altitude target down at the average descent speed.
                self.target_z += DESCENT_SPEED * 0.02          # 50 Hz tick
                self._publish_setpoint(pred[0], pred[1], self.target_z)
            else:
                # Lost the pad mid-descent: stop sinking, hold, then re-search.
                self._publish_setpoint(lp.x, lp.y, lp.z)
                if not seen_recently:
                    self.state = self.SEARCH
                    self.target_z = lp.z - HOVER_ALT
                    self.get_logger().warn('Lost during descent -> SEARCH')

        elif self.state == self.LANDED:
            return   # motors off, mission complete

    # ---------------------------------------------------------------- keyboard
    def handle_key(self, key):
        k = key.lower()
        if k == 'p':
            self.paused = True
            self.get_logger().info('PAUSED (hover)')
        elif k == 'r':
            self.paused = False
            self.get_logger().info('RESUMED')
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
    node = HoverController()
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
