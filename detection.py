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

# ----------------------------------------------------------------------------
# Tunables
# ----------------------------------------------------------------------------
# Try several dictionaries so detection works regardless of which family the
# world's markers use. The first one that matches wins per detection.
ARUCO_DICT_NAMES = [
    'DICT_4X4_50', 'DICT_5X5_250', 'DICT_5X5_1000',
    'DICT_6X6_250', 'DICT_4X4_250', 'DICT_ARUCO_ORIGINAL',
]
ARUCO_PARAMS = aruco.DetectorParameters()
# Loosen thresholds a bit -- helps with sim lighting / motion blur.
ARUCO_PARAMS.adaptiveThreshWinSizeMin = 3
ARUCO_PARAMS.adaptiveThreshWinSizeMax = 23
ARUCO_PARAMS.adaptiveThreshWinSizeStep = 10
ARUCO_PARAMS.minMarkerPerimeterRate = 0.02   # detect smaller / distant markers
COOLDOWN_SEC = 2.0

# Camera ROS2 topics (published by bridge.py). Adjust if your bridge uses
# different sensor names -- check with: ros2 topic list | grep camera
FRONT_CAM_TOPIC = '/drone/camera/front_camera'
DOWN_CAM_TOPIC  = '/drone/camera/down_camera'

TAKEOFF_ALT = 3.0     # meters above arming point
STEP_XY     = 0.5     # meters moved per W/A/S/D press
STEP_Z      = 0.5     # meters moved per U/J press
STEP_YAW    = math.radians(15)  # radians per Q/E press

HELP = """
==================== KEYBOARD CONTROL ====================
  W / S : forward / backward (relative to heading)
  A / D : strafe left / right
  U / J : up / down
  Q / E : yaw left / right
  L     : land
  T     : re-arm + takeoff
  X     : quit
=========================================================
"""


class DroneController(Node):
    def __init__(self):
        super().__init__('drone_controller')

        # --- QoS profiles -----------------------------------------------------
        # PX4 uXRCE-DDS uses BEST_EFFORT.
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        cam_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # --- PX4 publishers ---------------------------------------------------
        self.offboard_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', px4_qos)
        self.traj_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', px4_qos)
        self.cmd_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', px4_qos)

        # --- PX4 subscribers --------------------------------------------------
        self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1',
            self._pos_cb, px4_qos)
        self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status_v4',
            self._status_cb, px4_qos)

        # --- ArUco detection --------------------------------------------------
        self.cvbridge = CvBridge()
        # One detector per dictionary we want to try.
        self.detectors = []
        for name in ARUCO_DICT_NAMES:
            d = aruco.getPredefinedDictionary(getattr(aruco, name))
            self.detectors.append((name, aruco.ArucoDetector(d, ARUCO_PARAMS)))
        self.last_seen = {}   # (cam_label, dict, marker_id) -> last print monotonic time

        self.create_subscription(
            Image, FRONT_CAM_TOPIC, lambda m: self._img_cb(m, 'Front'), cam_qos)
        self.create_subscription(
            Image, DOWN_CAM_TOPIC, lambda m: self._img_cb(m, 'Down'), cam_qos)

        # --- State ------------------------------------------------------------
        self.local_pos = None         # latest VehicleLocalPosition
        self.nav_state = None
        self.arming_state = None

        # Target setpoint in NED (x=North, y=East, z=Down). Updated by keyboard.
        self.target = [0.0, 0.0, 0.0]
        self.target_yaw = 0.0
        self.have_origin = False      # captured arming-point position yet?
        self.tick = 0
        self.running = True
        self.lock = threading.Lock()

        # 50 Hz offboard heartbeat + setpoint stream
        self.create_timer(0.02, self._control_loop)

        self.get_logger().info('Drone controller up. Waiting for local position...')
        print(HELP)

    # ------------------------------------------------------------------ PX4 cb
    def _pos_cb(self, msg):
        self.local_pos = msg
        if not self.have_origin and msg.xy_valid and msg.z_valid:
            # Lock target to current spot, then command takeoff altitude.
            with self.lock:
                self.target = [msg.x, msg.y, msg.z - TAKEOFF_ALT]
                self.target_yaw = msg.heading
            self.have_origin = True
            self.get_logger().info(
                f'Origin captured at N={msg.x:.1f} E={msg.y:.1f} D={msg.z:.1f}. '
                f'Auto takeoff to {TAKEOFF_ALT} m.')
            self._start_takeoff()

    def _status_cb(self, msg):
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state

    # -------------------------------------------------------------- takeoff/arm
    def _start_takeoff(self):
        # Engage offboard, then arm. Setpoints are already streaming.
        self.send_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
                          param1=1.0, param2=6.0)   # PX4 offboard main mode
        self.send_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                          param1=1.0)               # arm

    def _land(self):
        self.send_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info('Landing...')

    def send_command(self, command, **params):
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.command = command
        msg.param1 = params.get('param1', 0.0)
        msg.param2 = params.get('param2', 0.0)
        msg.param3 = params.get('param3', 0.0)
        msg.param4 = params.get('param4', 0.0)
        msg.param5 = params.get('param5', 0.0)
        msg.param6 = params.get('param6', 0.0)
        msg.param7 = params.get('param7', 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.cmd_pub.publish(msg)

    # ----------------------------------------------------------- control stream
    def _control_loop(self):
        now_us = int(self.get_clock().now().nanoseconds / 1000)

        # Offboard heartbeat (must be >2 Hz for PX4 to stay in offboard).
        ob = OffboardControlMode()
        ob.timestamp = now_us
        ob.position = True
        ob.velocity = False
        ob.acceleration = False
        ob.attitude = False
        ob.body_rate = False
        self.offboard_pub.publish(ob)

        # Position setpoint.
        with self.lock:
            tx, ty, tz = self.target
            tyaw = self.target_yaw
        sp = TrajectorySetpoint()
        sp.timestamp = now_us
        sp.position = [float(tx), float(ty), float(tz)]
        sp.yaw = float(tyaw)
        self.traj_pub.publish(sp)

        # Re-assert offboard+arm for the first second in case the FMU missed it.
        if self.have_origin and self.tick < 50 and self.tick % 10 == 0:
            self._start_takeoff()
        self.tick += 1

    # ------------------------------------------------------------- keyboard nudge
    def handle_key(self, key):
        key = key.lower()
        with self.lock:
            yaw = self.target_yaw
            # Forward/right unit vectors in NED for the current heading.
            fwd = (math.cos(yaw), math.sin(yaw))
            left = (-math.sin(yaw), math.cos(yaw))

            if key == 'w':
                self.target[0] += STEP_XY * fwd[0]
                self.target[1] += STEP_XY * fwd[1]
            elif key == 's':
                self.target[0] -= STEP_XY * fwd[0]
                self.target[1] -= STEP_XY * fwd[1]
            elif key == 'a':
                self.target[0] += STEP_XY * left[0]
                self.target[1] += STEP_XY * left[1]
            elif key == 'd':
                self.target[0] -= STEP_XY * left[0]
                self.target[1] -= STEP_XY * left[1]
            elif key == 'u':
                self.target[2] -= STEP_Z          # NED: up is negative
            elif key == 'j':
                self.target[2] += STEP_Z
            elif key == 'q':
                self.target_yaw += STEP_YAW
            elif key == 'e':
                self.target_yaw -= STEP_YAW
            elif key == 'l':
                self._land()
            elif key == 't':
                self.have_origin = False          # recapture origin -> retakeoff
                self.tick = 0
            elif key == 'x':
                self.running = False

    # ----------------------------------------------------------- aruco detection
    def _img_cb(self, msg, cam_label):
        try:
            frame = self.cvbridge.imgmsg_to_cv2(msg, 'bgr8')
            frame = cv2.resize(frame, (640, 480))
        except CvBridgeError as e:
            self.get_logger().error(str(e))
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        now_mono = time.monotonic()

        # Try every dictionary; report the first that yields markers.
        for dict_name, det in self.detectors:
            corners, ids, _ = det.detectMarkers(gray)
            if ids is None:
                continue
            aruco.drawDetectedMarkers(frame, corners)
            for i, mid in enumerate(ids.flatten()):
                mid = int(mid)
                cx, cy = corners[i][0].mean(axis=0).astype(int)
                cv2.putText(frame, str(mid), (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                seen_key = (cam_label, dict_name, mid)
                if now_mono - self.last_seen.get(seen_key, 0) >= COOLDOWN_SEC:
                    print(f'[{cam_label} cam] ArUco marker detected '
                          f'-> ID: {mid}  ({dict_name})')
                    self.last_seen[seen_key] = now_mono
            break   # matched this dictionary; don't double-count others

        cv2.putText(frame, f'[{cam_label}] cam', (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        cv2.imshow(f'{cam_label} Camera', frame)
        cv2.waitKey(1)


# -------------------------------------------------------------- keyboard thread
def keyboard_thread(node):
    """Read single keypresses from the terminal in raw mode."""
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
    node = DroneController()

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
