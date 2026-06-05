#!/usr/bin/env python3
"""
drone_aruco_tracker.py
═══════════════════════════════════════════════════════════════════
UAV PX4 Visual Servoing: Tự động bay và căn chỉnh theo ArUco marker
sử dụng Offboard mode qua ROS2 + uXRCE-DDS.

State Machine:
  INIT → ARMING → TAKEOFF → APPROACH → VISUAL_SERVO → HOVER → RTL

Yêu cầu:
  1. PX4 SITL + Gazebo:     ~/px4_sim/launch_sim.sh x500_depth
  2. XRCE-DDS Agent:        MicroXRCEAgent udp4 -p 8888
  3. Chạy node này:         ros2 run visual_servoing drone_aruco_tracker

Frame conventions:
  - NED (North-East-Down): x=bắc, y=đông, z=xuống
  - Altitude: sp_z âm = bay lên (ví dụ -2.4m = 2.4m trên mặt đất)
═══════════════════════════════════════════════════════════════════
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np
import cv2
import cv2.aruco as aruco
from typing import Optional, Tuple

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
)
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


# ══════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════

class Config:
    # Camera OakD-Lite IMX214 (1920×1080, hFOV=69°=1.204rad)
    IMG_W   = 1920
    IMG_H   = 1080
    CAM_FX  = 1458.0
    CAM_FY  = 1458.0
    CAM_CX  = 960.0
    CAM_CY  = 540.0
    DIST    = np.zeros((4, 1), dtype=np.float32)

    # ArUco
    ARUCO_DICT   = aruco.DICT_4X4_1000
    MARKER_SIZE  = 0.10          # 10cm

    # Flight
    CRUISE_ALT   = 2.4           # m (drone bay ở độ cao này)
    APPROACH_X   = 3.0           # m dừng approach khi x > 3m
    TARGET_DIST  = 2.0           # m khoảng cách đến board khi hover
    HOVER_SEC    = 5.0           # giây hover sau khi lock

    # Visual servo gains
    K_LATERAL    = 2.0           # PID lateral (y NED)
    K_VERTICAL   = 1.5           # PID vertical (z NED)
    K_FORWARD    = 0.4           # PID forward (x NED)

    # Tolerances
    ALT_TOL_M    = 0.20          # m
    LOCK_ERR_PX  = 25            # pixel
    LOCK_DIST_M  = 0.25          # m

    # Safety
    SP_Z_MIN     = -6.0          # m NED (max altitude)
    SP_Z_MAX     = -0.3          # m NED (min safe altitude)


# ══════════════════════════════════════════════════════════════════
#  MAIN NODE
# ══════════════════════════════════════════════════════════════════

class DroneArucoTracker(Node):

    def __init__(self):
        super().__init__('drone_aruco_tracker')
        cfg = Config

        # QoS
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST, depth=1
        )

        # ── Publishers ──────────────────────────────────────────
        self.offboard_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', px4_qos)
        self.setpoint_pub  = self.create_publisher(
            TrajectorySetpoint,  '/fmu/in/trajectory_setpoint',  px4_qos)
        self.cmd_pub       = self.create_publisher(
            VehicleCommand,      '/fmu/in/vehicle_command',       px4_qos)

        # ── Subscribers ─────────────────────────────────────────
        self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position',
            self._on_position, px4_qos)
        self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status',
            self._on_status, px4_qos)
        self.create_subscription(
            Image, '/camera', self._on_camera, 10)

        # ── Camera setup ────────────────────────────────────────
        self.bridge = CvBridge()
        self.cam_matrix = np.array([
            [cfg.CAM_FX, 0,        cfg.CAM_CX],
            [0,        cfg.CAM_FY, cfg.CAM_CY],
            [0,        0,          1.0        ]
        ], dtype=np.float32)
        aruco_dict = aruco.getPredefinedDictionary(cfg.ARUCO_DICT)
        params = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(aruco_dict, params)

        # ── State ────────────────────────────────────────────────
        self.state           = 'INIT'
        self.pos             = np.zeros(3)      # NED [x, y, z]
        self.armed           = False
        self.offboard_active = False
        self.offboard_count  = 0

        # Setpoint (NED)
        self.sp = np.array([0.0, 0.0, -cfg.CRUISE_ALT])
        self.sp_yaw = 0.0

        # Vision
        self.marker_center: Optional[Tuple[int,int]] = None
        self.marker_dist:   Optional[float]          = None
        self.marker_detected = False

        # Hover timer
        self._hover_start = None

        # Control loop 20Hz
        self.timer = self.create_timer(0.05, self._loop)
        self.get_logger().info('🚁 DroneArucoTracker ready — state: INIT')

    # ══════════════════════════════════════════════════════════════
    #  CALLBACKS
    # ══════════════════════════════════════════════════════════════

    def _on_position(self, msg: VehicleLocalPosition):
        self.pos = np.array([msg.x, msg.y, msg.z])

    def _on_status(self, msg: VehicleStatus):
        self.armed           = (msg.arming_state == 2)   # ARMED
        self.offboard_active = (msg.nav_state    == 14)  # OFFBOARD

    def _on_camera(self, msg: Image):
        """Detect ArUco, update marker_center và marker_dist"""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = self.detector.detectMarkers(gray)

            if ids is not None and len(ids) > 0:
                # Dùng marker đầu tiên phát hiện được
                c  = corners[0][0]
                cx = int(c[:, 0].mean())
                cy = int(c[:, 1].mean())
                self.marker_center   = (cx, cy)
                self.marker_detected = True

                # PnP ước lượng khoảng cách
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                    corners[:1], Config.MARKER_SIZE,
                    self.cam_matrix, Config.DIST
                )
                self.marker_dist = float(np.linalg.norm(tvec[0]))
            else:
                self.marker_detected = False
                self.marker_center   = None
                self.marker_dist     = None

        except Exception as e:
            self.get_logger().warn(f'[Camera] {e}')

    # ══════════════════════════════════════════════════════════════
    #  CONTROL LOOP
    # ══════════════════════════════════════════════════════════════

    def _loop(self):
        # Luôn publish offboard heartbeat (≥2Hz, chúng ta 20Hz)
        self._pub_offboard_mode()

        # Dispatch state machine
        handler = getattr(self, f'_state_{self.state.lower()}', None)
        if handler:
            handler()
        else:
            self.get_logger().error(f'Unknown state: {self.state}')

        # Publish setpoint
        self._pub_setpoint()

    # ══════════════════════════════════════════════════════════════
    #  STATE HANDLERS
    # ══════════════════════════════════════════════════════════════

    def _state_init(self):
        """Warm-up stream 2s trước khi switch Offboard"""
        self.offboard_count += 1
        if self.offboard_count > 40:
            self._cmd_set_offboard()
            self._cmd_arm()
            self._transition('ARMING')

    def _state_arming(self):
        """Chờ drone arm và vào Offboard mode"""
        if self.armed and self.offboard_active:
            self.sp = np.array([0.0, 0.0, -Config.CRUISE_ALT])
            self._transition('TAKEOFF')

    def _state_takeoff(self):
        """Bay lên cruise altitude"""
        alt_error = abs(self.pos[2] - (-Config.CRUISE_ALT))
        if alt_error < Config.ALT_TOL_M:
            # Đạt altitude → bay về hướng board
            self.sp[0] = Config.APPROACH_X
            self._transition('APPROACH')

    def _state_approach(self):
        """Bay thẳng đến vị trí nhìn thấy board"""
        dist_to_approach = abs(self.pos[0] - Config.APPROACH_X)
        if dist_to_approach < 0.5 and self.marker_detected:
            self._transition('VISUAL_SERVO')
        elif dist_to_approach < 0.5 and not self.marker_detected:
            self.get_logger().warn('Reached approach zone but no marker — searching...')

    def _state_visual_servo(self):
        """PID visual feedback loop căn chỉnh drone theo marker"""
        if not self.marker_detected:
            self.get_logger().warn('[VS] Marker lost — holding position')
            return

        cx, cy = self.marker_center
        cfg = Config

        # Pixel errors (normalized [-0.5, +0.5])
        err_horiz = (cx - cfg.IMG_W / 2) / cfg.IMG_W   # + = marker bên phải
        err_vert  = (cy - cfg.IMG_H / 2) / cfg.IMG_H   # + = marker ở dưới

        # Điều chỉnh setpoint:
        #   err_horiz > 0 → marker bên phải camera → drone phải sang phải (+y NED)
        #   err_vert  > 0 → marker bên dưới camera → drone cần bay xuống thấp hơn (+z NED)
        self.sp[1] += cfg.K_LATERAL * err_horiz
        self.sp[2] += cfg.K_VERTICAL * err_vert
        self.sp[2] = np.clip(self.sp[2], cfg.SP_Z_MIN, cfg.SP_Z_MAX)

        # Forward/Backward theo khoảng cách
        if self.marker_dist is not None:
            dist_err = self.marker_dist - cfg.TARGET_DIST
            self.sp[0] += cfg.K_FORWARD * dist_err
            self.sp[0] = max(0.5, self.sp[0])  # Không bay lùi về gốc

        # Kiểm tra lock
        pixel_err = np.hypot(cx - cfg.IMG_W/2, cy - cfg.IMG_H/2)
        dist_locked = (
            self.marker_dist is not None and
            abs(self.marker_dist - cfg.TARGET_DIST) < cfg.LOCK_DIST_M
        )
        if pixel_err < cfg.LOCK_ERR_PX and dist_locked:
            self.get_logger().info(
                f'[VS] LOCKED! err={pixel_err:.0f}px, dist={self.marker_dist:.2f}m')
            self._transition('HOVER')

    def _state_hover(self):
        """Giữ vị trí đã lock trong HOVER_SEC giây"""
        if self._hover_start is None:
            self._hover_start = self.get_clock().now()
            self.get_logger().info(
                f'[HOVER] Holding for {Config.HOVER_SEC}s at '
                f'x={self.sp[0]:.2f} y={self.sp[1]:.2f} z={self.sp[2]:.2f}')

        elapsed = (self.get_clock().now() - self._hover_start).nanoseconds / 1e9
        if elapsed >= Config.HOVER_SEC:
            self._hover_start = None
            self._transition('RTL')

    def _state_rtl(self):
        """Return To Launch"""
        self.get_logger().info('[RTL] Sending Return To Launch command')
        self._cmd(VehicleCommand.VEHICLE_CMD_NAV_RETURN_TO_LAUNCH)

    # ══════════════════════════════════════════════════════════════
    #  PUBLISHERS & HELPERS
    # ══════════════════════════════════════════════════════════════

    def _pub_offboard_mode(self):
        msg = OffboardControlMode()
        msg.position     = True
        msg.velocity     = False
        msg.acceleration = False
        msg.timestamp    = self._ts()
        self.offboard_pub.publish(msg)

    def _pub_setpoint(self):
        msg = TrajectorySetpoint()
        msg.position  = [float(v) for v in self.sp]
        msg.yaw       = self.sp_yaw
        msg.timestamp = self._ts()
        self.setpoint_pub.publish(msg)

    def _cmd_set_offboard(self):
        self._cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, p1=1.0, p2=6.0)

    def _cmd_arm(self):
        self._cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, p1=1.0)

    def _cmd(self, command: int, p1: float = 0.0, p2: float = 0.0):
        msg = VehicleCommand()
        msg.command          = command
        msg.param1           = p1
        msg.param2           = p2
        msg.target_system    = 1
        msg.target_component = 1
        msg.source_system    = 1
        msg.source_component = 1
        msg.from_external    = True
        msg.timestamp        = self._ts()
        self.cmd_pub.publish(msg)

    def _transition(self, new_state: str):
        self.get_logger().info(f'State: {self.state} → {new_state}')
        self.state = new_state

    def _ts(self) -> int:
        return int(self.get_clock().now().nanoseconds / 1000)


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = DroneArucoTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
