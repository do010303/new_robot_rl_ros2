#!/usr/bin/env python3
"""
Gazebo State Mirror Node
========================
This node listens to the actual joint states published by the real Raspberry Pi
(`/pca9685_servo/joint_states`) and forwards them to Gazebo's ros2_control
JointTrajectoryController (`/arm_controller/joint_trajectory`), forcing the
simulated robot to perfectly mirror the real one.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class GazeboStateMirror(Node):
    def __init__(self):
        super().__init__('gazebo_state_mirror')

        # The names of the joints expected by our arm_controller (from controllers.yaml)
        # Order doesn't strictly matter as long as names match what's in the URDF.
        self.target_joint_names = [
            "Revolute 20", "Revolute 22", "Revolute 23", "Revolute 26", "Revolute 28", "Revolute 30"
        ]

        # The names of the joints published by the Raspberry Pi (from servos.yaml)
        self.pi_joint_names = [
            "base", "shoulder", "elbow", "wrist_roll", "wrist_pitch", "pen"
        ]

        # Mapping from Pi name to Gazebo/URDF name
        self.name_mapping = dict(zip(self.pi_joint_names, self.target_joint_names))

        # Publisher to Gazebo's joint trajectory controller
        self.traj_pub = self.create_publisher(
            JointTrajectory, 
            "/arm_controller/joint_trajectory", 
            10
        )

        # Subscriber to Raspberry Pi's joint states
        self.js_sub = self.create_subscription(
            JointState,
            "/pca9685_servo/joint_states",
            self.joint_states_callback,
            10
        )

        self.get_logger().info("Gazebo State Mirror started!")
        self.get_logger().info("Listening to Pi on: /pca9685_servo/joint_states")
        self.get_logger().info("Publishing to Gazebo on: /arm_controller/joint_trajectory")

    def joint_states_callback(self, msg: JointState):
        """
        Received real joint angles from the Pi.
        Convert and publish to Gazebo.
        """
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        
        point = JointTrajectoryPoint()
        
        # We only want to map the joints we know about
        for pi_name, position in zip(msg.name, msg.position):
            if pi_name in self.name_mapping:
                gazebo_name = self.name_mapping[pi_name]
                traj.joint_names.append(gazebo_name)
                point.positions.append(position)
                
        if not traj.joint_names:
            return  # No matching joints found in this message

        traj.points.append(point)
        self.traj_pub.publish(traj)


def main(args=None):
    rclpy.init(args=args)
    node = GazeboStateMirror()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
