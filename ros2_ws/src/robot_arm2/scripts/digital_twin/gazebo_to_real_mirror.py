#!/usr/bin/env python3
"""
Gazebo to Real Mirror Node (Sim-to-Real)
========================================
This node listens to the simulated joint states from Gazebo 
(`/joint_states` published by `robot_state_publisher` or `ros2_control`)
and forwards them as commands to the actual Raspberry Pi (`/pca9685_servo/command`),
forcing the physical robot to perfectly mirror the simulation.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class GazeboToRealMirror(Node):
    def __init__(self):
        super().__init__('gazebo_to_real_mirror')

        # The names of the joints published by Gazebo (from URDF)
        self.gazebo_joint_names = [
            "Revolute 20", "Revolute 22", "Revolute 23", "Revolute 26", "Revolute 28", "Revolute 30"
        ]

        # The names of the joints expected by the Raspberry Pi (from servos.yaml)
        self.pi_joint_names = [
            "base", "shoulder", "elbow", "wrist_roll", "wrist_pitch", "pen"
        ]

        # Mapping from Gazebo name to Pi name
        self.name_mapping = dict(zip(self.gazebo_joint_names, self.pi_joint_names))

        # Publisher to Raspberry Pi's command topic
        self.command_pub = self.create_publisher(
            JointState, 
            "/pca9685_servo/command", 
            10
        )

        # Subscriber to Gazebo's joint states
        # robot_state_publisher usually publishes to /joint_states
        self.js_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_states_callback,
            10
        )

        self.get_logger().info("Gazebo-to-Real Mirror started!")
        self.get_logger().info("Listening to Gazebo on: /joint_states")
        self.get_logger().info("Publishing to Pi on: /pca9685_servo/command")

    def joint_states_callback(self, msg: JointState):
        """
        Received simulated joint angles from Gazebo.
        Convert and publish to the Pi as a command.
        """
        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        
        # We only want to map the joints we know about
        for gazebo_name, position in zip(msg.name, msg.position):
            if gazebo_name in self.name_mapping:
                pi_name = self.name_mapping[gazebo_name]
                cmd.name.append(pi_name)
                cmd.position.append(position)
                
        if not cmd.name:
            return  # No matching joints found in this message

        self.command_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = GazeboToRealMirror()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
