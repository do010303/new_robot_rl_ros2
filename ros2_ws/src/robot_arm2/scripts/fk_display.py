#!/usr/bin/env python3
"""
Standalone terminal script to continuously display Forward Kinematics.
Subscribes to `/joint_states` and prints the end-effector position.
"""
import sys
import os
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

# Import the pure Python FK math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rl'))
from fk_ik_utils import fk, fk_full

C_RESET  = '\033[0m'
C_BOLD   = '\033[1m'
C_GREEN  = '\033[92m'
C_BLUE   = '\033[94m'
C_CYAN   = '\033[96m'
C_DIM    = '\033[2m'

JOINT_NAMES = [
    'Revolute 20', 'Revolute 22', 'Revolute 23',
    'Revolute 26', 'Revolute 28', 'Revolute 30'
]

class FKDisplay(Node):
    def __init__(self):
        super().__init__('fk_terminal_display')
        self.actual_positions = [0.0] * 6
        self.create_subscription(JointState, '/joint_states', self.js_cb, 10)
        self.timer = self.create_timer(0.2, self.print_fk)  # 5 Hz update

    def js_cb(self, msg):
        for i, name in enumerate(JOINT_NAMES):
            if name in msg.name:
                self.actual_positions[i] = msg.position[msg.name.index(name)]

    def print_fk(self):
        try:
            x, y, z = fk(self.actual_positions)
            
            # Use ANSI escape to clear terminal and move cursor to top-left
            # \033[H moves to 1,1; \033[J clears screen below cursor
            sys.stdout.write('\033[H\033[J')
            
            print(f"{C_BOLD}{C_CYAN}╔═════════════════════════════════════════════════╗")
            print(f"║          🤖 UAV ARM: REAL-TIME FK DISPLAY       ║")
            print(f"╚═════════════════════════════════════════════════╝{C_RESET}\n")

            print(f"{C_BOLD}Actual Joint Angles (rad):{C_RESET}")
            j_str = ', '.join(f'{v:+.4f}' for v in self.actual_positions)
            print(f"  [{C_BLUE}{j_str}{C_RESET}]\n")

            print(f"{C_BOLD}End-Effector Position (m):{C_RESET}")
            print(f"  X: {C_GREEN}{x:+.5f}{C_RESET}")
            print(f"  Y: {C_GREEN}{y:+.5f}{C_RESET}")
            print(f"  Z: {C_GREEN}{z:+.5f}{C_RESET}\n")

            pos, R = fk_full(self.actual_positions)
            print(f"{C_BOLD}End-Effector Orientation Matrix:{C_RESET}")
            for row in R:
                print(f"  {C_DIM}[{row[0]:+.5f}, {row[1]:+.5f}, {row[2]:+.5f}]{C_RESET}")
            
            print(f"\n{C_DIM}(Press Ctrl+C to quit){C_RESET}")
            sys.stdout.flush()
        except Exception as e:
            print(f"FK Computation Error: {e}")

def main():
    rclpy.init()
    node = FKDisplay()
    try:
        # Clear screen once cleanly before fast printing
        os.system('clear')
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        # Restore terminal state
        os.system('clear')
        print("FK Display closed.")

if __name__ == '__main__':
    main()
