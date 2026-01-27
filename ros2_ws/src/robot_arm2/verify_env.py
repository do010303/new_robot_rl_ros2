#!/usr/bin/env python3
import sys
import os
import rclpy

# Add scripts to path
sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))

from rl.drawing_environment import DrawingEnvironment
import drawing.drawing_config as config

def verify():
    print("="*60)
    print("🔍 Verifying Drawing Environment Configuration")
    print("="*60)
    
    print(f"Config SHAPE_TYPE: {config.SHAPE_TYPE}")
    print(f"Config POINTS_PER_EDGE: {config.POINTS_PER_EDGE}")
    print(f"Config TOTAL_WAYPOINTS: {config.TOTAL_WAYPOINTS}")
    
    rclpy.init()
    try:
        env = DrawingEnvironment(shape_type=config.SHAPE_TYPE, shape_size=0.15)
        shape = env._generate_shape()
        
        print("\n✅ Generated Shape in Environment:")
        print(f"  Name: {shape.name}")
        print(f"  Waypoints: {len(shape.waypoints)}")
        print(f"  Closed: {shape.closed}")
        
        print("\n📍 Waypoints:")
        for i, wp in enumerate(shape.waypoints):
            print(f"  P{i+1}: ({wp[0]:.3f}, {wp[1]:.3f}, {wp[2]:.3f})")
            
        # Verify it's a square (4 corners + return = 5 points)
        if len(shape.waypoints) == 5:
            print("\n✅ SUCCESS: Correct number of waypoints for Square (1 point/edge).")
        else:
            print(f"\n❌ FAILURE: Expected 5 waypoints, got {len(shape.waypoints)}.")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    verify()
