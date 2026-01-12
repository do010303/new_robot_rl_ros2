#!/usr/bin/env python3
"""
Test 7 Servos on PCA9685 (Channels 0-6)
Interactive control via user input.
"""
import time
import board
import busio
from adafruit_pca9685 import PCA9685

# ================== I2C ==================
i2c = busio.I2C(board.SCL, board.SDA)

# ================== PCA9685 ==================
pca = PCA9685(i2c)
pca.frequency = 50  # 50Hz for servo

# ================== Servo config (TD8120 - 360° servo) ==================
MIN_US = 500
MAX_US = 2500
PERIOD_US = 20000  # 20ms (50Hz)
MAX_ANGLE = 270    # TD8120 supports 270° rotation

NUM_SERVOS = 7  # Channels 0-6
current_angles = [90] * NUM_SERVOS  # All start at 90°

# ================== Functions ==================
def set_servo_angle(channel, angle, skip_mirror=False):
    """Set servo angle (0-360 degrees for TD8120)
    
    Servo 1 and 2 are mirrored - setting one sets both.
    """
    global current_angles
    
    if channel < 0 or channel >= NUM_SERVOS:
        print(f"⚠️ Invalid channel {channel}. Use 0-{NUM_SERVOS-1}")
        return
    
    angle = max(0, min(MAX_ANGLE, angle))  # Clamp to 0-360
    
    pulse_us = MIN_US + (angle / MAX_ANGLE) * (MAX_US - MIN_US)
    duty = int(pulse_us * 65535 / PERIOD_US)
    
    pca.channels[channel].duty_cycle = duty
    current_angles[channel] = angle
    
    # Mirror logic: CH1 and CH2 work together (no recursive call)
    if not skip_mirror and channel in [1, 2]:
        mirror_ch = 2 if channel == 1 else 1
        mirror_angle = MAX_ANGLE - angle
        mirror_pulse = MIN_US + (mirror_angle / MAX_ANGLE) * (MAX_US - MIN_US)
        mirror_duty = int(mirror_pulse * 65535 / PERIOD_US)
        pca.channels[mirror_ch].duty_cycle = mirror_duty
        current_angles[mirror_ch] = mirror_angle
        print(f"[OK] CH{channel}={angle:.0f}° + CH{mirror_ch}={mirror_angle:.0f}° (mirrored)")
    else:
        print(f"[OK] CH{channel} -> {angle:.0f}°")

def show_status():
    """Display current angles of all servos"""
    print("\n" + "="*40)
    print("Current Servo Positions:")
    for ch in range(NUM_SERVOS):
        print(f"  CH{ch}: {current_angles[ch]:5.1f}°")
    print("="*40 + "\n")

def home_all():
    """Move all servos to 90°"""
    print("\n🏠 Moving all servos to home (90°)...")
    for ch in range(NUM_SERVOS):
        set_servo_angle(ch, 90)
        time.sleep(0.1)
    print("✅ All servos at home position\n")

def sweep_all_joints():
    """Sweep all joints from 45° to 135° in a loop. Press 'q' to quit."""
    import sys
    import select
    
    print("\n🔄 Sweeping all joints 45° ↔ 135°...")
    print("   Press 'q' + Enter to stop\n")
    
    sweep_angles = list(range(45, 136, 5)) + list(range(135, 44, -5))  # 45->135->45
    
    try:
        while True:
            for ch in range(NUM_SERVOS):
                print(f"\n--- Joint {ch} ---")
                for angle in sweep_angles:
                    set_servo_angle(ch, angle)
                    time.sleep(0.05)
                    
                    # Check for 'q' input (non-blocking)
                    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                        key = sys.stdin.readline().strip().lower()
                        if key == 'q':
                            print("\n⏹ Sweep stopped by user")
                            home_all()
                            return
                
                # Return to 90° before next joint
                set_servo_angle(ch, 90)
                time.sleep(0.2)
            
            print("\n🔁 Cycle complete. Starting again... (press 'q' + Enter to stop)")
    
    except KeyboardInterrupt:
        print("\n⏹ Sweep interrupted")
        home_all()

# ================== Main ==================
try:
    print("="*50)
    print("   7-Servo Controller (PCA9685, CH0-6)")
    print("="*50)
    print("\nCommands:")
    print("  <channel> <angle>  - Set servo (e.g., '0 90')")
    print("  all <angle>        - Set all servos")
    print("  home               - All servos to 90°")
    print("  sweep              - Sweep all joints 45°↔135°")
    print("  status             - Show all positions")
    print("  q                  - Quit")
    print("-"*50)
    
    # Initialize all servos to home
    home_all()
    
    while True:
        user_input = input("Enter command: ").strip().lower()
        
        if user_input == 'q':
            print("\n👋 Exiting...")
            break
        elif user_input == 'home':
            home_all()
        elif user_input == 'sweep':
            sweep_all_joints()
        elif user_input == 'status':
            show_status()
        elif user_input.startswith('all '):
            try:
                angle = float(user_input.split()[1])
                print(f"\n🔧 Setting all servos to {angle}°...")
                for ch in range(NUM_SERVOS):
                    set_servo_angle(ch, angle)
                    time.sleep(0.05)
            except (ValueError, IndexError):
                print("⚠️ Usage: all <angle>")
        else:
            try:
                parts = user_input.split()
                if len(parts) == 2:
                    channel = int(parts[0])
                    angle = float(parts[1])
                    set_servo_angle(channel, angle)
                else:
                    print("⚠️ Usage: <channel> <angle> (e.g., '0 90')")
            except ValueError:
                print("⚠️ Invalid input. Use: <channel> <angle>")

except KeyboardInterrupt:
    print("\n\n⚠️ Interrupted by Ctrl+C")

finally:
    # Release all servos
    print("🔓 Releasing servos...")
    for ch in range(NUM_SERVOS):
        pca.channels[ch].duty_cycle = 0
    pca.deinit()
    print("✅ PCA9685 deinitialized")
