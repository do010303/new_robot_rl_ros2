import re

input_file = "best_waypoints.txt"
output_file = "best_waypoints_servo.txt"

def main():
    print(f"Reading {input_file}...")
    
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    with open(output_file, 'w') as out:
        out.write("BEST SOLUTIONS FOR SERVO CONTROL (Angles + 90 deg)\n")
        out.write("Format: Waypoint -> Servo Angles (0-180 range)\n\n")
        
        for line in lines:
            if "CMD (deg):" in line:
                # Extract angles
                match = re.search(r"CMD \(deg\):\s+(.*)", line)
                if match:
                    angles_str = match.group(1).strip()
                    # Split by space
                    angles = [float(x) for x in angles_str.split()]
                    
                    # Add 90 to each
                    servo_angles = [a + 90.0 for a in angles]
                    
                    # Format
                    servo_str = " ".join([f"{x:.2f}" for x in servo_angles])
                    
                    out.write(f"  SERVO (deg): {servo_str}\n")
                else:
                    out.write(line)
            else:
                out.write(line)

    print(f"Converted angles written to {output_file}")

if __name__ == "__main__":
    main()
