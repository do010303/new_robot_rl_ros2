import sys
import math
sys.path.append("/home/ducanh/new_rl_ros2/ros2_ws/src/visual_servoing/scripts/rl")
from fk_ik_utils import _T, _Rz, _Ry, _chain, _pos

# Fixed translations from fk_ik_utils
T_r6  = _T(-0.046528, 0.031724, 0.748891)
T_r18 = _T(-0.093, 0.0, -0.01)
T_r19 = _T(0.04889, -0.028138, -0.00625)
T_base = _chain(T_r6, T_r18, T_r19)

# Rev 20 Origin (Base)
T_j20_origin = _chain(T_base, _T(-0.034687, -0.0039, -0.0162))

# Rev 22 Origin (Shoulder)
T_r21 = _T(-0.048931, -0.007, -0.033724)
T_j22_origin = _chain(T_j20_origin, T_r21, _T(0.034687, -0.0192, -0.0039))

# Rev 23 Origin (Elbow)
T_j23_origin = _chain(T_j22_origin, _T(0.0, 0.0, -0.155))

# Rev 26 (Wrist Roll)
T_r24 = _T(-0.0039, 0.0192, -0.034687)
T_r25 = _T(0.03375, 0.0362, -0.042816)
T_j26_origin = _chain(T_j23_origin, T_r24, T_r25, _T(0.0, -0.00995, -0.0148))

# Rev 28 Origin (Wrist Pitch)
T_r27 = _T(0.0152, -0.023, -0.0425)
T_j28_origin = _chain(T_j26_origin, T_r27, _T(-0.00995, -0.0148, 0.0))

# EE Origin
T_r29 = _T(-0.0152, 0.0075, -0.075)
T_j30_origin = _chain(T_j28_origin, T_r29, _T(0.02045, 0.015, 0.0))
T_r32 = _T(0.0, 0.01225, -0.01)
T_r33 = _T(0.0, 0.0, -0.045)
T_ee_origin = _chain(T_j30_origin, T_r32, T_r33)

def dist(M1, M2):
    p1 = _pos(M1)
    p2 = _pos(M2)
    return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2 + (p2[2]-p1[2])**2)

print("--- DISTANCES (in cm) ---")
print(f"Base to Shoulder (L0 horizontal offset?): {dist(T_j20_origin, T_j22_origin)*100:.3f} cm")
# Wait, L0 in the kinematic solver is usually the Z-axis offset from base to shoulder.
p_base = _pos(T_j20_origin)
p_shoulder = _pos(T_j22_origin)
L0 = abs(p_shoulder[2] - p_base[2]) * 100
print(f"L0 (Z-offset Base to Shoulder): {L0:.3f} cm")

L1 = dist(T_j22_origin, T_j23_origin) * 100
print(f"L1 (Shoulder to Elbow): {L1:.3f} cm")

L2 = dist(T_j23_origin, T_j28_origin) * 100
print(f"L2 (Elbow to Wrist Pitch): {L2:.3f} cm")

L3 = dist(T_j28_origin, T_ee_origin) * 100
print(f"L3 (Wrist Pitch to EE): {L3:.3f} cm")
