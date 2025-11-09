import math
# from typing import List, Tuple
# ROTATION MATRIX R_INVERSE = R_TRANSPOSE

def rpy_to_rotation_matrix(roll:float, pitch:float, yaw:float) -> list[list[float]]:
    """Generate a rotation matrix from roll, pitch, yaw angles (in radians).

    Args:
        roll (float): Rotation around the x-axis.
        pitch (float): Rotation around the y-axis.
        yaw (float): Rotation around the z-axis.

    Returns:
        list[list[float]]: A 3x3 rotation matrix.
    """

    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    return [
        [cy * cp    , cy * sp * sr - sy * cr    , cy * sp * cr + sy * sr],
        [sy * cp    , sy * sp * sr + cy * cr    , sy * sp * cr - cy * sr],
        [-sp        , cp * sr                   , cp * cr]
    ]
    
def rotation_matrix_to_rpy(matrix: list[list[float]]) -> list[float, float, float]:
    """
    Convert a 3x3 rotation matrix to roll, pitch, yaw (in radians).
    Assumes rotation order: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    """
    r11, r12, r13 = matrix[0]
    r21, r22, r23 = matrix[1]
    r31, r32, r33 = matrix[2]

    # --- Check for gimbal lock case ---
    if abs(r31) != 1:
        pitch = -math.asin(r31)
        roll  = math.atan2(r32 / math.cos(pitch), r33 / math.cos(pitch))
        yaw   = math.atan2(r21 / math.cos(pitch), r11 / math.cos(pitch))
    else:
        # Gimbal lock: cos(pitch) = 0
        yaw = 0
        if r31 == -1:
            pitch = math.pi / 2
            roll  = yaw + math.atan2(r12, r13)
        else:
            pitch = -math.pi / 2
            roll  = -yaw + math.atan2(-r12, -r13)

    return roll, pitch, yaw

def print_matrix(matrix: list[list[float]]) -> None:
    """Prints a 3x3 matrix in a readable format.

    Args:
        matrix (list[list[float]]): A 3x3 matrix to print.
    """
    for row in matrix:
        print("[", "  ".join(f"{val: .4f}" for val in row), "]")
        
if __name__ == "__main__":
    # Example usage
    roll = math.radians(30)
    pitch = math.radians(45)
    yaw = math.radians(60)

    rotation_matrix = rpy_to_rotation_matrix(roll, pitch, yaw)
    print("Rotation Matrix:")
    print_matrix(rotation_matrix)

    r, p, y = rotation_matrix_to_rpy(rotation_matrix)
    print("\nRecovered RPY angles (in degrees):")
    print(f"Roll: {math.degrees(r):.2f}, Pitch: {math.degrees(p):.2f}, Yaw: {math.degrees(y):.2f}")