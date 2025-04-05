import vehicle_placement as VP
import numpy as np


def test_lateral_adjustment():
    """Tests the lateral adjustment function."""
    test_cases = [
        (0, 0, 0),  # Left edge, angle_offset = 0 -> 0
        (0, np.pi / 2, -np.pi / 2),  # Left edge, angle_offset = pi/2 -> -pi/2
        (1, 0, 0),  # Right edge, angle_offset = 0 -> 0
        (1, -np.pi / 2, np.pi / 2),  # Right edge, angle_offset = -pi/2 -> pi/2
        (0.5, np.pi / 4, np.pi / 4),  # Middle of lane, angle_offset = pi/4 -> pi/4
    ]

    for i, (latitude, angle_offset, expected) in enumerate(test_cases):
        result = VP.lateral_adjustment(latitude, angle_offset)
        # print(f"{i}: Expected {expected}, Actual {result}")
        assert result == expected, f"{i}: Expected {expected}, but got {result}"
    print("Vehicle Placement Test: All lateral adjustment tests PASSED")
        
def test_open_loop_adjustment():
    """Tests the open loop adjustment function."""
    # (Longitude, Latitude, Angle Offset, Expected Result)
    # Longitude = 0 or 1 indicates start or end of lane
    # Latitude = 0 or 1 indicates left or right edge of lane
    # Angle Offset is the angle in radians
    test_cases = [
        (0, 0.5, 0, 0),  # Start of lane, middle of lane, angle_offset = 0 -> 0
        (0, 0.5, np.pi / 2, np.pi / 2),  # Start of lane, middle of lane, angle_offset = pi/2 -> pi/2
        (1, 0.5, 0, 0),  # End of lane, middle of lane, angle_offset = 0 -> 0
        (1, 0.5, -np.pi / 2, -np.pi / 2),  # End of lane, middle of lane, angle_offset = -pi/2 -> -pi/2
        (0, 0, 0, 0),  # Start of lane, left edge, angle_offset = 0 -> 0
        (1, 1, 0, 0),  # End of lane, right edge, angle_offset = 0 -> 0
        (0, 0, np.pi / 2, 0),  # Start of lane, left edge, angle_offset = pi/2 -> 0
        (1, 1, -np.pi / 2, 0),  # End of lane, right edge, angle_offset = -pi/2 -> 0
        (0, 0, -3*np.pi / 2, -np.pi / 2),  # Start of lane, left edge, angle_offset = -3pi/2 -> -pi/2
        (1, 1, 3*np.pi / 2, np.pi / 2),  # End of lane, right edge, angle_offset = 3pi/2 -> pi/2
        (0.5, 0.5, np.pi / 4, np.pi / 4),  # Middle of lane, middle of lane, angle_offset = pi/4 -> pi/4
        (0.5, 0.5, -np.pi / 4, -np.pi / 4),  # Middle of lane, middle of lane, angle_offset = -pi/4 -> -pi/4
        (0, 0.5, -3*np.pi / 4, -np.pi / 2),  # Start of lane, middle of lane, angle_offset = -3pi/4 -> -pi/2
        (1, 0.5, 3*np.pi / 4, np.pi / 2),  # End of lane, middle of lane, angle_offset = 3pi/4 -> pi/2
        (0, 0, -np.pi / 2, -np.pi / 2),  # Start of lane, left edge, angle_offset = -pi/2 -> -pi/2
        (1, 1, np.pi / 2, np.pi / 2),  # End of lane, right edge, angle_offset = pi/2 -> pi/2
    ]

    for i, (longitude, latitude, angle_offset, expected) in enumerate(test_cases):
        result = VP.open_loop_adjustment(longitude, latitude, angle_offset)
        # print(f"{i}: Expected {angle_offset}, Actual {result}")
        assert result == expected, f"{i}: Expected {expected}, but got {result}"
    print("Vehicle Placement Test: All open loop adjustment tests PASSED")
        
def run_tests():
    test_lateral_adjustment()
    test_open_loop_adjustment()
    print("Vehicle Placement Test: All tests PASSED.\n")
    
if __name__ == "__main__":
    run_tests()
