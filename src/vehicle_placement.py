from point import Point
from lane import Lane
import numpy as np




def get_center_point_from_index(lane: Lane, index: int, fractional: float) -> Point:
        """Returns the center position at the given index with the fractional part.

        Args:
            lane (Lane): Lane object to use for the center_line points and the closed_loop flag.
            index (int): Index of the center point.
            fractional (float): Fractional part for interpolation.

        Returns:
            Point: The center position at the given index with the fractional part.
        """
        center_pos = None
        lane_center_line = lane.center_line
        if index >= len(lane_center_line) - 1:
            
            if lane.closed_loop:
                # If the lane is closed, the points on either side of the returned point should be the first and second center line points
                index = 0
            else:
                # If the lane is not closed, the center position returned should just be the last point in the center_line list
                center_pos = Point(lane_center_line[-1].x, lane_center_line[-1].y)
        
        if center_pos is None:
            # Finding the center line points on either side of the given fractional
            p1 = Point(lane_center_line[index].x, lane_center_line[index].y)
            p2 = Point(lane_center_line[index + 1].x, lane_center_line[index + 1].y)
            # Creating the point from p1 and p2 with the fractional part
            center_pos = Point(
                p1.x * (1 - fractional) + p2.x * fractional,
                p1.y * (1 - fractional) + p2.y * fractional,
            )
        return center_pos
    
def calculate_tangent(lane: Lane, center_pos: Point, index: int) -> Point:
        """Approximates the tangent at the given point using finite differences.

        Args:
            center_pos (Point): Point to calculate tangent from.
            index (int): Index of the center point.

        Returns:
            Point: The tangent at the given point.
        """
        tangent = None
        if index == len(lane.center_line) - 1:
            tangent = Point(
                center_pos.x - lane.center_line[index - 1].x,
                center_pos.y - lane.center_line[index - 1].y,
            )
        else:
            tangent = Point(
                lane.center_line[index + 1].x - center_pos.x,
                lane.center_line[index + 1].y - center_pos.y,
            )
        return tangent
    
def get_normal_and_direction_vectors(tangent: Point) -> list[Point]:
    """Noramlizes the tangent vector and rotates it 90 degrees to get the normal vector.
    Normalizes the tangent vector to get the direction vector at the center position.

    Args:
        tangent (Point): The tangent vector.

    Returns:
        Point: The normal vector.
        Point: The direction vector.
    """
    normal = None
    direction = None
    magnitude = (tangent.x**2 + tangent.y**2) ** 0.5
    if magnitude == 0:
        normal = Point(0, 0)
        direction = Point(0, 0)
    else:
        normal = Point(
            -tangent.y / magnitude, tangent.x / magnitude
        )  # Rotate 90 degrees
        direction = Point(tangent.x / magnitude, tangent.y / magnitude)
    return normal, direction



def lateral_adjustment(latitude: float, angle_offset: float) -> float:
    """Given a latitude and an angle_offset, returns an adjusted angle_offset to take into account if the latitude falls on the edge of the lane. 

    Args:
        lateral (float): Value 0 to 1 where 0 is the left edge and 1 is the right edge.
        angle_offset (float): The angle, in radians, the heading should be offset from parallel to the center line.

    Returns:
        float: The angle adjusted for being on the edge. If the latitude is 0 (on the left edge) and the angle is between 0 and pi, angle is multiplied by -1.
        If the latitude is 1 (on the right edge) and the angle offset is between 0 and -pi, angle is multiplied by -1. 
    """
    # If on the left edge and angle_offset is pointing outside the lane, reverse the angle_offset
    if latitude == 0 and (0 < angle_offset < np.pi):
        angle_offset *= -1
    # Else on the right edge and angle_offset is pointing outside the lane, reverse the angle_offset
    elif latitude == 1 and ((0 < angle_offset < -np.pi)):
        angle_offset *= -1
    return angle_offset

def open_loop_adjustment(longitude: float, latitude: float, angle_offset: float) -> float:
    """Adjusts the angle_offset to make sure that it wont be pointing outside the lane.
    
    Args:
        longitude (float): The longitude position down the length of the lane. 0 is the start, 1 is the end.
        latitude (float): The latitude position from left to right in the lane. 0 is the left edge of the lane, 1 is the right.
        angle_offset (float): The angle offset for the heading point in radians.
        
    Returns:
        float: The angle_offset adjusted for if its at the start or the end of the lane and its not a closed loop lane.
    """
    if (longitude == 0 or longitude == 1) and latitude == 0:
        if angle_offset > 0:
            angle_offset = 0
        elif angle_offset < (-np.pi / 2):
            angle_offset = -np.pi / 2
    elif (longitude == 0 or longitude == 1) and latitude == 1:
        if angle_offset < 0:
            angle_offset = 0
        elif angle_offset > (np.pi / 2):
            angle_offset = np.pi / 2
    return angle_offset

