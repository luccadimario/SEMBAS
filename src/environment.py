from lane import Lane
from point import Point
import numpy as np
import vehicle_placement as VP

class Environment:
    def __init__(self, lane: Lane):
        """Represents the environment in which the agent operates. The environment is represented by a lane object.

        Args:
            lane (Lane): Lane object representing the environment.
        """
        self.lane = lane
        
    def set_lane(self, lane: Lane):
        """Sets the lane of the environment.

        Args:
            lane (Lane): Lane object representing the environment.
        """
        self.lane = lane
        
    def point_in_lane(self, point: Point) -> bool:
        """Checks if a given point is within the lane.

        Args:
            point (Point): Point object representing the point to check.

        Returns:
            bool: True if the point is within the lane, False otherwise.
        """
        center_dist, _, _ = self.point_position_in_lane(point)
        return center_dist > 0
    
    def point_position_in_lane(self, point: Point) -> tuple[float, float, float]:
        """Calculates the position of a given point in the lane.

        Args:
            point (Point): Point object representing the point to check.

        Returns:
            tuple(float, float, float): Tuple of three floats representing: distance to center line, the distance to the left edge, distance to the right edge
        """
        
        return 0.0, 0.0, 0.0
    
    def position_from_coordiantes(self, longitude: float, latitude: float, angle_offset: float) -> tuple[Point, Point]:
        """Given a longitude and latitude, both from 0 to 1, returns a point inside the lane longitudinal distance along the lane and lateral distance between the lane edges.
        
        Args:
            longitude (float): Longitude distance from 0 to 1 where 0 is the start of the lane and 1 is the end of the lane.
            latitude (float): Lateral distance from 0 to 1 where 0 is the left edge of the lane and 1 is the right edge of the lane.
            angle_offset (float): Angle offset, in radians, from the center line of the lane to calculate the heading point.
        Returns:
            Point: The center point in the lane at the coordinates provided.
            Point: The heading point from the center point at the angle_offset from parallel to the center line.
        """
        # Ensuring both are from 0 to 1
        longitude = np.clip(longitude, 0, 1)
        latitude = np.clip(latitude, 0, 1)
        
        center_point, direction_vector = self.get_center_point(longitude, latitude)
        direction_point = self.get_direction_point(center_point=center_point, direction_vector=direction_vector, longitude=longitude)
        
        angle_offset = VP.lateral_adjustment(latitude, angle_offset)
        if not self.lane.closed_loop:
            angle_offset = VP.open_loop_adjustment(longitude, latitude, angle_offset)
            
        heading_point = direction_point.rotate_point_by_radians(center_point, angle_offset)

        return center_point, heading_point
    
    def get_direction_point(self, center_point: Point, direction_vector: Point, longitude: float):
        # Direction point is the center position offset by the direction vector
        straight_direction_pt = Point(
            center_point.x + direction_vector.x,
            center_point.y + direction_vector.y,
        )
        if not self.lane.closed_loop and longitude == 1:
            straight_direction_pt.rotate_point_by_radians(center_point, np.pi)
        return straight_direction_pt
    
    def get_center_point(self, longitude: float, latitude: float) -> tuple[Point, Point]:
        """Takes in a lane and coordinates of longitude and latitude.
        Returns the point at the coordinate location and the directional point that is parallel with the lane center line.
        

        Args:
            lane (Lane): Lane object to use the center line and closed loop information.
            longitude (float): Value from 0 to 1 representing how far along the lane for the point to be. 0 is the start of the lane and 1 is the end of the lane.
            latitude (float): Value from 0 to 1 representing the point placement from side to side of the lane. 0 is left and 1 is right.

        Returns:
            tuple(Point, Point): First point is Point located at longitude location down the lane, latitude location from side to side. Second point is Point located ahead of the first point parallel to the lane center line representing the heading point.
        """
        # Retrieving the point in the lane center line that is at the longitude position
        num_points = len(self.lane.center_line)
        index_float = longitude * (num_points - 1)
        index = int(index_float) # Always rounds down (by getting rid of the decimals)
        t = index_float - index
        
        center_point = VP.get_center_point_from_index(self.lane, index, t)
        tangent = VP.calculate_tangent(self.lane, center_point, index)
        normal, direction_vector = VP.get_normal_and_direction_vectors(tangent)
        center_point = self.find_final_center(center_point, normal, latitude)
        
        return center_point, direction_vector
    
    def find_final_center(
            self, center_pos: Point, normal: Point, lateral: float
        ) -> Point:
        """Finds the final position given the center position, normal vector, and lateral offset.
        Checks that the center position is within the lane boundaries and adjusts the position if necessary.

        Args:
            center_pos (Point): Center position.
            normal (Point): Normal vector.
            lateral (float): Lateral offset.

        Returns:
            Point: The final position.
        """
        # Compute lateral offset
        lateral_offset = (lateral - 0.5) * self.lane.lane_width  # Shift from center

        # Compute final position
        final_x = center_pos.x + normal.x * lateral_offset
        final_y = center_pos.y + normal.y * lateral_offset

        # Check if the point is outside the lane boundaries
        center_dist, _, _ = self.point_position_in_lane(Point(final_x, final_y))
        if center_dist < 0:
            center_offset = abs(lateral_offset) - abs(center_dist)
            final_x += normal.x * center_offset
            final_y += normal.y * center_offset

        return Point(final_x, final_y)
    
    
        

    

        
        
        
    
        
    
    