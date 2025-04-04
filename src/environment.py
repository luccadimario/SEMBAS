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
    
    def position_from_coordinates(self, longitude: float, latitude: float, angle_offset: float, heading_offset: float) -> tuple[Point, Point]:
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
        
        # center_point, direction_vector = self.get_lane_position_and_heading(longitude=longitude, latitude=latitude, angle_offset=angle_offset)
        center_point = VP.get_center_point(lane=self.lane, longitude=longitude, latitude=latitude)
        
        # Adjust the angle offset based on the lane's closed loop status
        angle_offset = VP.lateral_adjustment(latitude, angle_offset)
        if not self.lane.closed_loop:
            angle_offset = VP.open_loop_adjustment(longitude, latitude, angle_offset)
            
        direction_vector = VP.get_direction(self.lane.center_line, longitude)
        
        rotation_vector = VP.get_rotation_vector(direction_vector=direction_vector, angle_offset=angle_offset)
        
        # Heading point a short step ahead
        heading_point = Point(
            center_point.x + heading_offset * rotation_vector.x, 
            center_point.y + heading_offset * rotation_vector.y
        )
 
        return center_point, heading_point
    
    
        
    
    
    
    
        

    

        
        
        
    
        
    
    