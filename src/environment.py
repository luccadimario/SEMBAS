from lane import Lane
from point import Point

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
        # Placeholder for actual implementation
        return True
    
    def point_position_in_lane(self, point: Point) -> tuple[float, float, float]:
        """Calculates the position of a given point in the lane.

        Args:
            point (Point): Point object representing the point to check.

        Returns:
            tuple(float, float, float): Tuple of three floats representing the distance to the left edge, distance to the right edge and distance to the center line.
        """
        # Placeholder for actual implementation
        return 0.0, 0.0, 0.0