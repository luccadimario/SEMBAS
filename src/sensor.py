from point import Point
import math

class Sensor:
    def __init__(self, sensor_length: float, angle_offset: float): # Tested as of 3/29/2025
        """Represents a sensor in the environment. The sensor is represented by an origin point, end point, and sensor length.

        Args:
            sensor_length (float): Length of the sensor in feet.
            angle_offset (float): Angle offset of the sensor, in radians, from the center of the sensor group.
        """
        self.sensor_length = sensor_length
        self.angle_offset = angle_offset
        
    def update_sensor(self, origin_point: Point, direction_vector: Point):  # Tested as of 3/29/2025
        """Updates the sensor based on the given origin point and direction vector.

        Args:
            origin_point (Point): Point object representing the origin of the sensor.
            direction_vector (Point): Point object representing the direction vector of the sensor.
        """
        self.origin_point = origin_point
        self.end_point = self.calculate_end_point(direction_vector)
        
    def calculate_end_point(self, direction_vector: Point) -> Point: # Tested as of 3/29/2025
        """Calculates the end point of the sensor based on the origin point. Direction vector is the direction of the center of the sensor group.
        The sensors actual direction is calculated by adding the angle offset to the direction vector.
        The end point is calculated by adding the new direction vector to the origin point, scaled by the sensor length.

        Args:
            direction_vector (Point): Point object representing the direction vector of the center of the sensor group.

        Returns:
            Point: Point object representing the end point of the sensor.
        """
        # Calculate the new direction vector based on the angle offset
        new_direction_vector = Point(
            direction_vector.x * math.cos(self.angle_offset) - direction_vector.y * math.sin(self.angle_offset),
            direction_vector.x * math.sin(self.angle_offset) + direction_vector.y * math.cos(self.angle_offset)
        )
        if self.angle_offset == 0:
            # If angle offset is 0, the direction vector is unchanged
            new_direction_vector = direction_vector
        # Normalize the new direction vector
        new_direction_vector = new_direction_vector / new_direction_vector.norm()
        # Calculate the end point of the sensor
        end_point = self.origin_point + new_direction_vector * self.sensor_length
        # Return the end point of the sensor
        return end_point