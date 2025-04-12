from point import Point
import math


class Sensor:
    def __init__(
        self, sensor_length: float, angle_offset: float
    ):  # Tested as of 3/29/2025
        """Represents a sensor in the environment. The sensor is represented by an origin point, end point, and sensor length.

        Args:
            sensor_length (float): Length of the sensor in feet.
            angle_offset (float): Angle offset of the sensor, in radians, from the center of the sensor group.
        """
        self.sensor_length = sensor_length
        self.angle_offset = angle_offset

    def update_sensor(
        self, origin_point: Point, direction_vector: Point
    ):  # Tested as of 3/29/2025
        """Updates the sensor based on the given origin point and direction vector.

        Args:
            origin_point (Point): Point object representing the origin of the sensor.
            direction_vector (Point): Point object representing the direction vector of the sensor.
        """
        self.origin_point = origin_point
        self.end_point = self.calculate_end_point(direction_vector)

    def calculate_end_point(
        self, direction_vector: Point
    ) -> Point:  # Tested as of 3/29/2025
        """Calculates the end point of the sensor based on the origin point. Direction vector is the direction of the center of the sensor group.
        The sensors actual direction is calculated by adding the angle offset to the direction vector.
        The end point is calculated by adding the new direction vector to the origin point, scaled by the sensor length.

        Args:
            direction_vector (Point): Point object representing the direction vector of the center of the sensor group.

        Returns:
            Point: Point object representing the end point of the sensor.
        """
        ox, oy = self.origin_point.values()
        hx, hy = direction_vector.values()

        # Compute the initial heading angle
        initial_angle = math.atan2(hy - oy, hx - ox)

        # Convert angle offset to radians and apply rotation
        new_angle = initial_angle + self.angle_offset

        # Compute new heading point using the fixed sensor length
        new_x = ox + self.sensor_length * math.cos(new_angle)
        new_y = oy + self.sensor_length * math.sin(new_angle)

        return Point(new_x, new_y)
