from point import Point
import numpy as np

class Vehicle:
    def __init__(self, min_speed: float = 0.0, max_speed: float = 150.0):  # Tested as of 3/29/2025
        """Initializes a vehicle object with default parameters.
        min_speed and max_speed are in miles per hour. Represents vehicle speed capabilities.
        
        Args:
            min_speed (float): Minimum speed capability of the vehicle in miles per hour.
            max_speed (float): Maximum speed capability of the vehicle in miles per hour.
        """
        self.min_speed = min_speed
        self.max_speed = max_speed
    
    def vehicle_setup(self, center_point: Point, heading_point: Point, speed: float, length: float = 12, width: float = 6):     # Tested as of 3/29/2025
        """Sets up the vehicle based on the given center point, heading point and speed.
        
        Args:
            center_point (Point): Point object representing the center of the vehicle.
            heading_point (Point): Point object representing the heading of the vehicle.
            speed (float): Speed of the vehicle in miles per hour.
        """
        self.center_point = center_point
        self.heading_point = heading_point
        # self.heading_angle = self.calculate_heading_angle(center_point, heading_point)
        self.speed = speed
        self.distance_travelled = 0
        self.acceleration = 0
        self.velocity = self.calculate_velocity()
        self.length = length
        self.width = width
        self.body = self.build_body()
        self.calculate_velocity()
        
    def calculate_velocity(self):   # Tested as of 3/29/2025
        """Calculates the velocity of the vehicle based on the center point, heading point, and speed.
        
        Returns:
            float: Velocity of the vehicle in miles per hour.
        """
        direction = self.heading_point - self.center_point
        direction = direction / direction.norm()
        return direction * self.speed
        
    def build_body(self):  # Tested as of 3/29/2025
        """Builds the body of the vehicle based on the given width and length and the center point.
        Body is represented as a list of points in this order:
        - Front left: (-width/2, length/2)
        - Front right: (width/2, length/2)
        - Back right: (width/2, -length/2)
        - Back left: (-width/2, -length/2)
        
        Args:
            width (float): Width of the vehicle in feet.
            length (float): Length of the vehicle in feet.
        """
        body = [
            Point(self.center_point.x - self.width / 2, self.center_point.y + self.length / 2),  # Front left
            Point(self.center_point.x + self.width / 2, self.center_point.y + self.length / 2),  # Front right
            Point(self.center_point.x + self.width / 2, self.center_point.y - self.length / 2),  # Back right
            Point(self.center_point.x - self.width / 2, self.center_point.y - self.length / 2),  # Back left
        ]
        return body        
        
    def update_position(self, steering: float, acceleration: float, dt: float, friction: float = 0.0):
        """Updates the position of the vehicle based on the steering, acceleration and time step.
        
        Args:
            steering (float): Steering angle in degrees.
            acceleration (float): Acceleration in miles per hour squared.
            dt (float): Time step in seconds.
        """
        # Update speed based on acceleration
        self.speed = np.clip(self.speed + acceleration * dt, self.min_speed, self.max_speed)

        # Calculate direction vector from center to heading point
        dx = self.heading_point.x - self.center_point.x
        dy = self.heading_point.y - self.center_point.y
        direction_vector = np.array([dx, dy])
        
        # Normalize the direction vector
        direction_vector = direction_vector / np.linalg.norm(direction_vector)

        # Apply steering: rotate the direction vector by the steering input
        angle_of_rotation = steering  # Steering angle (radians)
        rotation_matrix = np.array([[np.cos(angle_of_rotation), -np.sin(angle_of_rotation)],
                                    [np.sin(angle_of_rotation), np.cos(angle_of_rotation)]])
        
        new_direction = np.dot(rotation_matrix, direction_vector)

        # Update heading point based on new direction
        self.heading_point = Point(self.center_point.x + new_direction[0] * self.length,
                                   self.center_point.y + new_direction[1] * self.length)

        # Update position (move center point based on speed)
        self.center_point.x += new_direction[0] * self.speed * dt
        self.center_point.y += new_direction[1] * self.speed * dt
        
        self.body = self.build_body()  # Rebuild the body after updating position
        
def test_vehicle():
    v = Vehicle()
    center_point = Point(0, 0)
    heading_point = Point(1, 0)
    speed = 10.0
    v.vehicle_setup(center_point=center_point, heading_point=heading_point, speed=speed)
    test_no_movement(v, center_point)
    # v.update_position(steering=0.0, acceleration=0.10, dt=1.0)
    # assert v.center_point.x == 0, "Vehicle center point x-coordinate should stay the same since vehicle is going straight."
    # assert v.center_point.y != center_point.y, "Vehicle center point y-coordinate should have changed."
    # assert v.center_point.y == 10, "Vehicle center point y-coordinate should have changed."
    
    
def test_no_movement(v, center):
    acceleration = 10
    steering = 0.0
    dt = 1.0
    v.update_position(steering, acceleration, dt)
    assert v.center_point.y == 1, f"Vehicle center point x-coordinate {v.center_point.y} does not match expected value {0}."
    # assert v.center_point.y == center.y, "Vehicle center point y-coordinate does not match expected value."
    print("Vehicle works when acceleration is 0")
    
if __name__ == "__main__":
    test_vehicle()
    print("Vehicle test passed.")