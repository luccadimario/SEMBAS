from point import Point

class Vehicle:
    def __init__(self):
        pass
    
    def vehicle_setup(self, center_point: Point, heading_point: Point, speed: float):
        """Sets up the vehicle based on the given center point, heading point and speed.
        
        Args:
            center_point (Point): Point object representing the center of the vehicle.
            heading_point (Point): Point object representing the heading of the vehicle.
            speed (float): Speed of the vehicle in miles per hour.
        """
        self.center_point = center_point
        self.heading_point = heading_point
        self.speed = speed
        
    def calculate_velocity(self):
        """Calculates the velocity of the vehicle based on the center point, heading point, and speed.
        
        Returns:
            float: Velocity of the vehicle in miles per hour.
        """
        direction = self.heading_point - self.center_point
        direction = direction / direction.norm()
        self.velocity = direction * self.speed
        
    def build_body(self, width: float = 6, length: float = 12):
        """Builds the body of the vehicle based on the given width and length and the center point.
        
        Args:
            width (float): Width of the vehicle in feet.
            length (float): Length of the vehicle in feet.
        """
        self.width = width
        self.length = length
        self.body = [
            Point(self.center_point.x - width / 2, self.center_point.y - length / 2),
            Point(self.center_point.x + width / 2, self.center_point.y - length / 2),
            Point(self.center_point.x + width / 2, self.center_point.y + length / 2),
            Point(self.center_point.x - width / 2, self.center_point.y + length / 2)
        ]
        
    def update_position(self, steering: float, acceleration: float, dt: float):
        """Updates the position of the vehicle based on the steering, acceleration and time step.
        
        Args:
            steering (float): Steering angle in degrees.
            acceleration (float): Acceleration in miles per hour squared.
            dt (float): Time step in seconds.
        """
        # Placeholder for actual implementation
        pass
        