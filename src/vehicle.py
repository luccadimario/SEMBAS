from point import Point
import numpy as np
import math

class Vehicle:
    
    mph_to_fps_conversion = 5280 / 3600
    fps_to_mph_conversion = 3600 / 5280
    
    mph2_to_fps2_conversion = 5280 / (3600 ** 2)
    fps2_to_mph2_conversion = (3600 ** 2) / 5280
    
    def __init__(self, vehicle_length_ft: float = 12, vehicle_width_ft: float = 6, min_speed_mph: float = 0.0, max_speed_mph: float = 150.0, heading_offset_ft: float = 15.0, sec_0_to_60: float = 8.0):  # Tested as of 3/29/2025
        """Initializes a vehicle object with default parameters.
        min_speed and max_speed are in miles per hour. Represents vehicle speed capabilities.
        The length and width parameters represent the vehicles dimensions in feet. 
        The heading_offset is how far, in feet, from the center point the heading point will be located.
        
        Vehicle motion calculations are done in feet and seconds. All passed in values are converted to feet and seconds before calculations are done.
        When retrieving the properties of speed and acceleration, they are converted from feet/second to miles/hour.
        
        Vehicle breaking and acceleration stats taken from: https://copradar.com/chapts/references/acceleration.html
        
        Args:
            vehicle_length_ft (float): Length of the vehicle in feet. Defaults to 12.0.
            vehicle_width_ft (float): Width of the vehicle in feet. Defaults to 6.0.
            min_speed (float): Minimum speed capability of the vehicle in miles per hour. Defaults to 150 mph.
            max_speed (float): Maximum speed capability of the vehicle in miles per hour. Defaults to 0 mph.
            heading_offset (float): The heading point offset from the center point in feet. Defaults to 15.0 feet.
            sec_0_to_60 (float): Number of seconds it takes the car to accelerate from 0 to 60 mph. Default is 8.0.
        """
        self.vehicle_length_ft = vehicle_length_ft
        self.vehicle_width_ft = vehicle_width_ft
        self.min_speed_fps = self.mph_to_fps(min_speed_mph)
        self.max_speed_fps = self.mph_to_fps(max_speed_mph)
        self.heading_offset_ft = heading_offset_ft
        # Calculating the max acceleration based on the number of seconds it takes the car to go from 0 to 60 miles per hour.
        # Acceleration = (Final speed - initial speed) / time
        self.max_acceleration_fps2 = self.mph_to_fps(60.0) / sec_0_to_60
        self.max_breaking_fps2 = 15.0
    
    def vehicle_setup(self, center_point: Point, heading_point: Point, speed_mph: float):     # Tested as of 3/29/2025
        """Sets up the vehicle based on the given center point, heading point and speed.
        
        Args:
            center_point (Point): Point object representing the center of the vehicle. Grid scale is in feet so X, Y should be in reference to feet.
            heading_point (Point): Point object representing the heading of the vehicle. Grid scale is in feet so X, Y should be in reference to feet.
            speed_mph (float): Speed of the vehicle in miles per hour. 
        """
        self.center_point = center_point
        self.heading_point = heading_point
        self.speed_fps = self.mph_to_fps(speed_mph)
        self.distance_travelled_ft = 0
        self.acceleration_fps2 = 0
        self.velocity_fps = self.calculate_velocity()
        self.body = self.build_body()
        
    def vehicle_capabilities_str(self):
        return (
            f"Vehicle Capabilities: "
            f"Speed: {self.fps_to_mph(self.min_speed_fps)} to {self.fps_to_mph(self.max_speed_fps)} mph -- "
            f"Max Acceleration: {self.max_acceleration_fps2} fps^2 = {self.fps2_to_mph2(self.max_acceleration_fps2)} mph^2 -- "
            f"Max Breaking: {self.max_breaking_fps2} fps^2 = {self.fps2_to_mph2(self.max_breaking_fps2)} mph^2"
        )
        
    def vehicle_state_str(self):
        return (
            f"Center: ({self.center_point.x}, {self.center_point.y}) -- "
            f"Heading: ({self.heading_point.x}, {self.heading_point.y}) -- "
            f"Speed: {self.speed_mph} mph -- "
            f"Acceleration: {self.acceleration_mph2} mph^2 -- "
            f"Distance travelled: {self.distance_travelled_ft} feet"
        )
        
    @property
    def speed_mph(self):
        """Speed in Miles Per Hour"""
        return self.fps_to_mph(self.speed_fps)
    
    @property
    def acceleration_mph2(self):
        """Acceleration in Miles Per Hour Squared"""
        return self.fps2_to_mph2(self.acceleration_fps2)
    
    @property
    def distance_travelled_miles(self):
        """Distance Travelled in Miles"""
        return self.distance_travelled_ft / 5280
    
    def mph_to_fps(self, value_mph: float):
        return value_mph * self.mph_to_fps_conversion
    
    def fps_to_mph(self, value_fps: float):
        return value_fps * self.fps_to_mph_conversion
    
    def mph2_to_fps2(self, value_mph2: float):
        return value_mph2 * self.mph2_to_fps2_conversion
    
    def fps2_to_mph2(self, value_fps2: float):
        return value_fps2 * self.fps2_to_mph2_conversion
        
    def calculate_velocity(self):   # Tested as of 3/29/2025
        """Calculates the velocity of the vehicle based on the center point, heading point, and speed.
        
        Returns:
            float: Velocity of the vehicle in miles per hour.
        """
        direction = self.heading_point - self.center_point
        direction = direction / direction.norm()
        return direction * self.speed_fps
        
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
            Point(self.center_point.x - self.vehicle_width_ft / 2, self.center_point.y + self.vehicle_length_ft / 2),  # Front left
            Point(self.center_point.x + self.vehicle_width_ft / 2, self.center_point.y + self.vehicle_length_ft / 2),  # Front right
            Point(self.center_point.x + self.vehicle_width_ft / 2, self.center_point.y - self.vehicle_length_ft / 2),  # Back right
            Point(self.center_point.x - self.vehicle_width_ft / 2, self.center_point.y - self.vehicle_length_ft / 2),  # Back left
        ]
        return body        
        
    def update_position(self, steering_rad: float, acceleration_mph2: float, dt_sec: float, friction: float = 0.0): 
        """Updates the position of the vehicle based on the steering, acceleration and time step.
        
        Args:
            steering (float): Steering angle in radians.
            acceleration (float): Acceleration in miles per hour squared.
            dt_sec (float): Time step in seconds.
            friction (float): Friction, in Newtons, subtracted from the acceleration.
        """        
        # Updating acceleration by clipping by the vehicle max breaking and acceleration capabilities
        self.acceleration_fps2 = np.clip(self.mph2_to_fps2(acceleration_mph2), -self.max_breaking_fps2, self.max_acceleration_fps2)
        
        
        # v = vo + a t 
        new_speed = self.speed_fps + (self.acceleration_fps2 * dt_sec)
        
        # Update speed based on acceleration and clipping based on the vehicles speed capabilities
        # new_speed = np.clip(new_speed, self.min_speed_fps, self.max_speed_fps)
        
        # Calculate direction vector from center to heading point
        dx = self.heading_point.x - self.center_point.x
        dy = self.heading_point.y - self.center_point.y
        direction_vector = np.array([dx, dy])
        
        # Normalize the direction vector
        direction_vector = direction_vector / np.linalg.norm(direction_vector)

        # Apply steering: rotate the direction vector by the steering input
        angle_of_rotation = steering_rad  # Steering angle (radians)
        rotation_matrix = np.array([[np.cos(angle_of_rotation), -np.sin(angle_of_rotation)],
                                    [np.sin(angle_of_rotation), np.cos(angle_of_rotation)]])
        
        new_direction = np.dot(rotation_matrix, direction_vector)
        
        prev_center = self.center_point.values()
        # Takes into account acceleration over time
        displacement = (self.speed_fps * dt_sec) + (0.5 * self.acceleration_fps2 * (dt_sec ** 2))
        # Update position (move center point based on speed)
        self.center_point.x += new_direction[0] * displacement
        self.center_point.y += new_direction[1] * displacement 
        
        # Update heading point based on new direction
        self.heading_point.x = self.center_point.x + new_direction[0] * self.heading_offset_ft
        self.heading_point.y = self.center_point.x + new_direction[1] * self.heading_offset_ft
        
        self.speed_fps = new_speed
        # Adding the distance travelled in feet
        self.distance_travelled_ft += self.center_point.distanceTo(Point(prev_center[0], prev_center[1]))
        # Rebuilding the body after updating position
        self.body = self.build_body()
        
