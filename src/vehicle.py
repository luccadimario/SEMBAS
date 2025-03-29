from point import Point
import numpy as np

class Vehicle:
    def __init__(self):
        pass
    
    def vehicle_setup(self, center_point: Point, heading_point: Point, speed: float, length: float = 12, width: float = 6):     # Tested as of 3/29/2025
        """Sets up the vehicle based on the given center point, heading point and speed.
        
        Args:
            center_point (Point): Point object representing the center of the vehicle.
            heading_point (Point): Point object representing the heading of the vehicle.
            speed (float): Speed of the vehicle in miles per hour.
        """
        self.center_point = center_point
        self.heading_point = heading_point
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
        speed = self.speed
        heading = self.heading_point
        # Kinematic bicycle model dynamics based on
        # "Kinematic and Dynamic Vehicle Models for Autonomous Driving Control Design" by
        # Jason Kong, Mark Pfeiffer, Georg Schildbach, Francesco Borrelli
        
        # distance between the rear wheels and the center of mass. This is needed to implement the kinematic bicycle model dynamics
        # lr = self.length / 2.0
        # lf = lr  # we assume the center of mass is the same as the geometric center of the entity
        
        # beta = np.arctan((lr / (lf + lr)) * np.tan(steering))
        
        # new_acceleration = acceleration - friction
        
        # # new_speed = np.clip(
        # #     speed + new_acceleration * dt, self.min_speed, self.max_speed
        # # )
        # new_speed = speed + new_acceleration * dt
        
        # new_heading = heading + (speed / lr) * np.sin(beta) * dt

        # angle = (heading + new_heading) / 2.0 + beta

        # new_center = (
        #     self.center_point + (speed + new_speed) * Point(np.cos(angle), np.sin(angle)) * dt / 2.0
        # )

        # new_velocity = Point(
        #     new_speed * np.cos(new_heading), new_speed * np.sin(new_heading)
        # )
        
        # self.distance_travelled += self.center_point.distanceTo(new_center)
        # self.center_point = new_center
        # self.heading_point = np.mod(new_heading, 2 * np.pi)  # wrap the heading angle between 0 and +2pi
        # self.velocity = new_velocity
        # self.speed = new_speed
        # self.acceleration = new_acceleration
        # self.body = self.build_body()
        
def test_vehicle():
    v = Vehicle()
    center_point = Point(0, 0)
    heading_point = Point(0, 1)
    speed = 10.0
    v.vehicle_setup(center_point=center_point, heading_point=heading_point, speed=speed)
    test_no_movement(v, center_point)
    v.update_position(steering=0.0, acceleration=0.10, dt=1.0)
    assert v.center_point.x == 0, "Vehicle center point x-coordinate should stay the same since vehicle is going straight."
    assert v.center_point.y != center_point.y, "Vehicle center point y-coordinate should have changed."
    assert v.center_point.y == 10, "Vehicle center point y-coordinate should have changed."
    
    
def test_no_movement(v, center):
    acceleration = 0.0
    steering = 0.0
    dt = 1.0
    v.update_position(steering, acceleration, dt)
    assert v.center_point.x == center.x, "Vehicle center point x-coordinate does not match expected value."
    assert v.center_point.y == center.y, "Vehicle center point y-coordinate does not match expected value."
    print("Vehicle works when acceleration is 0")
    
    
if __name__ == "__main__":
    test_vehicle()
    print("Vehicle test passed.")