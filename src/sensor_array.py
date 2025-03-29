from sensor import Sensor
from point import Point
from vehicle import Vehicle
from environment import Environment
import math
import numpy as np

class SensorArray:
    def __init__(self, num_sensors: int, sensor_length: float, sensor_angle_spread: float):
        """Represents an array of sensors in the environment. The sensor array is represented by a list of sensor objects.

        Args:
            num_sensors (int): Number of sensors in the array.
            sensor_length (float): Length of each sensor in feet.
            sensor_angle_spread (float): Angle spread of the sensors in radians.
        """
        self.num_sensors = num_sensors
        self.sensor_angle_spread = sensor_angle_spread
        self.setup_sensors(sensor_length)
        
    def setup_sensors(self, sensor_length: float):
        """Sets up the sensors in the array based on the number of sensors, sensor length and angle spread.
        Args:
            sensor_length (float): Length of each sensor in feet.
        """
        angle_max = self.sensor_angle_spread / 2
        angles = np.linspace(-angle_max, angle_max, self.num_sensors)
        self.sensors = [Sensor(sensor_length=sensor_length, angle_offset=angle) for angle in angles]
        
    def update_sensors(self, origin_point: Point, direction_vector: Point):
        """Updates the sensors in the array based on the given origin point and direction vector.

        Args:
            origin_point (Point): Point object representing the origin of the sensor array.
            direction_vector (Point): Point object representing the direction vector of the sensor array.
        """
        pass
    
    def sense(self, env: Environment, vehicle: Vehicle):
        """Senses the environment using the sensors in the array. The sensors are updated based on the vehicle's position and heading.

        Args:
            env (Environment): Environment object representing the environment.
            vehicle (Vehicle): Vehicle object representing the vehicle.
        """
        pass
    
        
            
