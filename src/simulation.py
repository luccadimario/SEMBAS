from point import Point
from vehicle import Vehicle
from environment import Environment
from sensors import SensorArray
import random
import numpy as np

class Simulation:
    def __init__(self, vehicle: Vehicle, environment: Environment, agent, dt: float = 0.1):
        """Represents a simulation of a vehicle in an environment. The simulation is represented by a vehicle object, environment object and agent object.

        Args:
            vehicle (Vehicle): Vehicle object representing the vehicle.
            environment (Environment): Environment object representing the environment.
            agent (Agent): Agent object representing the agent.
            dt (float): Time step for the simulation in seconds.
        """
        self.vehicle = vehicle
        self.environment = environment
        self.agent = agent
        self.dt = dt
        self.reset_sim_status()
        
    def reset_sim_status(self) -> None:
        """Resets the simulation status."""
        self.total_time_steps = 0
        self.vehicle_in_lane = True
        self.vehicle_in_motion = True
        
    def sim_reset(self, longitude: float, latitude: float, dir_angle_offset: float, speed: float) -> None:
        """Resets the simulation vehicle based on the given longitude, latitude, direction angle offset and speed.
        Also resets the simulation statuses.

        Args:
            longitude (float): Longitude of the vehicle placement on the lane from the start [0] to the end [1].
            latitude (float): Latitude of the vehicle placement in the vehicle from left [0] to right [1].
            dir_angle_offset (float): Direction angle offset of the vehicle in radians from the center of the lane.
            speed (float): Speed of the vehicle in miles per hour.
        """
        self.reset_sim_status()
        center_point, heading_point = self.environment.position_from_coordiantes(longitude=longitude, latitude=latitude, angle_offset=dir_angle_offset)
        self.vehicle.vehicle_setup(center_point, heading_point, speed)
        self.agent.sensors.update_sensors(self.vehicle.center_point, self.vehicle.heading_point)
        
    def sim_random_reset(self, speed_range: list[float]=[0,75]):
        longitude = random.uniform(0, 1)
        latitude = random.uniform(0, 1)
        dir_angle_offset = random.uniform(-np.pi, np.pi)
        speed = random.uniform(speed_range[0], speed_range[1])
        center_point, heading_point = self.environment.position_from_coordiantes(longitude=longitude, latitude=latitude, angle_offset=dir_angle_offset)
        self.vehicle.vehicle_setup(center_point=center_point, heading_point=heading_point, speed_mph=speed)
        
    def get_state(self) -> list[Point, float, list[float]]:
        """
        Returns the current state of the simulation, including the vehicle's heading, speed, and the sensor detections.
        """
        # Placeholder for actual implementation
        sensor_detections = self.agent.sensors.sense(self.environment, self.vehicle)
        return [self.vehicle.heading_point.x, self.vehicle.heading_point.y, self.vehicle.speed, sensor_detections]
    
    def sim_step(self) -> None:
        """Executes a single step in the simulation.
        1. Updates the sensors based on the vehicle's position and heading.
        2. Gets the current state of the simulation.
        3. Gets the action from the agent based on the current state.
        4. Updates the vehicle's position based on the action.
        5. Updates the simulation status.
        """
        self.agent.sensors.update_sensors(self.vehicle.center_point, self.vehicle.heading_point)
        
        state = self.get_state()
        
        steering, accleration = self.agent.decide(state)
        
        self.vehicle.update_position(steering, accleration, self.dt)
        
        self.update_sim_status()
        
    def update_sim_status(self) -> None:
        """
        Updates the simulation status based on the vehicle's position and heading.
        """
        self.total_time_steps += 1
        self.vehicle_in_lane = self.environment.point_in_lane(self.vehicle.center_point)
        self.vehicle_in_motion = self.vehicle.speed > 0
        
    def get_sim_status(self) -> tuple[float, bool, bool]:
        """Returns the current simulation status: the total time steps, vehicle in lane status, vehicle in motion status.
        """
        return self.total_time_steps, self.vehicle_in_lane, self.vehicle_in_motion
        
        
        
        
        
    