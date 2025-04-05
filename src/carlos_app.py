from point import Point
from vehicle import Vehicle
from lane import Lane
from environment import Environment
from simulation import Simulation
from sensor_array import SensorArray
import math
import layout_utils


# PLACEHOLDER AGENT CLASS FOR TESTING
class Agent:
    def __init__(self, sensor_array: SensorArray):
        self.sensors = sensor_array

    def decide(self, state):
        steering = 0.0
        acceleration = 0.0
        return steering, acceleration
    
    def sense(self, env: Environment, vehicle: Vehicle):
        """Senses the environment using the sensors in the array. The sensors are updated based on the vehicle's position and heading.
        Args:
            env (Environment): Environment object representing the environment.
            vehicle (Vehicle): Vehicle object representing the vehicle.
        """
        sensor_data = self.sensor_array.sense(env, vehicle)
        return sensor_data


#### Lane Initialization ####
def init_lane(file_path: str = None): # Tested as of 3/29/2025
    """Initializes a lane object based on the given file path. If no file path is provided, a file dialog will be opened to select the file.
    The file should contain the control points, lane width and closed loop parameter."""
    lane_ctrl_points, lane_width, closed_loop = layout_utils.load_lane_from_file(file_path)
    lane = Lane(control_points=lane_ctrl_points, lane_width=lane_width, closed_loop=closed_loop)
    return lane


#### Environment Initialization ####
def init_environment(lane): # Tested as of 3/29/2025
    """Initializes an environment object based on the given lane object."""
    env = Environment(lane)
    return env


#### Vehicle Initialization ####
def init_vehicle(center_point, heading_point, speed): # Tested as of 3/29/2025
    """Initializes a vehicle object based on the given center point, heading point and speed."""
    vehicle = Vehicle()
    vehicle.vehicle_setup(center_point=center_point, heading_point=heading_point, speed_mph=speed)
    return vehicle


#### Sensor Array Initialization ####
def init_sensor_array(num_sensors: int = 5, sensor_length: float = 10, sensor_angle_spread: float = math.pi / 2): # Tested as of 3/29/2025
    """Initializes a sensor array object based on the given number of sensors, sensor length and angle spread."""
    sensor_array = SensorArray(num_sensors=num_sensors, sensor_length=sensor_length, sensor_angle_spread=sensor_angle_spread)
    return sensor_array


#### Agent Initialization ####
def init_agent(sensor_array: SensorArray):
    agent = Agent(sensor_array)  # Placeholder for actual agent implementation
    return agent


#### Simulation Initialization ####
def init_simulation(vehicle: Vehicle, env: Environment, agent, dt: float):
    sim = Simulation(vehicle=vehicle, environment=env, agent=agent, dt=dt)
    return sim


#### Sim Execution ####
def execute_simulation(sim: Simulation):
    sim.sim_step()
    sim_status = sim.get_sim_status()
    return sim_status


def main():
    vehicle = init_vehicle(Point(0, 0), Point(0, 1), 10.0)
    lane = init_lane("./layouts/train_straight_layout_0.txt")
    env = init_environment(lane)
    sensor_array = init_sensor_array()
    agent = init_agent(sensor_array)
    sim = init_simulation(vehicle, env, agent, dt=0.1)



    
    
    
