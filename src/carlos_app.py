from point import Point
from vehicle import Vehicle
from lane import Lane
from environment import Environment
from simulation import Simulation
from sensor_array import SensorArray
import math
import layout_utils

# Lane Initialization:
lane_ctrl_points, lane_width, closed_loop = layout_utils.load_lane_from_file()
lane = Lane(control_points=lane_ctrl_points, lane_width=lane_width, closed_loop=closed_loop)

# Environment Initialization:
env = Environment(lane)

# Vehicle Initialization:
vehicle = Vehicle()
vehicle.setup(Point(0, 0), Point(1, 0), 0, 0)  # Placeholder for actual implementation

# Sensor Array Initialization:
num_sensors = 5
sensor_length = 10
sensor_angle_spread = math.pi / 4
sensor_array = SensorArray(num_sensors=num_sensors, sensor_length=sensor_length, sensor_angle_spread=sensor_angle_spread)

# Agent Initialization:
agent = None  # Placeholder for actual agent implementation

# Simulation Initialization:
dt = 0.1  # Time step in seconds
sim = Simulation(vehicle=vehicle, environment=env, agent=agent, dt=dt)


# Sim Execution:
sim.sim_step()
sim_status = sim.get_sim_status()
print(sim_status)  # Placeholder for actual implementation
sim.sim_reset(longitude=0.5, latitude=0.5, dir_angle_offset=0, speed=10)  # Placeholder for actual implementation
