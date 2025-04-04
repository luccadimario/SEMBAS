from simulation import Simulation
from vehicle import Vehicle
from lane import Lane
from carlos_app import Agent
from environment import Environment
from sensors import SensorArray
import layout_utils
import math
import graphics
from point import Point
import numpy as np



def init_sim():
    lane_ctrl_points, lane_width, closed_loop = layout_utils.load_lane_from_file("./layouts/train_straight_layout_0.txt")
    lane = Lane(control_points=lane_ctrl_points, lane_width=lane_width, closed_loop=closed_loop)
    env = Environment(lane)
    v = Vehicle()
    sensor_arr = SensorArray(num_sensors=5, sensor_length=50, sensor_angle_spread=math.pi)
    agent = Agent(sensor_array=sensor_arr)
    return Simulation(vehicle=v, environment=env, agent=agent)

def test_sim_reset_right_edge():
    sim = init_sim()
    longitude = 0.0
    latitude = 0.5
    speed = 25.0
    dir_angle_offset = 0.0
    sim.sim_reset(longitude=longitude, latitude=latitude, dir_angle_offset=dir_angle_offset, speed=speed)
    graphics.render_simulation(sim=sim)
    graphics.show("Initial reset")
    sim.sim_reset(longitude=0.5, latitude=1.0, dir_angle_offset=math.pi/4, speed=45.0)
    graphics.render_simulation(sim=sim)
    graphics.show("Vehicle .5 down lane, right edge, 45 deg angle towards center")
    
def test_sim_reset_left_edge():
    sim = init_sim()
    longitude = 0.0
    latitude = 0.5
    speed = 25.0
    dir_angle_offset = 0.0
    sim.sim_reset(longitude=longitude, latitude=latitude, dir_angle_offset=dir_angle_offset, speed=speed)
    graphics.render_simulation(sim=sim)
    graphics.show("Initial reset")
    sim.sim_reset(longitude=0.5, latitude=0.0, dir_angle_offset=math.pi/4, speed=45.0)
    graphics.render_simulation(sim=sim)
    graphics.show("Vehicle .5 down lane, left edge, 45 deg angle towards center")

def test_sim_random_reset():
    sim = init_sim()
    longitude = 0.0
    latitude = 0.5
    speed = 25.0
    dir_angle_offset = 0.0
    sim.sim_reset(longitude=longitude, latitude=latitude, dir_angle_offset=dir_angle_offset, speed=speed)
    graphics.render_simulation(sim=sim)
    graphics.show(title="Before random reset")
    sim.sim_random_reset()
    graphics.render_simulation(sim=sim)
    graphics.show(title="After random reset")
    
def test_sim_step():
    sim = init_sim()

def test_sim_status():
    sim = init_sim()

def run_tests():
    test_sim_reset_right_edge()
    test_sim_reset_left_edge()
    test_sim_random_reset()
    

if __name__ == "__main__":
    run_tests()