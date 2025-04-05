from simulation import Simulation
from vehicle import Vehicle
from lane import Lane
from carlos_app import Agent
from environment import Environment
from sensor_array import SensorArray
import layout_utils
import math
import graphics
from point import Point
import numpy as np

def init_sim():
    lane_ctrl_points, lane_width, closed_loop = layout_utils.load_lane_from_file("./layouts/open_loop_0.txt")
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
    print("Simulation Test: Visualization of right edge reset NEEDS VISUAL CONFIRMATION.")
    
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
    print("Simulation Test: Visualization of left edge reset NEEDS VISUAL CONFIRMATION.")

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
    print("Simulation Test: Random reset visualization NEEDS VISUAL CONFIRMATION.")
    
def test_sim_step():
    sim = init_sim()
    sim.sim_reset(longitude=0.5, latitude=0.5, dir_angle_offset=0.0, speed=25.0)
    # sim.agent.sensors.update_sensors(sim.vehicle.center_point, sim.vehicle.heading_point)
    original_x = sim.vehicle.center_point.x
    original_y = sim.vehicle.center_point.y
    graphics.render_simulation(sim=sim)
    graphics.show(title="Before simulation step")
    sim.sim_step()
    graphics.render_simulation(sim=sim)
    assert sim.total_time_steps == 1, "Total time steps should be 1 after one step."
    assert sim.vehicle.center_point.x != original_x, "Vehicle should have moved after one step."
    assert sim.vehicle.center_point.y != original_y, "Vehicle should have moved after one step."
    graphics.show(title="After simulation step")
    print("Simulation Test: Simulation step PASSED.")
    
def test_sim_status():
    sim = init_sim()
    sim.sim_random_reset()
    sim_status = sim.get_sim_status()
    assert sim_status[0] == 0, "Total time steps should be 0 after initialization."
    assert sim_status[1] == True, "Vehicle should be in lane after initialization."
    assert sim_status[2] == True, "Vehicle should be in motion after initialization."
    sim.sim_step()
    sim_status = sim.get_sim_status()
    assert sim_status[0] == 1, "Total time steps should be 1 after one step."
    print("Simulation Test: Sim status update PASSED.")
    
def test_get_state():
    sim = init_sim()
    sim.sim_random_reset()
    state = sim.get_state()
    assert len(state) == 4, "State should contain 4 elements: heading x, heading y, speed, and list of detection distances."
    assert isinstance(state[0], float), "First element of state should be a float."
    assert isinstance(state[1], float), "Second element of state should be a float."
    assert isinstance(state[2], float), "Third element of state should be a float."
    assert isinstance(state[3], list), "Fourth element of state should be a list."
    assert len(state[3]) == 5, "Fourth element of state should be a list of length 5 for 5 detection distances."
    print("Simulation Test: State retrieval PASSED.")
    

def run_tests():
    test_sim_reset_right_edge()
    test_sim_reset_left_edge()
    test_sim_random_reset()
    test_sim_status()
    test_sim_step()
    test_get_state()
    print("Simulation Test: All tests PASSED.\n")
    

if __name__ == "__main__":
    run_tests()
    # test_sim_step()
    