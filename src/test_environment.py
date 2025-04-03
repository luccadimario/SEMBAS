from vehicle import Vehicle
from lane import Lane
from environment import Environment
from point import Point
import graphics as G

def test_position_from_coordinates():
    control_points = [Point(100,100), Point(300,300)]
    closed_loop = False
    lane = Lane(control_points=control_points, lane_width=12, closed_loop=closed_loop)
    env = Environment(lane=lane)
    longitude = 0.0
    latitude = 0.5
    angle_offset = 0.0
    center, heading = env.position_from_coordiantes(longitude, latitude, angle_offset)
    v = Vehicle()
    v.vehicle_setup(center_point=center, heading_point=heading, speed_mph=25.0)
    G.plot_environment(env)
    G.plot_vehicle(v)
    G.show()
    
def run_tests():
    test_position_from_coordinates()
    
if __name__ == "__main__":
    run_tests()