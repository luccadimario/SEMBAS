from lane import Lane
from point import Point
from vehicle import Vehicle
from environment import Environment
import carlos_app as app
import graphics
import math
from sensor import Sensor

def test_init_lane():
    lane = app.init_lane("./layouts/train_straight_layout_0.txt")
    assert isinstance(lane, Lane), "Lane initialization failed."
    assert len(lane.control_points) > 0, "Lane control points are empty."
    assert lane.lane_width > 0, "Lane width is not positive."
    assert isinstance(lane.closed_loop, bool), "Lane closed loop parameter is not boolean."
    assert lane.control_points[0].x == 2.1387573383383085, "Lane control points do not match expected values."
    assert lane.control_points[0].y == 397.41216641207694, "Lane control points do not match expected values."
    assert lane.closed_loop == False, "Lane closed loop parameter does not match expected value."
    assert lane.lane_width == 12.0, "Lane width does not match expected value."
    print("Lane initialization test passed.")
    return lane  # Return the lane object for further testing
    
def test_init_environment():
    lane = app.init_lane("./layouts/train_straight_layout_0.txt")
    env = app.init_environment(lane)
    assert env is not None, "Environment initialization failed."
    assert isinstance(env.lane, Lane), "Environment lane is not a Lane object."
    print("Environment initialization test passed.")
    return env  # Return the environment object for further testing
    
def test_init_vehicle():
    center_point = Point(0, 0)
    heading_point = Point(0, 1)
    speed = 10.0
    vehicle = app.init_vehicle(center_point=center_point, heading_point=heading_point, speed=speed)
    assert vehicle is not None, "Vehicle initialization failed."
    assert hasattr(vehicle, 'center_point'), "Vehicle center point is not set."
    assert hasattr(vehicle, 'heading_point'), "Vehicle heading point is not set."
    assert hasattr(vehicle, 'speed'), "Vehicle speed is not set."
    assert vehicle.center_point.x == center_point.x, "Vehicle center point x-coordinate does not match."
    assert vehicle.center_point.y == center_point.y, "Vehicle center point y-coordinate does not match."
    assert vehicle.heading_point.x == heading_point.x, "Vehicle heading point x-coordinate does not match."
    assert vehicle.heading_point.y == heading_point.y, "Vehicle heading point y-coordinate does not match."
    assert vehicle.speed == speed, "Vehicle speed does not match."
    assert vehicle.body is not None, "Vehicle body is not built."
    assert vehicle.velocity is not None, "Vehicle velocity is not calculated."
    assert vehicle.body[0].x == -3.0, "Vehicle body points do not match expected values."
    assert vehicle.body[0].y == 6.0, "Vehicle body points do not match expected values."
    assert vehicle.velocity.x == 0.0, "Vehicle velocity does not match expected value."
    assert vehicle.velocity.y == 10.0, "Vehicle velocity does not match expected value."
    print("Vehicle initialization test passed.")
    return vehicle  # Return the vehicle object for further testing

def test_init_sensor_array():
    """Test function to initialize the sensor array."""
    sensor_array = app.init_sensor_array(num_sensors=5, sensor_length=50, sensor_angle_spread=math.pi / 2)
    assert sensor_array is not None, "Sensor array initialization failed."
    assert sensor_array.num_sensors == 5, "Sensor array number of sensors does not match."
    assert sensor_array.sensor_angle_spread == math.pi / 2, "Sensor array angle spread does not match."
    assert len(sensor_array.sensors) == 5, "Sensor array sensors list length does not match."
    assert all(isinstance(sensor, Sensor) for sensor in sensor_array.sensors), "Sensor array sensors are not Sensor objects."
    assert all(sensor.sensor_length == 50 for sensor in sensor_array.sensors), "Sensor array sensors length does not match."
    assert sensor_array.sensors[0].angle_offset == -math.pi / 4, "Sensor array first sensor angle does not match."
    assert sensor_array.sensors[4].angle_offset == math.pi / 4, "Sensor array last sensor angle does not match."
    assert sensor_array.sensors[2].angle_offset == 0, "Sensor array middle sensor angle does not match."
    test_sensor()
    print("Sensor array initialization test passed.")
    return sensor_array  # Return the sensor array object for further testing

def test_sensor():
    s = Sensor(sensor_length=50, angle_offset=-math.pi / 4)
    origin = Point(0, 0)
    direction = Point(-1, 1)
    s.update_sensor(origin, direction)
    assert s.origin_point == origin, "Sensor origin point does not match expected value."
    assert s.angle_offset == -math.pi / 4, "Sensor angle offset does not match expected value."
    assert s.end_point.x == 0, f"Sensor end point x-coordinate {s.end_point.x} does not match expected value."
    assert s.end_point.y == 50, "Sensor end point y-coordinate does not match expected value."
    print("Single sensor test passed.")
    
def sensor_tests():
    sensor_array = test_init_sensor_array()
    test_sensor()
    center_point = Point(0, 0)
    sensor_array.update_sensors(center_point, Point(0, 1))  # Update sensor positions based on vehicle
    assert sensor_array.sensors[0].origin_point == center_point, "Sensor origin point does not match vehicle center point."
    assert sensor_array.sensors[2].end_point.x == 0, "Sensor end point x-coordinate does not match expected value."
    print("Sensor tests passed.")

def test_graphics():
    env = test_init_environment()
    vehicle = test_init_vehicle()
    sensor_array = test_init_sensor_array()
    graphics.plot_environment(env)  # Plot the environment
    graphics.plot_vehicle(vehicle)  # Plot the vehicle
    sensor_array.update_sensors(vehicle.center_point, vehicle.heading_point)  # Update sensor positions based on vehicle
    graphics.plot_sensors(sensor_array)
    graphics.show()  # Show the plot
    
#### Main Test Function ####
def main_test():
    test_init_lane()
    test_init_environment()
    test_init_vehicle()
    
# main_test()
test_graphics()
# sensor_tests()
    
