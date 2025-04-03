import graphics
from point import Point
from sensors import Sensor, SensorArray
import matplotlib.pyplot as plt
import math

def test_calc_end_point():
    s = Sensor(sensor_length=50, angle_offset=math.pi/2)
    s.update_sensor(Point(200,200), Point(200,210))
    assert s.end_point.x == 150, "Sensor end point X does not match expected value."
    assert s.end_point.y == 200, "Sensor end point Y does not mathc expected value."
    print("End point calculated correctly")
    # graphics.plot_sensor(s)
    # ax = plt.gca()
    # ax.plot([200, 200], [200, 210], 'g-')
    # graphics.show([100, 300], [100, 300])
    
def test_angle_offsets():
    sa = SensorArray(num_sensors=5, sensor_length=50, sensor_angle_spread=math.pi)
    assert all(sensor.sensor_length==50 for sensor in sa.sensors), "Sensors length was not set correctly."
    assert sa.sensors[0].angle_offset == math.pi / 2, "Angle offset for first sensor to the left not correct."
    assert sa.sensors[2].angle_offset == 0.0, "Angle offset for the middle sensor is not correct"
    assert sa.sensors[4].angle_offset == -math.pi / 2, "Angle offset for the last sensor to the right not correct."
    print("Angle offsets calculated correctly")
    
def run_tests():
    test_calc_end_point()
    test_angle_offsets()
    
if __name__ == "__main__":
    run_tests()