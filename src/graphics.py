import matplotlib.pyplot as plt
from lane import Lane
from simulation import Simulation
from vehicle import Vehicle
from environment import Environment
from point import Point
import numpy as np

def list_points_as_values(point_list: list[Point]) -> tuple[list[float], list[float]]:
    """Takes in a list of points and returns 2 lists: x values and y values.
    
    Args:
        point_list (list[Point]): List of Point objects.
        
    Returns:
        tuple(list[float], list[float]): X values and Y values of the given list of points.
    """
    x = []
    y = []
    for p in point_list:
        x.append(p.x)
        y.append(p.y)
    return x, y
 
def plot_environment(environment: Environment): # Tested as of 3/29/2025
    """Plots the environment with its lane and vehicle."""
    lane = environment.lane
    plot_lane(lane)  # Plot the lane

def plot_lane(lane: Lane): # Tested as of 3/29/2025
    """Plots the lane with its center line, left edge and right edge."""
    ax = plt.gca()
    
    # Plot center line
    center_x, center_y = list_points_as_values(lane.center_line)
    ax.plot(center_x, center_y, 'k--', label='Center Line')
    
    # Plot control points
    ctrl_x, ctrl_y = list_points_as_values(lane.control_points)
    ax.plot(ctrl_x, ctrl_y, 'ro', label='Control Points')
    
    # Plot left edge
    left_x, left_y = list_points_as_values(lane.left_edge)
    ax.plot(left_x, left_y, 'b-', label='Left Edge')
    
    # Plot right edge
    right_x, right_y = list_points_as_values(lane.right_edge)
    ax.plot(right_x, right_y, 'm-', label='Right Edge')
    
    ax.fill(
        np.concatenate((left_x, right_x[::-1])),
        np.concatenate((left_y, right_y[::-1])),
        color="gray",
        alpha=0.5,
        label="Lane Area",
    )
    
    
def plot_vehicle(vehicle: Vehicle): # Tested as of 3/29/2025
    """Plots the vehicle with its body, center point and heading."""
    ax = plt.gca()
    # Plot vehicle body
    body_x, body_y = list_points_as_values(vehicle.body.corners)
    ax.fill(body_x, body_y, 'b', label='Vehicle Body')
    
    # Plot vehicle center point
    ax.plot(vehicle.center_point.x, vehicle.center_point.y, 'ro', label='Vehicle Center')
    
    # Plot vehicle heading with arrow
    ax.quiver(vehicle.center_point.x, vehicle.center_point.y,
              vehicle.heading_point.x - vehicle.center_point.x,
              vehicle.heading_point.y - vehicle.center_point.y,
              angles='xy', scale_units='xy', scale=0.1, color='g', label='Heading')
    
def plot_sensors(sensor_array): # Tested as of 3/29/2025
    """Plots the sensors in the sensor array."""
    ax = plt.gca()
    label = 'Sensor'
    for sensor in sensor_array.sensors:
        ax.plot([sensor.origin_point.x, sensor.end_point.x], [sensor.origin_point.y, sensor.end_point.y], 'r--', label=label)
        label = None
    
def show(): # Tested as of 3/29/2025
    """Displays the plot."""
    plt.tight_layout()
    plt.legend()
    plt.xlim(0, 400)
    plt.ylim(0, 400)
    plt.show()
    
def render(sim: Simulation):
    """Renders the simulation by plotting the lane and vehicle."""
    plt.clf()  # Clear the current figure
    env = sim.environment
    plot_environment(env)  # Plot the environment
    vehicle = sim.vehicle
    plot_vehicle(vehicle)  # Plot the vehicle
    plt.title("Simulation Visualization")
    show()
    