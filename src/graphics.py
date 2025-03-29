import matplotlib.pyplot as plt
from lane import Lane
from simulation import Simulation
from vehicle import Vehicle
from environment import Environment
 
def plot_environment(environment: Environment): # Tested as of 3/29/2025
    """Plots the environment with its lane and vehicle."""
    lane = environment.lane
    plot_lane(lane)  # Plot the lane

def plot_lane(lane: Lane): # Tested as of 3/29/2025
    """Plots the lane with its center line, left edge and right edge."""
    ax = plt.gca()
    # Plot center line
    ax.plot(lane.center_line[:, 0], lane.center_line[:, 1], 'k--', label='Center Line')
    
    # Plot control points
    cx = []
    cy = []
    for p in lane.control_points:
        cx.append(p.x)
        cy.append(p.y)
    ax.plot(cx, cy, 'ro', label='Control Points')
    
    # Plot left edge
    ax.plot(lane.left_edge[:, 0], lane.left_edge[:, 1], 'b-', label='Left Edge')
    
    # Plot right edge
    ax.plot(lane.right_edge[:, 0], lane.right_edge[:, 1], 'm-', label='Right Edge')
    
def plot_vehicle(vehicle: Vehicle): # Tested as of 3/29/2025
    """Plots the vehicle with its body, center point and heading."""
    ax = plt.gca()
    # Plot vehicle body
    vehicle_body = vehicle.body
    vehicle_x = [p.x for p in vehicle_body]
    vehicle_y = [p.y for p in vehicle_body]
    ax.fill(vehicle_x, vehicle_y, 'g', label='Vehicle Body')
    
    # Plot vehicle center point
    ax.plot(vehicle.center_point.x, vehicle.center_point.y, 'bo', label='Vehicle Center')
    
    # Plot vehicle heading with arrow
    ax.quiver(vehicle.center_point.x, vehicle.center_point.y,
              vehicle.heading_point.x - vehicle.center_point.x,
              vehicle.heading_point.y - vehicle.center_point.y,
              angles='xy', scale_units='xy', scale=1, color='r', label='Heading')
    
def show(): # Tested as of 3/29/2025
    """Displays the plot."""
    plt.tight_layout()
    plt.legend()
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
    