from sensor_array import SensorArray
from environment import Environment
from vehicle import Vehicle



class Agent:
    """Parent class for the agent. The agent is responsible for making decisions based on sensor data and the environment.
    The agent is represented by a sensor array and a decision-making function."""
    
    def __init__(self, sensor_array: SensorArray):
        self.sensors = sensor_array

    def decide(self, state):
        """throws an error if not implemented in child class."""
        raise NotImplementedError("decide() method not implemented in child class.")
    
    def sense(self, env: Environment, vehicle: Vehicle):
        """Senses the environment using the sensors in the array. The sensors are updated based on the vehicle's position and heading.
        Args:
            env (Environment): Environment object representing the environment.
            vehicle (Vehicle): Vehicle object representing the vehicle.
        """
        sensor_data = self.sensor_array.sense(env, vehicle)
        return sensor_data
    
class SimpleAgent(Agent):
    """Simple agent that makes decisions based on the sensor data."""
    
    def __init__(self, sensor_array: SensorArray):
        super().__init__(sensor_array)
    
    def decide(self, state):
        """Makes a decision based on the sensor data."""
        # Implement your decision-making logic here
        return 1.0, 1.0