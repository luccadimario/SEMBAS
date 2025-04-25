# SEMBAS-RL
Applying SEMBAS to training Reinforcement Learning algorithms.


# Setting up CARLOS:
These instructions are in linux.

1. Clone main branch

2. Install dependencies

To install the dependencies, run the following command from the main SEMBAS-RL folder:
```
pip install -r requirements
```
These need to have installed with no issues for the test of the steps.

4. From the src folder, run the "test_app.py" file. 
```
python test_app.py
```
or
```
py test_app.py
```
The output will be in the console and several plots will be displayed. 

See "Output from test_app.pdf" for what is displayed and in what order. If you get to "App Test: All tests PASSED" then it means everything is working how it should.

## Testing CARLOS Execution
Run the "carlos_app_pres_agent.py" from the src folder with the following command:
```
python carlos_app_pres_agent.py
```
You should see a plot show and the vehicle moving on the lane. If the vehicle leaves the lane, it will be randomly reset back on the lane and will continue. Its an endless loop. Use ctrl+C to escape.

# Initializing CARLOS Components
The following describes how to initialize each of the components in CARLOS:

## Lane -> located in lane.py
**_Requires: List of control points (```Point``` objects from ```point.py```), closed loop boolean flag, lane width as a float._**

The control points are the only necessary parameter. The lane width is defaulted to ```12.0``` and the closed loop flag is set to ```False``` (meaning it is not a continuous lane).

These can be hard coded such as:
```python
control_points = [Point(0, 0), Point (100, 100)]
lane_width = 11.0
closed_loop = False
```
OR you can load an existing lane from a file using the ```layout_utils.py``` function called ```load_lane_from_file```.
The method takes in a file path. The default is the file_path is None, meaning the user will select a file from the file system using the file explorer.
If a file path is provided and the file contains a valid layout, the method returns the control points, lane width, and closed loop flag.

Use the following command: 
```python
control_points, lane_width, closed_loop = layout_utils.load_lane_from_file(file_path)
```

Create the lane object with the following line:
```python
new_lane = Lane(control_points, lane_width, closed_loop)
```

## Environment -> located in environment.py
**_Requires: An Lane object_**
1. Create a lane object as described above.
2. Create the environment object:
```python
new_environment = Environment(new_lane)
```

## Vehicle -> located in vehicle.py
The vehicle has the following _optional_ parameters, with the types and default values as listed, in this order:
- ```vehicle_length_ft: float = 10``` -> Length of the vehicle in feet
- ```vehicle_width_ft: float = 6``` -> Width of the vehicle in feet
- ```min_speed_mph: float = 0.0``` -> THe minimum possible speed for the vehicle in miles per hour
- ```max_speed_mph: float = 150.0``` -> The maximum possible speed for the vehicle in miles per hour
- ```sec_0_to_60: float = 8.0``` -> How many seconds it takes for the vehicle to go 0 to 60 miles per hour

Initialize the vehicle with the following:
```python
new_vehicle = Vehicle(<insert parameters>)
```

## Sensor Array -> located in sensor_array.py
**_REQUIRED FOR AN INSTANCE OF AN AGENT_**
**_Requires: integer number of sensors, sensor length (float), and spread of the sensors as an angle in radians (float)_**
(Note: if youre confused, look at the paper, it has pictures)

Initialize the sensor array with the following:
```python
new_sensorarray = SensorArray(num_sensors, sensor_length, sensor_angle_spread)
```

## Agent -> parent located in agent.py
Every agent needs to be a child of the ```Agent``` class and must implement the following:
```python
decide(state) -> tuple(float, float)
compute_reward(state: Tensor, in_lane: bool, in_motion: bool) -> float
```
```decide``` must return a tuple containing the steering and acceleration for the vehicle
```compute_reward``` takes in the state as a tensor, the boolean for if the vehicle is in the lane and in motion and returns the reward value associated.

### Agent Types
The ```SimpleAgent``` does nothing but return ```steering``` of ```1.0``` (left turn) and ```acceleration``` of ```1.0```.


```PresentationAgent``` (located in ```presentation_agent.py``` returns ```steering``` as the angle offset of the sensor whose detection distance is the furthest and an ```acceleration``` of ```0.5```.

**_!!! This agent is used for testing carlos execution in the setup portion of this document !!!_**

## Simulation -> located in simulation.py
**_Requires: A Vehicle object, an Environment object, an Agent object, and a time step (float) for how long each simualtion step represents in seconds_**

Initialize the simulation with:
```python
sim = Simulation(vehicle, environment, agent, dt)
```
Note: ```dt``` is the time step and is defaulted to ```0.1```

# Implementing your own agent

# Implementing your own CARLOS Application/Execution script

# CARLOS Companion Layout Creator Application (CLC)
The CLC allows you to 
