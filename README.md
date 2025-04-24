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

# CARLOS Companion Layout Creator Application (CLC)
The CLC allows you to 
