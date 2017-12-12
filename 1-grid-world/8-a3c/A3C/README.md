# Tensorflow implementation of A3C for a 2D grid environment

## 1) Environment description  

The agent (red in color) should navigate to the purple circle (reward) by avoiding the moving obstacles (green triangles). 
The reward location is randomly placed in the 2D grid environment at every episode. The agent generalizes well to unseen reward locations as well as unseen grid sizes at test time.  

![1](https://github.com/akileshbadrinaaraayanan/A3C_grid_world/raw/master/img/sample.png)

## 2) Code organization
### A3C_tensorflow directory 

For train time, the below files are used:

grid_env_r1.py contains the implementation of A3C algorithm.

environment_a3c_r1.py is the code for 2D environment.

For test time, the below files are used:

grid_env_test.py - loads the saved model.

environment_a3c_load_weights.py : game logic for 2D environment.

renderenv_load_weights.py : To monitor how the agent behaves by rendering. 

### A3C_complex_environment directory

The environment is more complex in this case with more number of obstacles as well as obstacles that move in both the directions (left-to-right and right-to-left).
For train time, the below files are used:

grid_env_r3.py contains the implementation of A3C algorithm.

environment_a3c_r3.py is the code for 2D environment.

For test time, the below files are used:

grid_env_test.py - loads the saved model.

environment_a3c_load_weights.py : game logic for 2D environment.

renderenv_load_weights.py : To monitor how the agent behaves by rendering.

In both these cases, a non-uniform reward decay is used for the convergence of Reinforcement Learning (RL) agent.

## 3) How to train?
```
CUDA_VISIBLE_DEVICES="" python grid_env_r1.py
```
If you want to just test with pre-trained model (stored inside models directory). Rendering is also enabled at test time.
```
CUDA_VISIBLE_DEVICES="" python grid_env_test.py
```
## 4) Acknowledgement
The basic environment code is based on grid world environment [here](https://github.com/rlcode/reinforcement-learning/tree/master/1-grid-world)
