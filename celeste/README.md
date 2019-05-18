# Celeste RL

This project attempts to train a RL agent for speedrunning the game [Celeste](http://www.celestegame.com/).

It is made possible by the [libTAS](https://github.com/clementgallet/libTAS) tool which we repurpose to hook up with a neural network controller running in python.

## Approach

### Environment

The initial goal is an agent that learns to play individual screens based purely on pixel data. 

Each screen will have a hard-coded target pixel position relative to some origin. 
Screen scrolls will be tracked to maintain an estimate of the character's position.
The character position will not be given as input to the network. It will only be used to determine if the agent reaches the target.

The state will be a combination of:

* The pixel data for the last n frames.
* (maybe) some representation of the current input state or history of inputs.

### Algorithm

We will start with an A2C agent.

The reward for an episode is 1 if the agent reaches the target location and 0 otherwise, further scaled inversely by the number of frames used.

The network predicts at each frame a sigmoid for each button, thresholded at 0.5 for pressed/released. 

Questions:

* Do we do fixed episode length? Or continue until death/target is reached with a cap?
* Should distance to target be included in the reward? How to weight this value?
* How to compute the advantage baseline?
* How is the discount factor determined?
* Is there a benefit/use for maintaining a replay buffer with A2C?
