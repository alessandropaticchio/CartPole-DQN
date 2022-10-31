# Reinforcement Learning to solve the Cart Pole problem

This repository contains Pytorch code to train a Deep Q-Learning network that learns to balance
a CartPole.

Deep Q-Learning is a Reinforcement Learning algorithm that learns the optimal policy for a given problem, thanks to 
the help of a Deep Neural Network. 
The network will be our agent, operating in an environment (provided by [OpenAI Gym](https://github.com/openai/gym))
where it has to learn how to balance a pole on a cart.

## DQN Algorithm
In order to learn the optimal policy, namely the policy that keeps the policy in balance, our network
needs to approximate a function
![equation](https://latex.codecogs.com/svg.image?Q^*&space;=&space;State\text{&space;x&space;}Action&space;=>&space;\mathbb{R}),
such that we can easily pick a greedy policy:

![equation](https://latex.codecogs.com/svg.image?\pi^{*}(s)&space;=&space;argmax_a&space;\text{&space;}&space;Q^*(s,&space;a))

In order to learn such policy, we leverage the fact that every Q function respects the Bellman equation,
and hence we'll go minimizing:

![equation](https://latex.codecogs.com/svg.image?\delta&space;=&space;Q(s,a)&space;-&space;(r&space;&plus;&space;\gamma&space;max_a&space;Q(s',&space;a)))

## Policy Network and Target Network
When we compute the loss function, we need to compute the expected value of our next state ![equation](https://latex.codecogs.com/svg.image?V(s_{t&plus;1})&space;=&space;max_a&space;Q(s',&space;a)).
To do so, we use an older version of the Policy Network (the one we are currently training),
named Target Network. This adds stability to the learning procedure.

## Replay memory

[OpenAI Gym](https://github.com/openai/gym) helps us to generate as many episodes as we want.
We store every transition between states in a Replay Memory, where we sample from in the optimization phase.
In this way, at each optimization step, we observe transitions coming from different episodes.
It has been observed that this stabilizes training.

## How to run

I recommend setting the variable 

    num_episodes = 500+

in main.py to see meaningful results.


    pyenv install 3.9.14
    pyenv local 3.9.14
    poetry install
    python main.py
    


