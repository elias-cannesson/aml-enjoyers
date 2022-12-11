import os
import sys
from traceback import format_exc
from typing import *

import random
import pygame
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import copy

from collections import namedtuple, deque

from snake import Snake

import time

RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3

MOVE_STRAIGHT = [1, 0, 0]
MOVE_LEFT = [0, 1, 0]
MOVE_RIGHT = [0, 0, 1]

ClEANUP_SIZE = 2000

"""
Training:
Updating Q function with Bellman equation
Q(s, a) = r + gamma * max(Q(s', a')) where s' is next state and a' is next action

We want to minimize the Huber loss between the target and prediction Q values.

Loss = HuberLoss(r + gamma * max(Q(s', a')) - Q(s, a))

We calculate the loss and perform backpropagation to update the network over
a batch of experiences sampled from the replay buffer:

L = 1 / N * sum(HuberLoss(r + gamma * max(Q(s', a')) - Q(s, a)))
where
HuberLoss(x) = 0.5 * x^2 if |x| <= 1
                |x| - 0.5 otherwise

"""

__all__ = ["ReplayBuffer", "Agent"]

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayBuffer():
    def __init__(self, size: int) -> None:
        self.memory = deque([], maxlen=size)
    
    def add_experience(
        self, 
        state: Union[np.ndarray, torch.Tensor],
        action: torch.Tensor,
        next_state: Union[np.ndarray, torch.Tensor],
        reward: torch.Tensor,
        done: int
    ):
        self.memory.append(Transition(state, action, next_state, reward, done))
    
    def sample(
        self,
        batch_size: int
    ) -> Tuple:
        sample1 = random.sample(self.memory, batch_size)
        return sample1
    
    def cleanup(self):
        if len(self.memory) > ClEANUP_SIZE:
            for i in range(ClEANUP_SIZE):
                self.memory.popleft()

    def __len__(self):
        return len(self.memory)
    
    def get(self):
        return self.memory
    
# nn.Module class:
# https://pytorch.org/docs/stable/generated/torch.nn.Module.html

class DQN(nn.Module):
    def __init__(self, state_size: int, action_size: int, 
                hidden_size: int, seed: int) -> None:
        super(DQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # build network that maps states to actions

        x = F.relu(self.fc1(state))
        # second hidden layer with relu activation
        x = F.relu(self.fc2(x))
        # third hidden layer with Linear activation
        x = self.fc3(x)
        return x



class Agent:

    def __init__(
        self,
        state_size: int, # size of the state space
        action_size: int, # size of the action space
        hidden_size: int, # size of the hidden layer
        lr: float, # learning rate
        gamma: float, # discount factor
        epsilon: float, # epsilon for epsilon-greedy action selection
        # minimum value of epsilon
        batch_size: int, # size of the batch
        memory_size: int, # size of the replay buffer
        update_every: int, # how often to update the network
        device: str, # device to use for training, either 'cpu' or 'cuda'
        seed: Any, # random seed,
        epsilon_decay: float,
        epsilon_min: float
    ):
        self.state_size = state_size 
        self.action_size = action_size
        self.hidden_size = hidden_size

        self.lr = lr
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.batch_size = batch_size
        self.memory_size = memory_size
        self.update_every = update_every

        self.device = device

        self.seed = random.seed(seed)

        # Q-Networks to approximate Q-Value function for the given state
        self.qnetwork_local = DQN(state_size, action_size, hidden_size, seed)
        self.qnetwork_target = DQN(state_size, action_size, hidden_size, seed)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
    

        # Optimizer to update the weights of the local network via stochastic gradient descent
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        
        # Replay memory stores the experiences
        self.memory = ReplayBuffer(memory_size)
        self.t_step = 1

        self.loss = 0

        self.Q_expected = 0
        self.target = 0

    
    def save_model(self, path):
        torch.save(self.qnetwork_local.state_dict(), path)

    
    def act(
        self, 
        state: torch.Tensor,
    )-> int:
        # state = state.to(self.device)

        self.qnetwork_local.eval() # set the network to evaluation mode
        with torch.no_grad():
            action_values = self.qnetwork_local(state) # get predictions for all actions
        self.qnetwork_local.train() # set the network back to training mode
        action = [0, 0, 0] # straight, left, right
        # epsilon-greedy action selection

        if random.random() > self.epsilon:
            max_action_value = torch.argmax(action_values)
            max_action_value = max_action_value.item() # get the index of the highest predicted Q-value
           # print(action_values)
            action[max_action_value] = 1
            return action, action_values
            # return np.argmax(action_values.cpu().data.numpy()) 
            # return the action with the highest predicted Q-value
        else:
            action[random.randint(0, 2)] = 1
            return action, action_values 


    def train(
        self
    ): 
        if len(self.memory) < self.batch_size:
            return 0
        
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                self.learn_experiences_v2(experiences, self.batch_size)


    def replay_experiences(
        self
    ): 
        # train the q agent
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            size = self.batch_size
        else:
            experiences = self.memory.get()
            size = len(self.memory)
        self.learn_experiences_v2(experiences, size)
    

    def learn_experiences_v2(
        self,
        experiences: Tuple[torch.Tensor, Transition],
        batch_size: int
    ):  
    # --------------------------------------------------------------------------------------
    # convert experiences to tensors to feed to the torch models and optimizers
        states = []
        actions = []
        rewards = []
        next_states = []
        done = []
        for e in experiences:
            if e is not None:
                states.append(e.state)
                action = torch.Tensor(e.action)
                action = action.type(torch.int64)
                actions.append(action)
                rewards.append(e.reward)
                done.append(e.done)
                if e.done:
                    next_states.append(None)
                else:
                    next_states.append(e.next_state)
        states = torch.stack(states).to(self.device)
        actions = torch.stack(actions).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
    
    # --------------------------------------------------------------------------------------
    # get current q values (best action) for all actions in current states 
    # from the local model

        # Current model estimates
        state_action_values = self.qnetwork_local(states)
        
        # best state action values from the current model's estimates
        max_state_action_values = torch.stack(
            [torch.max(state_action_values[i]) for i in range(len(state_action_values))]
        )

    # --------------------------------------------------------------------------------------
    # Get max predicted Q values (for next states) from target model

        # since we only add a discount Q value for non final states, 
        # we need to know which states are non final
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, next_states)), 
            dtype=torch.bool).to(self.device)
        
        non_final_next_states = torch.stack(
            [s for s in next_states if s is not None]
        ).to(self.device) 

        # get the predicted Q values for the next states
        with torch.no_grad():
            next_q_values = self.qnetwork_target(non_final_next_states)
        
        next_state_values = torch.zeros(batch_size).to(self.device) # initialize to zeros 
        # (non-final next states will be updated)

        next_state_values[non_final_mask] = torch.max(next_q_values, dim=1)[0] 

        max_expected_state_action_values = next_state_values * self.gamma
        max_expected_state_action_values = max_expected_state_action_values + rewards

        try:
            self.optimizer.zero_grad()
            loss = F.mse_loss(max_state_action_values, max_expected_state_action_values)
            loss.backward()
            self.optimizer.step()

        except RuntimeError as e:
            print(e)
            print(format_exc())
            print("state_action_values: ", state_action_values.squeeze(), len(state_action_values), state_action_values.squeeze().shape)
            print(f"expected_state_action_values: {max_expected_state_action_values}, {len(max_expected_state_action_values)}, {max_expected_state_action_values.shape}")
            print("loss: ", loss)
            sys.exit()

        self.loss = loss.item() 

        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

        # update target network every update_every steps (load weights of local to target)
        if self.t_step % self.update_every == 0:
            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())


    def get_loss(self):
        return self.loss
    
    def remember(
        self, 
        state: Union[np.ndarray, torch.Tensor],
        action: int,
        next_state: Union[np.ndarray, torch.Tensor],
        reward: torch.Tensor,
        done: int
    ):
        self.memory.add_experience(state, action, next_state, reward, done)






