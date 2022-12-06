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
import torchvision.transforms as T
import numpy as np

from collections import namedtuple, deque

from snake import Snake

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
        action: int,
        next_state: Union[np.ndarray, torch.Tensor],
        reward: torch.Tensor,
        done: int
    ):
        self.memory.append(Transition(state, action, next_state, reward, done))
    
    def sample(
        self,
        batch_size: int
    ) -> Tuple:
        return random.sample(self.memory, batch_size)

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
        # this method defines the forward pass of the network
        #print(type(state), state.shape, state)
        #state = state.view(-1, 11)
        # first hidden layer with relu activation in order
        # to introduce non-linearity
        # input to first hidden layer is state
        x = F.relu(self.fc1(state))
        # second hidden layer with relu activation
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# class DQN(nn.Module):
#     def __init__(self, state_size: int, action_size: int, 
#                 hidden_size: int, seed: int) -> None:
#         super(DQN, self).__init__()
#         self.seed = torch.manual_seed(seed)
#         self.fc1 = nn.Linear(state_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, action_size)

#     def forward(self, state: torch.Tensor) -> torch.Tensor:
#         # this method defines the forward pass of the network
#         #print(type(state), state.shape, state)
#         #state = state.view(-1, 11)
#         # first hidden layer with relu activation in order
#         # to introduce non-linearity
#         # input to first hidden layer is state
#         x = F.relu(self.fc1(state))
#         # second hidden layer with relu activation
#         x = self.fc2(x)
#         return self.fc2(x)


class Agent:

    def __init__(
        self,
        state_size: int, # size of the state space
        action_size: int, # size of the action space
        hidden_size: int, # size of the hidden layer
        lr: float, # learning rate
        gamma: float, # discount factor
        epsilon: float, # epsilon for epsilon-greedy action selection
        epsilon_decay: float, # decay rate for epsilon
        epsilon_min: float, # minimum value of epsilon
        batch_size: int, # size of the batch
        memory_size: int, # size of the replay buffer
        update_every: int, # how often to update the network
        device: str, # device to use for training, either 'cpu' or 'cuda'
        seed: Any, # random seed
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
        self.qnetwork_local = DQN(state_size, action_size, hidden_size, seed).to(device)

        # Optimizer to update the weights of the local network via stochastic gradient descent
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        
        # Replay memory stores the experiences
        self.memory = ReplayBuffer(memory_size)
        self.t_step = 0

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

        action = [0, 0, 0, 0]
        # epsilon-greedy action selection
        if random.random() > self.epsilon:
            action_values2 = action_values.cpu().data.numpy()
            action[np.argmax(action_values2)] = 1
            return action, action_values
            # return np.argmax(action_values.cpu().data.numpy()) 
            # return the action with the highest predicted Q-value
        else:
            action[random.randint(0, 3)] = 1
            return action, action_values 
            # return random.choice(np.arange(self.action_size)) # return a random action

    def train(
        self
    ): 
        if len(self.memory) < self.batch_size:
            return 0
        
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                self.learn_experiences(experiences)


    def replay_experiences(
        self
    ): 
        # train the q agent
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
        else:
            experiences = self.memory.get()
        
        self.learn_experiences(experiences)

    
    def learn_experiences(
        self,
        experiences: Tuple[torch.Tensor]
    ):  
        states = [e.state for e in experiences if e is not None]
        actions = [e.action for e in experiences if e is not None]
        rewards = [e.reward for e in experiences if e is not None]
        next_states = [e.next_state for e in experiences if e is not None]
        done = [e.done for e in experiences if e is not None]

        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        done = torch.tensor(done, dtype=torch.float).to(self.device)
        states = torch.stack(states).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        actions = torch.stack(actions).to(self.device)

        Q_expected = self.qnetwork_local(states)
        target = Q_expected.clone()
        
        for i in range(len(done)):

            Q_targets = rewards[i]
            if not done[i]:
                Q_targets_next = torch.max(self.qnetwork_local(next_states[i]))
                Q_targets += (self.gamma * Q_targets_next)

            target[i][torch.argmax(actions[i]).item()] = Q_targets

        self.optimizer.zero_grad()
        # get loss between Q_expected and Q_targets

        loss = F.l1_loss(Q_expected, target)
        loss.backward() # backpropagate the loss

        self.optimizer.step() # update the weights of the local network

        self.loss = loss.item() 

        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

        self.target = target
        self.Q_expected = Q_expected


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






