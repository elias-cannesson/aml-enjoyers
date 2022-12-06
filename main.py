import os
import sys
from traceback import format_exc
from typing import *

import random
import pygame
from collections import namedtuple
from tqdm import tqdm

from dataclasses import dataclass

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# import matplotlib for line graph
import matplotlib.pyplot as plt


from snake import Snake
from learn import *


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)


START = False

BLOCK_SIZE = 30
BOUNDS = (600, 600)
WIDTH = BOUNDS[0]
HEIGHT = BOUNDS[1]

BATCH_SIZE = 64
GAMMA = 0.25
EPS_START = 0.975
EPS_END = 0.00001
EPS_DECAY = 0.9992
TARGET_UPDATE = 10
LR = 0.00001
LR_DECAY = 0.999
LR_MIN = 0.0001

MEMORY_SIZE = 100000
EPOCHS = 1000

HIDDEN_SIZE = 256



""" 
Deep Q Learning with PyTorch

Snake Game

Learning Algorithm: Deep Q Learning

First, we will initialize the replay memory capacity and the batch size.
Then, we will initialize the discount factor γ.
We will initialize the final value of ε and the rate at which we want to decrease ε.
We will also initialize the number of steps to copy the parameters of our Q-network 
to the target Q-network.

Then, we will initialize the two Q-networks and the optimizer.
We will also initialize the two states s and s′, the action a, the reward r,
and the final state s′.

For each step of the episode, we will select an action a using the ε-greedy policy.
We will execute this action in the environment and observe the reward r and the next state s′.

We will store the transition (s, a, r, s′) in the replay memory.
Then, we will sample a random batch of transitions (s, a, r, s′) from the replay memory.

We will set the target value for the Q-network as follows:
    If the episode ends at step t + 1, then the target value is simply the reward r.
    Otherwise, the target value is r + γ * maxa′Q(s′, a′; θ′).

We will set the loss as the mean squared error between the Q-network and the target value.
We will perform a gradient descent step to minimize this loss.

We will update the parameters of the target Q-network every C steps.

Finally, we will decrease the value of ε.

After training, we will save the parameters of the Q-network.

Note: The target Q-network is used to compute the target value.

"""

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))



def train(
    seed: Optional[int] = 42,
    d: Optional[int] = 1,
):
    global START
    print("training...")
    while True:
        pygame.init()
        if d:
            window = pygame.display.set_mode(BOUNDS)
            pygame.display.set_caption("Snake")
            font = pygame.font.SysFont("comicsansms", 20)
            clock = pygame.time.Clock()
        # state:
        # - snake direction (straight, left, right)
        # - food location (up, down, left, right)
        # - danger location (up, down, left, right)

        # so 12 values for state space
        state = np.zeros((1, 1))
        state_size = 12
        
        # define the Agent
        agent = Agent(state_size=state_size, action_size=4, hidden_size=HIDDEN_SIZE,
                        lr=LR, gamma=GAMMA, epsilon=EPS_START, batch_size=BATCH_SIZE,
                        epsilon_decay=EPS_DECAY, epsilon_min=EPS_END, memory_size=MEMORY_SIZE,
                        update_every=4, device='cpu', seed=seed)
        
        snake_skeleton = Snake(w=WIDTH,h=HEIGHT, color=RED,
                                csize=BLOCK_SIZE) 
        
        # create figure with 2 subplots
        fig, ax = plt.subplots(1, 4)
        # set figure size
        fig.set_size_inches(14.5, 8.5)
        # set figure title
        fig.suptitle('Training Results')

        ax[0].axis([0, EPOCHS, 0, 100])
        ax[1].axis([0, EPOCHS, 0, 1])
        ax[2].axis([0, EPOCHS, 0, 2])
        ax[3].axis([0, EPOCHS, 0, 30])

        # set x axis label
        ax[0].set_xlabel('Episode')
        # set y axis label
        ax[0].set_ylabel('Score')
        ax[0].set_title('Score per Episode')

        # set x axis label
        ax[1].set_xlabel('Episode')
        # set y axis label
        ax[1].set_ylabel('Epsilon')
        ax[1].set_title('Epsilon per Episode')

        # set x axis label
        ax[2].set_xlabel('Episode')
        # set y axis label
        ax[2].set_ylabel('Loss')
        ax[2].set_title('Loss per Episode')

        # set x axis label
        ax[3].set_xlabel('Episode')
        # set y axis label
        ax[3].set_ylabel('Steps')
        ax[3].set_title('Steps per Episode')

        plt.ion()

        max_score = 0
        max_reward = float('-inf')
        max_steps = 0

        for epoch in tqdm(range(EPOCHS + 1)):
            # create state
            state = snake_skeleton.get_state()
            score = 0
            total_reward = 0
            steps = 0
            while True:
                if d:
                    pygame.time.delay(1)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()

                # perform action via epsilon greedy policy
                action, action_values = agent.act(state) 
                steps += 1 

                # perform action and get reward, game over, and next state
                reward, game_over = snake_skeleton.update(action)
                next_state = snake_skeleton.get_state()

                # print(f" next state: {next_state}")

                total_reward += reward
                reward = torch.Tensor([reward]).unsqueeze(0) 

                # # train agent
                # agent.train(state=state, action=action, reward=reward, 
                #             next_state=next_state, done=game_over)
                action = action_values

                # print(f" experience: {experience}")

                agent.train()
                # agent.learn_experiences(experiences=[experience], gamma=GAMMA)

                # store transition in memory
                agent.remember(state=state, action=action, reward=reward, 
                               next_state=next_state, done=game_over)

                if game_over or steps > 1000:
                    snake_skeleton.reset()
                    agent.replay_experiences()
                    if score > max_score:
                        max_score = score
                        agent.save_model('models/snake.pth')
                    break
                
                score = snake_skeleton.get_score()
                if score > max_score:
                    max_score = score

                    # print("new max score: ", max_score)
                if d:
                    text = font.render(f"Score: {score}", True, WHITE)
                    window.fill((10, 10, 10))
                    window.blit(text, (WIDTH - 100, 10))
                    snake_skeleton.draw(window)
                    pygame.display.flip()

                # update line graph with score for epoch
            max_reward = max(max_reward, total_reward)
            max_steps = max(max_steps, steps)

            if max_score == 100:
                break

            # if epoch == 200 and max_score < 1:
            #     break
            # if epoch == 400 and max_score < 2:
            #     break

            if epoch % 50 == 0:
                print(f"epoch: {epoch}, score: {score}, steps: {steps}, total reward: {total_reward}")
                print(f"epsilon: {agent.epsilon}")
                print(f"max score: {max_score}")
                print(f"max reward: {max_reward}")
                print(f"max steps: {max_steps}")
                print(f"loss: {agent.loss}")
                print(f"last action: {action}")
                print(f"last direction: {snake_skeleton.direction}")


            ax[0].plot(epoch, score, 'ro', markersize=1)
            ax[1].plot(epoch, agent.epsilon, 'bo', markersize=1)
            ax[2].plot(epoch, agent.loss, 'go', markersize=1)
            ax[3].plot(epoch, steps, 'yo', markersize=1)
            plt.pause(0.001)
            plt.show()

            if d:
                plt.pause(0.001)
                window.fill((10, 10, 10))
                #snake_skeleton.draw(window)
                pygame.display.flip()
        break
    pygame.quit()
    return




def main(
    func: Optional[str] = 'train',
    d: Optional[int] = 0,
):
    global START
    if func == 'train':
        train(d=d)

    while True:
        line = input(f"Enter a command: ")
        if line == "quit":
            break
        elif line == "start":
            pygame.init()
            window = pygame.display.set_mode(BOUNDS)
            pygame.display.set_caption("Snake")

            START = True
            snake_skeleton = Snake(x=300, y=300, color=RED,
                                   bounds=BOUNDS, bsize=BLOCK_SIZE)

            run = True
            f = 0
            direction = None
            while run:
                if f == 0:
                    pygame.time.delay(400)
                    f = 1
                pygame.time.delay(50)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("shutting down pygame...")
                        run = False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_UP:
                            direction = "U"
                        if event.key == pygame.K_DOWN:
                            direction = "D"
                        if event.key == pygame.K_LEFT:
                            direction = "L"
                        if event.key == pygame.K_RIGHT:
                            direction = "R"
                
                if direction:
                    # make move
                    step = snake_skeleton.play(direction=direction)

                    if not step:
                        print("game over")
                        run = False
                        break

                    # draw
                    window.fill((10, 10, 10))
                    snake_skeleton.draw(pygame, window)
                    pygame.display.flip()
                pygame.quit()

    return


if __name__ == '__main__':

    try:
        cat = 'train'
        d = 0
        if len(sys.argv) > 1:
            cat = sys.argv[1]
            if len(sys.argv) > 2:
                d = int(sys.argv[2])
        if cat == 'train':
            main(func='train', d=d)
        elif cat == 'play':
            main(func='play', d=d)
        else:
            main('train')
    except Exception as e:
        print("An error occurred: ", e)
        print(format_exc())
        sys.exit(1)