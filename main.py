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

RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3

MOVE_STRAIGHT = [1, 0, 0]
MOVE_LEFT = [0, 1, 0]
MOVE_RIGHT = [0, 0, 1]

MODEL_PATH = './snake.pth'
START = False

BLOCK_SIZE = 50
BOUNDS = (1000, 800)
WIDTH = BOUNDS[0]
HEIGHT = BOUNDS[1]

BATCH_SIZE = 1000
GAMMA = 0.99

EPS_STEPS = 200

EPS_START = 0.99
EPS_END = 0.0001
EPS_DECAY = 0.98
TARGET_UPDATE = 200
LR = 0.001
LR_DECAY = 0.999
LR_MIN = 0.0001

MEMORY_SIZE = 300000
EPOCHS = 400
ACTION_SIZE = 3
STATE_SIZE = 11
# STATE_SIZE = 11 + (WIDTH // BLOCK_SIZE) * (HEIGHT // BLOCK_SIZE)
# STATE_SIZE = 7
HIDDEN_SIZE = 3136

CLEAN = 10000

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

direction_mask = lambda d: 0 if d[0] == 1 else 1 if d[1] == 1 else 2 if d[2] == 1 else 3


def plot_categories(
    ax: List[plt.Axes],
    categories: list,
):

    colors = ['r', 'b', 'g', 'y', 'c']

    for i in range(len(categories)):
        for j in range(len(categories[i])):
            ax[i].plot(j, categories[i][j], f'{colors[i]}o', markersize=1)
        
    return

def create_n_subplots(
    n: int,
    figsize: Tuple[int, int],
    xlabels: List[str],
    ylabels: List[str],
    dimensions: List[Tuple],
):
    fig, axs = plt.subplots(
        nrows=1,
        ncols=n,
        figsize=figsize,
    )
    for i, ax in enumerate(axs):
        ax.set_xlabel(xlabels[i])
        ax.set_ylabel(ylabels[i])
        ax.set_title(f"{xlabels[i]} vs {ylabels[i]}")
        ax.axis(dimensions[i])
    return fig, axs

# function to plot bar graph of values in a list
def plot_bar_graph(
    values: List[int],
    title: str,
    xlabel: str,
    ylabel: str
):
    # creating the bar plot
    fig = plt.figure()
    fig.set_size_inches(9, 7)
    ax = fig.add_subplot()

    ax.bar(range(len(values)), values, color='blue', width=0.2, 
            label=['right', 'down', 'left', 'up'])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return


def train(
    seed: Optional[Any] = 4567,
    d: Optional[int] = 1,
    load_model: Optional[bool] = False,
    save_model: Optional[bool] = False
):
    global START
    print("training...")
    while True:
        pygame.init()
        if d == 1 or d == 3:
            window = pygame.display.set_mode(BOUNDS)
            pygame.display.set_caption("Snake")

            font = pygame.font.SysFont("comicsansms", 20)

        agent = Agent(state_size=STATE_SIZE, action_size=ACTION_SIZE, hidden_size=HIDDEN_SIZE,
                      lr=LR, gamma=GAMMA, epsilon=EPS_START, batch_size=BATCH_SIZE,
                      memory_size=MEMORY_SIZE,update_every=TARGET_UPDATE, device='cpu', 
                      seed=seed, load_model=load_model, epsilon_decay=EPS_DECAY, 
                      epsilon_min=EPS_END)
                        
        snake_skeleton = Snake(w=WIDTH,h=HEIGHT, color=RED,
                                csize=BLOCK_SIZE) 
        
        of_interest = ["Score", "Epsilon", "Loss", "Steps", "Rewards"]
        dims = [100, 1, 2, 10, 1500, 1000]
        l_oi = len(of_interest)

        if d >= 2 or not d:
            # create figure with 2 subplots
            fig, ax = create_n_subplots(
                n=l_oi,
                figsize=(6, 6.5),
                xlabels=["Epochs"] * l_oi,
                ylabels=of_interest,
                dimensions = [(0, EPOCHS, 0, d ) for d in dims])

            plt.ion()

        direction_counts = [0, 0, 0, 0] # right, down, left, up

        max_score = 0
        max_reward = float('-inf')
        max_steps = 0

        for epoch in tqdm(range(EPOCHS + 1)):
            score = 0
            total_reward = 0
            steps = 0
            while True:
                if d == 1 or d == 3:
                    # pygame.time.delay(800)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()
                
                state = snake_skeleton.get_state()
                # perform action via epsilon greedy policy
                action, action_values = agent.act(state) 
                steps += 1 
                # get reward, done
                reward, game_over = snake_skeleton.update(action)

                direction_counts[snake_skeleton.direction] += 1

                # get next state
                next_state = snake_skeleton.get_state()

                total_reward += reward

                agent.train()

                # store transition in memory
                agent.remember(state=state, action=action, reward=torch.Tensor([reward]), 
                               next_state=next_state, done=game_over)

                if game_over:
                    last_direction = snake_skeleton.direction
                    snake_skeleton.reset()
                    agent.replay_experiences()
                    if score > max_score:
                        max_score = score
                        agent.save_model('./snake.pth')
                    break
                
                score = snake_skeleton.get_score()

                if score > max_score:
                    max_score = score

                if d == 1 or d == 3:
                    text = font.render(f"Score: {score}", True, WHITE)
                    text2 = font.render(f"Epoch: {epoch}", True, WHITE)
                    window.fill((10, 10, 10))
                    window.blit(text, (WIDTH - 100, 10))
                    window.blit(text2, (WIDTH - 100, 50))
                    snake_skeleton.draw(window)
                    pygame.display.flip()
            
            max_reward = max(max_reward, total_reward)
            max_steps = max(max_steps, steps)

            if max_score == 100:
                break

            # if epoch == 300:
            #     d = 3
            #     window = pygame.display.set_mode(BOUNDS)
            #     pygame.display.set_caption("Snake")

            #     font = pygame.font.SysFont("comicsansms", 20)   
                
            if epoch % 15 == 0:
                print(f"epoch: {epoch}, score: {score}, steps: {steps}, total reward: {total_reward}")
                print(f"epsilon: {agent.epsilon}")
                print(f"max score: {max_score}")
                print(f"max reward: {max_reward}")
                print(f"max steps: {max_steps}")
                print(f"loss: {agent.loss}")
                print(f"last action: {action}")
                print(f"last direction: {last_direction}")
                print(f"last action values: {action_values}")
                print(f"length of memory: {len(agent.memory)}")
                print(f"direction counts: {direction_counts}")

            if d == 3:
                plt.pause(0.001)
                window.fill((10, 10, 10))
                snake_skeleton.draw(window)
                pygame.display.flip()

                ax[0].plot(epoch, score, 'ro', markersize=1)
                ax[1].plot(epoch, agent.epsilon, 'bo', markersize=1)
                ax[2].plot(epoch, agent.loss, 'go', markersize=1)
                ax[3].plot(epoch, steps, 'yo', markersize=1)
                ax[4].plot(epoch, total_reward, 'co', markersize=1)
                plt.show()
            elif d == 1:
                window.fill((10, 10, 10))
                #snake_skeleton.draw(window)
                pygame.display.flip()
            
            elif d == 2 or not d:
                if d == 2:
                    plt.pause(0.001)
                ax[0].plot(epoch, score, 'ro', markersize=1)
                ax[1].plot(epoch, agent.epsilon, 'bo', markersize=1)
                ax[2].plot(epoch, agent.loss, 'go', markersize=1)
                ax[3].plot(epoch, steps, 'yo', markersize=1)
                ax[4].plot(epoch, total_reward, 'co', markersize=1)
                if d == 2:
                    plt.show()

        break

    if not d:
        plt.show()
        plot_bar_graph(values=direction_counts,
                        title='Direction counts',
                        xlabel='Epochs',
                        ylabel='Direction Counts')
    
    if save_model:
        try:
            agent.save_model(MODEL_PATH)
            print("Model saved")
        except Exception as e:
            print(f"Model not saved. Error: {e}")
    plt.show()
    pygame.quit()
    return


def main(
    func: Optional[str] = 'train',
    d: Optional[int] = 0,
    load_model: Optional[bool] = False,
    save_model: Optional[bool] = False
):
    global START
    if func == 'train':
        train(d=d, load_model=load_model, save_model=save_model)
    pygame.init()
    p = 0
    print("Enter 'start' or 'quit'")
    while True:
        line = input(f"Enter a command: ")
        if line == "quit":
            break
        elif line == "start":
            if not p:
                window = pygame.display.set_mode(BOUNDS)
                pygame.display.set_caption("Snake")
                font = pygame.font.SysFont("comicsansms", 20)
                clock = pygame.time.Clock()

            START = True
            snake_skeleton = Snake(w=WIDTH, h=HEIGHT, csize=BLOCK_SIZE,
                                    color=RED)

            run = True
            f = 0
            direction = RIGHT
            window.fill((10, 10, 10))
            text = font.render(f"Score: {snake_skeleton.score}", True, WHITE)
            window.blit(text, (WIDTH - 100, 10))
            snake_skeleton.draw(window)
            pygame.display.flip()
            while run:
                if f == 0:
                    pygame.time.delay(400)
                    f = 1
                pygame.time.delay(1000)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("shutting down pygame...")
                        run = False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_UP:
                            direction = UP
                        if event.key == pygame.K_DOWN:
                            direction = DOWN
                        if event.key == pygame.K_LEFT:
                            direction = LEFT
                        if event.key == pygame.K_RIGHT:
                            direction = RIGHT
                
                # make move
                # check if game over
                cont = snake_skeleton.play(direction=direction)

                # state = snake_skeleton.get_state_v2()

                # # print state in this format
                
                # print(
                #     f"state: {state}, direction: {direction}, score: {snake_skeleton.score}"
                # )
                # sys.exit()

                if not cont:
                    print("game over")
                    run = False
                    break
                # draw
                window.fill((10, 10, 10))
                text = font.render(f"Score: {snake_skeleton.score}", True, WHITE)
                snake_skeleton.draw(window)
                pygame.display.flip()
                
                #pygame.quit()
    pygame.quit()
    return

def print_program_usage():
    # incllude -e flag which can take a float value between 0-1
    print("Usage: python snake.py [-t|-l|-p] [-d 0|1|2|3] [-s model_path] \n"
          "[-e (value btwn 0 and 1)] [-ed (value btwn 0 and 1)] [-lr (value btwn 0 and 1)]\n"
          "[-ep (epochs for training: any positive integer)]")
    print("Options:")
    print("-t: train model")
    print("-l: load model")
    print("-s <model_path>: save model to model path")
    print("-p: play game")
    print("-d: display mode")
    print("-e: epsilon value (float between 0 and 1) (default is 0.99)")
    print("-ed: epsilon decay value (float between 0 and 1) (default is 0.999)")
    print("-lr: learning rate (float between 0 and 1) (default is 0.001)")
    print(f"-ep: epochs (any positive integer) (default is {EPOCHS})")
    print("0: no display")
    print("1: display snake game")
    print("2: display graphs")
    print("3: display snake game and graphs")

if __name__ == '__main__':
    try:
        if len(sys.argv) > 1:
            # parse through args, where -t indicates
            # user wants to train model, -l indicates
            # the user wants to load a saved model,
            # -p indicates user wants to play game,
            # and -d followed by 0, 1, 2, or 3
            # indicates value of d

            # default is to train model
            func = 'train'
            load_model = False
            save_model = False
            d = 0
            if '-t' in sys.argv and '-p' in sys.argv:
                print("Please only choose one of the following: -t for train, -p for play")
                print_program_usage()
                sys.exit(1)
            
            if '-h' in sys.argv or '--help' in sys.argv:
                print_program_usage()
                sys.exit(1)

            for i in range(1, len(sys.argv)):
                if sys.argv[i] == '-t':
                    func = 'train'
                elif sys.argv[i] == '-l':
                    load_model = True
                elif sys.argv[i] == '-p':
                    func = 'play'
                elif sys.argv[i] == '-d':
                    d = int(sys.argv[i + 1])
                elif sys.argv[i] == '-e':
                    EPS_START = float(sys.argv[i + 1])
                elif sys.argv[i] == '-ed':
                    EPS_DECAY = float(sys.argv[i + 1])
                elif sys.argv[i] == '-lr':
                    LR = float(sys.argv[i + 1])
                elif sys.argv[i] == '-s':
                    save_model = True
                    MODEL_PATH = sys.argv[i + 1]
                elif sys.argv[i] == '-ep':
                    EPOCHS = int(sys.argv[i + 1])
            
            main(func=func, d=d, load_model=load_model, save_model=save_model)

    except Exception as e:
        print("An error occurred: ", e)
        print(format_exc())
        sys.exit(1)