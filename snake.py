import random
import pygame
import numpy as np

import torch

from dataclasses import dataclass

__all__ = ['Snake']


direction_mask = lambda d: 0 if d[0] == 1 else 1 if d[1] == 1 else 2 if d[2] == 1 else 3


def round_up(n, multiple):
    return n + (multiple - n % multiple) % multiple

@dataclass
class Point:
    x: int
    y: int


class Snake():

    def __init__(
        self,
        w: int,
        h: int,
        csize: int,
        color: tuple
    ):
        self.w = w
        self.h = h
        self.x = round_up(w // 2, csize)
        self.y = round_up(h // 2, csize)
        self.csize = csize

        self.head = Point(self.x, self.y)
        self.cells = [self.head,
                      Point(self.head.x - csize, self.head.y),
                      Point(self.head.x - (2 * csize), self.head.y)]
        self.direction = 0
        self.score = 0
        self.size = 3

        self.steps = 0

        self.color = color

        self.new_food()
        
    def reset(self):
        self.size = 3
        self.direction = 0
        self.score = 0
        self.head = Point(self.x, self.y)
        self.cells = [self.head,
                     Point(self.head.x - self.csize, self.head.y),
                     Point(self.head.x - (2 * self.csize), self.head.y)]
        self.new_food()
    
    def new_food(self):
        self.food = Point(round_up(random.randint(0, self.w - self.csize), self.csize),
                          round_up(random.randint(0, self.h - self.csize), self.csize))
        if self.food in self.cells:
            self.new_food()
    
    def has_collided(self):
        if self.head.x >= self.w or self.head.x <= 0:
            return True
        elif self.head.y >= self.h or self.head.y <= 0:
            return True
        else:
            for cell in self.cells[1:]:
                if self.head.x == cell.x and self.head.y == cell.y:
                    return True
        return False
    
    def move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == "U":
            y -= self.csize
        elif direction == "D":
            y += self.csize
        elif direction == "L":
            x -= self.csize
        elif direction == "R":
            x += self.csize
        self.head = Point(x, y)
    
    def update(self, action):
        reward = 0
        game_over = 0

        x = self.head.x
        y = self.head.y
        direction = direction_mask(action)

        if direction == 0:
            # go right
            x += self.csize
        elif direction == 1:
            # go down
            y += self.csize
        elif direction == 2:
            # go left
            x -= self.csize
        elif direction == 3:
            # go up
            y += self.csize

        self.head = Point(x, y)
        self.cells.insert(0, self.head)

        if self.has_collided():
            game_over = 1
            reward = -10
            return reward, game_over
        
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.new_food()
        else:
            self.cells.pop()
        
        return reward, game_over
        
    
    def get_head_location(self):
        return self.head
    
    def get_food_location(self):
        return self.food
    
    def get_state(self):
        """
        s = [
            1: danger up,
            1: danger right,
            1: danger left,
            1: danger down,
            1: food up,
            1: food down,
            1: food right,
            1: food left,
            1: move right,
            1: move left,
            1: move down,
            1: move up
        ]
        """

        state = [ 0 ] * 12

        # danger up
        if self.head.y - self.csize <= 0:
            state[0] = 1
        # danger right
        if self.head.x + self.csize >= self.w:
            state[1] = 1
        # danger left
        if self.head.x - self.csize <= 0:
            state[2] = 1
        # danger down
        if self.head.y + self.csize >= self.h:
            state[3] = 1
        
        # food up
        if self.food.y < self.head.y:
            state[4] = 1
        # food down
        if self.food.y > self.head.y:
            state[5] = 1
        # food right
        if self.food.x > self.head.x:
            state[6] = 1
        # food left
        if self.food.x < self.head.x:
            state[7] = 1
        
        # move right
        if self.direction == 0:
            state[8] = 1
        # move left
        if self.direction == 2:
            state[9] = 1
        # move down
        if self.direction == 1:
            state[10] = 1
        # move up
        if self.direction == 3:
            state[11] = 1

        return torch.tensor(np.array(state, dtype=int), dtype=torch.float32)
        


        # # danger straight
        # if self.head.x == self.w or self.head.x == 0:
        #     state[0] = 1
        # elif self.head.y == self.h or self.head.y == 0:
        #     state[0] = 1
        # else:
        #     for cell in self.cells[1:]:
        #         if self.head.x == cell.x and self.head.y == cell.y:
        #             state[0] = 1
        #             break
        # # danger right
        # if self.head.y == self.h or self.head.y == 0:
        #     state[1] = 1
        # else:
        #     for cell in self.cells[1:]:
        #         if self.head.x + self.csize == cell.x and self.head.y == cell.y:
        #             state[1] = 1
        #             break
        # # danger left
        # if self.head.y == self.h or self.head.y == 0:
        #     state[2] = 1
        # else:
        #     for cell in self.cells[1:]:
        #         if self.head.x - self.csize == cell.x and self.head.y == cell.y:
        #             state[2] = 1
        #             break
        # # food straight
        # if self.head.x == self.food.x:
        #     state[3] = 1
        # # food right
        # if self.head.y == self.food.y:
        #     state[4] = 1
        # # food left
        # if self.head.y == self.food.y:
        #     state[5] = 1
        # # move up
        # if self.head.y - self.csize >= 0:
        #     state[6] = 1
        # # move right
        # if self.head.x + self.csize < self.w:
        #     state[7] = 1
        # # move left
        # if self.head.x - self.csize >= 0:
        #     state[8] = 1
        # # move down
        # if self.head.y + self.csize < self.h:
        #     state[9] = 1
        # return state
    
    def get_score(self):
        return self.score
    
    def draw(self, 
        game: pygame.Surface,
    ):
        for i, cell in enumerate(self.cells):
            pygame.draw.rect(game, self.color, (cell.x, cell.y, self.csize, self.csize))
        pygame.draw.rect(game, (255, 255, 0), (self.food.x, self.food.y, self.csize, self.csize))
    

    
    

        

    

    
    

        

    