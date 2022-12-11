import random
import pygame
import numpy as np

import torch

from dataclasses import dataclass

__all__ = ['Snake']


direction_mask = lambda d: RIGHT if d[0] == 1 else DOWN if d[1] == 1 else LEFT if d[2] == 1 else UP

WHITE = (255, 255, 255)

RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3

DIRECTIONS = [RIGHT, DOWN, LEFT, UP]

MOVE_STRAIGHT = [1, 0, 0]
MOVE_LEFT = [0, 1, 0]
MOVE_RIGHT = [0, 0, 1]

def convert_direction_to_move(
    direction: int,
    action: list
) -> int:
    i = DIRECTIONS.index(direction)

    if action == MOVE_STRAIGHT:
        return direction
    elif action == MOVE_LEFT:
        return DIRECTIONS[(i - 1) % 4]
    elif action == MOVE_RIGHT:
        return DIRECTIONS[(i + 1) % 4]

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
        self.direction = RIGHT
        self.score = 0
        self.size = 3

        self.steps = 0

        self.color = color

        self.init_easy_food()
        
    def reset(self):
        self.size = 3
        self.direction = RIGHT
        self.score = 0
        self.head = Point(self.x, self.y)
        self.cells = [self.head,
                     Point(self.head.x - self.csize, self.head.y),
                     Point(self.head.x - (2 * self.csize), self.head.y)]
    
    def new_food(self):
       # self.food = Point(rand_point, rand_point)
        self.food = Point(round_up(random.randint(0, self.w - (2*self.csize)), self.csize),
                          round_up(random.randint(0, self.h - (2*self.csize)), self.csize))
        if self.food in self.cells:
            self.new_food()
    
    def init_easy_food(self):
        self.food = Point(round_up(random.randint(self.csize*4, self.w - (4*self.csize)), self.csize),
                          round_up(random.randint(self.csize*4, self.h - (4*self.csize)), self.csize))
        if self.food in self.cells:
            self.init_easy_food()
    
    def has_collided(self, point: Point = None):
        if not point:
            point = self.head
        if point.x > self.w - self.csize or point.x < 0:
            return True
        elif point.y > self.h - self.csize or point.y < 0:
            return True
        else:
            for cell in self.cells[1:]:
                if point.x == cell.x and point.y == cell.y:
                    return True
        return False
    
    def move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == RIGHT:
            x += self.csize
        elif direction == DOWN:
            y += self.csize
        elif direction == LEFT:
            x -= self.csize
        elif direction == UP:
            y -= self.csize
        self.direction = direction
        self.head = Point(x, y)
    
    def update(self, action):
        reward = 0
        game_over = 0
        
        direction = convert_direction_to_move(self.direction, action)
        self.move(direction)
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
    
    def play(self, direction):
        self.move(direction)
        self.cells.insert(0, self.head)

        if self.has_collided():
            return False
        if self.head == self.food:
            self.score += 1
            self.new_food()
        else:
            self.cells.pop()
        return True
    
    
    def get_head_location(self):
        return self.head
    
    def get_food_location(self):
        return self.food
    
    def get_state(self):
        """
        s = [
            1: danger straight,
            1: danger right,
            1: danger left,

            1: food left,
            1: food right,
            1: food up,
            1: food down,
            
            1: move right,
            1: move left,
            1: move down,
            1: move up
        ]
        """

        state = [ 0 ] * 11

        left_block = Point(self.head.x - self.csize, self.head.y)
        right_block = Point(self.head.x + self.csize, self.head.y)
        up_block = Point(self.head.x, self.head.y - self.csize)
        down_block = Point(self.head.x, self.head.y + self.csize)

        # danger straight, right, left
        if self.direction == RIGHT:
            if self.has_collided(right_block):
                state[0] = 1
            if self.has_collided(down_block):
                state[1] = 1
            if self.has_collided(up_block):
                state[2] = 1
        elif self.direction == LEFT:
            if self.has_collided(left_block):
                state[0] = 1
            if self.has_collided(up_block):
                state[1] = 1
            if self.has_collided(down_block):
                state[2] = 1
        elif self.direction == UP:
            if self.has_collided(up_block):
                state[0] = 1
            if self.has_collided(right_block):
                state[1] = 1
            if self.has_collided(left_block):
                state[2] = 1
        elif self.direction == DOWN:
            if self.has_collided(down_block):
                state[0] = 1
            if self.has_collided(left_block):
                state[1] = 1
            if self.has_collided(right_block):
                state[2] = 1


        # food left, right, up, down
        state[3] = 1 if self.food.x < self.head.x else 0
        state[4] = 1 if self.food.x > self.head.x else 0
        state[5] = 1 if self.food.y < self.head.y else 0
        state[6] = 1 if self.food.y > self.head.y else 0

        # moving right
        state[7] = self.direction == RIGHT
        state[8] = self.direction == DOWN
        state[9] = self.direction == LEFT
        state[10] = self.direction == UP
        return torch.tensor(np.array(state, dtype=int), dtype=torch.float32)

    # def get_state(self):
    #     # make a grid of self.csize by width and height
    #     # if the snake's body is in that cell, make it 1
    #     # if the food is in that cell, make it 1
    #     # else 0
    #     # concatenate another tensor of the direction the snake is moving
    #     # if bound is near snake, make it 1
    #     # else 0

    #     # make a grid of self.csize by width and height
    #     grid = torch.zeros((self.width // self.csize, self.height // self.csize))
    #     # if the snake's body is in that cell, make it 1
    #     for cell in self.cells:
    #         grid[cell.x // self.csize, cell.y // self.csize] = 1
    #     # if the food is in that cell, make it 1
    #     grid[self.food.x // self.csize, self.food.y // self.csize] = 1

    #     # concatenate another tensor of the direction the snake is moving
    #     direction = torch.zeros(4)
    #     direction[self.direction] = 1



    def get_score(self):
        return self.score
    
    def draw(self, 
        game: pygame.Surface,
        actions: torch.Tensor
    ):
        pygame.draw.rect(game, WHITE, (self.head.x, self.head.y, self.csize, self.csize))
        for i, cell in enumerate(self.cells[1:]):
            pygame.draw.rect(game, self.color, (cell.x, cell.y, self.csize, self.csize))
        pygame.draw.rect(game, (255, 255, 0), (self.food.x, self.food.y, self.csize, self.csize))
        
        # # draw the values of each action
        # for i, action in enumerate(actions):
        #     font = pygame.font.SysFont('Arial', 20)
        #     text = font.render(str(action), True, (255, 255, 255))
        #     game.blit(text, (self.head.x + 10, self.head.y + 10 + (i * 20)))

    
    

        

    

    
    

        

    