# CS 374 - Average Machine Learning Enjoyers Final Project 
## Elias Cannesson, Austin Alcancia, Nghi Nguyen, Harsha Rauniyar

# Summary
This project delves into the exploration of the combination of neural networks from Machine Learning, and Q-learning (reinforcement learning) from the concepts of Artificial Intelligence and Game Theory. In this project, we use Deep Q Networks (DQN) to get a two dimensional snake game to play the game successfully. To this end, we developed a snake game in Python using pygame, and made use of Deep Q networks to train an agent to play the game by itself by learning from past experience. We explore the performance of a trained agent ( the snake) playing the original game using our DQN algorithm. We then conduct some experiments on the different variables available(learning rate, discount rate, epsilon decay) to find out cases where the agent performs the best. 

# Requirements
-Python  
-Pygame  
-Pytorch  
-matplotlib  
-tqdm  
-torchvision  
-numpy  
Make sure you have all of these downloaded.  

# How to run the program
Simply clone the repository, make sure you have the requirements and run the following command according to the command manual to run the program.  

# Program command manual
Run the main.py file with the parameters specified below, if the options are not specified, it will use the default value.
Run main.py -h for detailed version of the command manual  

Usage: python snake.py [-t|-l|-p] [-d 0|1|2|3] [-s model_path]  
    [-e (value btwn 0 and 1)] [-ed (value btwn 0 and 1)] [-lr (value btwn 0 and 1)]  
    [-ep (epochs for training: any positive integer)] [-hs (hidden size)]  
    [-g (gamma / discount-factor for future actions)] [-bs (batch size of training data)]  
    [-tu (target update, must be int greater than 0)] [-... other options]

Options:  
    -t: train model  
    -l: load model  
    -s <model_path>: save model to model path  
    -p: play game  
    -e: epsilon value (float between 0 and 1) (default is 0.99)  
    -ed: epsilon decay value (float between 0 and 1) (default is 0.999)  
    -lr: learning rate (float between 0 and 1) (default is 0.001)  
    -ep: epochs (any positive integer)  
    -hs: hidden size (any positive integer)  
    -g: gamma (float between 0 and 1)  
    -bs: batch size (any positive integer)  
    -tu: target update (any positive integer)    
    -d: display mode  
    0: no display  
    1: display snake game  
    2: display graphs  
    3: display snake game and graphs  

Example:  
python main.py -t -d 3 -e 0.99 -ed 0.991 -lr 0.0009 -ep 500 -s "./final1.pth" -hs 512 -bs 2500 -tu 200