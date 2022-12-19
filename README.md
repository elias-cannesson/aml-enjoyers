# CS 374 - Average Machine Learning Enjoyers Final Project 
## Elias Cannesson, Austin Alcancia, Nghi Nguyen, Harsha Rauniyar

# Summary
This project delves into the exploration of the combination of neural networks from Machine Learning, and Q-learning (reinforcement learning) from the concepts of Artificial Intelligence and Game Theory. In this project, we use Deep Q Networks (DQN) to get a two dimensional snake game to play the game successfully. To this end, we developed a snake game in Python using pygame, and made use of Deep Q networks to train an agent to play the game by itself by learning from past experience. We explore the performance of a trained agent ( the snake) playing the original game. 

# Program command manual
Run main.py -h for detailed version of the command manual  

Usage: python snake.py [-t|-l|-p] [-d 0|1|2|3] [-s model_path]  
    [-e (value btwn 0 and 1)] [-ed (value btwn 0 and 1)] [-lr (value btwn 0 and 1)]  
    [-ep (epochs for training: any positive integer)] [-hs (hidden size)]  
    [-g (gamma / discount-factor for future actions)] [-bs (batch size of training data)]  
    [-tu (target update, must be int greater than 0)]  

Options:  
    -t: train model  
    -l: load model  
    -s <model_path>: save model to model path  
    -p: play game  
    -e: epsilon value (float between 0 and 1) (default is 0.99)  
    -ed: epsilon decay value (float between 0 and 1) (default is 0.999)  
    -lr: learning rate (float between 0 and 1) (default is 0.001)  
    -ep: epochs (any positive integer) (default is {EPOCHS})  
    -hs: hidden size (any positive integer) (default is {HIDDEN_SIZE})  
    -g: gamma (float between 0 and 1) (default is {GAMMA})  
    -bs: batch size (any positive integer) (default is {BATCH_SIZE})  
    -tu: target update (any positive integer) (default is {TARGET_UPDATE})  
    -d: display mode  
    0: no display  
    1: display snake game  
    2: display graphs  
    3: display snake game and graphs  