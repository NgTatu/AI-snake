import pygame
import random
import time
import math

from tqdm import tqdm
import numpy as np
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression


################ THE FUNCTIONS #######################

def display_snake(snake_position):
    for position in snake_position:
        pygame.draw.rect(display,red,pygame.Rect(position[0],position[1],10,10))
        
def display_apple(apple_position,apple):
  
    
    pygame.draw.rect(display,red,pygame.Rect(apple_position[0],apple_position[1],10,10))
    pygame.draw.rect(display,black,pygame.Rect(100,200,100,100))
  
    pygame.draw.rect(display,black,pygame.Rect(200,0,100,100))
    pygame.draw.rect(display,black,pygame.Rect(300,200,100,100))
    pygame.draw.rect(display,black,pygame.Rect(200,400,100,100))


# STARTING POSITION WITH ML
def starting_positions2():
    
    snake_start = [50,50]
    
    snake_position = [[100,100],[90,100]]
    
    #apple_position = [200,150]
    #apple_position = [250,300]
    #apple_position = [250,350]
    #apple_position = [300,450]
    apple_position = [310,450]
    
    score = 0
    
    return snake_start, snake_position, apple_position, score


def apple_distance_from_snake(apple_position, snake_position):
    return np.linalg.norm(np.array(apple_position)-np.array(snake_position[0]))



def generate_snake2(snake_start, snake_position, apple_position, button_direction, score):

    if button_direction == 1:
        snake_start[0] += 10
    elif button_direction == 0:
        snake_start[0] -= 10
    elif button_direction == 2:
        snake_start[1] += 10
    else:
        snake_start[1] -= 10
        

    snake_position.insert(0,list(snake_start))
    snake_position.pop()
        
    return snake_position, apple_position, score



def collision_with_boundaries(snake_start):
    if snake_start[0]>=500 or snake_start[0]<0 or snake_start[1]>=500 or snake_start[1]<0 or (snake_start[0]>100 and snake_start[0]<200 and snake_start[1]>200 and snake_start[1]<300) or (snake_start[0]>200 and snake_start[0]<300 and snake_start[1]>0 and snake_start[1]<100)or (snake_start[0]>300 and snake_start[0]<400 and snake_start[1]>200 and snake_start[1]<300) or (snake_start[0]>200 and snake_start[0]<300 and snake_start[1]>400 and snake_start[1]<500) :                     
        return 1
    else:
        return 0    




def collision_with_self(snake_position):
    snake_start = snake_position[0]
    if snake_start in snake_position[1:]:
        return 1
    else:
        return 0

def blocked_directions(snake_position):
    current_direction_vector = np.array(snake_position[0])-np.array(snake_position[1])
    
    left_direction_vector = np.array([current_direction_vector[1], -current_direction_vector[0]])
    right_direction_vector = np.array([-current_direction_vector[1], current_direction_vector[0]])
    
    is_front_blocked = is_direction_blocked(snake_position, current_direction_vector)
    is_left_blocked = is_direction_blocked(snake_position, left_direction_vector)
    is_right_blocked = is_direction_blocked(snake_position, right_direction_vector)
    
    return is_front_blocked, is_left_blocked, is_right_blocked

def is_direction_blocked(snake_position, current_direction_vector):
    next_step = snake_position[0]+ current_direction_vector
    snake_start = snake_position[0]+ current_direction_vector
    if collision_with_boundaries(snake_start) == 1 or collision_with_self(snake_position) == 1:
        return 1
    else:
        return 0

def generate_next_direction(snake_position, angle_with_apple):
    direction = 0
    
    if angle_with_apple > 0:
        direction = 1
    elif angle_with_apple < 0:
        direction = -1
    else:
        direction = 0
        
    current_direction_vector = np.array(snake_position[0])-np.array(snake_position[1])
    left_direction_vector = np.array([current_direction_vector[1],-current_direction_vector[0]])
    right_direction_vector = np.array([-current_direction_vector[1], current_direction_vector[0]])
    
    new_direction = current_direction_vector
    if direction == -1:
        new_direction = left_direction_vector
    if direction == 1:
        new_direction = right_direction_vector

    button_direction = generate_button_direction(new_direction)
    
    return direction, button_direction

def generate_button_direction(new_direction):
    button_direction = 0
    if new_direction.tolist() == [10,0]:
        button_direction = 1
    elif new_direction.tolist() == [-10,0]:
        button_direction = 0
    elif new_direction.tolist() == [0,10]:
        button_direction = 2
    else:
        button_direction = 3
        
    return button_direction

def angle_with_apple(snake_position, apple_position):
    apple_direction_vector = np.array(apple_position)-np.array(snake_position[0])
    snake_direction_vector = np.array(snake_position[0])-np.array(snake_position[1])
    
    norm_of_apple_direction_vector = np.linalg.norm(apple_direction_vector)
    norm_of_snake_direction_vector = np.linalg.norm(snake_direction_vector)
    if norm_of_apple_direction_vector == 0:
        norm_of_apple_direction_vector = 10
    if norm_of_snake_direction_vector == 0:
        norm_of_snake_direction_vector = 10
        
    apple_direction_vector_normalized = apple_direction_vector/norm_of_apple_direction_vector
    snake_direction_vector_normalized = snake_direction_vector/norm_of_snake_direction_vector
    angle = math.atan2(apple_direction_vector_normalized[1] * snake_direction_vector_normalized[0] - apple_direction_vector_normalized[0] * snake_direction_vector_normalized[1], apple_direction_vector_normalized[1] * snake_direction_vector_normalized[1] + apple_direction_vector_normalized[0] * snake_direction_vector_normalized[0]) / math.pi
    return angle



def play_game2(snake_start, snake_position, apple_position, button_direction, score,apple):
    crashed = False
    while crashed is not True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True
        display.fill(background)
    
        display_apple(apple_position,apple)
        display_snake(snake_position)
    
        snake_position, apple_position, score = generate_snake2(snake_start, snake_position, apple_position, button_direction, score)
        pygame.display.set_caption("SCORE: "+str(score))
        pygame.display.update()
        clock.tick(20)
    
        return snake_position, apple_position, score   

    




def run_game_with_ML(model):
    max_score = 3
    avg_score = 0
    test_games = 5
    steps_per_game = 150
    
    for _ in tqdm(range(test_games)):
        snake_start, snake_position, apple_position, score = starting_positions2()

        for _ in range(steps_per_game):
            is_front_blocked, is_left_blocked ,is_right_blocked = blocked_directions(snake_position)
            angle = angle_with_apple(snake_position, apple_position)
            predictions = []
            for i in range(-1,2):
                predictions.append(model.predict(np.array([is_left_blocked,is_front_blocked, is_right_blocked,angle,i]).reshape(-1,5,1)))
            
            predicted_direction = np.argmax(np.array(predictions))-1
            
            new_direction = np.array(snake_position[0]) - np.array(snake_position[1])
            if predicted_direction == -1:
                new_direction = np.array([new_direction[1],-new_direction[0]])
            if predicted_direction == 1:
                new_direction = np.array([-new_direction[1],new_direction[0]])
            
            button_direction = generate_button_direction(new_direction)

            snake_position, apple_position, score = play_game2(snake_start, snake_position, apple_position, button_direction, score, apple_image)                       
            if collision_with_boundaries(snake_position[0]) == 1 or collision_with_self(snake_position) == 1:
                avg_score += score
                break
                                         
            if score > max_score:
                max_score = score
        
        avg_score += score
            


def train_model():
    network = input_data(shape=[None, 5, 1], name='input')
    network = fully_connected(network, 25, activation='relu')
    network = fully_connected(network, 10, activation='relu')
    network = fully_connected(network, 1, activation='tanh')
    network = regression(network, optimizer='adam', learning_rate=1e-3, loss='mean_square', name='target')
    model = tflearn.DNN(network)
    
    return model

############## LOAD THE MODEL TRAINED BY TRAIN.PY ##################

model = train_model()
model.load("model.tfl")


############### THE MAIN ################################

display_width = 500
display_height = 500
green = (0,255,0)
red = (255,0,0)
black = (0,0,0)
background = (0,255,0)

pygame.init()
display=pygame.display.set_mode((display_width,display_height))

clock=pygame.time.Clock()
pygame.draw.rect(display,black,pygame.Rect(100,200,100,100))


run_game_with_ML(model)
                                        

pygame.quit()
