import pygame
import random
import time
import math

from tqdm import tqdm
import numpy as np
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression

################# THE FUNCTIONS ###############################

def display_snake(snake_position):
    for position in snake_position:
        pygame.draw.rect(display,red,pygame.Rect(position[0],position[1],10,10))
        
def display_apple(apple_position,apple):
  
    
    pygame.draw.rect(display,red,pygame.Rect(apple_position[0],apple_position[1],10,10))
    pygame.draw.rect(display,black,pygame.Rect(100,200,100,100))
  
    pygame.draw.rect(display,black,pygame.Rect(200,0,100,100))
    pygame.draw.rect(display,black,pygame.Rect(300,200,100,100))
    pygame.draw.rect(display,black,pygame.Rect(200,400,100,100))
    
def starting_positions(apple):
    snake_start = [0, 0]
    snake_position = [[100,100],[90,100]]
    
    apple_num_i = 0
    apple_position = [apple[apple_num_i][0], apple[apple_num_i][1]]
    score = 0
    
    return snake_start, snake_position, apple_position, score, apple_num_i




def apple_distance_from_snake(apple_position, snake_position):
    return np.linalg.norm(np.array(apple_position)-np.array(snake_position[0]))

def generate_snake(snake_start, snake_position, apple_position, button_direction, score, apple_num_i, apple):

    if button_direction == 1:
        snake_start[0] += 10
    elif button_direction == 0:
        snake_start[0] -= 10
    elif button_direction == 2:
        snake_start[1] += 10
    else:
        snake_start[1] -= 10
        
    if snake_start == apple_position:
        apple_position, score, apple_num_i = collision_with_apple(apple_position, score, apple_num_i, apple)
        snake_position.insert(0,list(snake_start))
        snake_position.pop()

    else:
        snake_position.insert(0,list(snake_start))
        snake_position.pop()
        
    return snake_position, apple_position, score, apple_num_i




def collision_with_boundaries(snake_start):
    if snake_start[0]>=500 or snake_start[0]<0 or snake_start[1]>=500 or snake_start[1]<0 or (snake_start[0]>100 and snake_start[0]<200 and snake_start[1]>200 and snake_start[1]<300) or (snake_start[0]>200 and snake_start[0]<300 and snake_start[1]>0 and snake_start[1]<100)or (snake_start[0]>300 and snake_start[0]<400 and snake_start[1]>200 and snake_start[1]<300) or (snake_start[0]>200 and snake_start[0]<300 and snake_start[1]>400 and snake_start[1]<500) :                     
        return 1
    else:
        return 0    


def collision_with_apple(apple_position, score, apple_num_i, apple):
    apple_num_i += 1
    apple_position = [apple[apple_num_i][0],apple[apple_num_i][1]]
    score += 1
    
    return apple_position, score, apple_num_i

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

def play_game(snake_start, snake_position, apple_position, button_direction, score, apple, apple_num_i):
    crashed = False
    while crashed is not True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True
        display.fill(background)
    
        display_apple(apple_position,apple)
        display_snake(snake_position)
  
    
        snake_position, apple_position, score, apple_num_i = generate_snake(snake_start, snake_position, apple_position, button_direction, score, apple_num_i, apple)                    
        pygame.display.set_caption("SCORE: "+str(score))
        pygame.display.update()
        clock.tick(20)
    
        return snake_position, apple_position, score, apple_num_i



    
def generate_training_data():
    training_data_x = []
    training_data_y = []
    training_games = 5
    steps_per_game = 500
    
    for _ in tqdm(range(training_games)):
        apple = get_apple()
        snake_start, snake_position, apple_position, score, apple_num_i = starting_positions(apple)
        prev_apple_distance = apple_distance_from_snake(apple_position, snake_position)
        prev_score = score
        
        for _ in range(steps_per_game):
            angle = angle_with_apple(snake_position, apple_position)
            direction, button_direction = generate_next_direction(snake_position, angle)
            snake_position, apple_position, score, apple_num_i = play_game(snake_start, snake_position, apple_position, button_direction, score, apple, apple_num_i)                 
            is_front_blocked, is_left_blocked ,is_right_blocked = blocked_directions(snake_position)
            
            training_data_x.append([is_left_blocked,is_front_blocked, is_right_blocked, angle, direction])
            
            if collision_with_boundaries(snake_position[0]) == 1 or collision_with_self(snake_position) == 1:
                training_data_y.append(-1)
                break
            else:
                current_snake_distance = apple_distance_from_snake(apple_position, snake_position)
                if score > prev_score or current_snake_distance < prev_apple_distance:
                    training_data_y.append(1)
                else:
                    training_data_y.append(0)
                prev_apple_distance = current_snake_distance
                prev_score = score
    return training_data_x, training_data_y

def get_apple():
    a1 = [random.randrange(10,20)*10,random.randrange(10,20)*10]
    a2 = [random.randrange(20,30)*10,random.randrange(10,20)*10]
    a3 = [random.randrange(20,30)*10,random.randrange(20,30)*10]
    a4 = [random.randrange(20,30)*10,random.randrange(30,40)*10]
    a5 = [random.randrange(30,40)*10,random.randrange(30,40)*10]
    a6 = [random.randrange(40,50)*10,random.randrange(40,50)*10]
    
    apple = [a1, a2, a3, a4, a5, a6, [500, 500]]
    #apple = [[200, 150], [250, 300], [400, 400], [500, 500]]
    return apple





def train_model():
    network = input_data(shape=[None, 5, 1], name='input')
    network = fully_connected(network, 25, activation='relu')
    network = fully_connected(network, 10, activation='relu')
    network = fully_connected(network, 1, activation='tanh')
    network = regression(network, optimizer='adam', learning_rate=1e-3, loss='mean_square', name='target')
    model = tflearn.DNN(network)
    
    return model


################# THE MAIN ###########################

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




training_data_x, training_data_y = generate_training_data()
model = train_model()
model.fit(np.array(training_data_x).reshape(-1,5,1),np.array(training_data_y).reshape(-1,1), n_epoch = 3, shuffle = True)

model.save("model.tfl")

pygame.quit()
