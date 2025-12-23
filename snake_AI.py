import pygame
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

pygame.init()
WIDTH, HEIGHT = 400, 400
CELL_SIZE = 20
WHITE, BLACK, GREEN, DARK_GREEN, RED = (255, 255, 255), (0, 0, 0), (0, 255, 0), (60, 155, 60), (255, 0, 0)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake DRL")
clock = pygame.time.Clock()

MAX_MEMORY = 10000
BATCH_SIZE = 64
LR = 0.001
GAMMA = 0.9
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995

def spawn_food(snake):
    while True:
        x = random.randrange(0, WIDTH, CELL_SIZE)
        y = random.randrange(0, HEIGHT, CELL_SIZE)
        if (x, y) not in snake:
            return x, y


def get_state(snake, food, direction):
    head_x, head_y = snake[-1]

    danger_up = (head_x, head_y - CELL_SIZE) in snake or head_y - CELL_SIZE < 0
    danger_down = (head_x, head_y + CELL_SIZE) in snake or head_y + CELL_SIZE >= HEIGHT
    danger_left = (head_x - CELL_SIZE, head_y) in snake or head_x - CELL_SIZE < 0
    danger_right = (head_x + CELL_SIZE, head_y) in snake or head_x + CELL_SIZE >= WIDTH

    food_dx = food[0] - head_x
    food_dy = food[1] - head_y

    dir_up = direction == (0, -CELL_SIZE)
    dir_down = direction == (0, CELL_SIZE)
    dir_left = direction == (-CELL_SIZE, 0)
    dir_right = direction == (CELL_SIZE, 0)

    state = [
        int(danger_up), int(danger_down), int(danger_left), int(danger_right),
        np.sign(food_dx), np.sign(food_dy),
        int(dir_up), int(dir_down), int(dir_left), int(dir_right)
    ]

    return np.array(state, dtype=int)


def is_collision(pos, snake):
    x, y = pos
    return x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT or pos in snake


class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        return self.linear3(x)


class Agent:
    def __init__(self):
        self.model = DQN(10, 128, 4)
        self.memory = deque(maxlen=MAX_MEMORY)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.epsilon = EPSILON_START

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        state0 = torch.tensor(state, dtype=torch.float)
        with torch.no_grad():
            pred = self.model(state0)
        return torch.argmax(pred).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) < BATCH_SIZE:
            mini_sample = self.memory
        else:
            mini_sample = random.sample(self.memory, BATCH_SIZE)

        states, actions, rewards, next_states, dones = zip(*mini_sample)

        states = torch.tensor(np.array(states), dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.bool)

        pred = self.model(states)
        target = pred.clone()
        with torch.no_grad():
            next_pred = self.model(next_states)

        for idx in range(len(mini_sample)):
            q_new = rewards[idx]
            if not dones[idx]:
                q_new += GAMMA * torch.max(next_pred[idx])
            target[idx][actions[idx]] = q_new

        self.optimizer.zero_grad()
        loss = nn.MSELoss()(pred, target)
        loss.backward()
        self.optimizer.step()

        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY


def train():
    agent = Agent()
    generation = 0
    while True:
        snake = [(WIDTH // 2, HEIGHT // 2)]
        direction = (0, -CELL_SIZE)
        food = spawn_food(snake)
        score = 0
        snake_length = 1
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            state = get_state(snake, food, direction)
            action = agent.get_action(state)

            if action == 0 and direction != (0, CELL_SIZE):
                next_dir = (0, -CELL_SIZE)
            elif action == 1 and direction != (0, -CELL_SIZE):
                next_dir = (0, CELL_SIZE)
            elif action == 2 and direction != (CELL_SIZE, 0):
                next_dir = (-CELL_SIZE, 0)
            elif action == 3 and direction != (-CELL_SIZE, 0):
                next_dir = (CELL_SIZE, 0)
            else:
                next_dir = direction

            direction = next_dir
            head_x, head_y = snake[-1]
            new_head = (head_x + direction[0], head_y + direction[1])

            reward = 0
            if is_collision(new_head, snake):
                reward = -10
                done = True
            else:
                snake.append(new_head)
                if new_head == food:
                    score += 1
                    reward = 10
                    food = spawn_food(snake)
                    snake_length += 1
                if len(snake) > snake_length:
                    snake.pop(0)

            next_state = get_state(snake, food, direction)
            agent.remember(state, action, reward, next_state, done)
            agent.train_long_memory()

            screen.fill(BLACK)
            pygame.draw.rect(screen, RED, (*food, CELL_SIZE, CELL_SIZE))
            for i, segment in enumerate(snake):
                color = DARK_GREEN if i % 2 == 0 else GREEN
                pygame.draw.rect(screen, color, (*segment, CELL_SIZE, CELL_SIZE))

            pygame.display.flip()
            clock.tick(100)

        generation += 1
        print(f"Episode {generation} | Score: {score} | Epsilon: {agent.epsilon:.2f}")


train()
