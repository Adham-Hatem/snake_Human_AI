import pygame
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import heapq

pygame.init()
WIDTH, HEIGHT = 200, 200
CELL_SIZE = 20
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake DRL")
clock = pygame.time.Clock()

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
DARK_GREEN = (60, 155, 60)
RED = (255, 0, 0)

MAX_MEMORY = 10000
BATCH_SIZE = 64
LR = 0.001
GAMMA = 0.9
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.998

def food_reachable(snake, food):
    head = snake[-1]
    body = set(snake[1:])

    open_list = []
    heapq.heappush(open_list, (0 + manhattan(head, food), 0, head))

    visited = set()

    while open_list:
        f, g, current = heapq.heappop(open_list)
        if current == food:
            return True
        if current in visited:
            continue
        visited.add(current)

        x, y = current
        for dx, dy in [(0, -CELL_SIZE), (0, CELL_SIZE), (-CELL_SIZE, 0), (CELL_SIZE, 0)]:
            nx, ny = x + dx, y + dy
            next_pos = (nx, ny)
            if 0 <= nx < WIDTH and 0 <= ny < HEIGHT and next_pos not in body and next_pos not in visited:
                g_next = g + 1
                f_next = g_next + manhattan(next_pos, food)
                heapq.heappush(open_list, (f_next, g_next, next_pos))

    return False


def manhattan(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def spawn_food(snake):
    while True:
        x = random.randrange(0, WIDTH, CELL_SIZE)
        y = random.randrange(0, HEIGHT, CELL_SIZE)
        if (x, y) not in snake:
            return x, y

def get_state(snake, food, direction):
    head_x, head_y = snake[-1]
    body = snake[:-1]

    danger_up = (head_x, head_y - CELL_SIZE) in body or head_y - CELL_SIZE < 0
    danger_down = (head_x, head_y + CELL_SIZE) in body or head_y + CELL_SIZE >= HEIGHT
    danger_left = (head_x - CELL_SIZE, head_y) in body or head_x - CELL_SIZE < 0
    danger_right = (head_x + CELL_SIZE, head_y) in body or head_x + CELL_SIZE >= WIDTH

    food_dx = (food[0] - head_x) / WIDTH
    food_dy = (food[1] - head_y) / HEIGHT

    def look(dx, dy):
        x, y = head_x, head_y
        distance = 0
        while True:
            x += dx
            y += dy
            distance += 1
            if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT or (x, y) in body:
                return distance / (WIDTH // CELL_SIZE)

    dist_up = look(0, -CELL_SIZE)
    dist_down = look(0, CELL_SIZE)
    dist_left = look(-CELL_SIZE, 0)
    dist_right = look(CELL_SIZE, 0)
    dist_up_left = look(-CELL_SIZE, -CELL_SIZE)
    dist_up_right = look(CELL_SIZE, -CELL_SIZE)
    dist_down_left = look(-CELL_SIZE, CELL_SIZE)
    dist_down_right = look(CELL_SIZE, CELL_SIZE)

    def free_space_count():
        visited = set(snake)
        queue = deque([snake[-1]])
        count = 0
        while queue:
            x, y = queue.popleft()
            for dx, dy in [(0, -CELL_SIZE), (0, CELL_SIZE), (-CELL_SIZE, 0), (CELL_SIZE, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < WIDTH and 0 <= ny < HEIGHT and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
                    count += 1
        return count / ((WIDTH // CELL_SIZE) * (HEIGHT // CELL_SIZE))

    tail_x, tail_y = snake[0]
    tail_distance = (abs(head_x - tail_x) + abs(head_y - tail_y)) / (WIDTH + HEIGHT)

    state = [
        dist_up, dist_down, dist_left, dist_right,
        dist_up_left, dist_up_right, dist_down_left, dist_down_right,
        food_dx, food_dy,
        int(direction == (0, -CELL_SIZE)),
        int(direction == (0, CELL_SIZE)),
        int(direction == (-CELL_SIZE, 0)),
        int(direction == (CELL_SIZE, 0)),
        int(danger_up), int(danger_down), int(danger_left), int(danger_right),
        free_space_count(),
        tail_distance
    ]

    return np.array(state, dtype=float)


class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        return self.linear3(x)

class Agent:
    def __init__(self):
        self.model = DQN(20, 128, 4)
        self.memory = deque(maxlen=MAX_MEMORY)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_END

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0)
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
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.bool).unsqueeze(1)

        pred = self.model(states)
        with torch.no_grad():
            next_pred = self.model(next_states)
            target_q = rewards + GAMMA * torch.max(next_pred, dim=1, keepdim=True)[0] * (~dones)

        loss = nn.MSELoss()(pred.gather(1, actions), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= EPSILON_DECAY

def train():
    agent = Agent()
    generation = 0
    speed = 250
    unreachable_counter = 0
    unreachable_threshold = 3
    counter = 0

    while True:
        snake = [(WIDTH // 2, HEIGHT // 2)]
        direction = (0, -CELL_SIZE)
        food = spawn_food(snake)
        score = 0
        snake_length = 1
        done = False
        steps_since_food = 0
        path_exists = True
        needs_bfs = True

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        speed += 5
                    elif event.key == pygame.K_DOWN:
                        speed = max(1, speed - 5)
                    elif event.key == pygame.K_LEFT:
                        speed = 30
                    elif event.key == pygame.K_RIGHT:
                        speed = 250

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
            reward += 0.01

            will_eat = (new_head == food)
            body = snake if will_eat else snake[1:]

            if new_head in body or new_head[0] < 0 or new_head[0] >= WIDTH or new_head[1] < 0 or new_head[1] >= HEIGHT:
                reward = -50
                done = True
                print("\nDeath By wall/tail")
            else:
                snake.append(new_head)

                if needs_bfs or not path_exists:
                    path_exists = food_reachable(snake, food)
                    needs_bfs = False

                if not path_exists:
                    unreachable_counter += 1
                    if unreachable_counter >= unreachable_threshold:
                        reward -= 5
                        print("Food Not Reachable")
                else:
                    unreachable_counter = 0


                if will_eat:
                    score += 1
                    reward = 50
                    food = spawn_food(snake)
                    snake_length += 1
                    steps_since_food = 0
                    needs_bfs = True

                else:
                    steps_since_food += 1

                if len(snake) > snake_length:
                    snake.pop(0)

            if steps_since_food > 300:
                reward = -30
                done = True
                print("\nDeath By Starvation")

            next_state = get_state(snake, food, direction)
            agent.remember(state, action, reward, next_state, done)
            agent.train_long_memory()

            screen.fill(BLACK)
            pygame.draw.rect(screen, RED, (*food, CELL_SIZE, CELL_SIZE))
            for i, segment in enumerate(snake):
                color = DARK_GREEN if i % 2 == 0 else GREEN
                pygame.draw.rect(screen, color, (*segment, CELL_SIZE, CELL_SIZE))
            pygame.display.flip()
            clock.tick(speed)

        generation += 1
        print(f"Generation {generation} | Score: {score} | Epsilon: {agent.epsilon:.2f}")

        if generation % 50 == 0:
            agent.epsilon = min(1.0, agent.epsilon + 0.03)
            counter += 1

        if generation % 100 == 0:
            agent.epsilon_min *= 0.9
            agent.epsilon_min = max(agent.epsilon_min, 0.01)

train()
