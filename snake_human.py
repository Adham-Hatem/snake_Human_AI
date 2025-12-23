import pygame
import random

pygame.init()

WIDTH, HEIGHT = 400, 400
CELL_SIZE = 20

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
DARK_GREEN = (60, 155, 60)
RED = (255, 0, 0)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake")

clock = pygame.time.Clock()

snake = [(WIDTH // 2, HEIGHT // 2)]
direction = (0, -CELL_SIZE)
snake_length = 1

def spawn_food(snake):
    while True:
        x = random.randrange(0, WIDTH, CELL_SIZE)
        y = random.randrange(0, HEIGHT, CELL_SIZE)
        food_pos = (x, y)
        if food_pos not in snake:
            return food_pos

food = spawn_food(snake)

running = True
while running:
    screen.fill(BLACK)

    next_direction = direction

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                if direction != (0, CELL_SIZE):
                    next_direction = (0, -CELL_SIZE)
            elif event.key == pygame.K_DOWN:
                if direction != (0, -CELL_SIZE):
                    next_direction = (0, CELL_SIZE)
            elif event.key == pygame.K_LEFT:
                if direction != (CELL_SIZE, 0):
                    next_direction = (-CELL_SIZE, 0)
            elif event.key == pygame.K_RIGHT:
                if direction != (-CELL_SIZE, 0):
                    next_direction = (CELL_SIZE, 0)

    direction = next_direction

    head_x, head_y = snake[-1]
    new_head = (head_x + direction[0], head_y + direction[1])

    if new_head == food:
        snake_length += 1
        food = spawn_food(snake)

    if len(snake) > snake_length:
        snake.pop(0)

    if new_head[0] < 0 or new_head[0] >= WIDTH or new_head[1] < 0 or new_head[1] >= HEIGHT \
            or new_head in snake:
        print("Game Over! Resetting...")
        snake = [(WIDTH // 2, HEIGHT // 2)]
        direction = (0, -CELL_SIZE)
        next_direction = direction
        snake_length = 3
        food = spawn_food(snake)
        continue

    elif new_head in snake[:-1]:
        print("Game Over! Hit self.")
        snake = [(WIDTH // 2, HEIGHT // 2)]
        direction = (0, -CELL_SIZE)
        next_direction = direction
        snake_length = 3
        food = spawn_food(snake)

    snake.append(new_head)

    pygame.draw.rect(screen, RED, (*food, CELL_SIZE, CELL_SIZE))

    for i, segment in enumerate(snake):
        color = DARK_GREEN if i % 2 == 0 else GREEN
        pygame.draw.rect(screen, color, (*segment, CELL_SIZE, CELL_SIZE))

    pygame.display.flip()
    clock.tick(10)  # FPS

pygame.quit()
