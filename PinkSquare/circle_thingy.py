import pygame
import math
import time

# initialize Pygame
pygame.init()

# set up the display
WINDOW_SIZE = (400, 400)
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Circle Drawing")

# define colors
BLACK = (21, 21, 21)
WHITE = (245, 245, 245)
Blue = (54, 68, 95)


def draw_circle(radius, size, screen):
    for y in range(-radius, radius + 1):
        for x in range(-radius, radius + 1):
            if 3 <= math.sqrt(x ** 2 + y ** 2) < radius:
                rect = pygame.Rect((x + radius) * size, (y + radius) * size, size, size)
                pygame.draw.rect(screen, WHITE, rect)
                pygame.display.flip()
                # time.sleep(0.02)
            else:
                rect = pygame.Rect((x + radius) * size, (y + radius) * size, size, size)
                pygame.draw.rect(screen, Blue, rect)

    pygame.display.flip()
    time.sleep(3)


# main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # draw the circle
    screen.fill(BLACK)
    draw_circle(4, 20, screen)
    pygame.display.flip()

# quit Pygame
pygame.quit()