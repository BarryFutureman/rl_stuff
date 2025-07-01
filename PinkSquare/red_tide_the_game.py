import pygame
import random
import pygame.math as pgm
import math
from enum import Enum
from collections import namedtuple
import numpy as np

from block_and_entity import *
from human_user_input_system import *

# Ideas
#  - A time variable in the organism's brain so it knows how old it is


pygame.init()

SPEED = 30


class WorldSimulation:
    def __init__(self, w=640, h=640):
        self.screen_width = w
        self.screen_height = h
        self.block_size = 128
        self.w = w//self.block_size
        self.h = h//self.block_size

        self.cam_block_size = self.block_size
        self.cam_zoom = 1
        self.world_pivot = pgm.Vector2(0, 0)
        self.world_pivot_offset = pgm.Vector2(0, 0)

        self.user = HumanUser(self)

        self.left_mouse_down = False
        self.debug_display = False
        self.score = 0

        # init display
        self.display = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('JellyTide')
        self.clock = pygame.time.Clock()
        self.clockwise = [(0, 0), (0, -1), (1, 0), (0, 1), (-1, 0)]
        self.blocks = []
        self.entities = []
        self.organisms = []

        self.generate_map(self.screen_width, self.screen_height)

    def generate_map(self, w, h):
        for y in range(0, h // self.block_size):
            row = []
            for x in range(0, w // self.block_size):
                row.append(Block(self.block_size, (x, y), (4, 28, 49)))
            self.blocks.append(row)

    def set_game(self):
        self.score = 0

        self.blocks = []
        self.entities = []
        self.organisms = []

        self.generate_map(self.screen_width, self.screen_height)

        jelly = OrganismJelly(self)
        # self.place_organism(jelly, 1, 1)
        self.place_organism(jelly, random.randint(0, self.w - 3),
                                        random.randint(0, self.h - 3))
        self.user.entity_in_control = jelly

        for i in range(1):
            algae = SunlightAlgae(self)
            while not self.place_entity(algae, random.randint(1, self.w - 1),
                                        random.randint(1, self.h - 1)):
                continue

    def place_entity(self, entity, x, y):
        entity.pos = pgm.Vector2(x, y)
        blocks = entity.get_occupied_blocks()
        if not blocks:
            return False
        for b in entity.get_occupied_blocks():
            if b.occupying_entity:
                return False
            b.occupying_entity = entity
        self.entities.append(entity)
        return True

    def place_organism(self, entity, x, y):
        self.organisms.append(entity)
        self.place_entity(entity, x, y)

    def get_block(self, x, y):
        x = int(x)
        y = int(y)
        if x >= self.w or x < 0 or y >= self.h or y < 0:
            return None
        else:
            return self.blocks[int(y)][int(x)]

    def destroy_entity_on_block(self, block):
        entity = block.occupying_entity
        if entity:
            block.occupying_entity = None
            self.entities.remove(entity)
            del entity
            return True
        else:
            return False

    def play_step(self, actions):
        # For AI
        reward = 0
        done = False
        score = 0

        # Events ==================
        events = pygame.event.get()

        self.user.update(events)

        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    global SPEED
                    if SPEED <= 30:
                        SPEED = 1440
                    else:
                        SPEED = 30
                elif event.key == pygame.K_r:
                    self.set_game()
                elif event.key == pygame.K_t:
                    if self.debug_display:
                        self.debug_display = False
                    else:
                        self.debug_display = True

            elif event.type == pygame.MOUSEWHEEL:
                zoom_amount = 0.1
                if event.y < 0:
                    if self.cam_zoom < 1.5:
                        self.cam_zoom = min(1.5, self.cam_zoom + zoom_amount)
                        self.world_pivot = self.world_pivot_offset * (
                                    1 / self.cam_zoom) - (
                                                       1 - self.cam_zoom) * pgm.Vector2(
                            self.screen_width, self.screen_height) / 2 * (1 / self.cam_zoom)
                else:
                    if self.cam_zoom > 0.1:
                        self.cam_zoom = max(0.1, self.cam_zoom - zoom_amount)
                        self.world_pivot = self.world_pivot_offset * (
                                    1 / self.cam_zoom) - (
                                                       1 - self.cam_zoom) * pgm.Vector2(
                            self.screen_width, self.screen_height) / 2 * (1 / self.cam_zoom)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.left_mouse_down = True
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.left_mouse_down = False

            elif event.type == pygame.MOUSEMOTION:
                render_scale = 1 / self.cam_zoom
                if self.left_mouse_down:
                    rel = event.rel
                    self.world_pivot_offset += pgm.Vector2(rel) / render_scale
                    self.world_pivot = self.world_pivot_offset * render_scale - (
                            1 - self.cam_zoom) * pgm.Vector2(self.screen_width,
                                                             self.screen_height) / 2 * (
                                               1 / self.cam_zoom)

        for o in self.organisms:
            o.actions = actions
        for e in self.entities:
            e.update()

        self._update_ui()

        if SPEED <= 30:
            self.clock.tick(SPEED)

        reward += self.user.entity_in_control.get_reward()
        done = not self.user.entity_in_control.is_alive or len(self.entities) < 2
        if done:
            # max_h = self.user.entity_in_control.max_health
            # reward = 0 # -1 + ((max(self.user.entity_in_control.t, max_h) - max_h) // 60) ** 2
            if len(self.entities) < 2:
                if reward > 0:
                    reward += 10
                for row in self.blocks:
                    for b in row:
                        b.color = pgm.Vector3(60, 125, 40)
            else:
                reward = 0
            # """
                for row in self.blocks:
                    for b in row:
                        b.color = pgm.Vector3(125, 20, 40)
            self._update_ui()
            # """

        self.score += reward
        score = self.score # self.user.entity_in_control.t
        # print((reward, done, score))
        # print(f"Health: {self.user.entity_in_control.curr_health} Age: {self.user.entity_in_control.t}")
        return reward, done, score

    def _update_ui(self):
        self.display.fill((0, 0, 0))
        b_size = self.block_size
        render_scale = 1 / self.cam_zoom
        curr_b_size = self.block_size * render_scale
        font = pygame.font.Font('Assets/Fonts/roguehero.ttf',
                                int(12 * render_scale))
        if not self.debug_display:
            for row in self.blocks:
                for block in row:
                    pygame.draw.rect(self.display, block.get_edge_color(),
                                     block.get_rect(self))
                    pygame.draw.rect(self.display, block.get_display_color(),
                                     pygame.Rect(
                                         self.world_pivot.x + block.pos.x * render_scale * b_size,
                                         self.world_pivot.y + block.pos.y * render_scale * b_size,
                                         curr_b_size - 1 * render_scale,
                                         curr_b_size - 1 * render_scale))

            for e in self.entities:
                e.draw(self.display, render_scale)

        if self.debug_display:
            for row in self.blocks:
                for block in row:
                    if block.text_label:
                        text = font.render(str(block.text_label), True,
                                           (50, 159, 50))
                        self.display.blit(text, pygame.Rect(
                                         self.world_pivot.x + block.pos.x * render_scale * b_size,
                                         self.world_pivot.y + block.pos.y * render_scale * b_size,
                                         curr_b_size - 1 * render_scale,
                                         curr_b_size - 1 * render_scale))
                        block.text_label = None

        # text = font.render("Score: " + str(0), True, BLACK)
        # hunger_text = font.render("Hunger: " + str(1), True, BLACK)
        # self.display.blit(text, [0, 0])
        # self.display.blit(hunger_text, [0, 30])
        pygame.display.flip()

    def _move(self, action):
        pass


if __name__ == '__main__':
    game = WorldSimulation()
    game.set_game()

    # game loop
    while True:
        game.user.entity_in_control.get_sense_data()
        game.play_step(game.user.get_entity_actions())
