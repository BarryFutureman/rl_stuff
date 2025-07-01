from red_tide_the_game import *
import pygame
import pygame.math as pgm


class HumanUser:
    def __init__(self, world):
        self.world = world

        self.entity_in_control = None
        self.entity_actions = [0, 0, 0]
        self.hover_block = None

    def update(self, events):
        if not self.entity_in_control:
            return

        direction = 0
        reach_point = -1
        reach_action = 0

        # Hover control
        self.hover_block = None
        for block in self.entity_in_control.reachable_blocks:
            if block and block.get_rect(self.world).collidepoint(pygame.mouse.get_pos()):
                block.mix_extra_color(pgm.Vector3(95, 95, 95))
                self.hover_block = block
                reach_point = self.entity_in_control.reachable_blocks.index(block)

        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    direction = 1
                elif event.key == pygame.K_d:
                    direction = 2
                elif event.key == pygame.K_s:
                    direction = 3
                elif event.key == pygame.K_a:
                    direction = 4
                if event.key == pygame.K_UP:
                    direction = 1
                elif event.key == pygame.K_RIGHT:
                    direction = 2
                elif event.key == pygame.K_DOWN:
                    direction = 3
                elif event.key == pygame.K_LEFT:
                    direction = 4
                elif event.key == pygame.K_1:
                    reach_action = 1
                    """if self.hover_block:
                        self.entity_in_control.eat(self.entity_in_control.reachable_blocks[reach_point])
                    else:
                        self.hover_block = None"""

        self.entity_actions[0] = direction
        self.entity_actions[1] = reach_point
        self.entity_actions[2] = reach_action

    def get_entity_actions(self):
        return self.entity_actions
