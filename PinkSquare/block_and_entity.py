from red_tide_the_game import *
import pygame
import pygame.math as pgm


class Entity:
    def __init__(self, world):
        self.world = world
        self.pos = pgm.Vector2(0, 0)
        # self.blocks_occupied = []

    def get_pushed(self, d):
        raise NotImplementedError


class OrganismJelly(Entity):
    def __init__(self, world):
        super().__init__(world)
        self.body_size = 3
        self.base_image = pygame.image.load(
            "Assets/PNGs/CellgularOrganismCubic.png").convert_alpha()
        self.reachable_blocks = []

        self.actions = (0, 0, 0)
        self.max_health = 50
        self.curr_health = self.max_health
        self.is_alive = True
        self.t = 0
        self.reward = 0
        self.last_dist = 100

    def update(self):
        actions = self.actions

        self.move(actions[0])
        self.reachable_blocks = self.get_blocks_reach()
        # self.get_blocks_sense()
        self.apply_reach_action(actions[1], actions[2])

        self.add_health(-1)
        self.t += 1

        # Debug Stuff:
        b1 = self.world.get_block(self.pos.x, self.pos.y)
        b1.text_label = str(self.t)
        b2 = self.world.get_block(self.pos.x + 1, self.pos.y + 1)
        b2.text_label = str(self.curr_health)

    def apply_reach_action(self, point, action):
        if point >= 12 or point < 0:
            return
        selected_block = self.reachable_blocks[point]
        if selected_block is None or action == 0:
            return

        # == Apply Health reduction ==
        self.add_health(-1)
        # ============================

        if action == 1:
            self.eat(selected_block)

    def get_pushed(self, d):
        return False

    def move(self, d):
        # Apply Health reduction
        if d != 0:
            self.add_health(-1)
        # ======================

        if not self.push(d):
             #self.add_reward(-0.01)
            return False
        # elif d != 0:
            # self.add_reward(0.001)

        for b in self.get_occupied_blocks():
            b.occupying_entity = None
        self.pos += pgm.Vector2(self.world.clockwise[d])
        for b in self.get_occupied_blocks():
            b.occupying_entity = self

        # Add reward fpr getting close
        """
        if len(self.world.entities) >= 2:
            center = pgm.Vector2(self.pos.x + 1, self.pos.y + 1)
            for i in range(1, len(self.world.entities)):
                dist = center.distance_to(self.world.entities[i].pos)
                if dist < 8:
                    if self.last_dist > dist:
                        if dist > 2.5:
                            self.add_reward(0.01 / dist)
                        else:
                            self.add_reward(0.1)

                self.last_dist = dist
        """
        return True

    def push(self, d):
        d_vector = pgm.Vector2(self.world.clockwise[d])
        for i in range(self.body_size):
            if d_vector.x != 0:
                b = self.world.get_block(
                    self.pos.x + self.body_size//2 + (self.body_size//2 + 1) * d_vector.x,
                    self.pos.y + i)
            elif d_vector.y != 0:
                b = self.world.get_block(
                    self.pos.x + i,
                    self.pos.y + self.body_size//2 + (self.body_size//2 + 1) * d_vector.y)
            else:
                return False
            if b:
                b.extra_color = pgm.Vector3(255, 255, 255)
                if b.occupying_entity and not b.occupying_entity.get_pushed(d):
                    return False
            else:
                return False
        return True

    def get_occupied_blocks(self):
        lst = []
        for y in range(self.body_size):
            for x in range(self.body_size):
                b = self.world.get_block(self.pos.x + x, self.pos.y + y)
                if b:
                    lst.append(b)
        return lst

    def draw(self, display, render_scale):
        block_size = self.world.block_size

        draw_image = pygame.transform.smoothscale(self.base_image,
                                                  (
                                                  block_size * self.body_size * render_scale,
                                                  block_size * self.body_size * render_scale))
        draw_rect = draw_image.get_rect()
        draw_rect.topleft = (
                    self.world.world_pivot + self.pos * render_scale * block_size)
        display.blit(draw_image, draw_rect)

        """
        self.get_blocks_r()
        self.get_blocks_l()
        self.get_blocks_up()
        self.get_blocks_low()
        self.get_blocks_body()"""

    def eat(self, block):
        block.mix_extra_color(pgm.Vector3(235, 20, 50))
        if block.occupying_entity and isinstance(block.occupying_entity, SunlightAlgae):
            self.add_health(50)
            self.reward = 10
            print(self.world.destroy_entity_on_block(block))
        else:
            self.add_reward(-0.2)

    def add_health(self, amount):
        self.curr_health = min(self.max_health, self.curr_health + amount)
        if self.curr_health <= 0:
            self.is_alive = False

    def get_blocks_r(self):
        for i in range(self.body_size):
            b = self.world.get_block(self.pos.x + self.body_size,
                                     self.pos.y + i)
            if b:
                b.extra_color = pgm.Vector3(215, 235, 255)

    def get_blocks_l(self):
        for i in range(self.body_size):
            b = self.world.get_block(self.pos.x - 1,
                                     self.pos.y + i)
            if b:
                b.extra_color = pgm.Vector3(215, 235, 255)

    def get_blocks_up(self):
        for i in range(self.body_size):
            b = self.world.get_block(self.pos.x + i,
                                     self.pos.y - 1)
            if b:
                b.extra_color = pgm.Vector3(215, 235, 255)

    def get_blocks_low(self):
        for i in range(self.body_size):
            b = self.world.get_block(self.pos.x + i,
                                     self.pos.y + self.body_size)
            if b:
                b.extra_color = pgm.Vector3(215, 235, 255)

    def get_blocks_body(self):
        for y in range(self.body_size):
            for x in range(self.body_size):
                b = self.world.get_block(self.pos.x + x,
                                         self.pos.y + y)
                if b:
                    b.extra_color = pgm.Vector3(215, 235, 255)

    def get_blocks_reach(self):
        radius = 3
        block_lst = []
        for y in range(-radius, radius + 1):
            for x in range(-radius, radius + 1):
                if self.body_size//2 + 1 <= math.sqrt(x ** 2 + y ** 2) < radius - 0.5:
                    pos_x = (self.pos.x + x + radius // 2)
                    pos_y = (self.pos.y + y + radius // 2)
                    b = self.world.get_block(pos_x, pos_y)
                    if b:
                        b.mix_extra_color(pgm.Vector3(25, 25, 25))
                        block_lst.append(b)
                    else:
                        block_lst.append(None)
        return block_lst

    def get_blocks_sense(self):
        return self.get_blocks_reach()
        radius = 8
        block_lst = []
        for y in range(-radius, radius + 1):
            for x in range(-radius, radius + 1):
                if math.sqrt(x ** 2 + y ** 2) < radius:
                    pos_x = (self.pos.x - self.body_size + x + radius // 2)
                    pos_y = (self.pos.y - self.body_size + y + radius // 2)
                    b = self.world.get_block(pos_x,
                                             pos_y)
                    if b:
                        b.mix_extra_color(pgm.Vector3(25, 44, 66))
                        block_lst.append(b)
                    else:
                        block_lst.append(None)
        return block_lst

    def get_sense_data(self):
        data_lst = []
        blocks = self.get_blocks_sense()
        for block in blocks:
            if block is None:
                data_lst.append(-1)
                continue

            stuff = block.occupying_entity
            if stuff is None:
                data_lst.append(0)
            elif isinstance(stuff, OrganismJelly):
                data_lst.append(0)
            elif isinstance(stuff, SunlightAlgae):
                data_lst.append(2)
            else:
                data_lst.append(0)

        for d in range(len(data_lst)):
            if blocks[d]:
                blocks[d].text_label = str(data_lst[d])
        return data_lst

    def get_reward(self):
        r = self.reward
        self.reward = 0
        return r

    def add_reward(self, amount):
        self.reward += amount


class SunlightAlgae(Entity):
    def __init__(self, world, image=None):
        super().__init__(world)
        self.body_size = 1
        self.base_image = image
        if not image:
            self.base_image = pygame.image.load(
                "Assets/PNGs/Algae_Generated_Leaf_Like.png").convert_alpha()

        self.energy = 0

    def draw(self, display, render_scale):
        block_size = self.world.block_size

        draw_image = pygame.transform.smoothscale(self.base_image,
                                                  (
                                                      block_size * self.body_size * render_scale,
                                                      block_size * self.body_size * render_scale))
        draw_rect = draw_image.get_rect()
        draw_rect.topleft = (
                self.world.world_pivot + self.pos * render_scale * block_size)
        display.blit(draw_image, draw_rect)

    def update(self):
        self.energy += 1
        if self.energy >= 100:
            self.reproduce()

    def reproduce(self):
        return

    def move(self, d):
        if not self.push(d):
            return False

        self.get_occupied_block_single().occupying_entity = None
        self.pos += pgm.Vector2(self.world.clockwise[d])
        self.get_occupied_block_single().occupying_entity = self

        return True

    def get_occupied_blocks(self):
        lst = []
        for y in range(self.body_size):
            for x in range(self.body_size):
                b = self.world.get_block(self.pos.x + x,
                                                self.pos.y + y)
                if b:
                    lst.append(b)
        return lst

    def get_occupied_block_single(self):
        return self.world.get_block(self.pos.x, self.pos.y)

    def get_pushed(self, d):
        return self.move(d)

    def push(self, d):
        d_vector = pgm.Vector2(self.world.clockwise[d])
        for i in range(self.body_size):
            if d_vector.x != 0:
                b = self.world.get_block(
                    self.pos.x + d_vector.x,
                    self.pos.y + i)
            elif d_vector.y != 0:
                b = self.world.get_block(
                    self.pos.x + i,
                    self.pos.y + d_vector.y)
            else:
                return False
            if b:
                b.extra_color = pgm.Vector3(285, 285, 285)
                if b.occupying_entity and not b.occupying_entity.get_pushed(d):
                    return False
            else:
                return False
        return True


class HorizontalAlgae(SunlightAlgae):
    def __init__(self, world, image=None):
        super().__init__(world)

    def reproduce(self):
        algae = HorizontalAlgae(self.world, self.base_image)
        self.world.place_entity(algae, self.pos.x + 1, self.pos.y)
        algae = HorizontalAlgae(self.world, self.base_image)
        self.world.place_entity(algae, self.pos.x - 1, self.pos.y)

        self.world.destroy_entity_on_block(self.get_occupied_block_single())


class Block:
    def __init__(self, size, p, c):
        self.size = size
        self.pos = pgm.Vector2(p)
        self.occupying_entity = None
        self.color = pgm.Vector3(c)
        self.extra_color = None

        self.text_label = None

    def get_rect(self, world):
        render_scale = 1/world.cam_zoom
        return pygame.Rect(
            world.world_pivot.x + self.pos.x * render_scale * self.size,
            world.world_pivot.y + self.pos.y * render_scale * self.size,
            self.size * render_scale, self.size * render_scale)

    def __repr__(self):
        return str(self.pos)

    def get_display_color(self):
        if self.occupying_entity:
            # self.extra_color = pgm.Vector3(85, 98, 30)
            self.mix_extra_color(pgm.Vector3(25, 25, 25))

        c = self.color / 3
        if self.extra_color:
            new_c = pgm.Vector3(int((c.x + self.extra_color.x * 4) / 5),
                                int((c.y + self.extra_color.y * 4) / 5),
                                int((c.z + self.extra_color.z * 4) / 5))
            self.extra_color = None
            return new_c
        else:
            return c

    def get_edge_color(self):
        return self.color

    def mix_extra_color(self, c):
        if self.extra_color:
            new_c = pgm.Vector3(int((c.x + self.extra_color.x) / 2),
                                int((c.y + self.extra_color.y) / 2),
                                int((c.z + self.extra_color.z) / 2))
        else:
            new_c = pgm.Vector3(c)
        self.extra_color = new_c