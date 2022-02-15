# !pip install opencv-python
# !pip install pillow
# !pip install gym
# !pip install matplotlib

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Box, Discrete
import numpy as np
import cv2
import random
import time

display_font = cv2.FONT_HERSHEY_COMPLEX_SMALL

# Constant for determining the screen size.
SIZE_OF_THE_SCREEN = 424, 430
# Border
BORDER = 40


class Collidable:
    def get_right_border(self):
        pass

    def get_left_border(self):
        pass

    def get_down_border(self):
        pass

    def get_up_border(self):
        pass


class SpaceShip(Collidable):
    def __init__(self):
        self.offset = 210
        self.h = 40
        self.low_end = SIZE_OF_THE_SCREEN[1] - self.h - 5
        self.d = 5
        self.ship_color = (100, 100, 100)

    def set_left_limit(self, val):
        self.left_limit = max(0, val)

    def set_right_limit(self, val):
        self.right_limit = min(SIZE_OF_THE_SCREEN[1], val)

    def get_right_border(self):
        return self.offset + self.d * 4

    def get_left_border(self):
        return self.offset

    def get_down_border(self):
        return self.low_end

    def get_up_border(self):
        return self.low_end + self.h + 10

    def move_right(self):
        if self.get_right_border() >= self.right_limit:
            return
        self.offset += 6

    def move_left(self):
        if self.get_left_border() <= self.left_limit:
            return
        self.offset -= 6

    def get_tip(self):
        return [self.offset + 3 * self.d // 2, self.low_end - self.h // 5]


class BadBlock(Collidable):
    def __init__(self):
        self.center = [random.randint(20, SIZE_OF_THE_SCREEN[0] - 20), 20]
        self.r = 10
        self.hp = 2
        self.colors = [(200, 200, 0), (200, 100, 0), (200, 0, 0)]
        self.velocity = [random.choice([-4, 4]), 2]

    def get_up_border(self):
        return self.center[1] + self.r

    def get_down_border(self):
        return self.center[1] - self.r

    def get_left_border(self):
        return self.center[0] - self.r

    def get_right_border(self):
        return self.center[0] + self.r

    def move(self):
        prv1, prv2 = self.center
        self.center[0] += self.velocity[0]
        self.center[1] += self.velocity[1]
        if self.get_left_border() <= 0 or self.get_right_border() >= SIZE_OF_THE_SCREEN[0]:
            self.velocity[0] *= -1
            self.center = [prv1, prv2]
        if self.get_up_border() >= SIZE_OF_THE_SCREEN[1]:
            self.hp = -1

    def hit(self):
        # Return true if dead
        self.hp -= 1
        return self.hp < 0


class Bullet(Collidable):
    def __init__(self, center_x, center_y):
        self.center = [center_x, center_y]
        self.r = 4
        self.color = (255, 255, 255)

    def get_up_border(self):
        return self.center[1] + self.r

    def get_down_border(self):
        return self.center[1] - self.r

    def get_left_border(self):
        return self.center[0] - self.r

    def get_right_border(self):
        return self.center[0] + self.r

    def move(self):
        self.center[1] -= 4


def check_collide(A: Collidable, B: Collidable):
    l1, l2 = A.get_left_border(), B.get_left_border()
    r1, r2 = A.get_right_border(), B.get_right_border()
    if max(l1, l2) > min(r1, r2):
        return False
    d1, d2 = A.get_down_border(), B.get_down_border()
    u1, u2 = A.get_up_border(), B.get_up_border()
    if max(d1, d2) > min(u1, u2):
        return False
    return True


class ShootNDodgeEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        self.canvas = np.zeros((SIZE_OF_THE_SCREEN[0] + BORDER * 2, SIZE_OF_THE_SCREEN[1] + BORDER * 2, 3),
                               dtype=np.int32)
        self.start_time = time.time()
        self.capture = 0
        self.kill_count = 0
        self.hit_count = 0

        self.space_ship = SpaceShip()
        moving_range = (SIZE_OF_THE_SCREEN[0] // 2 - SIZE_OF_THE_SCREEN[0] // 3,
                        SIZE_OF_THE_SCREEN[0] // 2 + SIZE_OF_THE_SCREEN[0] // 3)
        self.space_ship.set_left_limit(moving_range[0])
        self.space_ship.set_right_limit(moving_range[1])
        self.bad_blocks = []
        self.Lambda = 0.02

        self.time_left = np.random.exponential(scale=1 / self.Lambda, size=1)[0]

        self.bullets = []
        self.INTER_SHOOTING_TIME = 20
        self.time_to_shoot = self.INTER_SHOOTING_TIME

        # Observation and Action spaces
        self.observation_space = Box(0.0, 255.0,
                                     (SIZE_OF_THE_SCREEN[0] + BORDER * 2, SIZE_OF_THE_SCREEN[1] + BORDER * 2, 3))
        self.action_space = Discrete(3)

    def seed(self, seed=None):
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(seed)

    def render(self, mode='human', close=False):
        assert mode not in self.metadata, "Invalid mode. Must be either human or rgb_array"
        if mode == 'human':
            cv2.imshow("Game", self.canvas)
            cv2.waitKey(10)
        elif mode == 'rgb_array':
            return self.canvas

    def reset(self):
        self.__init__()
        self.draw_all()
        return self.canvas

    def close(self):
        cv2.destroyAllWindows()

    def calculate_current_score(self):
        time_elapsed = round(time.time() - self.start_time)
        return time_elapsed + 3 * self.hit_count + 100 * self.kill_count

    def clear_canvas(self):
        self.canvas = np.zeros((SIZE_OF_THE_SCREEN[0] + BORDER * 2, SIZE_OF_THE_SCREEN[1] + BORDER * 2, 3),
                               dtype=np.int32)

    def draw_rect(self, sX, sY, LX, LY, color):
        for x in range(sX, sX + LX):
            for y in range(sY, sY + LY):
                self.canvas[BORDER + x, BORDER + y] = np.array(color, dtype=np.int32)

    def draw_all(self):

        border_cell = np.array([255, 255, 255], dtype=np.int32)
        for x in range(SIZE_OF_THE_SCREEN[0]):
            self.canvas[BORDER - 1, BORDER + x] = border_cell
            self.canvas[BORDER + SIZE_OF_THE_SCREEN[1], BORDER + x] = border_cell
        for x in range(SIZE_OF_THE_SCREEN[1]):
            self.canvas[BORDER + x, BORDER - 1] = border_cell
            self.canvas[BORDER + x, BORDER + SIZE_OF_THE_SCREEN[0]] = border_cell

        self.draw_rect(self.space_ship.offset, self.space_ship.low_end, self.space_ship.d, self.space_ship.h,
                       self.space_ship.ship_color)
        self.draw_rect(self.space_ship.offset + self.space_ship.d, self.space_ship.low_end - self.space_ship.h // 5,
                       self.space_ship.d,
                       self.space_ship.h, self.space_ship.ship_color)
        self.draw_rect(self.space_ship.offset + 2 * self.space_ship.d, self.space_ship.low_end, self.space_ship.d,
                       self.space_ship.h, self.space_ship.ship_color)
        # DRAW BULLETS
        for bullet in self.bullets:
            self.draw_rect(bullet.center[0] - bullet.r, bullet.center[1] - bullet.r, 2 * bullet.r, 2 * bullet.r,
                           bullet.color)

        # DRAW BLOCKS
        for block in self.bad_blocks:
            self.draw_rect(block.center[0] - block.r, block.center[1] - block.r, 2 * block.r, 2 * block.r,
                           block.colors[block.hp])

        # DISPLAY SCORE
        score_text = "Score: {}".format(self.calculate_current_score())
        self.canvas = cv2.putText(self.canvas, score_text, (10, 20), display_font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    def step(self, target):
        self.clear_canvas()
        prv_kill_count = self.kill_count
        prv_hit_count = self.hit_count

        terminal = False

        # HANDLE MOVEMENT OF SPACESHIP
        if target == 0:
            self.space_ship.move_left()
        if target == 2:
            self.space_ship.move_right()

        # HANDLE BLOCK MOVEMENT
        for block in self.bad_blocks:
            block.move()

        # HANDLE BLOCK CREATION
        self.time_left -= 1
        if self.time_left <= 0 or len(self.bad_blocks) == 0:
            self.bad_blocks.append(BadBlock())
            self.time_left = self.time_left = np.random.exponential(scale=1 / self.Lambda, size=1)[0]

        # HANDLE SHOOTING
        self.time_to_shoot -= 1
        if self.time_to_shoot == 0:
            self.time_to_shoot = self.INTER_SHOOTING_TIME
            self.bullets.append(Bullet(center_x=self.space_ship.get_tip()[0], center_y=self.space_ship.get_tip()[1]))

        # MOVE BULLETS
        new_bullets = []
        for bullet in self.bullets:
            bullet.move()
            if bullet.center[0] < 1 or bullet.center[1] < 1:
                continue
            collides = False
            for block in self.bad_blocks:
                if check_collide(block, bullet):
                    is_dead = block.hit()
                    if is_dead:
                        self.kill_count += 1
                    else:
                        self.hit_count += 1
                    collides = True
            if not collides:
                new_bullets.append(bullet)
        self.bullets = new_bullets

        # HANDLE BLOCK DYING
        new_blocks = []
        for block in self.bad_blocks:
            if block.hp >= 0:
                new_blocks.append(block)
        self.bad_blocks = new_blocks

        # HANDLE DYING
        for block in self.bad_blocks:
            if check_collide(block, self.space_ship):
                terminal = True

        self.draw_all()
        sub = -10 if terminal else 0
        return self.canvas, (self.kill_count - prv_kill_count) * 3 + (
                self.hit_count - prv_hit_count) * 1 + sub, terminal, "additional_info"
