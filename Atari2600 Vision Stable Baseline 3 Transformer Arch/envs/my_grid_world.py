import math
import random

import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Game stats
        self._agent_location = None
        self._target_location = None
        self.max_player_health: int = int(math.sqrt(2 * self.size**2) + 5)
        self.curr_player_health = self.max_player_health
        self.bg_color = (255, 255, 255)

        low = np.array(
            [
                0,
                0,
                0,
                0,
                0,
            ]
        ).astype(np.float32)
        high = np.array(
            [
                self.size - 1,
                self.size - 1,
                self.size - 1,
                self.size - 1,
                self.max_player_health
            ]
        ).astype(np.float32)
        self.obs_vec_size = low.shape[0]

        self.states = []
        self.memory_length = 2

        # useful range is -1 .. +1, but spikes can be higher
        # My edit here
        self.observation_space = spaces.Box(np.tile(low, (self.memory_length, 1)),
                                            np.tile(high, (self.memory_length, 1)))

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = "rgb_array"

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        agent_location_vec_x = self._agent_location[0] / (self.size - 1)
        agent_location_vec_y = self._agent_location[1] / (self.size - 1)
        target_location_vec_x = self._target_location[0] / (self.size - 1)
        target_location_vec_y = self._target_location[1] / (self.size - 1)
        health_vec = self.curr_player_health / self.max_player_health

        observation = np.array([agent_location_vec_x, agent_location_vec_y,
                                target_location_vec_x, target_location_vec_y,
                                health_vec], dtype=float)

        # Update the states
        self.states.pop(0)
        self.states.append(observation)

        return observation

    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset states
        self.states = [np.zeros((self.obs_vec_size,)) for _ in range(self.memory_length)]

        # Reset player
        self.size = random.randint(4, 16)
        self.max_player_health = int(math.sqrt(2 * self.size**2) + 5)
        self.curr_player_health = self.max_player_health

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        observation = np.array(self.states, dtype=np.float32)
        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 0

        if terminated:
            reward = 1
            self.bg_color = (188, 250, 120)

        # Reduce player health each step
        self.curr_player_health -= 1
        if self.curr_player_health <= 0:
            terminated = True
            reward = -1
            self.bg_color = (249, 180, 128)

        self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        observation = np.array(self.states, dtype=np.float32)
        return observation, reward, terminated, False, info

    def render(self):
        return self._render_frame()

    def _render_frame(self):
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(self.bg_color)
        pix_square_size = (
                self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (235, 128, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (198, 85, 107),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


