from gym.core import Wrapper
from gym.spaces import Box
import numpy as np


class FrameBuffer(Wrapper):
    # TODO: Feel free to update the frame buffer as you please
    # The code provided below is only a template
    def __init__(self, env, n_frames=4):
        """A gym wrapper that reshapes, crops and scales image into the desired shapes"""
        super(FrameBuffer, self).__init__(env)
        height, width, n_channels = env.observation_space.shape
        obs_shape = (height, width, n_channels * n_frames)
        frame_buffer_shape = (height, width, n_channels * n_frames)
        self.observation_space = Box(0.0, 1.0, obs_shape)
        self.framebuffer = np.zeros(frame_buffer_shape, 'float32')

    def reset(self):
        """resets breakout, returns initial frames"""
        self.framebuffer = np.zeros_like(self.framebuffer)
        self.update_buffer(self.env.reset())
        return self.framebuffer

    def step(self, action):
        """plays breakout for 1 step, returns frame buffer"""
        new_img, reward, done, info = self.env.step(action)
        self.update_buffer(new_img)
        return self.framebuffer, reward, done, info

    def update_buffer(self, img):
        offset = self.env.observation_space.shape[2]
        axis = 2
        cropped_framebuffer = self.framebuffer[:, :, :-offset]
        self.framebuffer = np.concatenate(
            [img, cropped_framebuffer], axis=axis)
