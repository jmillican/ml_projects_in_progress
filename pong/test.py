import os
os.environ['MPLBACKEND'] = 'TkAgg'
os.environ['SDL_VIDEODRIVER'] = 'cocoa'  # Force Cocoa on macOS


import ale_py
import gymnasium as gym

gym.register_envs(ale_py)
env = gym.make('ALE/Pong', render_mode="human", continuous=True)