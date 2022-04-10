import numpy as np
import gym 


if __name__ == "__main__":
    env = gym.make("HalfCheetah-v3",exclude_current_positions_from_observation=False)
    env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        env.render()
