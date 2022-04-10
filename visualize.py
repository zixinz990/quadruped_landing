import numpy as np
import gym 


if __name__ == "__main__":
    init_qpos = np.array([0,6,1,0,0,0,0])
    """
    qpos: [x, height, rotation, angles of each joint]
    """
    env = gym.make("HalfCheetah-v3",exclude_current_positions_from_observation=False, init_qpos = init_qpos)
    env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        env.render()
