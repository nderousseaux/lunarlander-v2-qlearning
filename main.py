import matplotlib.pyplot as plt
import gym
import gym_pytris


env = gym.make("My-awesome-tetris-v0",  apply_api_compatibility=True)


for episode in range(10):
    env.reset()

    while True:
        env.render()
        
        action = env.action_space.sample()

        next_state, reward, done, truncated, info = env.step(action)
                    
        if done:
            break