"""
Implement first time Monte carlo policy evaluation base on Sutton and David Silver course
Gotta implement it to learn it ;)

Using gym blackjack env for test
ref:
- gym env:
https://github.com/openai/gym/blob/master/gym/envs/toy_text/blackjack.py

"""

import gym

env = gym.make('Blackjack-v0')

print("Action space: {}:\n 0 = stick, 1 = hit".format(env.action_space))
print("Observation space: {}:\n (current sum, dealer showing card, usable ace)".format(env.observation_space))

# Demo Game to explain game mechanic

env.seed(1)  # Set env random seed
observation = env.reset()  # First observation

print("First env observation: {}".format(observation))
# Observing (13, 1, False), We should hit since 13 is too small
observation, reward, done, info = env.step(1)  # hit = 1
print(observation)  # (18, 1, False), We get a card 5 but now we decide to stop
print(reward)  # reward obtained
print(done)  # finish episode or not?
print(info)  # Empty

print("---Next move---")

observation, reward, done, info = env.step(0)  # Stick = 0
print(observation)  # same since we stick
print(reward)  # 1 reward obtained
print(done)  # finish episode = true
print(info)  # Empty


