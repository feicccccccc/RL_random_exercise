"""
Implement first time Monte carlo policy evaluation base on Sutton and David Silver course
Gotta implement it to learn it ;)

Using gym blackjack env for test
ref:
- gym env:
https://github.com/openai/gym/blob/master/gym/envs/toy_text/blackjack.py

"""
import numpy as np
import gym
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

env = gym.make('Blackjack-v0')

'''
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
'''

# Start First time Visit MC Policy evaluation

# Hyper Parameter
num_episode = 500000
num_visit = np.zeros((22, 11, 2))  # number of state, for simplicity leave 0 in the array
value_func_estimation = np.zeros((22, 11, 2))  # estimation of value function
discount_factor = 0.9

# Naive Policy
def policy_naive(obs):
    """
    Stick if player's sum = 20, 21
    :param obs: Observation from Env 0-21
    :return: Action 0/1
    """
    if obs[0] >= 20:
        action = 0
    else:
        action = 1
    return action

env.seed(940920)

for episode in range(num_episode):
    observation = env.reset()
    observation = observation[0], observation[1], 1 if observation[2] else 0
    # Run one game
    history = []
    while True:
        action = policy_naive(observation)
        next_observation, reward, done, info = env.step(action)
        # turn usable ace into small more easy to work with
        next_observation = next_observation[0], next_observation[1], 1 if next_observation[2] else 0
        # Count how many time visit the state
        num_visit[observation[0]][observation[1]][observation[2]] = num_visit[observation[0]][observation[1]][observation[2]] + 1
        history.append((observation, action, reward))
        print("Forward ", (observation, action, reward))
        observation = next_observation
        if done:
            break

    reward = 0
    for t, backward in enumerate(reversed(history)):
        # print("backward ", len(history) - t, backward)
        reward = discount_factor * reward + backward[2]
        # First time visit
        # Updating backward
        if backward not in history[:len(history) - t - 1]:
            cur_card_sum = backward[0][0]
            cur_dealer = backward[0][1]
            cur_usable_ace = backward[0][2]

            # Incremental update for value function
            value_func_estimation[cur_card_sum][cur_dealer][cur_usable_ace] = \
                value_func_estimation[cur_card_sum][cur_dealer][cur_usable_ace] + \
                (reward - value_func_estimation[cur_card_sum][cur_dealer][cur_usable_ace]) / num_visit[cur_card_sum][cur_dealer][cur_usable_ace]

    if episode % 100:
        print("game {}".format(episode))

# Plot the Value function
fig = plt.figure()
ax = fig.gca(projection='3d')
ax2 = fig.gca(projection='3d')
X = np.arange(11)
Y = np.arange(22)
X, Y = np.meshgrid(X, Y)
Z = value_func_estimation

surf = ax.plot_surface(X, Y, Z[:, :, 0], cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_zlim(-1.01, 1.01)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
