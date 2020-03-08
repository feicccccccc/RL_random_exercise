import numpy as np

# Hyper parameter:
discount_factor = 0.9
location1_avg_request = 3
location1_avg_return = 3
location2_avg_request = 3
location2_avg_return = 3
accuracy = 0.01
reward_for_rental = 10
reward_for_moving = -2
max_car = 20

# Step 1: Init Parameter

# Total state space (0-20) at each side, element = Value function
state = np.zeros((max_car + 1, max_car + 1))

# Action, the number of car to return
# entry in [i,j] representing number of car to move when there's i car on left and j car on the right.
car_to_return = np.zeros((max_car + 1, max_car + 1))


def get_prob_poisson(avg, n):
    """
    Return the probability base on poisson distribution.
    Poisson distribution is the discrete probability distribution of the number of events occurring in a given time period,
    given the average number of times the event occurs over that time period.
    :param avg: average lambda
    :param n: number of time
    :return: probability
    """
    return np.power(avg, n) / np.math.factorial(n) * np.exp((-avg))


prop_sum = 0

for i in range(10):
    prop_sum = prop_sum + get_prob_poisson(3, i)
    print(" avg: {}, number: {}, prob: {} ".format(3, i, get_prob_poisson(3, i)))

print("Sum of probability of getting [0:9] event happen: ", prop_sum)

# Step 2: Policy Evaluation
error = 0

while True:
    for i in state:
        print(i)
    break

