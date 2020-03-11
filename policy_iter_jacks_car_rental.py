import numpy as np

# Hyper parameter:
discount_factor = 0.9
location1_avg_request = 3
location1_avg_return = 3
location2_avg_request = 4
location2_avg_return = 2
accuracy = 50
reward_for_rental = 10
reward_for_moving = -2
max_car = 20

# Step 1: Init Parameter

# Total state space (0-20) at each side, element = Value function
state_value_func = np.zeros((max_car + 1, max_car + 1))

# Action, the number of car to return
# entry in [i,j] representing number of car to move when there's i car on left and j car on the right.
# Action (-5,5). -5 -> move 5 car to loc1, 7 -> move 5 car to loc2
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
    return np.power(avg, n) * np.exp((-avg)) / np.math.factorial(n)


prop_sum = 0

for i in range(10):
    prop_sum = prop_sum + get_prob_poisson(3, i)
    print(" avg: {}, number: {}, prob: {} ".format(4, i, get_prob_poisson(3, i)))

print("Sum of probability of getting [0:9] event happen: ", prop_sum)


# Step 2: Policy Evaluation
def expected_reward(state, action):
    """
    Return the one step expected return base on Bellman Equation with current policy (action)
    :param state: pair of integer representing number of car at [loc1, loc2]
    :param action: -5 <= action <= 5. -5 -> move 5 car to loc1, 7 -> move 5 car to loc2
    :return: Expected return for current policy
    """

    reward = 0

    # inner min to prevent more car to move to the location
    # outer max to prevent car number to be less than 0
    new_state_loc1 = max(min(state[0] - action, max_car), 0)
    new_state_loc2 = max(min(state[1] + action, max_car), 0)

    # (negative) reward to move cars
    # assuming moving car will generate cost no matter success or not
    moving_cost = reward_for_moving * abs(action)
    reward = reward + moving_cost

    # All possible state after taking the action
    # sum of event happen from 0 times to 9 times is > 0.99 so let's just assume it cover all case.

    for rent_loc1 in range(10):
        for return_loc1 in range(10):
            for rent_loc2 in range(10):
                for return_loc2 in range(10):
                    # possible request
                    valid_rent_loc1 = min(rent_loc1, new_state_loc1)
                    valid_rent_loc2 = min(rent_loc2, new_state_loc2)

                    credit = (valid_rent_loc1 + valid_rent_loc2) * reward_for_rental

                    # new state
                    # update the new_state from env response to the action (next state)
                    # inner min to prevent more car to move to the location
                    # outer max to prevent car number to be less than 0

                    # Assuming there's always car in the market to be return...
                    # And bad naming...
                    # original state -> new (intermediate state) -> transit to the new (final) state (taking env repsonse into account)
                    # This is not so accurate tho, but than the state should take number of car on the market as consideration

                    next_state_loc1 = int(max(min(new_state_loc1 - valid_rent_loc1 + return_loc1, max_car), 0))
                    next_state_loc2 = int(max(min(new_state_loc2 - valid_rent_loc2 + return_loc1, max_car), 0))

                    # cumulate the updated expected base on bellman equation
                    total_prop = get_prob_poisson(location1_avg_request, rent_loc1) * \
                                 get_prob_poisson(location2_avg_request, rent_loc2) * \
                                 get_prob_poisson(location1_avg_return, return_loc1) * \
                                 get_prob_poisson(location2_avg_return, return_loc2)

                    reward = reward + total_prop * (credit + discount_factor * state_value_func[next_state_loc1][next_state_loc2])
                    #print("rent1 {} return1 {} rent2 {} return 2 {} P: {} : Reward {}".format(rent_loc1, return_loc1, rent_loc2, return_loc2, total_prop, reward))


    return reward


def policy_evaluation():
    """
    policy_evaluation
    :return: None
    """
    while True:

        error = 0

        for index, x in np.ndenumerate(state_value_func):
            #print("(Loc1 car, Loc2 car): {}, Value function: {}".format(index, x))
            print(".", end="")

            old_value_func = state_value_func[index[0]][index[1]]

            # Update the value function
            state_value_func[index[0]][index[1]] = expected_reward([index[0], index[1]], car_to_return[index[0]][index[1]])

            error = max(error, abs(state_value_func[index[0], index[1]] - old_value_func))
        print("\nCurrent Max error in Value func: {}".format(error))
        if error < accuracy:
            break

def policy_improvement():
    """
    policy_improvement
    :return: Bool, if the policy become stable which by the policy improvement theorem, it is the optimal policy
    """
    policy_stable = True
    for index, x in np.ndenumerate(state_value_func):
        old_action = car_to_return[index[0]][index[1]]

        max_q = -10000
        argmax_q = 0

        # Search through possible action to find the best action for current state
        # car on left: index[0]
        # possible car to rent at loc1: min(index[0],5) // account for state that have lesser car than 5
        # possible car to rent at loc2: -min(index[1],5) // negative for moving to loc1
        # +1 to accommodate the range() offset
        min_action = -min(index[1], 5)
        max_action = min(index[0]+1, 5+1)
        for possible_action in range(min_action, max_action):
            new_q = expected_reward([index[0], index[1]], possible_action)
            if index[0] == 10:
                test = None
            if new_q > max_q:
                # look ahead for the new action value
                max_q = new_q
                argmax_q = possible_action

        car_to_return[index[0]][index[1]] = argmax_q
        print(".", end="")

        if car_to_return[index[0]][index[1]] != old_action:
            policy_stable = False
    return policy_stable

iter = 0
while True:
    policy_evaluation()
    print("Policy Eva complete at iter {}".format(iter))
    print(state_value_func)
    terminate =  policy_improvement()
    print("Policy improvement complete at iter {}".format(iter))
    print(car_to_return)
    iter = iter + 1
    if terminate:
        break