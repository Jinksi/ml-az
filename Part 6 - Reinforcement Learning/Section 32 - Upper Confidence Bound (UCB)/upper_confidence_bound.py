# Upper Confidence Bound

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# Import the dataset
# each row in this dataset is a round over time
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
dataset

# Implementing UCB
# N = rounds
N = len(dataset)
# d = options available
d = len(dataset.values[0])

# Vector of size d containing 0
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
ads_selected = []
total_reward = 0

for n in range(0, N):
    # each round
    max_upper_confidence_bound = 0
    ad = 0

    for i in range(0, d):
        # each option
        if numbers_of_selections[i] > 0:
            # after first round
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            # UCB formula
            # n + 1 to accound for array indices
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_confidence_bound = average_reward + delta_i
        else:
            # set a very large upper_confidence_bound on the first round
            upper_confidence_bound = 1e400

        if upper_confidence_bound > max_upper_confidence_bound:
            # update max_upper_confidence_bound if this upper_confidence_bound is larger
            max_upper_confidence_bound = upper_confidence_bound
            ad = i

    # add this selected add to the list of ads_selected
    ads_selected.append(ad)
    # increment the numbers_of_selections
    numbers_of_selections[ad] += 1
    # get the reward
    reward = dataset.values[n, ad]
    # increment the sums_of_rewards
    sums_of_rewards[ad] += reward
    # increment the total reward
    total_reward += reward

plt.bar(range(0, len(numbers_of_selections)), numbers_of_selections)
plt.title('Numbers of Selections')
plt.show()

plt.hist(ads_selected)
plt.title('Histogram of ads selected')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

total_reward

# Random Selection for comparision
import random
N = 10000
d = 10
ads_selected_random = []
total_reward_random = 0
for n in range(0, N):
    ad = random.randrange(d)
    ads_selected_random.append(ad)
    reward = dataset.values[n, ad]
    total_reward_random += reward

# Visualising the results
plt.hist(ads_selected_random)
plt.title('Ads selections by random')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

total_reward, total_reward_random
plt.bar(['UCB Total Reward', 'Random Selection Total Reward'], [total_reward, total_reward_random])
plt.ylabel('Reward')
plt.show()
