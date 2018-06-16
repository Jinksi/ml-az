# Thompson Sampling

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# Import the dataset
# each row in this dataset is a round over time
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
dataset

# Implementing Thompson Sampling
# N = rounds
N = len(dataset)
# d = options available
d = len(dataset.values[0])
N, d
# Vector of size d containing 0
ads_selected = []
# Vector of d length with 0 value
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
total_reward = 0

for n in range(0, N):
    # each round
    ad = 0
    max_random_draw = 0
    for i in range(0, d):
        # each option
        # select a random number in beta distribution
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if random_beta > max_random_draw:
            # update max_random_draw if this random_beta is larger
            max_random_draw = random_beta
            ad = i

    # add this selected add to the list of ads_selected
    ads_selected.append(ad)
    # get the reward
    reward = dataset.values[n, ad]
    # numbers_of_rewards_0
    if reward == 1:
        numbers_of_rewards_1[ad] += 1
    else:
        numbers_of_rewards_0[ad] += 1
    # increment the total reward
    total_reward += reward

plt.bar(range(0, len(numbers_of_rewards_1)), numbers_of_rewards_1)
plt.title('Numbers of Rewards 1')
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
plt.bar(['Thompson Sampling Total Reward', 'Random Selection Total Reward'], [total_reward, total_reward_random])
plt.ylabel('Reward')
plt.show()
