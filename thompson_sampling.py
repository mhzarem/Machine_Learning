import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
data = pd.read_csv('data/Ads_CTR_Optimisation.csv')
# to familiar with data
a = data.describe()

# implementing thompson
N = 10000
d = 10
numbers_of_rewards_1 = [0]*d
numbers_of_rewards_0 = [0]*d
ads_selected = []
total_reward = 0
for n in range(N):
    max_random = 0
    ad = 0
    for i in range(d):
        random_beta = random.betavariate(numbers_of_rewards_1[i]+1, numbers_of_rewards_0[i]+1)
        print("random_beta:{},max_random:{},I:{},N:{}".format(random_beta, max_random, i, n))
        if random_beta > max_random:
            max_random = random_beta
            ad = i

    ads_selected.append(ad)
    reward = data.values[n, ad]

    if reward == 1:
        numbers_of_rewards_1[ad] += 1
    else:
        numbers_of_rewards_0[ad] += 1

    total_reward += data.values[n, ad]

plt.hist(ads_selected)
plt.show()
