import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
data = pd.read_csv('data/Ads_CTR_Optimisation.csv')
# to familiar with data
a = data.describe()

# random selection ads and compute reward
N = 10000
d = 10
ads_selected = []
total_reward = 0
for i in range(N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = data.values[i, ad]
    total_reward += reward

plt.hist(ads_selected)
plt.show()

# implementing UCB
N = 10000
d = 10
numbers_of_selections = [0]*d
sums_of_rewards = [0]*d
ads_selected = []
total_reward = 0
for n in range(N):
    max_upper_bound = 0
    ad = 0
    for i in range(d):
        if numbers_of_selections[i] > 0:
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = np.sqrt(3/2*np.log(n+1)/numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        # print("upper_bound:{},max_upper_bound:{},I:{},N:{}".format(upper_bound, max_upper_bound, i, n))
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i

    ads_selected.append(ad)
    numbers_of_selections[ad] += 1
    sums_of_rewards[ad] += data.values[n, ad]
    total_reward += data.values[n, ad]

plt.hist(ads_selected)
plt.show()
