import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori
data = pd.read_csv('data/Market_Basket_optimisation.csv', header=None)
transaction = []
for i in range(0, 7501):
    transaction.append([str(data.values[i, j]) for j in range(20)])
min_support = 0.003
min_confidence = 0.2
min_lift = 3
rules = apriori(transaction, min_support=min_support, min_confidence=min_confidence, min_lift=min_lift, max_length=2)
results = np.array(list(rules))





