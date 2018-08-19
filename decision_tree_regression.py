import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
data = pd.read_csv('data/Position_Salaries.csv')
X = data.iloc[:, 1:2].values
Y = data.iloc[:, 2].values

regressor = DecisionTreeRegressor(min_samples_leaf=2)
regressor.fit(X, Y)

# visualising
x_grid = np.arange(min(X), max(X), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(X, Y, color='red')
plt.plot(x_grid, regressor.predict(x_grid), color='blue')
plt.show()