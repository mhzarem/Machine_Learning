import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
data = pd.read_csv('data/Position_Salaries.csv')
X = data.iloc[:, 1:2].values
Y = data.iloc[:, 2].values

regressor = RandomForestRegressor(n_estimators=100000, random_state=0)
regressor.fit(X, Y)

# visualising
x_grid = np.arange(min(X), max(X), 0.01)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(X, Y, color='red')
plt.plot(x_grid, regressor.predict(x_grid), color='blue')
plt.show()