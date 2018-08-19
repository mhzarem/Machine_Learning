import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

data = pd.read_csv('data/Position_Salaries.csv')
X = data.iloc[:, 1:2].values
Y = data.iloc[:, 2].values

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X.reshape(-1, 1))
Y = sc_Y.fit_transform(Y.reshape(-1, 1))



regression = SVR(kernel='rbf')
regression.fit(X, Y)

plt.scatter(sc_X.inverse_transform(X), sc_Y.inverse_transform(Y), color='red')
plt.plot(sc_X.inverse_transform(X), sc_Y.inverse_transform( regression.predict(X)), color='blue')
plt.show()