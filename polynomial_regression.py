import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
data = pd.read_csv('data/Position_Salaries.csv')
X = data.iloc[:, 1:2].values
Y = data.iloc[:, 2].values
# plt.scatter(X, Y)
# plt.show()

# make linear regression
regression_linear = LinearRegression()
regression_linear.fit(X, Y)

# make polynomial regression
degree = 4
regression_poly = PolynomialFeatures(degree=degree)
X_poly = regression_poly.fit_transform(X)
# plt.scatter(X_poly[:, -1], Y)
# plt.show()

regression_linear_2 = LinearRegression()
regression_linear_2.fit(X_poly, Y)

# visualising the models

Y_lin = regression_linear.predict(X)
Y_poly = regression_linear_2.predict(X_poly)

plt.scatter(X, Y)
plt.plot(X, Y_lin)
plt.plot(X, Y_poly)
plt.title('degree of polynomial is :{}'.format(degree))
plt.show()
