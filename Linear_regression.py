import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
data = pd.read_csv('data/Salary_Data.csv')
X = data.iloc[:, 0].values
Y = data.iloc[:, 1].values
plt.scatter(X, Y)
plt.show()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# fitting linear regression
from sklearn.linear_model import LinearRegression
regression = LinearRegression(n_jobs=-1)
X_train = X_train.reshape(-1, 1)
Y_train = Y_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)

regression.fit(X_train, Y_train)

y_predicted = regression.predict(X_test.reshape(-1, 1))
y_predicted =y_predicted.reshape(-1, 1)
# plot train set
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regression.predict(X_train), color='blue')
plt.show()

# plot test set
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_test, regression.predict(X_test), color='blue')
plt.show()

# compute rms
from sklearn.metrics import *
rms = mean_squared_error(Y_test, y_predicted)/np.size(y_predicted)
lms = mean_squared_log_error(Y_test, y_predicted)
print(rms)
print(lms)