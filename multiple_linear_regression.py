import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
data = pd.read_csv('data/50_Startups.csv')
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values
label = LabelEncoder()
hot = OneHotEncoder(categorical_features=[3], sparse=False)
X[:, 3] = label.fit_transform(X[:, 3])
X = hot.fit_transform(X)
X = X[:, 1:]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
regression = LinearRegression()
regression.fit(X_train, Y_train)
y_pred = regression.predict(X_test)
rms = mean_squared_error(Y_test, y_pred)/np.size(y_pred)
print(rms)


# try optimal model using Backward Elimination
import statsmodels.formula.api as sm
# add one colume to X for b0

X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
X_optimal = X[:, [0, 1, 2, 3, 4, 5]]
ols = sm.OLS(endog=Y, exog=X_optimal).fit()
ols.summary()


