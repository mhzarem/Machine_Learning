import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

data = pd.read_csv('data/Social_Network_Ads.csv')

X = data.iloc[:, 1:4].values
Y = data.iloc[:, -1].values

label = LabelEncoder()
X[:, 0] = label.fit_transform(X[:, 0])

# it's important to scaling
sc = StandardScaler()
X[:, 1:3] = sc.fit_transform(X[:, 1:3])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

classifier_without_gender = KNeighborsClassifier(n_neighbors=2, metric='minkowski', p=2)
classifier_with_gender = KNeighborsClassifier(n_neighbors=2, metric='minkowski', p=2)

classifier_with_gender.fit(X_train, Y_train)
classifier_without_gender.fit(X_train[:, 1:], Y_train)

y_pre_with = classifier_with_gender.predict(X_test)
y_pre_without = classifier_without_gender.predict(X_test[:, 1:])
# confusion matrix
cm_with = confusion_matrix(Y_test, y_pre_with)
cm_without = confusion_matrix(Y_test, y_pre_without)

# visualising training set

X_set, Y_set = X_train[:, 1:], Y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.001),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.001))

plt.contourf(X1, X2, classifier_without_gender.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.7, cmap=ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

d = np.unique(Y_set)
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.legend()
plt.title('training set')
plt.show()

# visualising test set

X_set, Y_set = X_test[:, 1:], Y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.001),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.001))

plt.contourf(X1, X2, classifier_without_gender.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.7, cmap=ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

d = np.unique(Y_set)
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.legend()
plt.title('test set')
plt.show()

