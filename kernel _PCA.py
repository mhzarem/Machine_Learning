import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA,PCA

from matplotlib.colors import ListedColormap

data = pd.read_csv('data/Social_Network_Ads.csv')

X = data.iloc[:, [2, 3]].values
Y = data.iloc[:, -1].values
# pre processing
# it's important to scaling
sc = StandardScaler()
X = sc.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)


# using  PCA for dimension reduction

k_pca = KernelPCA(n_components=2, kernel='rbf')
X_train_transformed = k_pca.fit_transform(X_train)
X_test_transformed = k_pca.transform(X_test)
classifier = LogisticRegression()
classifier.fit(X_train_transformed, Y_train)
y_pre = classifier.predict(X_test_transformed)

cm = confusion_matrix(Y_test, y_pre)
accuracy = accuracy_score(Y_test, y_pre)
# visualising data

X_set, Y_set = X_train_transformed, Y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X2.shape),
             alpha=0.7, cmap=ListedColormap(('red', 'blue')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

d = np.unique(Y_set)
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c=ListedColormap(('red', 'blue'))(i), label=j)
plt.legend()
plt.show()

