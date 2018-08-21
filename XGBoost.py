import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Importing the data_set
data_set = pd.read_csv('data/Churn_Modelling.csv')
X = data_set.iloc[:, 3:13].values
y = data_set.iloc[:, 13].values

# Encoding categorical data
label_encoder_X_1 = LabelEncoder()
X[:, 1] = label_encoder_X_1.fit_transform(X[:, 1])
label_encoder_X_2 = LabelEncoder()
X[:, 2] = label_encoder_X_2.fit_transform(X[:, 2])
one_hot_encoder = OneHotEncoder(categorical_features=[1])
X = one_hot_encoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the data_set into the Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)


classifier = XGBClassifier()
classifier.fit(X_train, Y_train)
y_pre = classifier.predict(X_test)


cm = confusion_matrix(Y_test, y_pre)
accuracy = accuracy_score(Y_test, y_pre)

accuracy_cross = cross_val_score(classifier, X, y, cv=10)
accuracy_cross.mean()
accuracy_cross.std()










