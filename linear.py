import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

companies = pd.read_csv('C:/Users/Ukrai/Desktop/123.csv',sep=';')
X = companies.iloc[:, :-1].values
y = companies.iloc[:, :4].values
companies.head()

sns.heatmap(companies.corr())

X = X[:, 1:]

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
X[:, 3] = LabelEncoder().fit_transform(X[:, 3])

X = ColumnTransformer([("Name_Of_Your_Step", OneHotEncoder(),[3])], remainder="passthrough").fit_transform(X)
print(X[0])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
y_pred

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)