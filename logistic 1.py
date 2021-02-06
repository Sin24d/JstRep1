import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
#создал данные
x = np.arange(10).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
model = LogisticRegression(solver='liblinear', random_state=0).fit(x, y)
#регрессия
print(model.classes_)
print(model.intercept_)
print(model.coef_)
model.predict_proba(x)
#она будет не правильной
model.predict(x)
model.score(x, y)
confusion_matrix(y, model.predict(x))
#создание матрицы конфуза
print(classification_report(y, model.predict(x)))
model = LogisticRegression(solver='liblinear', C=10.0, random_state=0).fit(x,y)
#(нормализовали)
model.predict(x)
model.score(x, y)
confusion_matrix(y, model.predict(x))
print(classification_report(y, model.predict(x)))
############лучше в jupyter