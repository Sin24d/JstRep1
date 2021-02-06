import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([15, 11, 2, 8, 25, 32])
#преобразовать входной массив, xчтобы он содержал дополнительный столбец (столбцы) со значениями 𝑥² (и, в конечном итоге, другими функциями).
transformer = PolynomialFeatures(degree=2, include_bias=False)
#Перед применением transformerего нужно подогнать .fit():
transformer.fit(x)
#После transformerустановки он готов к созданию нового измененного входа. Вы подаете заявку .transform()на это:
x_ = transformer.transform(x)
#Вы также можете использовать .fit_transform()для замены трех предыдущих операторов только одним:
x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)
print(x_)

model = LinearRegression().fit(x_, y)

r_sq = model.score(x_, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('coefficients:', model.coef_)
x_ = PolynomialFeatures(degree=2, include_bias=True).fit_transform(x)
print(x_)

model = LinearRegression(fit_intercept=False).fit(x_, y)
r_sq = model.score(x_, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('coefficients:', model.coef_)

y_pred = model.predict(x_)
print('predicted response:', y_pred, sep='\n')