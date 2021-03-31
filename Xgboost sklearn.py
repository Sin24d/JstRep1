import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np

boston = load_boston()
print(boston.data.shape)

print(boston.feature_names)

print(boston.DESCR)

dframe = pd.DataFrame(boston.data)
dframe.columns = boston.feature_names
dframe.head()

dframe['PRICE'] = boston.target
#data.info()
dframe.describe()

X, y = dframe.iloc[:, :-1], dframe.iloc[:, -1]

#https://xgboost.readthedocs.io/en/latest/python/python_intro.html
dmatrix_dframe = xgb.DMatrix(data=X, label=y)#Dmatrix helps XGBst

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=120)#20%

xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,max_depth = 5, alpha = 10, n_estimators = 10)

xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))#mean_sqaured_error
print("RMSE: %f" % (rmse))

params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,'max_depth': 5, 'alpha': 10}#k-fold Cross Validation using XGBoost
#carefull, can destroy you PC:D
cv_results = xgb.cv(dtrain=dmatrix_dframe, params=params, nfold=3, num_boost_round=50, early_stopping_rounds=10, metrics="rmse", as_pandas=True, seed=120)
#carefull, can destroy you PC:D

#cv_results.head()
print((cv_results["test-rmse-mean"]).tail(1))#Final extracted info!?
#carefull, can destroy you PC:D
xg_reg = xgb.train(params=params, dtrain=dmatrix_dframe, num_boost_round=10)#booooosted деревья и важность функций

import matplotlib.pyplot as plt
xgb.plot_tree(xg_reg,num_trees=1)#num_trees=0
plt.rcParams['figure.figsize'] = [50, 20]
plt.show()#Важность каждого столбца функций в исходном наборе данных

xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [7, 7]
plt.show()
#Шпаргалочка на ХГбуст