# первая нейронная сеть с keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# загрузить набор данных
dataset = loadtxt('pima-indians-diabetes.data.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]

# Определить модель keras
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# компилируем модель keras
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# поместите модель keras в набор данных
model.fit(X, y, epochs=150, batch_size=10)

# оцениваем модель keras
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

# поместите модель keras в набор данных без индикаторов выполнения
model.fit(X, y, epochs=150, batch_size=10, verbose=0)
# оцениваем модель keras
_, accuracy = model.evaluate(X, y, verbose=0)
#######################################################EVALUATIONNNN

# делаем вероятностные прогнозы с помощью модели
predictions = model.predict(X)
# раунд прогнозов
rounded = [round(x[0]) for x in predictions]

# делаем прогнозы классов с помощью модели
predictions = model.predict_classes(X)
# суммируем первые 5 случаев
for i in range(5):
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))

