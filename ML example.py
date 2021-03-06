from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

#Summarize the Dataset
# shape
#print(dataset.shape)
#150,5
# head
#print(dataset.head(20))
# descriptions
#print(dataset.describe())
# class distribution
print(dataset.groupby('class').size())

##Data Visualization
# box and whisker plots
#Это дает нам гораздо более четкое представление о распределении входных атрибутов
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()

# histograms
#Мы также можем создать гистограмму каждой входной переменной, чтобы получить представление о распределении
dataset.hist()
pyplot.show()

# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()
#Обратите внимание на диагональное сгруппирование некоторых пар атрибутов. Это говорит о высокой корреляции и предсказуемой взаимосвязи.

#Выделяю набор данных проверки.
#Настраиваю тестовый жгут для использования 10-кратной перекрестной проверки.
#Создаю несколько различных моделей для прогнозирования видов на основе измерений цветов.
#Выбераю лучшую модель.
# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

#Логистическая регрессия (LR)
#Линейный дискриминантный анализ (LDA)
#K-Ближайшие соседи (KNN).
#Деревья классификации и регрессии (CART).
#Гауссовский наивный байесовский (NB).
#Машины опорных векторов (SVM).
# Алгоритмы выборочной проверки
#выбираем лучшую модель
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# оценим каждую по очереди
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Сравниваем алгоритмы
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

# Прогнозируем на основе набора данных проверки
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Оцениваем прогнозы
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
# Мы можем видеть, что точность составляет 0,966 или около 96% для удерживаемого набора данных.

