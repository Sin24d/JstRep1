import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv("../input/college-basketball-dataset/cbb.csv")

array = ['2019']
df_19=df.loc[df['YEAR'].isin(array)]

df_19.info()

df_19.sample(15)

df_19.describe()

print(df_19.shape)

print(df_19.info())

df_19.columns
#Посмотрим, сыграли ли все команды одинаковое количество игр
df_19['Games_Played'] = df_19['W'] + df_19['G']
sns.distplot(df_19['Games_Played'])
#!Hе все команды сыграли одинаковое количество игр.

df_19['W_ratio'] = df_19['W'] / df_19['G']
df_19.sort_values(by='W_ratio', ascending=False).head(20)
df_19['CONF'].value_counts()
df_19['CONF'].count()

#Сколько команд добрались до ивента "March Madness"?
df_19['SEED'].notna().sum()
d=df_19['SEED'].notna().sum()/df_19['TEAM'].count()
print ("Percentage of college teams that make it to the March Madness: "+"{:.2%}".format(d))
df_19['POSTSEASON'].unique()
df_19['POSTSEASON'].value_counts()
df_19['SEED'].value_counts()

#Лучше закодировать в интегральное значение, чтобы провести другой анализ
d = {'Champions' : 1, '2ND' : 2, 'F4' : 3, 'E8' : 8, 'R68' : 5, 'S16' : 5, 'R32' : 6, 'R64' : 7}
df_19['POSTSEASON_Value'] = df_19['POSTSEASON'].map(d)
df_19.head(10)

#Тепловая карта 
#Матрица корреляций Что важнее: рейтинг нападения или защиты?
plt.figure(figsize=(20,20))  # on this line I just set the size of figure to 12 by 10.
p=sns.heatmap(df_19.corr(), annot=True,cmap='RdYlGn',square=True)
corr_mat = df_19.corr()
corr_mat['W_ratio']
so = corr_mat['W_ratio'].sort_values(kind="quicksort", ascending=False)
print(so)
#Победители баскетбольного турнира NCAA
array1 = ['Champions']
df1=df.loc[df['POSTSEASON'].isin(array1)]
df1.sort_values(by='YEAR', ascending=False)

array2 = ['Virginia', 'Villanova', 'North Carolina','Duke']
df2=df.loc[df['TEAM'].isin(array2)]
df2.sort_values(['TEAM', 'YEAR'], ascending=[False, False])

#Диаграмма рассеяния для отбора на турнир NCAA 
#Давайте сначала очистим данные до двоичного кода, если команда подходит или не подходит для данных за 2015-2019 годы.
df['Not_Qualified'] = pd.isna(df['SEED'])
sns.scatterplot(y=df['EFG_O'], x=df['EFG_D'], hue=df['Not_Qualified'])
#Диаграмма разброса показывает, что для команды важно иметь высокий процент результативных бросков с игры (EFG_O) и низкий допустимый процент эффективных попаданий с игры (EFG_D), чтобы иметь больше шансов пройти квалификацию (2015-2019).