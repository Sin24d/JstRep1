import pandas as pd
import io
#data=pd.read_csv('C:/Users/Ukrai/Desktop/Tesla3stats2.csv','rb')
data=pd.read_csv('C:/Users/Ukrai/Desktop/Tesla3stats2.csv', sep='[;,]', engine='python')
df = pd.DataFrame(data)

print(len(df.FullName.unique()),'FullName uniq')
print(len(df.VIN.unique()),'VIN uniq')
print(len(df.LotNumber.unique()),'LotNumber uniq')

#df.drop_duplicates()
#не чистит полноценно, поскольку уникальних вин только 857, слишком много дублей
dfnodubs=df.drop_duplicates(subset=['VIN'])

#предполагаю что для последующего иследования где конечная точка будет связанна с предсказанием цены нам не потребуются:
#-Lotnumber-useless!?
#-Gearbox-allsimilar
#-Fuel-allsimilar
#-EstimatedValue-unfortunately was empty
#-RepairCost-unfortunately was empty
#later clear more
dfprimarclear1=dfnodubs.drop(['LotNumber', 'Gearbox','Fuel','EstimatedValue','RepairCost'], axis=1)
dfprimarclear=dfprimarclear1.fillna(0)#.reset_index()
dfprimarclear.head(5)

import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
heater1=dfprimarclear.Milage.str.extract('(\d+)').fillna(0).rename(columns={0: "Milage"}).Milage.astype(float)#Only numbers
heater2=dfprimarclear[['Price','Condition']]
heater=pd.concat([heater2,heater1], axis=1).groupby(['Condition']).mean()

df_heatmap = heater.pivot_table(values='Price',index='Condition',columns='Milage',aggfunc=np.mean)
sns.heatmap(df_heatmap,annot=True)
plt.show()
#На первый взгялд странный хитмап демонстрирует нам прямую зависимость между состоянием авто в показателях пробега и гтовности
#к показателю цены. Маленький пробег и незначительный урон данного ТС будет всегда в стоимости больше 26к$, когда авто что не заводится
#и не едет имеет слабую корреляцию с ценовым показателем изза разброса, а пробег не может быть узнан через отсудствие зажигания авто

a=0
workingDF=dfprimarclear.reset_index()
while a<len(workingDF['Auction']):
    if workingDF.loc[a,'Auction']=='\xa0IAAI\xa0':
        workingDF.at[a, 'Auction']=1
    elif workingDF.loc[a,'Auction']=='\xa0Copart\xa0':
        workingDF.at[a, 'Auction']=0
    elif workingDF.loc[a,'Auction']=='IAAI':
        workingDF.at[a, 'Auction']=1
    elif workingDF.loc[a,'Auction']=='Copart':
        workingDF.at[a, 'Auction']=0
    else:
        print('rechek it')
    a+=1
#workingDF['Auction'].head(15)
#workingDF.head(15)

print(workingDF.Condition.unique())##
#print(workingDF.Documents.unique())
print(workingDF.PrimaryDamage.unique())
print(workingDF.Color.unique())
print(workingDF.Keys.unique())##
print(workingDF.DriveUnit.unique())##
temp=0
while temp<len(workingDF['Condition']):
    if workingDF.loc[temp,'Condition']=='Stationary':
        workingDF.at[temp, 'Condition']=float(0)
    elif workingDF.loc[temp,'Condition']=='Run and Drive':
        workingDF.at[temp, 'Condition']=float(1)
    elif workingDF.loc[temp,'Condition']=='Run And Drive':
        workingDF.at[temp, 'Condition']=float(1)
    elif workingDF.loc[temp,'Condition']=='Engine Start Program':
        workingDF.at[temp, 'Condition']=float(2)
    elif workingDF.loc[temp,'Condition']=='Enhanced Vehicles':
        workingDF.at[temp, 'Condition']=float(3)
    elif workingDF.loc[temp,'Condition']=='Starts':
        workingDF.at[temp, 'Condition']=float(4)
    elif workingDF.loc[temp,'Condition']=='Неизвестно':
        workingDF.at[temp, 'Condition']=float(5)
    if workingDF.loc[temp,'Keys']=='Missing':
        workingDF.at[temp, 'Keys']=float(0)
    elif workingDF.loc[temp,'Keys']=='Present':
        workingDF.at[temp, 'Keys']=float(1)
    elif workingDF.loc[temp,'Keys']=='Yes':
        workingDF.at[temp, 'Keys']=float(1)
    elif workingDF.loc[temp,'Keys']=='No':
        workingDF.at[temp, 'Keys']=float(0)
    elif workingDF.loc[temp,'Keys']=='Exempt':
        workingDF.at[temp, 'Keys']=float(0)
    elif workingDF.loc[temp,'Keys']==0:
        workingDF.at[temp, 'Keys']=float(0)
    if workingDF.loc[temp,'DriveUnit']=='All Wheel Drive':
        workingDF.at[temp, 'DriveUnit']=float(1)
    if workingDF.loc[temp,'DriveUnit']=='Rear Wheel Drive':
        workingDF.at[temp, 'DriveUnit']=float(2)
    elif workingDF.loc[temp,'DriveUnit']=='Rear-Wheel Drive':
        workingDF.at[temp, 'DriveUnit']=float(2)
    elif workingDF.loc[temp,'DriveUnit']=='Не указан':
        workingDF.at[temp, 'DriveUnit']=float(3)
    temp+=1
#print(workingDF.PrimaryDamage.unique())
#print(workingDF.Color.unique())
#print(workingDF.Keys.unique())##
#print(workingDF.DriveUnit.unique())##
#workingDF.loc[0,'DriveUnit']
#ENCODING EVEN WORSE THAT MONKEY
workingDF.head(10)

import seaborn as sns; sns.set_theme()
import numpy as np;

heatmaparray=[[],[],[],[],[],[]]
ar=0
while ar<len(workingDF.Price):
    heatmaparray[0].append(workingDF.Price[ar])
    heatmaparray[1].append(workingDF.Auction[ar])
    heatmaparray[2].append(workingDF.ProductionDate[ar])
    heatmaparray[3].append(workingDF.Condition[ar])
    heatmaparray[4].append(workingDF.DriveUnit[ar])
    heatmaparray[5].append(workingDF.Keys[ar])
    ar+=1
    
uniform_data=np.corrcoef(heatmaparray)
ax = sns.heatmap(uniform_data, vmin=0, vmax=1, annot=True,linewidths=.5,cmap="YlGnBu")
#Не такое приятное к лицо, но 0=цена,1=аукцион,2=ДатаПродукции,3=Состояние,4=Привод,5=Ключи(Есть/Нет)
#Есть некая легкая связь между Ценой и ДатойПродукции(СтаростьюАвто) и еще больше наличием ключей, что очень интерестно для более подробного изучения

import matplotlib.pyplot as plt

plt.plot(workingDF['Price'])
plt.suptitle('Everage Price')
plt.show()
print(workingDF['Price'].describe())
#Сердняя цена на авто в диапазоне 22к$, основной диапазон от 18к$ к 26к$. Возможно ли купить Авто за 13к?=Судя по всему да, но состояние будет страшное

