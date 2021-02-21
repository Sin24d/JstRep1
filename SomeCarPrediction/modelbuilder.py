import pandas as pd
import io
data=pd.read_csv('C:/Users/Ukrai/Desktop/Tesla3stats2.csv', sep='[;,]', engine='python')
df = pd.DataFrame(data)
df

print(len(df.VIN.unique()),'VIN uniq')
dfnodubs=df.drop_duplicates(subset=['VIN'])
dfprimarclear1=dfnodubs.drop(['LotNumber', 'Gearbox','Fuel','EstimatedValue','RepairCost','Notes'], axis=1)
dfprimarclear=dfprimarclear1.fillna(0).reset_index()
#heater1=dfprimarclear.Milage.str.extract('(\d+)').fillna(0).rename(columns={0: "Milage"}).Milage.astype(float)#Only numbers
qe=0
for q in dfprimarclear.Milage:
    qw = [int(w) for w in q.split() if w.isdigit()]
    if not qw:
        dfprimarclear.at[qe,'Milage']=float(0.0)
    else:
        dfprimarclear.at[qe,'Milage']=float(qw[0])
    qe+=1
temp=0
while temp<len(dfprimarclear['Keys']):
    if dfprimarclear.loc[temp,'Keys']=='Missing':
        dfprimarclear.at[temp, 'Keys']=float(0)
    elif dfprimarclear.loc[temp,'Keys']=='Present':
        dfprimarclear.at[temp, 'Keys']=float(1)
    elif dfprimarclear.loc[temp,'Keys']=='Yes':
        dfprimarclear.at[temp, 'Keys']=float(1)
    elif dfprimarclear.loc[temp,'Keys']=='No':
        dfprimarclear.at[temp, 'Keys']=float(0)
    elif dfprimarclear.loc[temp,'Keys']=='Exempt':
        dfprimarclear.at[temp, 'Keys']=float(0)
    elif dfprimarclear.loc[temp,'Keys']==0:
        dfprimarclear.at[temp, 'Keys']=float(0)
    temp+=1
        
cleaned=dfprimarclear[['DriveUnit','PrimaryDamage','Condition','Price','ProductionDate','Milage','Keys']]
cleaned.head(5)

dfc = cleaned.copy()
target = 'Condition'
encode = ['DriveUnit','PrimaryDamage']
#encoding
for col in encode:
    dummy = pd.get_dummies(dfc[col], prefix=col)
    dfc = pd.concat([dfc,dummy], axis=1)
    del dfc[col]

target_mapper = {'Stationary':0, 'Run and Drive':1, 'Run And Drive':1, 'Engine Start Program':2,'Enhanced Vehicles':3,'Starts':4,'Неизвестно':5}
def target_encode(val):
    return target_mapper[val]

dfc['Condition'] = dfc['Condition'].apply(target_encode)
#X Y separation
X = dfc.drop('Condition', axis=1)
Y = dfc['Condition']
#Randomforest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, Y)
#Saving
import pickle
pickle.dump(clf, open('Car_tesla_clf.pkl', 'wb'))

cleaned.Condition.unique()
#cleaned.DriveUnit.unique()
#cleaned.PrimaryDamage.unique()
#cleaned.Price.describe()
#cleaned.ProductionDate.unique()
#cleaned.Milage
#cleaned.Keys.unique()
#cleaned.groupby(['Condition']).mean()

#cleaned.to_csv (r'CarTesla3.csv', index = False)

