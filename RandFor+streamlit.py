#streamlit run JUSTTRY.py
import pandas as pd
data=pd.read_csv('C:/Users/Ukrai/Desktop/JeepWrangCSV.csv', sep='[;,]', engine='python')
df = pd.DataFrame(data)
dfn=df.drop_duplicates(subset=['VIN'])
worksheet=dfn[['Price','Auction','ProductionDate','Condition','Milage','PrimaryDamage','SecondaryDamage','Gearbox','DriveUnit','Keys']].fillna(0).reset_index()
qwe=0
for i in worksheet.Milage:
    qw = [int(w) for w in i.split() if w.isdigit()]
    if not qw:
        worksheet.at[qwe,'Milage']=float(0.0)
    else:
        worksheet.at[qwe,'Milage']=float(qw[0])
    qwe+=1

###########Играемся лесами с реграсиями
dataframeWS = worksheet.copy()
target = 'Price'
encode = ['Auction','Condition','PrimaryDamage','SecondaryDamage','Gearbox','DriveUnit','Keys']

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
Модель предсказания цены авто
""")
st.sidebar.header('Пожалуйста выбоерите желаймое')

# Вносимые нами данные будут играться с моделью
uf = st.sidebar.file_uploader("Сюда путь СиСВ", type=["csv"])
if uf is not None:
    input_df = pd.read_csv(uf)
else:
    def user_input_features():
        Auction = st.sidebar.selectbox('Auction',('\xa0IAAI\xa0','\xa0Copart\xa0'))
        ProductionDate = st.sidebar.slider('ProductionDate', min(worksheet.ProductionDate),max(worksheet.ProductionDate),2013)
        Condition=st.sidebar.selectbox('Condition',('Stationary','Run and Drive','Starts','Run And Drive','Enhanced Vehicles','Engine Start Program','Неизвестно'))
        Milage=st.sidebar.slider('Milage',int(min(worksheet.Milage)),int(max(worksheet.Milage)),785000)
        PrimaryDamage=st.sidebar.selectbox('PrimaryDamage',('Unknown','Rollover','Roof','Front End','Rear','Repossession','Right Front','Right Rear','Left Rear','Right Side','Minor Dent/Scratches','All Over','Rear End','Flood','Left and Right Side','Burn','Side','Top/Roof','Fresh Water','Left Side','Water/Flood','Left Front','Suspension','Normal Wear','Engine Damage','Frame','Engine Burn','Total Burn','Interior Burn','Transmission Damage','Burn - Engine','Frame Damage','Undercarriage','Partial Repair','Stripped','Biohazard','None','Mechanical','Hail','Front and Rear','Theft','Damage History','Electrical','Storm Damage','Exterior Burn','Missing/Altered Vin'))
        SecondaryDamage=st.sidebar.selectbox('SecondaryDamage',('Не указано','Right Side','Suspension','Front and Rear','Front End','Rear','Left Side','Damage History','Normal Wear','Side','Electrical','Roof','Minor Dent/Scratches','Rear End','Biohazard/Chemical','Undercarriage','Top/Roof','All Over','Biohazard','Right Front','Right Rear','Frame Damage','Hail','Left and Right Side','Left Rear','Engine Burn','Mechanical','Missing/Altered Vin','Exterior Burn','Interior Burn','Water/Flood','Transmission Damage','Left Front','Engine Damage','Rollover','Burn','Storm Damage','Flood','Burn - Interior'))
        Gearbox=st.sidebar.selectbox('Gearbox',('Manual','Automatic'))
        DriveUnit=st.sidebar.selectbox('DriveUnit',('4X4 Drive','4X4 W/Rear Wheel Drv','Rear Wheel Drive','Не указан'))
        Keys=st.sidebar.selectbox('Keys',('Present','Missing','Yes','Exempt','No',0))
        data = {'Auction': Auction,
                'ProductionDate':ProductionDate,
                'Condition':Condition,
                'Milage':Milage,
                'PrimaryDamage':PrimaryDamage,
                'SecondaryDamage':SecondaryDamage,
                'Gearbox':Gearbox,
                'DriveUnit':DriveUnit,
                'Keys':Keys,}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()
    
# Сочетает в себе функции пользовательского ввода со всем набором данных, будет полезно на этапе кодирования
playing=worksheet.drop(columns=['index','Price'])
WorkingDF = pd.concat([input_df,playing],axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
encode = ['Auction','Condition','PrimaryDamage','SecondaryDamage','Gearbox','DriveUnit','Keys']
for col in encode:
    dummy = pd.get_dummies(WorkingDF[col], prefix=col)
    WorkingDF = pd.concat([WorkingDF,dummy], axis=1)
    del WorkingDF[col]
WorkingDF = WorkingDF[:1] # Выбирает только первую строку (данные вводимые пользователем)

# Displays the user input features
st.subheader('Переменные пользователя')

if uf is not None:
    st.write(WorkingDF)
else:
    st.write('Ожидание загрузки файла CSV. В настоящее время используется пример входных параметров (показан ниже).')
    st.write(WorkingDF)

# Читает в сохраненной модели классификации
load_clf = pickle.load(open('JeepPKLtwo.pkl', 'rb'))

# Применяйте модель, чтобы делать прогнозы
prediction = load_clf.predict(WorkingDF)
prediction_proba = load_clf.predict_proba(WorkingDF)

st.subheader('Прогноз')
st.write(prediction)

st.subheader('Вероятность предсказания')
st.write(prediction_proba)