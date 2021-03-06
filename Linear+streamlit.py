import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

st.write("""
# Price prediction
""")
st.write('---')

# Загружаем нужные нам данные для совершения предсказания
Jeep = datasets.load_boston()
X = pd.DataFrame(Jeep.data, columns=Jeep.feature_names)
Y = pd.DataFrame(Jeep.target, columns=["Price"])#MEDV

# Sidebar
st.sidebar.header('Уточним параметры ввода')


def user_input_features():
    Auction = st.sidebar.selectbox('Auction', ('\xa0IAAI\xa0', '\xa0Copart\xa0'))
    ProductionDate = st.sidebar.slider('ProductionDate', min(X.ProductionDate), max(X.ProductionDate),2013)
    Condition = st.sidebar.selectbox('Condition', ('Stationary', 'Run and Drive', 'Starts', 'Run And Drive', 'Enhanced Vehicles', 'Engine Start Program','Неизвестно'))
    Milage = st.sidebar.slider('Milage', int(min(X.Milage)), int(max(X.Milage)), 785000)
    PrimaryDamage = st.sidebar.selectbox('PrimaryDamage', (
    'Unknown', 'Rollover', 'Roof', 'Front End', 'Rear', 'Repossession', 'Right Front', 'Right Rear', 'Left Rear',
    'Right Side', 'Minor Dent/Scratches', 'All Over', 'Rear End', 'Flood', 'Left and Right Side', 'Burn', 'Side',
    'Top/Roof', 'Fresh Water', 'Left Side', 'Water/Flood', 'Left Front', 'Suspension', 'Normal Wear', 'Engine Damage',
    'Frame', 'Engine Burn', 'Total Burn', 'Interior Burn', 'Transmission Damage', 'Burn - Engine', 'Frame Damage',
    'Undercarriage', 'Partial Repair', 'Stripped', 'Biohazard', 'None', 'Mechanical', 'Hail', 'Front and Rear', 'Theft',
    'Damage History', 'Electrical', 'Storm Damage', 'Exterior Burn', 'Missing/Altered Vin'))
    SecondaryDamage = st.sidebar.selectbox('SecondaryDamage', (
    'Не указано', 'Right Side', 'Suspension', 'Front and Rear', 'Front End', 'Rear', 'Left Side', 'Damage History',
    'Normal Wear', 'Side', 'Electrical', 'Roof', 'Minor Dent/Scratches', 'Rear End', 'Biohazard/Chemical',
    'Undercarriage', 'Top/Roof', 'All Over', 'Biohazard', 'Right Front', 'Right Rear', 'Frame Damage', 'Hail',
    'Left and Right Side', 'Left Rear', 'Engine Burn', 'Mechanical', 'Missing/Altered Vin', 'Exterior Burn',
    'Interior Burn', 'Water/Flood', 'Transmission Damage', 'Left Front', 'Engine Damage', 'Rollover', 'Burn',
    'Storm Damage', 'Flood', 'Burn - Interior'))
    Gearbox = st.sidebar.selectbox('Gearbox', ('Manual', 'Automatic'))
    DriveUnit = st.sidebar.selectbox('DriveUnit',
                                     ('4X4 Drive', '4X4 W/Rear Wheel Drv', 'Rear Wheel Drive', 'Не указан'))
    Keys = st.sidebar.selectbox('Keys', ('Present', 'Missing', 'Yes', 'Exempt', 'No', 0))
    data = {'Auction': Auction,
            'ProductionDate': ProductionDate,
            'Condition': Condition,
            'Milage': Milage,
            'PrimaryDamage': PrimaryDamage,
            'SecondaryDamage': SecondaryDamage,
            'Gearbox': Gearbox,
            'DriveUnit': DriveUnit,
            'Keys': Keys, }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel
st.header('Уточним параметры ввода')
st.write(df)
st.write('---')

#Делаем ригрессию
model = RandomForestRegressor()
model.fit(X, Y)
#Подттверждаем модель для произведения предсказания
prediction = model.predict(df)

st.header('Предасказание цены')
st.write(prediction)
st.write('---')

# https://github.com/slundberg/shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st.header('Важность выбраных эелементов')
plt.title('Важность на базе SHAP иследования')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')