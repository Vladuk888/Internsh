import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import numpy as np
from sklearn.model_selection import cross_val_score


def load_data():
    data = pd.read_csv('train.csv')
    return data

data = load_data()

duplicated = data[data.duplicated(keep=False)]
duplicated = duplicated.sort_values(by=['gender', 'height', 'weight'], ascending= False)
duplicated.head()
data.drop_duplicates(keep = 'first', inplace = True)
st.write('Total {} datapoints remaining with {} features'.format(data.shape[0], data.shape[1]))


X = data.drop(['cardio', 'id'], axis=1,)
data.drop('id', axis=1, inplace = True)
y = data['cardio']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=65)

# Обучение модели случайного леса
best_model = None
best_accuracy = 0

for est in range(1,15):
    model = RandomForestClassifier(
        random_state=42,
        n_estimators=est,
        max_depth=5,
        #min_samples_split=8,
        #min_samples_leaf=3,
        #max_features='auto'
    )
    cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='accuracy')
    model.fit(X_train, y_train)
    result = model.score(X_test, y_test)

    if result > best_accuracy:
        best_model = model
        best_accuracy = result
st.write('Точность модели на тестовых данных:', best_accuracy)
st.write("Средняя точность:", cv_scores.mean())

test_score = best_model.score(X_test, y_test)
st.write('accuracy on test:', test_score)

y_pred = model.predict(X_test)

def calc_metrics(prefix, model, X_test, y_test, y_pred):

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.write('Accuracy:', accuracy)
    st.write('Precision:', precision)
    st.write('Recall:', recall)
    st.write('F1-Score:', f1)

calc_metrics('RF', model, X_test, y_test, y_pred)

st.title('Предсказание сердечно-сосудистых заболеваний')

st.sidebar.header('Введите данные пациента:')
# Ввод данных
age = st.sidebar.number_input('Возраст', min_value=1, max_value=100)
gender = st.sidebar.radio('Пол', ('Мужчина', 'Женщина'))
height = st.sidebar.number_input('Рост (в см)', min_value=50, max_value=250)
weight = st.sidebar.number_input('Вес (в кг)', min_value=10, max_value=500)
cholesterol = st.sidebar.selectbox('Холестерин', ('Низкий','Нормальный', 'Высокий'))
smoke = bool(st.sidebar.checkbox('Курит ли пациент?'))
alco = bool(st.sidebar.checkbox('Пьет ли пациент?'))
gluc = st.sidebar.selectbox('Уровень глюкозы', ('Низкий','Нормальный', 'Высокий'))
ap_hi = st.sidebar.number_input('Верхнее давление (ap_hi)', min_value=1)
ap_lo = st.sidebar.number_input('Нижнее давление (ap_lo)', min_value=1)
active = st.sidebar.radio('Активность', ('Активный', 'Не активный'))

# Преобразование входных данных в числовые значения
gender_mapping = {'Мужчина': 1, 'Женщина': 2}
cholesterol_mapping = {'Низкий': 1,'Нормальный': 2, 'Высокий': 3}
gluc_mapping = {'Низкий': 1,'Нормальный': 2, 'Высокий': 3}
active_mapping = {'Активный': 1, 'Не активный': 0}
smoke_mapping = {'Не курит': 0, 'Курит': 1}
alco_mapping = {'Не пьет': 0, 'Пьет': 1}

gender = gender_mapping[gender]
cholesterol = cholesterol_mapping[cholesterol]
gluc = gluc_mapping[gluc]
active = active_mapping[active]
smoke = int(smoke)
alco = int(alco)

# Создание массива с введенными данными
user_data = [age, gender, height, weight, cholesterol, smoke, alco, gluc, ap_hi, ap_lo, active]

# Предсказание
if st.sidebar.button('Предсказать'):
    st.write('user_data:', user_data)
    prediction = model.predict([user_data])
    st.write(prediction)
    if prediction[0] == 1:
        st.error('Пациент имеет риск сердечно-сосудистого заболевания.')
    else:
        st.success('Пациент не имеет риска сердечно-сосудистого заболевания.')
st.subheader('Данные из CSV:')
st.write('Список признаков в данных из CSV:', data.columns.tolist())

st.write(data)
