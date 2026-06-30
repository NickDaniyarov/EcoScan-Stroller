import streamlit as st
import pandas as pd
import numpy as np
import requests
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import joblib

st.set_page_config(page_title="EcoScan: Астма-прогноз", layout="wide")
st.title("🫁 Индекс безопасности для астматиков")

# Параметры
THINGSPEAK_URL = "https://api.thingspeak.com/channels/3293999/feeds.json?results=150"
MODEL_PATH = "asthma_model.h5"
SCALER_PATH = "scaler.pkl"

@st.cache_data(ttl=300)  # кэшируем данные на 5 минут
def load_data():
    """Загружает данные из ThingSpeak"""
    try:
        r = requests.get(THINGSPEAK_URL).json()
        df = pd.DataFrame(r['feeds'])
        
        fields = ['field1', 'field2', 'field3', 'field6', 'field7']
        for f in fields:
            df[f] = pd.to_numeric(df[f], errors='coerce')
        
        df['created_at'] = pd.to_datetime(df['created_at'])
        df = df.dropna(subset=fields)
        
        # Нормализация
        df['co2_norm'] = df['field2'] / 2000
        df['co_norm'] = df['field1'] / 50
        df['voc_norm'] = df['field3'] / 100
        df['no2_norm'] = df['field6'] / 200
        df['pm_norm'] = df['field7'] / 100
        
        return df
    except Exception as e:
        st.error(f"Ошибка загрузки: {e}")
        return pd.DataFrame()

@st.cache_resource  # важный момент: модель сохраняется в кэш и не пересоздаётся
def get_or_train_model(df):
    """
    Загружает сохранённую модель или обучает новую
    """
    X = df[['co2_norm', 'co_norm', 'voc_norm', 'no2_norm', 'pm_norm']].values[:-1]
    y = df[['co2_norm', 'co_norm', 'voc_norm', 'no2_norm', 'pm_norm']].values[1:]
    
    if len(X) < 30:
        return None
    
    # Пробуем загрузить существующую модель
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            return model
        except:
            pass
    
    # Если модели нет — обучаем
    with st.spinner("🔄 Первое обучение нейросети (займёт 20–30 секунд)..."):
        model = Sequential([
            Dense(64, activation='relu', input_shape=(5,)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(5)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=50, verbose=0, batch_size=16)
        
        # Сохраняем модель
        model.save(MODEL_PATH)
        
    return model

def calculate_hazard_index(row):
    """Вычисляет индекс опасности для астматика"""
    weights = {'pm': 0.35, 'no2': 0.25, 'co': 0.15, 'co2': 0.15, 'voc': 0.10}
    
    pm_hazard = min(1, row.pm_norm)
    no2_hazard = min(1, row.no2_norm)
    co_hazard = min(1, row.co_norm)
    co2_hazard = min(1, row.co2_norm)
    voc_hazard = min(1, row.voc_norm)
    
    hazard = (
        weights['pm'] * pm_hazard +
        weights['no2'] * no2_hazard +
        weights['co'] * co_hazard +
        weights['co2'] * co2_hazard +
        weights['voc'] * voc_hazard
    ) * 100
    
    return min(100, hazard)

def get_recommendation(hazard_index):
    """Возвращает рекомендацию на основе индекса"""
    if hazard_index < 20:
        return {'color': 'green', 'level': 'Низкий риск', 'message': 'Воздух чистый.', 'advice': 'Можно гулять без ограничений.'}
    elif hazard_index < 40:
        return {'color': 'lightgreen', 'level': 'Умеренный риск', 'message': 'Качество воздуха удовлетворительное.', 'advice': 'Избегайте оживлённых дорог.'}
    elif hazard_index < 60:
        return {'color': 'yellow', 'level': 'Повышенный риск', 'message': 'Возможны лёгкие раздражения.', 'advice': 'Сократите прогулку до 30 минут.'}
    elif hazard_index < 80:
        return {'color': 'orange', 'level': 'Высокий риск', 'message': 'Качество воздуха плохое.', 'advice': 'Гуляйте только при необходимости, используйте маску.'}
    else:
        return {'color': 'red', 'level': 'Критический риск', 'message': 'Воздух опасен!', 'advice': 'Оставайтесь в помещении.'}

def predict_future(model, last_values, hours_ahead=6):
    """Прогнозирует будущие значения"""
    predictions = []
    current = last_values.copy()
    
    for _ in range(hours_ahead):
        pred = model.predict(current.reshape(1, 5), verbose=0)[0]
        predictions.append(pred)
        current = pred
    
    return np.array(predictions)

# ========== ОСНОВНОЙ ИНТЕРФЕЙС ==========
data = load_data()

if data.empty:
    st.warning("⚠️ Нет данных в ThingSpeak")
    st.stop()

# Отображаем текущие показатели
st.subheader("📊 Текущие показатели")
last = data.iloc[-1]
cols = st.columns(5)
metrics = [("CO₂", last['field2'], "ppm"), ("CO", last['field1'], "ppm"),
           ("VOC", last['field3'], "ppb"), ("NO₂", last['field6'], "ppb"),
           ("Пыль PM", last['field7'], "µg/m³")]

for i, (name, val, unit) in enumerate(metrics):
    with cols[i]:
        st.metric(name, f"{val:.1f} {unit}")

# Индекс опасности
current_hazard = calculate_hazard_index(last)
rec = get_recommendation(current_hazard)

st.subheader("🫁 Индекс опасности")
c1, c2 = st.columns([1, 2])
with c1:
    st.markdown(f"""
    <div style="background-color: {rec['color']}; padding: 20px; border-radius: 15px; text-align: center;">
        <h1>{current_hazard:.0f}<span style="font-size: 20px;">/100</span></h1>
        <h3>{rec['level']}</h3>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.info(rec['message'])
    st.success(f"💡 {rec['advice']}")

# Прогноз (только если есть данные)
if len(data) >= 30:
    st.subheader("🔮 Прогноз на 6 часов")
    
    with st.spinner("Загрузка модели..."):
        model = get_or_train_model(data)
    
    if model:
        last_values = data[['co2_norm', 'co_norm', 'voc_norm', 'no2_norm', 'pm_norm']].iloc[-1].values
        future_norm = predict_future(model, last_values, 6)
        
        # Денормализация и расчёт индексов
        hazard_hours = []
        for norm in future_norm:
            class Temp: pass
            row = Temp()
            row.pm_norm = norm[4]
            row.no2_norm = norm[3]
            row.co_norm = norm[1]
            row.co2_norm = norm[0]
            row.voc_norm = norm[2]
            hazard_hours.append(calculate_hazard_index(row))
        
        now = datetime.now()
        hours = [(now + timedelta(hours=i+1)).strftime("%H:%M") for i in range(6)]
        
        # График
        fig = go.Figure(go.Bar(x=hours, y=hazard_hours,
                                marker_color=['green' if h<40 else 'orange' if h<80 else 'red' for h in hazard_hours]))
        fig.update_layout(title="Прогноз индекса опасности", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Таблица
        forecast_data = []
        for i, hour in enumerate(hours):
            rec_h = get_recommendation(hazard_hours[i])
            forecast_data.append({"Время": hour, "Индекс": f"{hazard_hours[i]:.0f}",
                                  "Риск": rec_h['level'], "Совет": rec_h['advice']})
        st.dataframe(pd.DataFrame(forecast_data), use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ Недостаточно данных для прогноза (нужно минимум 30 записей)")
else:
    st.info(f"📊 Для точного прогноза нейросети нужно 30 записей. Сейчас в канале: {len(data)} записей.")

# Исторический тренд
st.subheader("📈 Тренды")
recent = data.tail(50).copy()
recent['hazard'] = recent.apply(calculate_hazard_index, axis=1)

fig = go.Figure()
fig.add_trace(go.Scatter(x=recent['created_at'], y=recent['hazard'],
                          mode='lines', name='Индекс', line=dict(color='red')))
fig.update_layout(height=300)
st.plotly_chart(fig, use_container_width=True)

st.caption("📌 Прогноз основан на данных ЭкоСкан")
