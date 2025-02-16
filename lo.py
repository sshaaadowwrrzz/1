import streamlit as st
import sqlite3
import pandas as pd
import requests
import logging
import plotly.express as px
import joblib
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from pulp import LpMaximize, LpProblem, LpVariable, lpSum
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Класс для работы с базой данных
class DatabaseManager:
    def __init__(self, db_name):
        self.db_name = db_name
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()
        self._initialize_db()

    def _initialize_db(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS soil_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                Культура TEXT,
                Уровень_pH REAL,
                Азот REAL,
                Фосфор REAL,
                Калий REAL,
                Урожайность REAL,
                Широта REAL,
                Долгота REAL,
                Дата TEXT
            )
        ''')
        self.conn.commit()

    def insert_data(self, Культура, Уровень_pH, Азот, Фосфор, Калий, Урожайность, Широта, Долгота, Дата):
        self.cursor.execute('''
            INSERT INTO soil_data 
            (Культура, Уровень_pH, Азот, Фосфор, Калий, Урожайность, Широта, Долгота, Дата)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (Культура, Уровень_pH, Азот, Фосфор, Калий, Урожайность, Широта, Долгота, Дата))
        self.conn.commit()

    def fetch_data(self):
        return pd.read_sql("SELECT * FROM soil_data", self.conn)

    def close(self):
        self.conn.close()

# Класс для работы с API погоды
class WeatherAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://weatherapi-com.p.rapidapi.com/current.json"

    @st.cache_data(ttl=3600)
    def get_weather(_self, city):
        headers = {
            "X-RapidAPI-Key": _self.api_key,
            "X-RapidAPI-Host": "weatherapi-com.p.rapidapi.com"
        }
        try:
            response = requests.get(f"{_self.base_url}?q={city}", headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Ошибка при запросе погоды: {e}")
            return None

# Класс для предсказательной модели
class YieldPredictor:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.models = {
            "Случайный лес": RandomForestRegressor(n_estimators=100, random_state=42),
            "Градиентный бустинг": GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        self.scaler = StandardScaler()
        self.explainer = None

    def train_model(self):
        df = self.db_manager.fetch_data()
        if len(df) < 10:
            logging.warning("Недостаточно данных для обучения модели.")
            return None

        X = df[['Уровень_pH', 'Азот', 'Фосфор', 'Калий']]
        y = df['Урожайность']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        trained_models = {}
        for name, model in self.models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            trained_models[name] = (model, mse)

        joblib.dump(trained_models, "models.pkl")
        joblib.dump(self.scaler, "scaler.pkl")

        # Инициализация SHAP explainer
        self.explainer = shap.Explainer(trained_models["Случайный лес"][0], X_train_scaled)

        return trained_models

    def predict_yield(self, Уровень_pH, Азот, Фосфор, Калий):
        try:
            trained_models = joblib.load("models.pkl")
            self.scaler = joblib.load("scaler.pkl")

            X_new = self.scaler.transform([[Уровень_pH, Азот, Фосфор, Калий]])
            predictions = {}
            for name, (model, _) in trained_models.items():
                predictions[name] = model.predict(X_new)[0]

            return predictions
        except FileNotFoundError:
            logging.warning("Модели не обучены.")
            return None

    def explain_prediction(self, Уровень_pH, Азот, Фосфор, Калий):
        if self.explainer is None:
            return None

        X_new = self.scaler.transform([[Уровень_pH, Азот, Фосфор, Калий]])
        shap_values = self.explainer(X_new)
        return shap_values

# Класс для оптимизации ресурсов
class ResourceOptimizer:
    def __init__(self, db_manager):
        self.db_manager = db_manager

    def optimize_resources(self, бюджет, культура):
        df = self.db_manager.fetch_data()
        df = df[df['Культура'] == культура]

        if df.empty:
            return None

        prob = LpProblem("Оптимизация_ресурсов", LpMaximize)
        Азот = LpVariable("Азот", lowBound=0)
        Фосфор = LpVariable("Фосфор", lowBound=0)
        Калий = LpVariable("Калий", lowBound=0)

        # Цены на удобрения в тенге (примерные значения)
        ЦЕНЫ = {
            "Азот": 150,  # тенге за кг
            "Фосфор": 200,
            "Калий": 180
        }

        # Целевая функция
        prob += lpSum([
            Азот * df['Азот'].mean(),
            Фосфор * df['Фосфор'].mean(),
            Калий * df['Калий'].mean()
        ])

        # Ограничение по бюджету
        prob += lpSum([
            Азот * ЦЕНЫ["Азот"],
            Фосфор * ЦЕНЫ["Фосфор"],
            Калий * ЦЕНЫ["Калий"]
        ]) <= бюджет

        prob.solve()

        return {
            "Азот (кг)": round(Азот.varValue, 2),
            "Фосфор (кг)": round(Фосфор.varValue, 2),
            "Калий (кг)": round(Калий.varValue, 2)
        }

# Основной класс приложения
class AgricultureApp:
    def __init__(self):
        self.db_manager = DatabaseManager("agriculture.db")
        self.weather_api = WeatherAPI("f07561ab93mshb26859914e2780bp19d0e8jsn18df37d76fd5")
        self.yield_predictor = YieldPredictor(self.db_manager)
        self.resource_optimizer = ResourceOptimizer(self.db_manager)
        self._setup_streamlit()

    def _setup_streamlit(self):
        st.set_page_config(page_title="Агроаналитика", layout="wide")
        st.title("📊 Интеллектуальная система управления сельскохозяйственными ресурсами")

        self.menu = [
            "Ввод данных", "Анализ", "Прогнозирование", 
            "Оптимизация ресурсов", "Экспорт данных", "Рекомендации"
        ]
        self.choice = st.sidebar.selectbox("Меню", self.menu)

        if self.choice == "Ввод данных":
            self._input_data()
        elif self.choice == "Анализ":
            self._analyze_data()
        elif self.choice == "Прогнозирование":
            self._predict_yield()
        elif self.choice == "Оптимизация ресурсов":
            self._optimize_resources()
        elif self.choice == "Экспорт данных":
            self._export_data()
        elif self.choice == "Рекомендации":
            self._show_recommendations()

    def _input_data(self):
        st.subheader("🌱 Ввод агрономических данных")
        with st.form("data_form"):
            col1, col2 = st.columns(2)
            with col1:
                Культура = st.selectbox("Культура", ["Пшеница", "Кукуруза", "Подсолнечник", "Рис", "Ячмень"])
                Уровень_pH = st.slider("Уровень pH", 4.0, 9.0, 6.5)
                Азот = st.number_input("Азот (кг/га)", 0.0, 1000.0)
                Фосфор = st.number_input("Фосфор (кг/га)", 0.0, 1000.0)
            with col2:
                Калий = st.number_input("Калий (кг/га)", 0.0, 1000.0)
                Урожайность = st.number_input("Урожайность (ц/га)", 0.0, 1000.0)
                Широта = st.number_input("Широта", -90.0, 90.0, 43.2567)
                Долгота = st.number_input("Долгота", -180.0, 180.0, 76.9286)
                Дата = st.date_input("Дата", datetime.now())

            if st.form_submit_button("Сохранить"):
                if Урожайность < 0 or Уровень_pH < 0 or Азот < 0 or Фосфор < 0 or Калий < 0:
                    st.error("Ошибка: Введены некорректные данные. Значения не могут быть отрицательными.")
                else:
                    self.db_manager.insert_data(Культура, Уровень_pH, Азот, Фосфор, Калий, Урожайность, Широта, Долгота, Дата.strftime('%Y-%m-%d'))
                    st.success("✅ Данные успешно сохранены!")

    def _analyze_data(self):
        st.subheader("📈 Аналитическая панель")
        df = self.db_manager.fetch_data()

        if not df.empty:
            # Тепловая карта
            st.subheader("🌍 Тепловая карта урожайности")
            m = folium.Map(location=[df['Широта'].mean(), df['Долгота'].mean()], zoom_start=5)
            heat_data = [[row['Широта'], row['Долгота'], row['Урожайность']] for _, row in df.iterrows()]
            HeatMap(heat_data, radius=15).add_to(m)
            folium_static(m)

            # Графики
            fig = px.scatter_matrix(
                df,
                dimensions=['Уровень_pH', 'Азот', 'Фосфор', 'Калий', 'Урожайность'],
                color='Культура',
                title="Матрица корреляции параметров"
            )
            st.plotly_chart(fig)

            # Статистика
            st.subheader("📊 Статистика по культурам")
            st.dataframe(df.groupby('Культура').agg({
                'Урожайность': ['mean', 'max', 'min'],
                'Азот': 'mean'
            }).style.format("{:.2f}"))

        else:
            st.warning("⚠️ Недостаточно данных для анализа")

    def _predict_yield(self):
        st.subheader("🔮 Прогноз урожайности")
        trained_models = self.yield_predictor.train_model()

        if trained_models:
            with st.form("predict_form"):
                col1, col2 = st.columns(2)
                with col1:
                    Уровень_pH = st.slider("Уровень pH", 4.0, 9.0, 6.5)
                    Азот = st.number_input("Азот (кг/га)", 0.0, 1000.0)
                with col2:
                    Фосфор = st.number_input("Фосфор (кг/га)", 0.0, 1000.0)
                    Калий = st.number_input("Калий (кг/га)", 0.0, 1000.0)

                if st.form_submit_button("Рассчитать"):
                    predictions = self.yield_predictor.predict_yield(Уровень_pH, Азот, Фосфор, Калий)
                    if predictions:
                        for model, value in predictions.items():
                            st.success(f"**{model}**: {value:.2f} ц/га")

                        # Объяснение предсказания
                        shap_values = self.yield_predictor.explain_prediction(Уровень_pH, Азот, Фосфор, Калий)
                        if shap_values:
                            st.subheader("Объяснение предсказания с помощью SHAP")
                            fig, ax = plt.subplots()
                            shap.plots.waterfall(shap_values[0], show=False)
                            st.pyplot(fig)
                    else:
                        st.error("Ошибка прогнозирования")
        else:
            st.warning("⚠️ Требуется минимум 10 записей для обучения моделей")

    def _optimize_resources(self):
        st.subheader("⚖️ Оптимизация ресурсов")
        with st.form("optim_form"):
            культура = st.selectbox("Культура", ["Пшеница", "Кукуруза", "Подсолнечник", "Рис", "Ячмень"])
            бюджет = st.number_input("Бюджет (тенге)", 0, 1000000, 100000)

            if st.form_submit_button("Оптимизировать"):
                result = self.resource_optimizer.optimize_resources(бюджет, культура)
                if result:
                    st.subheader("Результаты оптимизации:")
                    df = pd.DataFrame.from_dict(result, orient='index', columns=['Значение'])
                    st.dataframe(df.style.format("{:.2f}"))
                else:
                    st.error("Нет данных для выбранной культуры")

    def _export_data(self):
        st.subheader("📤 Экспорт данных")
        df = self.db_manager.fetch_data()
        if not df.empty:
            st.download_button(
                label="Скачать CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name="агро_данные.csv",
                mime="text/csv"
            )
        else:
            st.warning("Нет данных для экспорта")

    def _show_recommendations(self):
        st.subheader("📌 Рекомендации")
        df = self.db_manager.fetch_data()
        if not df.empty:
            avg_yield = df['Урожайность'].mean()
            st.markdown(f"""
            ### Средняя урожайность: **{avg_yield:.2f} ц/га**
            #### Советы по улучшению:
            - Оптимальный уровень pH: 6.0-7.0
            - Баланс NPK: 4:2:1
            - Используйте сидераты для улучшения почвы
            - Учитывайте сезонные изменения и климатические условия
            """)
        else:
            st.warning("Нет данных для анализа")

    def run(self):
        self.db_manager.close()

if __name__ == "__main__":
    app = AgricultureApp()
    app.run()
