import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Configuración
st.set_page_config(page_title="Predicción Nota Cálculo", layout="centered")
st.title("📘 Predicción de Nota Final en Cálculo")
st.markdown("Modelos: Ridge + Fórmula 60/40 (Diagnóstico 60%)")

# Cargar dataset
@st.cache_data
def cargar_dataset():
    try:
        df = pd.read_csv("dataset_estudiantes_final.csv")
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"❌ Error cargando dataset: {e}")
        return None

df = cargar_dataset()
if df is None or df.empty:
    st.stop()

# Renombrar columnas
df = df.rename(columns={
    'Nota_Aritmetica': 'aritmetica',
    'Nota_Algebra': 'algebra',
    'Nota_Geometria_Plana': 'geometria_plana',
    'Nota_Trigonometria': 'trigonometria',
    'Nota_Progresiones': 'progresiones',
    'Calificacion_Diagnostico': 'diagnostico',
    'Calificacion_Calculo': 'calculo'
})

# Variables
X = df[['aritmetica', 'algebra', 'geometria_plana', 'trigonometria', 'progresiones', 'diagnostico']]
y = df['calculo']

# Entrenamiento Ridge
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo_ridge = Ridge(alpha=1.0)
modelo_ridge.fit(X_train, y_train)
y_pred_ridge = modelo_ridge.predict(X_test)
y_pred_ridge = np.clip(y_pred_ridge, 0, 100)

# Predicción 60/40
promedios_5 = X_test[['aritmetica', 'algebra', 'geometria_plana', 'trigonometria', 'progresiones']].mean(axis=1)
y_pred_manual = 0.6 * X_test['diagnostico'] + 0.4 * promedios_5
y_pred_manual = np.clip(y_pred_manual, 0, 100)

# --- Predicción Personalizada ---
st.subheader("🔍 Predicción Personalizada")
with st.form("formulario_prediccion"):
    aritmetica = st.number_input("Aritmética", 0.0, 100.0)
    algebra = st.number_input("Álgebra", 0.0, 100.0)
    geometria = st.number_input("Geometría Plana", 0.0, 100.0)
    trigonometria = st.number_input("Trigonometría", 0.0, 100.0)
    progresiones = st.number_input("Progresiones", 0.0, 100.0)
    diagnostico = st.number_input("Diagnóstico", 0.0, 100.0)
    submit = st.form_submit_button("Predecir")

    if submit:
        entrada = [[aritmetica, algebra, geometria, trigonometria, progresiones, diagnostico]]
        pred_ridge = modelo_ridge.predict(entrada)[0]
        pred_ridge = np.clip(pred_ridge, 0, 100)
        prom_5 = np.mean([aritmetica, algebra, geometria, trigonometria, progresiones])
        pred_manual = 0.6 * diagnostico + 0.4 * prom_5
        pred_manual = np.clip(pred_manual, 0, 100)

        st.success(f"📈 Nota predicha (Ridge): {pred_ridge:.2f}")
        st.info(f"📊 Nota predicha (Fórmula 60/40): {pred_manual:.2f}")

# --- Métricas Ridge ---
mse = mean_squared_error(y_test, y_pred_ridge)
r2 = r2_score(y_test, y_pred_ridge)
st.subheader("📊 Métricas del Modelo Ridge")
col1, col2 = st.columns(2)
col1.metric("MSE", f"{mse:.2f}")
col2.metric("R²", f"{r2:.2f}")

# --- Coeficientes ---
st.subheader("📌 Coeficientes del Modelo Ridge")
st.dataframe(pd.DataFrame({
    "Variable": X.columns,
    "Coeficiente": modelo_ridge.coef_
}))

# --- Gráfico Real vs Predicho ---
st.subheader("📈 Real vs Predicho (Ridge)")
fig1, ax1 = plt.subplots()
sns.scatterplot(x=y_test, y=y_pred_ridge, ax=ax1, label="Predicho Ridge")
sns.lineplot(x=y_test, y=y_test, color='red', label='Ideal', ax=ax1)
ax1.set_xlabel("Nota Real")
ax1.set_ylabel("Nota Predicha")
ax1.legend()
st.pyplot(fig1)

# --- Distribución de Notas Finales ---
st.subheader("📊 Distribución de Notas Finales (Reales)")
fig2, ax2 = plt.subplots()
sns.histplot(df['calculo'], bins=10, kde=True, ax=ax2)
ax2.set_xlabel("Nota Final")
ax2.set_ylabel("Frecuencia")
st.pyplot(fig2)

# --- Matriz de Correlación ---
st.subheader("🔗 Matriz de Correlación")
num_df = df.select_dtypes(include='number')
fig3, ax3 = plt.subplots()
sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax3)
st.pyplot(fig3)

# --- Dataset completo opcional ---
if st.checkbox("👀 Mostrar dataset completo"):
    st.dataframe(df)

# --- Tabla de predicciones ---
if st.checkbox("📋 Mostrar tabla de predicciones (Ridge)"):
    st.dataframe(pd.DataFrame({"Real": y_test.values, "Predicho": y_pred_ridge}))


