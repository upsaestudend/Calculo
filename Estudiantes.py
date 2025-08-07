import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Cargar dataset
df = pd.read_csv("dataset_estudiantes_final.csv")
df.columns = df.columns.str.strip()

# 2. Renombrar columnas
df = df.rename(columns={
    'Nota_Aritmetica': 'aritmetica',
    'Nota_Algebra': 'algebra',
    'Nota_Geometria_Plana': 'geometria_plana',
    'Nota_Trigonometria': 'trigonometria',
    'Nota_Progresiones': 'progresiones',
    'Calificacion_Diagnostico': 'diagnostico',
    'Calificacion_Calculo': 'calculo'
})

# 3. Variables independientes y dependiente
X = df[['aritmetica', 'algebra', 'geometria_plana', 'trigonometria', 'progresiones', 'diagnostico']]
y = df['calculo']

# 4. Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Modelo Ridge
modelo_ridge = Ridge(alpha=1.0)
modelo_ridge.fit(X_train, y_train)

# 6. Predicción Ridge
y_pred_ridge = modelo_ridge.predict(X_test)
y_pred_ridge = np.clip(y_pred_ridge, 0, 100)

# 7. Predicción Fórmula 60/40
promedios_5 = X_test[['aritmetica', 'algebra', 'geometria_plana', 'trigonometria', 'progresiones']].mean(axis=1)
y_pred_manual = 0.6 * X_test['diagnostico'] + 0.4 * promedios_5
y_pred_manual = np.clip(y_pred_manual, 0, 100)

# 8. Métricas Ridge
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print("📊 Métricas Modelo Ridge")
print(f"Error cuadrático medio (MSE): {mse_ridge:.2f}")
print(f"Coeficiente de determinación (R²): {r2_ridge:.2f}")
print("\n📌 Coeficientes Ridge:")
for var, coef in zip(X.columns, modelo_ridge.coef_):
    print(f"{var}: {coef:.4f}")

# 9. Gráfico Real vs Predicho (Ridge)
plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_test, y=y_pred_ridge, label="Predicho Ridge")
sns.lineplot(x=y_test, y=y_test, color='red', label='Ideal')
plt.xlabel("Nota Real")
plt.ylabel("Nota Predicha")
plt.title("Real vs Predicho (Ridge)")
plt.legend()
plt.tight_layout()
plt.show()

# 10. Distribución de Notas Reales
plt.figure(figsize=(6, 4))
sns.histplot(df['calculo'], bins=10, kde=True)
plt.xlabel("Nota Final (Calculo)")
plt.ylabel("Frecuencia")
plt.title("Distribución de Notas Finales")
plt.tight_layout()
plt.show()

# 11. Matriz de Correlación (solo numéricas)
num_df = df.select_dtypes(include='number')
plt.figure(figsize=(8, 6))
sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm")
plt.title("Matriz de Correlación")
plt.tight_layout()
plt.show()

# 12. Comparación Ridge vs 60/40
plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_test, y=y_pred_ridge, label="Ridge")
sns.scatterplot(x=y_test, y=y_pred_manual, label="Fórmula 60/40")
plt.xlabel("Nota Real")
plt.ylabel("Nota Predicha")
plt.title("Comparación: Ridge vs Fórmula 60/40")
plt.legend()
plt.tight_layout()
plt.show()

