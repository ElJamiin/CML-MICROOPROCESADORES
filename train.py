import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# =======================================================
# 1. Carga y Preprocesamiento de Datos
# =======================================================

print("Cargando y preprocesando datos...")

# Carga de datos: El archivo parece no tener cabecera para la primera columna
# por eso se usa index_col=0. El separador es la coma (default para CSV).
# El nombre de la columna 'maxtemp' tiene solo minúsculas.

# Nota: Si el archivo no tiene encabezados, necesitarías agregar names=['Col1', 'Date', ...]
# Asumiendo que pandas lo leyó correctamente:
df = pd.read_csv('data.csv')

# --- 1.1 Limpieza y Selección de Características ---
# Reemplazar los nombres de características de la plantilla anterior por los reales del archivo
features = ['mintemp', 'pressure', 'humidity', 'mean wind speed'] 
target = 'maxtemp'

# Eliminar filas con valores faltantes (NaN) en las columnas clave
df.dropna(subset=features + [target], inplace=True)

# Las características categóricas ('weather', 'cloud') se omiten por simplicidad,
# pero en un proyecto real se convertirían a numéricas (One-Hot Encoding).

X = df[features]
y = df[target]

# --- 1.2 División y Escalado ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Opcional: Escalar los datos (aunque Random Forest es menos sensible, es buena práctica)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =======================================================
# 2. Entrenamiento del Modelo (Random Forest Regressor)
# =======================================================

print(f"Iniciando entrenamiento del Random Forest con {len(X_train)} muestras...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
print("Entrenamiento finalizado.")


# =======================================================
# 3. Evaluación y Predicción
# =======================================================

y_pred = rf_model.predict(X_test_scaled)

# Calcular métricas
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Resultados de Evaluación ---")
print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
print(f"R2 Score: {r2:.4f}")


# =======================================================
# 4. Generación de Artefactos (Para CML)
# =======================================================

# 4.1. Guardar el Modelo Entrenado (rf_model.pkl)
model_filename = 'rf_model.pkl'
joblib.dump(rf_model, model_filename)
print(f"Modelo guardado como {model_filename}.")

# 4.2. Generar el Gráfico de Predicción (prediction_plot.png)
plt.figure(figsize=(10, 6))
# Gráfico de dispersión de Predicciones vs. Valores Reales
plt.scatter(y_test, y_pred, alpha=0.6, color='darkblue')
# Línea perfecta y=x (lo que el modelo debería predecir)
max_val = max(y_test.max(), y_pred.max())
min_val = min(y_test.min(), y_pred.min())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
plt.title('Random Forest: Predicción vs. Temperatura Real')
plt.xlabel('Temperatura Máxima Real')
plt.ylabel('Temperatura Máxima Predicha')
plt.grid(True)
plot_filename = 'prediction_plot.png'
plt.savefig(plot_filename)
print(f"Gráfico de predicción guardado como {plot_filename}.")

# 4.3. Guardar las Métricas (metrics.txt)
metrics_filename = 'metrics.txt'
with open(metrics_filename, 'w') as f:
    f.write("Random Forest Regressor - Predicción de Temperatura Máxima\n")
    f.write("-" * 50 + "\n")
    f.write(f"Características utilizadas: {features}\n")
    f.write(f"MSE (Error Cuadrático Medio): {mse:.2f}\n")
    f.write(f"R2 Score (Coeficiente de Determinación): {r2:.4f}\n")
print(f"Métricas guardadas en {metrics_filename}.")

# =======================================================
# Fin del Script
# =======================================================
