# Importar librerías estándar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor # Importa el modelo de Random Forest
from sklearn.metrics import mean_squared_error, r2_score # Para evaluar el modelo
import joblib # Para guardar el modelo entrenado

# -----------------------------------------------------
# Carga de Datos
# -----------------------------------------------------
# Carga tu dataset principal
df = pd.read_csv('data.csv', sep=';') # Ajusta el separador si es necesario

# -----------------------------------------------------
# Preparación de Variables para Random Forest
# -----------------------------------------------------
# 🚨 Suposición: 'Temperature' es tu variable a predecir (Objetivo y)
# y 'Feature1', 'Feature2', etc., son tus variables de entrada (Características X)

# Define las variables a usar
features = ['Feature1', 'Feature2', 'Feature3'] # Reemplaza con tus columnas reales
target = 'Temperature' # Reemplaza con el nombre de tu columna de temperatura

X = df[features]
y = df[target]

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------------------------------
# Entrenamiento del Modelo
# -----------------------------------------------------
print("Iniciando entrenamiento del Random Forest...")
# Crea y entrena el modelo Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

print("Entrenamiento finalizado.")

# -----------------------------------------------------
# Evaluación del Modelo
# -----------------------------------------------------
y_pred = rf_model.predict(X_test)

# Calcular métricas
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Métricas en el conjunto de prueba:")
print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# -----------------------------------------------------
# Generación de Artefactos
# -----------------------------------------------------
# 1. Guardar el modelo entrenado (artefacto .pkl)
model_filename = 'rf_model.pkl'
joblib.dump(rf_model, model_filename)

# 2. Generar un gráfico de predicciones vs. valores reales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title('Predicciones de Temperatura vs. Valores Reales')
plt.xlabel('Valores Reales (Temperatura)')
plt.ylabel('Predicciones (Temperatura)')
# Guardar la imagen (El archivo que sube GitHub Actions)
plot_filename = 'prediction_plot.png'
plt.savefig(plot_filename)
print(f"Gráfico de predicción guardado como {plot_filename}.")

# 3. Guardar métricas en un archivo de texto (otro artefacto)
metrics_filename = 'metrics.txt'
with open(metrics_filename, 'w') as f:
    f.write(f"Random Forest Regressor Metrics:\n")
    f.write(f"MSE: {mse:.2f}\n")
    f.write(f"R2 Score: {r2:.2f}\n")
print(f"Métricas guardadas en {metrics_filename}.")
