# --- LIBRERÍAS ADAPTADAS PARA RANDOM FOREST Y PERSISTENCIA ---
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import TimeSeriesSplit

# --- CONFIGURACIÓN DEL MODELO ---
prediction_hour = 1 # Predicción a 1 hora (H+1)

# MODO EXPERIMENTAL: Desactivar lags para probar modelo sin persistencia
USE_LAGS = True  # ← Cambiar a True para activar lags

# Características que queremos "desfasar" (Lags)
lag_features = ['ts', 'hr', 'p0']
lag_steps = [1, 2, 3] # Desfases de 1, 2 y 3 horas

file_path = "dataset_ml.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"No se encuentra: {file_path}")

# Cargar dataset
df = pd.read_csv(file_path, sep=";")

# --- [CORRECCIÓN CRÍTICA] PREPROCESAMIENTO DE FECHA E ÍNDICE ---
print("--- Procesando Fechas e Índice ---")

# 1. Convertir 'momento' a objeto datetime real
df['momento'] = pd.to_datetime(df['momento'])

# 2. Establecer 'momento' como el índice del DataFrame
df.set_index('momento', inplace=True)

# 3. Ordenar el índice (vital para Time Series)
df.sort_index(inplace=True)

# 4. [IMPORTANTE] Remuestreo a Hora (Resample)
# Tus datos originales son por minuto (01:00, 01:01).
# Tus lags son [1, 2, 3]. Sin esto, el modelo haría lags de 1 minuto.
# Al hacer resample('H'), promediamos los datos para tener 1 fila por hora.
print("Datos originales (filas):", df.shape[0])
df = df.resample('H').mean()
print("Datos después de resample horario (filas):", df.shape[0])

# Eliminar filas que hayan quedado vacías tras el resample (si hay huecos grandes)
df = df.dropna()

print("VERIFICACIÓN DEL DATASET")
print(df.shape)
print(df.columns)

# 1. Mostrar primeras filas
print("Primeras 5 filas del dataset:")
print(df.head(), "\n")

# --- INGENIERÍA DE CARACTERÍSTICAS TEMPORALES ---
print("--- Extrayendo características temporales (Hora y Día del Año) ---")
# AHORA SÍ funcionará porque el índice es DatetimeIndex
df['hour'] = df.index.hour
df['dayofyear'] = df.index.dayofyear

# --- DEFINICIÓN DE CARACTERÍSTICAS (X) Y OBJETIVO (Y) ---
target = 'ts'

# Características iniciales
features = ['hr', 'p0', 'hour', 'dayofyear']

# Crear las características desfasadas (lags) - SOLO SI ESTÁ ACTIVADO
if USE_LAGS:
    print("Modo: CON Lags (ts_lag_1h, ts_lag_2h, ts_lag_3h...)")
    for feature in lag_features:
        if feature in df.columns:
            for h in lag_steps:
                new_col_name = f'{feature}_lag{h}h'
                df[new_col_name] = df[feature].shift(h)
                features.append(new_col_name)
        else:
            print(f"Advertencia: La columna {feature} no existe para crear lags.")
else:
    print("🔸 Modo: SIN Lags (solo variables físicas y temporales)")

# Crear la columna objetivo desfasada (shift negativo para traer el futuro al presente)
df[f'ts_future_H{prediction_hour}'] = df[target].shift(-prediction_hour)

# Filtrar el DataFrame final y eliminar filas nulas generadas por los shifts
df_model = df[features + [f'ts_future_H{prediction_hour}']].dropna()

X = df_model[features]
y = df_model[f'ts_future_H{prediction_hour}']

print(f"\n--- Características (X) a usar en el modelo ---")
print(X.columns.tolist())
print("-" * 60)

# --- DIVISIÓN DE DATOS: 60% TRAIN, 20% VALIDATION, 20% TEST ---
print("\n--- División de Datos: 60% Train | 20% Validation | 20% Test ---")

total_size = len(df_model)
train_size = int(total_size * 0.6)
val_size = int(total_size * 0.2)
test_size = total_size - train_size - val_size

# División temporal (respetando el orden cronológico)
X_train = X[:train_size]
X_val = X[train_size:train_size + val_size]
X_test = X[train_size + val_size:]

y_train = y[:train_size]
y_val = y[train_size:train_size + val_size]
y_test = y[train_size + val_size:]

print(f"Train:      {len(X_train):6d} filas ({len(X_train)/total_size*100:.1f}%)")
print(f"Validation: {len(X_val):6d} filas ({len(X_val)/total_size*100:.1f}%)")
print(f"Test:       {len(X_test):6d} filas ({len(X_test)/total_size*100:.1f}%)")
print(f"Total:      {total_size:6d} filas")

# --- MODELO RANDOM FOREST REGRESSOR ---
print("\n--- Entrenando Random Forest Regressor ---")
# Ajusté min_samples_leaf a 2 porque al reducir los datos a horas tenemos menos filas
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# --- FUNCIÓN AUXILIAR PARA CALCULAR MÉTRICAS ---
def calcular_metricas(y_real, y_pred, set_name):
    mae = mean_absolute_error(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_real, y_pred)

    print(f"\n{'='*60}")
    print(f"MÉTRICAS - {set_name}")
    print(f"{'='*60}")
    print(f"MAE  (Error Absoluto Medio):       {mae:.4f} °C")
    print(f"RMSE (Raíz Error Cuadrático):      {rmse:.4f} °C")
    print(f"R²   (Coeficiente Determinación):  {r2:.4f}")
    print(f"{'='*60}")
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

# --- EVALUACIÓN ---
print("\n--- Evaluando Modelo ---")
y_train_pred = rf_model.predict(X_train)
metricas_train = calcular_metricas(y_train, y_train_pred, "TRAIN SET")

y_val_pred = rf_model.predict(X_val)
metricas_val = calcular_metricas(y_val, y_val_pred, "VALIDATION SET")

y_test_pred = rf_model.predict(X_test)
metricas_test = calcular_metricas(y_test, y_test_pred, "TEST SET")

# --- ANÁLISIS DE OVERFITTING ---
diff_r2 = metricas_train['R2'] - metricas_val['R2']
print(f"\nDiferencia R² (Train - Val): {diff_r2:.4f}")
if diff_r2 > 0.10:
    print("⚠️ ALERTA: Overfitting fuerte detectado.")
else:
    print("✅ Ajuste correcto entre Train y Validación.")

# --- GUARDAR MODELO ---
model_filename = f'random_forest_H{prediction_hour}_hourly.joblib'
joblib.dump(rf_model, model_filename)
joblib.dump(X.columns.tolist(), f'features_H{prediction_hour}_hourly.joblib')
print(f"\nModelo guardado: {model_filename}")

# --- GRÁFICO FINAL RAPIDO ---
plt.figure(figsize=(12, 5))
plt.plot(y_test.index, y_test.values, label='Real', alpha=0.6)
plt.plot(y_test.index, y_test_pred, label='Predicción', alpha=0.7, color='orange')
plt.title(f'Predicción Temperatura (H+{prediction_hour}) - Test Set')
plt.legend()
plt.tight_layout()
plt.savefig('resultado_final.png')
print("Gráfico generado: resultado_final.png")
