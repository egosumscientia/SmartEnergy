# SmartEnergy Optimizer

Dashboard en Streamlit para monitoreo energetico, generacion de datos sintéticos, entrenamiento de un modelo de deteccion de anomalías y calculo de metricas sobre PostgreSQL.

## Stack
- Python 3.11+ (pandas, numpy, scikit-learn, plotly, streamlit)
- SQLAlchemy + psycopg2 (PostgreSQL)
- IsolationForest para deteccion de anomalías

## Estructura clave
- `core/`: configuracion, motor SQLAlchemy, modelos ORM.
- `scripts/`: bootstrap, reseteo y generacion de datos (`simulate_data`, `seed_db`, `reset_db`).
- `ml/`: entrenamiento (`train_model.py`) y deteccion + registro de metricas (`detect_anomalies.py`).
- `app/`: app Streamlit (`dashboard_app.py`) con acciones de dev, filtros y graficos.
- `tests/`: esqueletos de pruebas (pendientes).

## Requisitos
- Python 3.11+ y PostgreSQL accesible.
- Variable de entorno `DATABASE_URL` en un `.env` en la raiz, ejemplo:
  ```
  DATABASE_URL=postgresql://smartuser:smartpwd@127.0.0.1:5432/smartenergy
  ```

## Estado actual de la BD (verificado)
- `DATABASE_URL`: `postgresql://smartuser:smartpwd@127.0.0.1:5432/smartenergy`
- Base en uso: `smartenergy`
- Version del servidor: `PostgreSQL 18.0 on x86_64-windows`
- Tablas en esquema `public`: 2
- Ultima verificacion de conexion: `2025-12-27 13:04:38.425538-05:00` con `psycopg2`/SQLAlchemy

## Instalacion
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Flujo de uso
1) Crear tablas (solo una vez):
   ```bash
   python -m scripts.seed_db
   ```
2) Generar datos simulados:
   ```bash
   python -m scripts.simulate_data
   ```
3) Entrenar modelo y guardar bundle en `models/anomaly_detector.pkl`:
   ```bash
   python -m ml.train_model
   ```
4) Calcular metricas y guardarlas en la tabla `metricas`:
   ```bash
   python -m ml.detect_anomalies
   ```
5) Ejecutar dashboard:
   ```bash
   streamlit run app/dashboard_app.py
   ```

## Comandos utiles
- Resetear tablas y regenerar 10k registros:
  ```bash
  python -m scripts.reset_db
  ```
- Regenerar solo datos (respeta maximo 20k registros):
  ```bash
  python -m scripts.simulate_data
  ```

## Notas
- La app asume que existen datos y el modelo entrenado; de lo contrario mostrara advertencias.
- La tasa de contaminacion y ratio de anomalías en los scripts se eligen al azar en cada ejecucion.
- Directorios `data/` y `models/` estan ignorados en git por defecto.
