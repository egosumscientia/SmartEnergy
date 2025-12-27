# SmartEnergy Optimizer

Dashboard en Streamlit para monitoreo energetico, generacion de datos sinteticos, entrenamiento de un modelo de deteccion de anomalias y calculo de metricas sobre PostgreSQL.

## Stack
- Python 3.11+ (pandas, numpy, scikit-learn, plotly, streamlit)
- SQLAlchemy + psycopg2 (PostgreSQL)
- IsolationForest para deteccion de anomalias

## Estructura clave
- `core/`: configuracion, motor SQLAlchemy, modelos ORM.
- `scripts/`: bootstrap, reseteo y generacion de datos (`simulate_data`, `seed_db`, `reset_db`).
- `ml/`: entrenamiento (`train_model.py`) y deteccion + registro de metricas (`detect_anomalies.py`).
- `app/`: app Streamlit (`dashboard_app.py`) con acciones en GUI, filtros y graficos.
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

## Flujo de uso (GUI)
1) Levantar el dashboard:
   ```bash
   streamlit run app/dashboard_app.py
   ```
2) Panel superior: muestra registros en BD, si existe modelo entrenado y el ultimo analisis (%).
3) Botones rapidos (ejecutan scripts internos):
   - Generar dataset: inserta ~10 000 registros simulados (limita a 20k totales) y crea tablas si faltan.
   - Entrenar modelo: entrena IsolationForest con los datos actuales y registra metricas en BD.
   - Recalcular metricas: vuelve a correr la deteccion con el modelo ya entrenado y guarda metricas nuevas.
   - Reiniciar base de datos: trunca `registros` y `metricas`, y repuebla con 10 000 registros nuevos.
4) Tabs de analisis:
   - Analisis general: KPIs de anomalias y gauge de salud.
   - Variables energeticas: series de tiempo e histogramas.
   - Comparativos: scatter matrix de variables.
   - Deteccion de anomalias: superpone predicciones del modelo.
   - Tabla de registros: vista y descarga (max 1000 filas) de los datos filtrados.

## Flujo CLI alternativo
1) Crear tablas (solo una vez): `python -m scripts.seed_db`
2) Generar datos simulados: `python -m scripts.simulate_data`
3) Entrenar modelo: `python -m ml.train_model`
4) Calcular metricas: `python -m ml.detect_anomalies`
5) Ejecutar dashboard: `streamlit run app/dashboard_app.py`

## Comandos utiles
- Resetear tablas y regenerar 10k registros: `python -m scripts.reset_db`
- Regenerar solo datos (respeta maximo 20k registros): `python -m scripts.simulate_data`

## Notas
- La app funciona sin modelo inicial; usa los botones para generarlo si falta.
- La tasa de contaminacion y ratio de anomalias en los scripts se eligen al azar en cada ejecucion.
- Directorios `data/` y `models/` estan ignorados en git por defecto.
