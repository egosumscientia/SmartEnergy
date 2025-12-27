# SmartEnergy Optimizer

Aplicación Streamlit para monitoreo energético casi en tiempo real sobre PostgreSQL: genera datos simulados, entrena un IsolationForest para detección de anomalías y calcula métricas (totales, anomalías detectadas, porcentaje y estado Normal/Precaución/Crítico) que se persisten en la tabla `metricas`.

La aplicación aborda el monitoreo continuo de variables eléctricas en entornos industriales, donde desviaciones en voltaje, corriente, potencia o temperatura pueden indicar degradación de la calidad de la energía, ineficiencias o fallas incipientes. SmartEnergy Optimizer automatiza la detección de comportamientos anómalos, consolida métricas de estado energético y presenta una visión clara de la salud del sistema para mantenimiento preventivo y decisiones operativas.

## Stack
- Python 3.11+ (pandas, numpy, scikit-learn, plotly, streamlit)
- SQLAlchemy + psycopg2 (PostgreSQL)
- IsolationForest para detección de anomalías

## Estructura clave
- `core/`: configuración, motor SQLAlchemy, modelos ORM.
- `scripts/`: bootstrap, reseteo y generación de datos.
  - `simulate_data`: genera ~10k registros batch (histórico).
  - `stream_realtime`: genera datos en tiempo real en ciclos pequeños.
  - `auto_train_loop`: reentrena modelo y recalcula métricas en bucle.
  - `seed_db` / `reset_db`: crear tablas / limpiar y repoblar.
- `ml/`: entrenamiento (`train_model.py`) y detección + registro de métricas (`detect_anomalies.py`).
- `app/`: app Streamlit (`dashboard_app.py`) con panel de estado, filtros, gráficos y descargas.
- `tests/`: esqueletos de pruebas.

## Requisitos
- Python 3.11+ y PostgreSQL accesible.
- Variable `DATABASE_URL` en un `.env` en la raíz, ejemplo:
  ```
  DATABASE_URL=postgresql://smartuser:smartpwd@127.0.0.1:5432/smartenergy
  ```

## Flujo en tiempo real (recomendado)
1) Levanta el generador continuo:
   ```bash
   python -m scripts.stream_realtime --interval 1 --batch-size 5 --anomaly-ratio 0.12 --max-registros 20000
   ```
   - `interval`: segundos entre ciclos.
   - `batch-size`: registros por ciclo.
   - `anomaly-ratio`: probabilidad de anomalía por variable (puede afectar varias variables a la vez).
   - `max-registros`: límite duro; se borran los más antiguos si se excede.
2) Reentrena y recalcula métricas de forma automática (en otra terminal):
   ```bash
   python -m scripts.auto_train_loop --interval 300
   ```
   - `interval`: segundos entre ciclos (default 300s). Evita correr más de un loop a la vez.
3) Corre el dashboard:
   ```bash
   streamlit run app/dashboard_app.py
   ```
   La página se auto-actualiza cada pocos segundos sin recargar completa, leyendo siempre de la BD.

## Flujo batch (histórico)
1) Generar datos simulados (10k, respeta máximo 20k):
   ```bash
   python -m scripts.simulate_data
   ```
2) Entrenar modelo:
   ```bash
   python -m ml.train_model
   ```
3) Calcular métricas:
   ```bash
   python -m ml.detect_anomalies
   ```
4) Dashboard:
   ```bash
   streamlit run app/dashboard_app.py
   ```

## Flujo GUI
1) Levanta el dashboard: `streamlit run app/dashboard_app.py`.
2) Acciones principales:
   - Entrenar modelo: usa los datos actuales y recalcula métricas.
   - Recalcular métricas: reevalúa el modelo entrenado.
3) Zona de mantenimiento (expander):
   - Generar dataset histórico (batch 10k).
   - Reiniciar base de datos (limpia `registros` y `metricas` y repuebla 10k).
4) Filtros: rango de fechas, variable, estado Normal/Anómalo, muestreo para gráficos (máx 5k filas) y límite de tabla.
5) Tabs:
   - Análisis general: KPIs y gauge de salud.
   - Variables energéticas: series e histogramas.
   - Comparativos: scatter matrix.
   - Detección de anomalías: predicciones del modelo.
   - Tabla de registros: vista y descarga CSV.

## Comandos útiles
- Resetear tablas y regenerar 10k registros: `python -m scripts.reset_db`
- Regenerar solo datos (batch): `python -m scripts.simulate_data`
- Generación continua: `python -m scripts.stream_realtime ...`
- Reentrenar en loop: `python -m scripts.auto_train_loop ...`

## Notas
- El dashboard funciona sin modelo inicial; usa los botones para generarlo si falta.
- El generador en tiempo real mantiene el límite de registros eliminando los más antiguos.
- Directorios `data/` y `models/` están ignorados en git por defecto.
