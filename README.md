# SmartEnergy Optimizer (MVP)

Aplicacion Streamlit para monitoreo energetico casi en tiempo real sobre PostgreSQL: genera datos simulados, entrena un IsolationForest para deteccion de anomalias y calcula metricas (totales, anomalias detectadas, porcentaje y estado Normal/Precaucion/Critico) que se persisten en la tabla `metricas`.

Alcance: demo/MVP con datos mock. Los valores provienen de generadores estadisticos (gaussianas con perturbaciones). No modelan fisica real; los KPIs reflejan ese set sintetico.

## Stack
- Python 3.11+ (pandas, numpy, scikit-learn, plotly, streamlit)
- SQLAlchemy + psycopg2 (PostgreSQL)
- IsolationForest para deteccion de anomalias

## Estructura clave
- `core/`: configuracion, motor SQLAlchemy, modelos ORM.
- `scripts/`: bootstrap, reseteo y generacion de datos.
  - `simulate_data`: genera ~10k registros batch (historico).
  - `stream_realtime`: genera datos en tiempo real en ciclos pequenos.
  - `auto_train_loop`: reentrena modelo y recalcula metricas en bucle.
  - `seed_db` / `reset_db`: crear tablas / limpiar y repoblar.
- `ml/`: entrenamiento (`train_model.py`) y deteccion + registro de metricas (`detect_anomalies.py`).
- `app/`: app Streamlit (`dashboard_app.py`) con panel de estado, filtros, graficos y descargas.
- `tests/`: esqueletos de pruebas.

## Requisitos
- Python 3.11+ y PostgreSQL accesible.
- Variable `DATABASE_URL` en un `.env` en la raiz, ejemplo:
  ```
  DATABASE_URL=postgresql://smartuser:smartpwd@127.0.0.1:5432/smartenergy
  ```

## Flujo en tiempo real (recomendado)
1) Generador continuo:
   ```bash
   python -m scripts.stream_realtime --interval 1 --batch-size 5 --anomaly-ratio 0.12 --max-registros 20000
   ```
   - `interval`: segundos entre ciclos.
   - `batch-size`: registros por ciclo.
   - `anomaly-ratio`: probabilidad de anomalia por variable (puede afectar varias variables a la vez).
   - `max-registros`: limite duro; se borran los mas antiguos si se excede.
2) Reentrenar y recalcular en loop (otra terminal):
   ```bash
   python -m scripts.auto_train_loop --interval 300
   ```
   - `interval`: segundos entre ciclos (default 300s). No corras mas de un loop a la vez.
3) Dashboard:
   ```bash
   streamlit run app/dashboard_app.py
   ```
   Auto-actualiza cada pocos segundos sin recargar completa, leyendo la BD.

## Flujo batch (historico)
1) Generar datos simulados (10k, respeta max 20k):
   ```bash
   python -m scripts.simulate_data
   ```
2) Entrenar modelo:
   ```bash
   python -m ml.train_model
   ```
3) Calcular metricas:
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
   - Entrenar modelo: usa los datos actuales y recalcula metricas.
   - Recalcular metricas: reevalua el modelo entrenado.
3) Zona de mantenimiento (expander):
   - Generar dataset historico (batch 10k).
   - Reiniciar base de datos (limpia `registros` y `metricas` y repuebla 10k).
4) Filtros: rango de fechas, variable, estado Normal/Anomalo, muestreo para graficos (max 5k filas) y limite de tabla.
5) Tabs:
   - Analisis general: KPIs y gauge de salud.
   - Variables energeticas: series e histogramas.
   - Comparativos: scatter matrix.
   - Deteccion de anomalias: predicciones del modelo.
   - Tabla de registros: vista y descarga CSV.

## Datos y supuestos
- Datos 100% simulados, con distribuciones gaussianas y perturbaciones programadas (no fisica real).
- Tasas de anomalias y patrones cambian en cada corrida; los KPIs reflejan ese set sintetico.
- Timestamps en tiempo actual; se recorta la tabla cuando se supera `max-registros`.
- El modelo usa aislamiento con contamination ajustada a la tasa observada en los datos mock.

## Comandos utiles
- Resetear tablas y regenerar 10k registros: `python -m scripts.reset_db`
- Regenerar solo datos (batch): `python -m scripts.simulate_data`
- Generacion continua: `python -m scripts.stream_realtime ...`
- Reentrenar en loop: `python -m scripts.auto_train_loop ...`

## Notas
- Dashboard funciona sin modelo inicial; usa los botones para generarlo si falta.
- No productivo; orientado a demo/MVP con datos sinteticamente plausibles.
- Directorios `data/` y `models/` estan ignorados en git por defecto.
