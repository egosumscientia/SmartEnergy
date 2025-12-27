# SmartEnergy Optimizer

Aplicacion Streamlit para monitoreo energetico en tiempo casi real sobre PostgreSQL: genera datos simulados, entrena un IsolationForest para deteccion de anomalias y calcula metricas (totales, anomalias detectadas, porcentaje y estado Normal/Precaucion/Critico) que se persisten en la tabla `metricas`.

La aplicación aborda el problema del monitoreo continuo de variables eléctricas en entornos industriales, donde desviaciones en voltaje, corriente, potencia o temperatura pueden indicar degradación de la calidad de la energía, ineficiencias o fallas incipientes. SmartEnergy Optimizer automatiza la detección de estos comportamientos anómalos, consolida métricas de estado energético y presenta una visión clara de la salud del sistema para soporte a mantenimiento preventivo y toma de decisiones operativas.

## Stack
- Python 3.11+ (pandas, numpy, scikit-learn, plotly, streamlit)
- SQLAlchemy + psycopg2 (PostgreSQL)
- IsolationForest para deteccion de anomalias

## Estructura clave
- `core/`: configuracion, motor SQLAlchemy, modelos ORM.
- `scripts/`: bootstrap, reseteo y generacion de datos (`simulate_data`, `seed_db`, `reset_db`).
- `ml/`: entrenamiento (`train_model.py`) y deteccion + registro de metricas (`detect_anomalies.py`).
- `app/`: app Streamlit (`dashboard_app.py`) con panel de estado, filtros, graficos y descargas.
- `tests/`: esqueletos de pruebas (pendientes).

## Requisitos
- Python 3.11+ y PostgreSQL accesible.
- Variable de entorno `DATABASE_URL` en un `.env` en la raiz, ejemplo:
  ```
  DATABASE_URL=postgresql://smartuser:smartpwd@127.0.0.1:5432/smartenergy
  ```

## Que calcula la app
- Modelo: IsolationForest entrenado sobre corriente, voltaje, potencia_activa y temperatura_motor.
- Deteccion: asigna etiqueta de anomalia por prediccion del modelo (-1) y compara contra la columna `estado` (Normal/Anomalo).
- Metricas guardadas en tabla `metricas`: total de registros, anomalias detectadas, porcentaje detectado, estado (Normal/Precaucion/Critico segun porcentaje/precision) y timestamp. El dashboard muestra el ultimo analisis y el delta vs el anterior.

## Flujo de uso (GUI)
1) Levantar el dashboard:
   ```bash
   streamlit run app/dashboard_app.py
   ```
2) Panel superior: muestra registros en BD, si existe modelo entrenado y el ultimo analisis (%) con delta vs analisis previo.
3) Botones rapidos (ejecutan scripts internos):
   - Generar dataset: inserta ~10 000 registros simulados (limita a 20k totales) y crea tablas si faltan.
   - Entrenar modelo: entrena IsolationForest con los datos actuales y registra nuevas metricas.
   - Recalcular metricas: corre deteccion con el modelo entrenado y guarda metricas.
   - Reiniciar base de datos: trunca `registros` y `metricas`, y repuebla con 10 000 registros.
4) Filtros: rango de fechas, variable a graficar, estados Normal/Anomalo, muestreo opcional (max 5k filas) y limite de filas de la tabla.
5) Tabs de analisis:
   - Analisis general: KPIs de anomalias y gauge de salud.
   - Variables energeticas: series de tiempo e histogramas.
   - Comparativos: scatter matrix de variables.
   - Deteccion de anomalias: superpone predicciones del modelo.
   - Tabla de registros: vista y descarga (respeta limite de filas configurado).

## Flujo CLI alternativo
1) Crear tablas: `python -m scripts.seed_db`
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
