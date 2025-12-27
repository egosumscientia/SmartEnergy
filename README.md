# SmartEnergy Optimizer

Aplicación Streamlit para monitoreo energético en tiempo casi real sobre PostgreSQL: genera datos simulados, entrena un IsolationForest para detección de anomalías y calcula métricas (totales, anomalías detectadas, porcentaje y estado Normal/Precaución/Crítico) que se persisten en la tabla `metricas`.

La aplicación aborda el problema del monitoreo continuo de variables eléctricas en entornos industriales, donde desviaciones en voltaje, corriente, potencia o temperatura pueden indicar degradación de la calidad de la energía, ineficiencias o fallas incipientes. SmartEnergy Optimizer automatiza la detección de estos comportamientos anómalos, consolida métricas de estado energético y presenta una visión clara de la salud del sistema para soporte a mantenimiento preventivo y toma de decisiones operativas.

## Stack
- Python 3.11+ (pandas, numpy, scikit-learn, plotly, streamlit)
- SQLAlchemy + psycopg2 (PostgreSQL)
- IsolationForest para detección de anomalías

## Estructura clave
- `core/`: configuración, motor SQLAlchemy, modelos ORM.
- `scripts/`: bootstrap, reseteo y generación de datos (`simulate_data`, `seed_db`, `reset_db`, `strip_accents_db`).
- `ml/`: entrenamiento (`train_model.py`) y detección + registro de métricas (`detect_anomalies.py`).
- `app/`: app Streamlit (`dashboard_app.py`) con panel de estado, filtros, gráficos y descargas.
- `tests/`: esqueletos de pruebas (pendientes).

## Requisitos
- Python 3.11+ y PostgreSQL accesible.
- Variable de entorno `DATABASE_URL` en un `.env` en la raíz, ejemplo:
  ```
  DATABASE_URL=postgresql://smartuser:smartpwd@127.0.0.1:5432/smartenergy
  ```

## Qué calcula la app
- Modelo: IsolationForest entrenado sobre corriente, voltaje, potencia_activa y temperatura_motor.
- Detección: asigna etiqueta de anomalía por predicción del modelo (-1) y compara contra la columna `estado` (Normal/Anómalo).
- Métricas guardadas en tabla `metricas`: total de registros, anomalías detectadas, porcentaje detectado, estado (Normal/Precaución/Crítico según porcentaje/precisión) y timestamp. El dashboard muestra el último análisis y el delta vs el anterior.

## Flujo de uso (GUI)
1) Levantar el dashboard:
   ```bash
   streamlit run app/dashboard_app.py
   ```
2) Panel superior: muestra registros en BD, si existe modelo entrenado y el último análisis (%) con delta vs análisis previo.
3) Botones rápidos (ejecutan scripts internos):
   - Generar dataset: inserta ~10 000 registros simulados (limita a 20k totales) y crea tablas si faltan.
   - Entrenar modelo: entrena IsolationForest con los datos actuales y registra nuevas métricas.
   - Recalcular métricas: corre detección con el modelo entrenado y guarda métricas.
   - Reiniciar base de datos: trunca `registros` y `metricas`, y repuebla con 10 000 registros.
4) Filtros: rango de fechas, variable a graficar, estados Normal/Anómalo, muestreo opcional (máx. 5k filas) y límite de filas de la tabla.
5) Tabs de análisis:
   - Análisis general: KPIs de anomalías y gauge de salud.
   - Variables energéticas: series de tiempo e histogramas.
   - Comparativos: scatter matrix de variables.
   - Detección de anomalías: superpone predicciones del modelo.
   - Tabla de registros: vista y descarga (respeta límite de filas configurado).

## Flujo CLI alternativo
1) Crear tablas: `python -m scripts.seed_db`
2) Generar datos simulados: `python -m scripts.simulate_data`
3) Entrenar modelo: `python -m ml.train_model`
4) Calcular métricas: `python -m ml.detect_anomalies`
5) Ejecutar dashboard: `streamlit run app/dashboard_app.py`

## Comandos útiles
- Resetear tablas y regenerar 10k registros: `python -m scripts.reset_db`
- Regenerar solo datos (respeta máximo 20k registros): `python -m scripts.simulate_data`
- Limpiar acentos en columnas de texto de la BD: `python -m scripts.strip_accents_db`

## Notas
- La app funciona sin modelo inicial; usa los botones para generarlo si falta.
- La tasa de contaminación y ratio de anomalías en los scripts se eligen al azar en cada ejecución.
- Directorios `data/` y `models/` están ignorados en git por defecto.
