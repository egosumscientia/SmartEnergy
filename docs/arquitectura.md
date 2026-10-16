# Arquitectura implementada

SmartEnergy organiza el backend en el paquete raíz `smartenergy/`. Streamlit sigue
siendo la presentación operativa de esta etapa. No se introducen `domain/`,
microservicios, API HTTP ni frontend adicional.

`AGENTS.md` y la [guía de ingeniería](./guia-de-ingenieria.md) referencian esta
estructura; el [roadmap](./roadmap-tecnico.md) registra el avance por prompt.

## Responsabilidades

| Módulo | Responsabilidad |
|---|---|
| `smartenergy/config.py` | Carga explícita de variables de entorno, `.env` del repositorio y rutas locales |
| `smartenergy/application/monitoring.py` | Datos y estado de consulta para las interfaces |
| `smartenergy/application/model_jobs.py` | Coordinación independiente de entrenamiento y detección persistente |
| `smartenergy/analysis/measurements.py` | Normalización, filtros, orden, muestreo, agregaciones y suavizado |
| `smartenergy/analysis/anomalies.py` | Predicción y evaluación compartida de anomalías |
| `smartenergy/analysis/training.py` | Ajuste del modelo a mediciones recibidas como entrada |
| `smartenergy/infrastructure/database.py` | Base ORM, motor y fábrica de sesiones con ciclo de vida explícito |
| `smartenergy/infrastructure/models.py` | Modelos ORM `Registro` y `Metrica` |
| `smartenergy/infrastructure/persistence.py` | Consultas y escrituras PostgreSQL |
| `smartenergy/infrastructure/model_artifact.py` | Carga bloqueada y publicación atómica del bundle |
| `smartenergy/operations/` | Entradas explícitas de esquema, datos, mantenimiento y procesos ML |
| `app/dashboard_app.py` | Interacción, estado visual, gráficas y descargas Streamlit |

La presentación consume aplicación y análisis; no contiene SQL ni administra
sesiones. Aplicación coordina infraestructura y análisis. Análisis no importa UI,
SQLAlchemy ni persistencia. Las operaciones no se ejecutan desde el monitoreo.

## Persistencia y recursos

`Registro` y `Metrica` comparten una única `Base` declarativa. Se conservan nombres,
columnas, tipos y valores por defecto; esta reorganización no incluye migraciones ni
cambios de esquema.

La importación de módulos no carga `.env`, crea un motor, conecta, crea tablas,
carga modelos ni ejecuta tareas. `load_settings()` lee la configuración cuando una
entrada la necesita. El motor se crea bajo demanda y se reutiliza en el proceso;
`dispose_engine()` lo cierra explícitamente y también está registrado para el cierre
del proceso. Las sesiones de consulta y escritura se cierran en su ámbito de uso.

El bundle conserva `model`, `scaler` y `features`. Lectores y escritores mantienen
el mismo archivo de bloqueo; la publicación sigue escribiendo un archivo temporal y
reemplazando atómicamente el artefacto.

## Compatibilidad temporal

- `app/dashboard_data.py` delega las funciones históricas de transformación a
  `smartenergy.analysis.measurements`.
- `ml/train_model.py` y `ml/detect_anomalies.py` delegan en sus entradas canónicas.
- Los seis módulos de `scripts/` conservan los comandos `python -m scripts...` y
  delegan en `smartenergy.operations`.
- El bucle ML conserva `--interval`; el generador continuo conserva
  `--interval`, `--batch-size`, `--anomaly-ratio` y `--max-registros`.

Estos wrappers no contienen implementaciones paralelas y no ejecutan operaciones al
importarse. Su retiro requiere comprobar consumidores externos, actualizar comandos,
documentación y pruebas, y disponer de una vía de reversión; corresponde a una etapa
posterior autorizada.
