# Contrato HTTP y de presentación (H00)

Este documento concreta el contrato de consulta de la API y la interfaz de P03 a
partir del dashboard y de `smartenergy/application/`. La API FastAPI y el frontend
se implementaron en H01–H02; Streamlit sigue disponible y no se retira hasta una
etapa posterior aprobada.

## Límites y comportamiento

- La API sólo ofrece monitoreo de lectura; no expone generación de datos,
  entrenamiento, detección persistente, mantenimiento ni escritura.
- La API adapta los servicios de aplicación y las funciones existentes de
  `smartenergy/analysis/`. No repite SQL, filtros, agregaciones ni fórmulas en los
  controladores o en JavaScript.
- La población filtrada original es la fuente de contadores y evaluación ML. La
  muestra o serie transformada sólo sirve para representar gráficas.
- La ausencia o indisponibilidad de ML no bloquea las vistas de mediciones.
- La distribución prevista es bajo el mismo origen HTTP para API y frontend. Hasta
  que se apruebe una política de publicación distinta, la ejecución se limita a
  acceso local; no se habilita exposición pública ni autenticación ficticia.

## Recursos HTTP

### `GET /api/v1/monitoring`

Devuelve en una respuesta el estado de monitoreo, métricas, datos para las cinco
vistas, filas recientes y datos de gráficas. Los valores filtrados alimentan
Análisis general, Variables energéticas, Comparativos, Detección de anomalías y
Tabla de registros; la respuesta no depende de la pestaña seleccionada.

Parámetros:

| Nombre | Tipo y regla | Valor por defecto / efecto |
|---|---|---|
| `date_from`, `date_to` | `YYYY-MM-DD`; deben enviarse juntos y `date_from <= date_to` | Sin parámetros, reproducir el intervalo inicial del dashboard: mínimo y máximo de fechas válidas; si no hay fechas válidas, no aplicar filtro temporal |
| `states` | Repetible; valores `Normal`, `Anomalo`, `Sin estado` | Omitido: incluir los estados normalizados disponibles. `states=` representa selección vacía y produce cero filas. Un estado desconocido es inválido |
| `variable` | `corriente`, `voltaje`, `potencia_activa`, `temperatura_motor` | `corriente` |
| `sample_plots` | Booleano | `true`; respetar `prepare_plot_records` y sus límites actuales |
| `smoothing_method` | `Ninguno`, `Media movil`, `Downsample por ventana` | `Ninguno` |
| `rolling_window` | Entero de 1 a 5000 | `60`; sólo aplica a `Media movil` |
| `resample_minutes` | Entero de 1 a 240 | `5`; sólo aplica a `Downsample por ventana` |
| `table_limit` | Entero de 100 a 20000 | `1000`; limita sólo las filas recientes devueltas, no la población ni el CSV |

Las fechas son inclusivas y se comparan con la fecha almacenada en cada registro,
sin conversión de zona horaria, igual que `filter_records`. Si ambos extremos se
envían explícitamente, sólo se incluyen registros con fecha válida dentro del
intervalo. Los filtros vacíos producen una respuesta satisfactoria con población
filtrada vacía, no un error.

### `GET /api/v1/monitoring/records.csv`

Usa los mismos filtros `date_from`, `date_to` y `states`. Descarga todos los
registros originales normalizados de la población filtrada, no sólo `table_limit`
ni los puntos de gráfica. Conserva el orden ascendente de timestamp e id y las
columnas, en este orden: `id`, `timestamp`, `corriente`, `voltaje`,
`potencia_activa`, `temperatura_motor`, `estado`, `label_anomalia`. La respuesta
es `text/csv; charset=utf-8`, sin columna de índice. El tamaño potencialmente grande
del CSV es una limitación conocida que deberá mantenerse visible; no se reduce
silenciosamente la población existente.

## Esquema JSON de monitoreo

Los nombres de campos JSON usan `snake_case`; las cadenas visibles de la interfaz
pueden estar localizadas. Los campos numéricos sin valor se serializan como `null`,
nunca como `NaN`, infinito ni cero sustituto.

```json
{
  "database": "available",
  "data": {
    "status": "available",
    "record_count": 1,
    "filtered_count": 1,
    "has_filtered_data": true,
    "date_range": {"from_date": "2026-10-15", "to_date": "2026-10-15"},
    "available_states": ["Normal", "Anomalo"]
  },
  "model": {
    "status": "available",
    "modified_at": "2026-10-15T12:00:00",
    "evaluation": {
      "status": "available",
      "total": 1,
      "detected": 1,
      "true_positives": 1,
      "precision": 100.0,
      "recall": 100.0,
      "coverage": 100.0,
      "state": "Normal"
    }
  },
  "persisted_metrics": {
    "status": "available",
    "latest": {
      "total": 300,
      "anomalies": 14,
      "percentage": 4.67,
      "state": "Precaucion",
      "recorded_at": "2026-10-15T12:00:00"
    },
    "previous_percentage": 3.9
  },
  "population": {
    "anomalies_detected": 1,
    "coverage_percent": 100.0
  },
  "chart": {
    "variable": "corriente",
    "unit": null,
    "source_count": 1,
    "point_count": 1,
    "histograms": {
      "corriente": [
        {
          "state": "Anomalo",
          "bins": [{"lower": 24.0, "upper": 26.0, "count": 1}]
        }
      ]
    },
    "comparison_histograms": {
      "corriente": [
        {
          "state": "Anomalo",
          "bins": [{"lower": 24.0, "upper": 26.0, "count": 1}]
        }
      ]
    },
    "prediction_status": "available",
    "points": [
      {
        "timestamp": "2026-10-15T10:00:00",
        "values": {
          "corriente": 25.0,
          "voltaje": 220.0,
          "potencia_activa": 5.0,
          "temperatura_motor": 45.0
        },
        "state": "Anomalo",
        "label_anomaly": 1,
        "predicted_anomaly": 1
      }
    ]
  },
  "table": {
    "limit": 1000,
    "returned_count": 1,
    "order": "timestamp_desc_id_desc",
    "records": [
      {
        "id": 1,
        "timestamp": "2026-10-15T10:00:00",
        "corriente": 25.0,
        "voltaje": 220.0,
        "potencia_activa": 5.0,
        "temperatura_motor": 45.0,
        "estado": "Anomalo",
        "label_anomalia": 1
      }
    ]
  }
}
```

El ejemplo ilustra la forma (las listas de bins muestran un elemento representativo),
no datos reales ni unidades aprobadas. `unit` es `null`
hasta que exista una definición de unidades autorizada. El objeto `evaluation` se
calcula mediante `evaluate_model` sobre todos los registros originales filtrados;
`population` conserva esos denominadores. `predicted_anomaly` corresponde sólo a
puntos representados y puede ser `null` cuando ML no esté disponible o no pueda
evaluar esos puntos. No sustituye los contadores de la población original.

`chart.points` procede de `prepare_plot_records` con los parámetros recibidos,
respetando el muestreo, downsample y suavizado existentes. `source_count` identifica
la población original filtrada y `point_count` la serie final representada. Cada
punto contiene las cuatro variables para permitir la vista Comparativos; la variable
seleccionada es `values[variable]`. No se calculan estadísticas de la población a
partir de `chart.points`. `chart.histograms` contiene los conteos de 40 bins con
bordes comunes por estado y variable, calculados en el backend sobre la serie
representada para la vista de evolución; `chart.comparison_histograms` usa sólo
puntos completos, como la matriz comparativa del dashboard. JavaScript sólo los
dibuja. `data.available_states` lista las opciones de estado admitidas, incluyendo
`"Sin estado"` si existe en los registros; así los filtros siguen disponibles
aunque el filtro actual no produzca coincidencias. Las filas de `table.records` son
las más recientes primero mediante
`recent_records`.

Estados de disponibilidad:

| Situación | Respuesta y representación |
|---|---|
| Sin registros y modelo ausente | `200`; `data.status: "empty"`, conteos cero, sin serie/filas; `model.status: "absent"` y evaluación `not_available` |
| Registros disponibles y modelo ausente | `200`; datos y gráficas disponibles; `model.status: "absent"` y ML `not_available` |
| Sin registros y modelo disponible | `200`; `data.status: "empty"`, `model.status: "available"` y evaluación `not_available` por población vacía |
| Registros disponibles y modelo disponible | `200`; `model.status: "available"` y evaluación calculada sólo si la población filtrada es no vacía y las características requeridas están completas |
| Artefacto presente pero inválido/no cargable o evaluación no posible | `200`; `model.status: "unavailable"` o evaluación `not_available`, con código público estable, sin excepción, ruta ni traza |
| Filtros válidos sin coincidencias | `200`; `data.status: "empty"` para la población filtrada, sin datos ficticios; el estado general de datos puede seguir indicando registros disponibles |
| Consulta primaria no disponible/fallida | `503` genérico; no aparentar una colección vacía |
| Fallo opcional al leer métricas persistidas | Datos de monitoreo siguen disponibles; `persisted_metrics.status: "unavailable"` y valores opcionales `null` |

Un timestamp se serializa en ISO 8601 conservando su offset cuando la fuente lo
contenga. Si la fuente es naive, se devuelve sin offset y no se presume UTC ni se
convierte a la zona del navegador. Las métricas históricas y fechas de modificación
no adquieren semántica de zona horaria que el backend no proporciona.

## Errores

Los errores usan una forma estable y no incluyen SQL, nombres de host, rutas
locales, credenciales, trazas ni mensajes de excepción:

```json
{
  "error": {
    "code": "validation_error",
    "message": "Uno o más parámetros no son válidos.",
    "details": [{"field": "table_limit", "reason": "out_of_range"}]
  }
}
```

- `422 Unprocessable Entity`: tipos, enum, límites, rango de fechas o combinación
  de parámetros inválidos. `details` sólo identifica campo y razón segura.
- `503 Service Unavailable`: consulta primaria no disponible; mensaje genérico.
- `500 Internal Server Error`: fallo no previsto; mensaje genérico, sin detalles
  internos. No convertir una excepción inesperada en estado vacío.

## Correspondencia funcional e implementación

El contrato conserva las cinco vistas, los cuatro escenarios de disponibilidad,
filtros, muestreo, orden, poblaciones y CSV de Streamlit. La API reutiliza
`smartenergy.application.monitoring`, `smartenergy.analysis.measurements` y
`smartenergy.analysis.anomalies`; no agrega rutas de escritura ni fórmulas de
negocio en JavaScript. Los errores de consulta primaria se distinguen de una
colección vacía en la capa de monitoreo y se exponen como `503`.

La implementación está en `smartenergy/interfaces/api/`; `frontend/` contiene la
interfaz estática que la misma aplicación sirve. Las dependencias directas fijadas
son FastAPI 0.143.2, Uvicorn 0.54.0 y HTTPX 0.28.1. Las pruebas HTTP usan
`httpx.ASGITransport` y mocks, sin conectarse a PostgreSQL.

ECharts 6.1.0 procede del paquete oficial `echarts` de npm; se verificó la
integridad SHA-512 publicada antes de incorporar el bundle local. Se incluyen
`LICENSE` y `NOTICE` de Apache-2.0 en `frontend/vendor/`; no se usa CDN.
Streamlit permanece disponible y la sustitución/retirada corresponde a P04, que
sigue pendiente.
