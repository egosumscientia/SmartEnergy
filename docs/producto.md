# Alcance actual del producto

La firma oficial del producto es **SmartEnergy by makeAutomatic**. Su incorporación
a la interfaz corresponde a P03; P02 no modifica la presentación.

SmartEnergy es un prototipo de monitoreo de mediciones energéticas. Los datos de
demostración son sintéticos y no acreditan exactitud metrológica, predictiva,
eléctrica ni industrial. La adquisición desde instrumentos, el roadmap funcional
y la arquitectura industrial siguen pendientes de aprobación.

## Consulta

El dashboard Streamlit conserva cinco vistas:

1. Análisis general.
2. Variables energéticas.
3. Comparativos.
4. Detección de anomalías.
5. Tabla y descarga CSV.

Filtros, muestras, agregaciones y suavizado trabajan sobre los registros consultados.
Los indicadores de población original no se sustituyen por el conteo de puntos
agregados o representados. Los datos ausentes y filtros vacíos muestran estados
vacíos; no se inventan mediciones ni predicciones.

## Disponibilidad independiente

Las cinco pestañas permanecen disponibles cuando no hay datos o no hay modelo. Se
conservan los cuatro escenarios de monitoreo:

| Datos | Modelo | Comportamiento |
|---|---|---|
| Ausentes | Ausente | Vistas vacías informativas |
| Disponibles | Ausente | Análisis de mediciones sin predicciones |
| Ausentes | Disponible | Vistas vacías; ML no bloquea la interfaz |
| Disponibles | Disponible | Mediciones y evaluación ML opcional |

Un modelo inválido, no cargable o incompatible se informa como indisponibilidad de
ML; no oculta las demás vistas. Las métricas persistidas son un historial opcional y
distinto de la evaluación en memoria de la población filtrada.

## Operaciones y límites

La interfaz no permite administrar la base, generar datos, entrenar ni guardar
métricas. Esas tareas son entradas independientes que requieren invocación y
permisos explícitos. Las etiquetas de anomalía y métricas del prototipo no equivalen
a un diagnóstico industrial ni a una validación con instrumentos reales.
