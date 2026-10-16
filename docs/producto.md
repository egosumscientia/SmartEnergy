# SmartEnergy by makeAutomatic

**SmartEnergy by makeAutomatic** es una plataforma de monitoreo y análisis del
consumo eléctrico concebida para instalaciones industriales, comerciales e
inmobiliarias. Su alcance comercial previsto abarca máquinas, motores, líneas de
producción, tableros y circuitos de esas instalaciones.

El funcionamiento previsto consiste en medir variables mediante analizadores de
energía conectados a una infraestructura de adquisición, normalizar y registrar
las mediciones, y ofrecer su visualización y análisis a través de servicios y una
interfaz web. Está orientado inicialmente al **monitoreo, registro, visualización
y análisis**, sin control activo de cargas eléctricas.

## Finalidad y capacidades previstas

El producto busca permitir que las organizaciones:

- Conozcan el consumo por equipo, circuito, área o proceso.
- Consulten voltajes, corrientes, potencias, energía y otras variables disponibles.
- Analicen históricos, demanda y comportamiento de las cargas.
- Identifiquen consumos elevados, desperdicios y oportunidades de eficiencia.
- Comparen períodos operativos y estimen costos energéticos.
- Tomen decisiones basadas en mediciones reales.

Estas capacidades definen la orientación del producto; no implican que estén
implementadas o validadas en su totalidad. El estado actual es un prototipo con
PostgreSQL, servicios de consulta, API FastAPI y cinco vistas web con ECharts.
FastAPI/ECharts es la única interfaz oficial tras retirar Streamlit en P04. Las funciones concretas
implementadas y sus límites se detallan a continuación.

La adquisición física, la trazabilidad multicanal, el análisis de demanda y consumo
acumulado por activo, las comparaciones entre períodos operativos y la estimación
de costos siguen pendientes de implementación y validación. La vista Comparativos
actual muestra relaciones entre variables; no acredita esas capacidades futuras.
El Acrel ADW210 es la referencia inicial aprobada para diseñar datos sintéticos y
el futuro contrato común de adquisición, según la [arquitectura](arquitectura.md).
No existe todavía un adaptador Modbus ni validación con instrumentos reales.
Los datos de demostración son sintéticos y no acreditan exactitud metrológica,
predictiva, eléctrica ni industrial. El resto del roadmap funcional, las reglas
industriales y el despliegue productivo requieren aprobación posterior.

## Consulta

La interfaz web mantiene cinco vistas de consulta:

1. Análisis general.
2. Variables energéticas.
3. Comparativos.
4. Detección de anomalías.
5. Tabla y descarga CSV.

La interfaz web se sirve desde la misma aplicación FastAPI que ofrece la API de
monitoreo. ECharts se carga desde un bundle local con licencia y aviso incluidos.
Las agregaciones de negocio se producen en backend; JavaScript sólo presenta los
datos y permite ajustar filtros.

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

La API diferencia fallo de consulta primaria (`503`) de una población realmente
vacía (`200` con estado vacío). Un fallo al consultar métricas persistidas es
opcional y no bloquea las mediciones.

## Operaciones y límites

La interfaz no permite administrar la base, generar datos, entrenar ni guardar
métricas. Esas tareas son entradas independientes que requieren invocación y
permisos explícitos. Las etiquetas de anomalía y métricas del prototipo no equivalen
a un diagnóstico industrial ni a una validación con instrumentos reales.
