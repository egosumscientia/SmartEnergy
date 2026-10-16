# Pruebas y evidencia

## Suite automatizada

Con las dependencias ya instaladas en `.venv`, ejecuta desde la raíz:

```bash
.venv/bin/python -B -m unittest discover -v
```

## Evidencia P04

Resultado final: `.venv/bin/python -B -m unittest discover -v` — **62 pruebas
aprobadas en 12,795 s, sin omisiones**. `pip check`, `node --check` para los tres
módulos JavaScript y `git diff --check` pasaron. Se comprobaron 43 archivos Python
mediante AST, sin importaciones activas de Streamlit, y el grafo de 31 requisitos
resueltos no contiene Streamlit. Los enlaces locales y las rutas canónicas de los
comandos documentados existen. `uvicorn --help` funciona y las importaciones/rutas
FastAPI se prueban sin efectos. Los cuatro comandos shell de Compose pasan `sh -n`
sin ejecutarse. No hay secretos, datos ni modelos incluidos en los cambios.

La suite final se ejecuta después de retirar Streamlit, autorefresh, Altair, pydeck,
Plotly, Matplotlib y los restantes requisitos exclusivos antiguos del `.venv`,
sin instalar dependencias. Los casos de HTTP y navegador requieren ejecución
fuera del sandbox que bloquea esas capacidades; no utilizan PostgreSQL ni Docker.

La cobertura AppTest se sustituyó por `tests.test_frontend_monitoring` y casos
adicionales en `tests.test_api_monitoring`, conservando las reglas de negocio:

- Chrome real a 1365 y 390 px: cinco vistas navegables, ECharts local, cuatro
  combinaciones datos/modelo, gráficas y mensajes vacíos, tabla, filtros de fechas,
  variable, muestreo y suavizado/downsampling, URL CSV y errores de consulta.
- HTTP en memoria: modelo inválido, variables y fechas nulas, denominadores tras
  agregación y rechazo de POST/PUT/PATCH/DELETE sin invocar administración.
- Se conservan los filtros, límites de muestreo, orden y población CSV de la antigua
  prueba de seguridad en `tests.test_monitoring_safety`.
- Las ocho entradas administrativas canónicas siguen cubiertas con mocks; no se
  ejecuta mantenimiento, entrenamiento ni detección sobre recursos reales.

La revisión visual del agente usa capturas de Chrome con fixtures temporales en
memoria. No sustituye la aprobación manual de cierre H03 por el responsable.
Dockerfile/Compose se revisan sólo como texto; su ejecución no se verifica.

La regresión `tests.test_frontend_table` ejecuta la interfaz real en Chrome headless
con respuestas de la aplicación bajo mocks, sin servidor ni base real. Comprueba
la tabla sin registros, con datos, tras aplicar filtros sin resultados y al
restablecerlos; verifica filas, mensaje visible/oculto y ausencia de errores
JavaScript. Falló antes de corregir `populateTable` y pasó después. La revisión
visual en Chrome confirmó la tabla con filas y el mensaje sin resultados.

Para ejecutar sólo esa regresión:

```bash
.venv/bin/python -B -m unittest -v tests.test_frontend_table
```

Requiere `google-chrome` o `chromium` ya disponible en PATH; si falta, unittest
marca esa prueba como omitida y no acredita la verificación de navegador.
No instala dependencias y utiliza un perfil y HTML temporales aislados.

Los casos se ejecutan con `unittest`; fixtures, mocks y archivos temporales evitan
acceder a PostgreSQL, escribir registros o publicar artefactos ML. Las pruebas de
navegador usan Chrome con respuestas de aplicación simuladas.

Para seleccionar componentes:

```bash
.venv/bin/python -B -m unittest -v \
  tests.test_api_monitoring tests.test_measurements tests.test_anomalies tests.test_monitoring \
  tests.test_model_jobs tests.test_models tests.test_model_artifact
```

## Contratos cubiertos

- Nombres, columnas, tipos, claves y metadatos ORM.
- Filtros, orden, muestreo, valores ausentes, suavizado y agregaciones.
- Predicciones, denominadores, métricas, umbrales y parámetros de entrenamiento.
- Independencia de la aplicación de monitoreo respecto de Streamlit.
- Creación diferida/reutilización/cierre de recursos e importación sin efectos.
- Argumentos y efectos aislados de entradas canónicas batch, continuas y ML.
- Publicación atómica y bloqueo compartido/exclusivo del modelo.
- Cinco pestañas, cuatro combinaciones de disponibilidad de datos/modelo, filtros
  vacíos y ausencia de acciones administrativas en la web.
- Contrato HTTP, validación, errores 503/422, CSV, sólo lectura, entrega de estáticos
  y cierre de ciclo de vida, con HTTPX ASGITransport y mocks.

## Cierre manual H03

El responsable confirmó expresamente que su revisión personal de SmartEnergy fue
satisfactoria y autorizó el cierre definitivo de H03 y P04, el commit y el push a
`origin/main`. La revisión se realizó en la demostración FastAPI/Uvicorn disponible
en `http://127.0.0.1:8000/`, con 2.016 registros sintéticos exclusivamente en memoria
y sin modelo Isolation Forest. La demostración se mantiene funcionando.

Antes de esa aprobación pasaron 17 pruebas focalizadas de interfaz y API. La
comprobación HTTP de la demostración confirmó carga de la marca y ECharts local,
2.016 registros disponibles, filtro sin resultados, CSV con 2.016 filas y rechazo
con `405` de POST/PUT/PATCH/DELETE. Las pruebas de Chrome verificaron las cinco
vistas, filtros, estados vacíos y ausencia de errores JavaScript relevantes.
Esta evidencia automatizada complementa la revisión personal confirmada; no la
sustituye. El resultado de la suite completa de cierre se registra en Evidencia P04.

No se modificaron PostgreSQL, datos persistentes, artefactos ML ni el laboratorio
Docker. No se ejecutaron migraciones ni entrenamiento. La aprobación de esta UI
no acredita validación industrial, adquisición Modbus ni coherencia multicanal del
simulador con el ADW210. Su adaptación futura y la migración de automatizaciones
externas no inventariadas conservan sus pendientes documentados.

## Revisión manual posterior a P03

Tras publicar P03 (`02e701b`), el responsable de la revisión confirmó una
inspección visual y funcional satisfactoria en Chrome con 2.016 registros
sintéticos exclusivamente en memoria:

- Se inspeccionaron las cinco pestañas.
- Las gráficas de evolución, histogramas y comparativos se visualizaron correctamente.
- La tabla presentó los registros sintéticos.
- La ausencia de Isolation Forest se comunicó como modelo no disponible; las
  demás vistas permanecieron operativas sin modelo.

La demostración no utilizó PostgreSQL ni mediciones industriales reales y no
entrenó modelos. Esta evidencia manual no acredita exactitud industrial ni
validación del modelo. Como ajuste posterior, el indicador pasa a denominarse
«Registros disponibles», válido para fuentes reales o sintéticas, sin cambiar
el valor ni los cálculos del contador.

## Límites de evidencia

Las pruebas de API usan mocks y no demuestran conectividad o permisos contra una
instancia PostgreSQL real ni validez industrial. La revisión manual en navegador
usa respuestas de prueba y no accede a la base real. P03 no ejecuta Docker/Compose,
migraciones, reset, generación persistente, entrenamiento real ni detección
persistente. Dockerfile y Compose sólo se revisan estáticamente. No se presenta la
simulación como evidencia de funcionamiento con instrumentos.
