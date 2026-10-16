# Pruebas y evidencia

## Suite automatizada

Con las dependencias ya instaladas en `.venv`, ejecuta desde la raíz:

```bash
.venv/bin/python -B -m unittest discover -v
```

En el cierre de P03, `.venv/bin/python -B -m unittest discover -v` pasó:
**68 pruebas aprobadas** en 10,808 s, sin omisiones. Se ejecutó fuera del sandbox
porque sus restricciones bloquean Chrome y la ejecución HTTP en memoria.
`pip check` no detectó dependencias rotas; `node --check` pasó para los tres
módulos JavaScript y `git diff --check` pasó.

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
renderizado usan Streamlit AppTest con las fronteras de aplicación simuladas.

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
- Delegación de wrappers y argumentos de entradas batch, continuas y ML.
- Publicación atómica y bloqueo compartido/exclusivo del modelo.
- Cinco pestañas, cuatro combinaciones de disponibilidad de datos/modelo, filtros
  vacíos y controles administrativos bloqueados.
- Contrato HTTP, validación, errores 503/422, CSV, sólo lectura, entrega de estáticos
  y cierre de ciclo de vida, con HTTPX ASGITransport y mocks.

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
