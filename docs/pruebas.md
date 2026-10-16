# Pruebas y evidencia

## Suite automatizada

Con las dependencias ya instaladas en `.venv`, ejecuta desde la raíz:

```bash
.venv/bin/python -B -m unittest discover -v
```

Los casos se ejecutan con `unittest`; fixtures, mocks y archivos temporales evitan
acceder a PostgreSQL, escribir registros o publicar artefactos ML. Las pruebas de
renderizado usan Streamlit AppTest con las fronteras de aplicación simuladas.

Para seleccionar componentes:

```bash
.venv/bin/python -B -m unittest -v \
  tests.test_measurements tests.test_anomalies tests.test_monitoring \
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

## Límites de evidencia

Estas son pruebas unitarias y de interfaz aislada; no demuestran conectividad o
permisos contra una instancia PostgreSQL real ni validez industrial. P02 no ejecuta
Docker/Compose, migraciones, reset, generación persistente, entrenamiento real ni
detección persistente. Dockerfile y Compose sólo se revisan estáticamente. No se
presenta la simulación como evidencia de funcionamiento con instrumentos.
