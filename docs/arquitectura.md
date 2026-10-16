# Arquitectura implementada

SmartEnergy organiza el backend en el paquete raíz `smartenergy/`. Streamlit sigue
disponible y coexiste con una API FastAPI de monitoreo y un frontend web local. Esta
etapa no retira Streamlit ni introduce `domain/` o microservicios.

`AGENTS.md` y la [guía de ingeniería](./guia-de-ingenieria.md) referencian esta
estructura; el [roadmap](./roadmap-tecnico.md) registra el avance por prompt.
El contrato y su implementación están documentados en
[contrato HTTP](./contrato-http.md). La API es de sólo lectura y entrega los
recursos estáticos locales de la interfaz.

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
| `smartenergy/interfaces/api/` | Adaptación HTTP de sólo lectura, validación, errores y entrega de frontend |
| `frontend/` | Cinco vistas de consulta HTML/CSS/JavaScript con ECharts local |
| `app/dashboard_app.py` | Interacción, estado visual, gráficas y descargas Streamlit |

La presentación consume aplicación y análisis; no contiene SQL ni administra
sesiones. Aplicación coordina infraestructura y análisis. Análisis no importa UI,
SQLAlchemy ni persistencia. Las operaciones no se ejecutan desde el monitoreo.
La API reutiliza los servicios de monitoreo y transformaciones existentes. El
frontend sólo representa la respuesta: histogramas y comparaciones agregadas se
calculan en backend. Ambos caminos de presentación permanecen disponibles; P04,
que podría sustituir Streamlit, sigue pendiente.

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

## Decisión de adquisición: referencia ADW210 y contrato común previsto

En el cierre de P03 se aprueba el **Acrel ADW210 como primer analizador eléctrico
de referencia para diseñar los datos sintéticos y el futuro contrato de adquisición
de mediciones**. Admite configuraciones de hasta cuatro circuitos trifásicos o doce
monofásicos y comunicación Modbus RTU mediante RS-485, según la
[ficha del fabricante](https://www.acrelpower.com/adw210-ac-multi-channels-iot-smart-energy-meter-product/).
Es una referencia inicial, no una dependencia estructural de SmartEnergy.

El flujo conceptual previsto es:

```text
Fuente sintética o analizador físico
  → Adaptador de adquisición y normalización
  → Modelo común de mediciones
  → Servicios de aplicación y persistencia
  → API HTTP
  → Interfaz web de monitoreo y análisis
```

Simuladores y futuros adaptadores Modbus deberán entregar la misma representación
común, independiente del fabricante y del origen. El contrato deberá identificar
instalación, dispositivo, circuito, fase (o alcance agregado explícito), variable,
valor, unidad y marca de tiempo. Deberá distinguir la fuente sintética o real y
los valores calculados o estimados, conservando su procedencia y método de cálculo.
La semántica temporal, calidad y ausencia de valores deberá definirse expresamente,
sin inventar ceros ni atribuir una medición agregada a cada circuito o fase.

El catálogo previsto contempla tensión, corriente, potencia activa, reactiva y
aparente, energía activa y reactiva, factor de potencia, frecuencia y otras
variables que la configuración efectiva exponga. **No se supone que todas estén
disponibles individualmente por circuito.** Antes de implementar deberá contrastarse
cada variable y su alcance con el mapa de registros Modbus, el manual y las
capacidades reales de la configuración elegida. Este documento no establece
registros, direcciones, escalas ni especificaciones eléctricas adicionales.

Los datos sintéticos deberán conservar relaciones eléctricas razonables de acuerdo
con la topología y las hipótesis declaradas, estados operativos, acumulación de
energía y evolución temporal. Se deberán cubrir funcionamiento normal, variación
de carga, arranque y parada de motores, inactividad, desbalance y anomalías, con
criterios verificables y sin equiparar simulación con validación industrial.

Esta decisión es documental y arquitectónica: el esquema `Registro` y el generador
actual conservan su contrato histórico y no implementan aún este modelo multicanal.
No se modifica PostgreSQL, no se incorpora un driver Modbus y no se sustituye el
generador en P03. La adaptación pendiente y sus criterios están en el
[roadmap técnico](roadmap-tecnico.md#i-pendiente-técnico-adquisición-y-simulador).
