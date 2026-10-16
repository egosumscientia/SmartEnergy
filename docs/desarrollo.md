# Desarrollo

El desarrollo cotidiano está orientado a Debian nativo y al `.venv` del repositorio.
Las dependencias directas están fijadas en `requirements.txt`; FastAPI, Uvicorn y
HTTPX se añadieron para P03. ECharts 6.1.0 se distribuye desde
`frontend/vendor/`, junto con su licencia y aviso Apache, sin CDN.

## Configuración

1. Prepara el entorno virtual y las dependencias conforme al entorno autorizado.
2. Copia `config/env.example` a `.env` en la raíz.
3. Configura `DATABASE_URL` para la base y el rol autorizados. No incluyas
   credenciales en Git ni en reportes.
4. El archivo local `.env` está excluido del control de versiones.

La configuración se carga explícitamente desde la raíz del repositorio, no desde el
directorio de trabajo. `wait_for_db` conserva su contrato previo y sólo lee las
variables del proceso; en particular, no carga `.env` por su cuenta.

## Dashboard

```bash
streamlit run app/dashboard_app.py
```

El dashboard es de consulta. No inicializa esquema ni genera datos, y mantiene
deshabilitadas las acciones administrativas también en sus manejadores. Necesita
una base existente y autorizada para lectura; sus estados vacíos no se rellenan con
mediciones ficticias. La carga del modelo es opcional.

## Interfaz web de consulta

```bash
.venv/bin/uvicorn smartenergy.interfaces.api.main:app --host 127.0.0.1 --port 8000
```

Abre `http://127.0.0.1:8000/`. La API lee la base sólo al recibir solicitudes y
expone únicamente rutas GET; no inicializa esquema, escribe registros, entrena ni
persiste detecciones. Mantén el host en loopback: no hay autenticación ni
autorización para exposición en red. Antes de consultar, verifica `DATABASE_URL`,
destino y rol de sólo lectura. La ausencia de la base se informa como error de
servicio, no como datos vacíos. Streamlit permanece operativo durante P03.

## Operaciones canónicas

| Comando | Efecto explícito |
|---|---|
| `python -m smartenergy.operations.seed_db` | `create_all` para los modelos ORM existentes |
| `python -m smartenergy.operations.wait_for_db` | Espera una consulta de disponibilidad |
| `python -m smartenergy.operations.simulate_data` | Inserta un lote y aplica la retención existente |
| `python -m smartenergy.operations.stream_realtime` | Inserta lotes continuamente |
| `python -m smartenergy.operations.reset_db` | Trunca tablas y delega en el generador batch |
| `python -m smartenergy.operations.train_model` | Lee registros y publica el bundle ML |
| `python -m smartenergy.operations.detect_anomalies` | Evalúa registros y persiste una métrica |
| `python -m smartenergy.operations.auto_train_loop --interval 300` | Repite entrenamiento y detección |

Las operaciones de escritura y mantenimiento requieren invocación deliberada.
Verifica destino, rol, permisos y alcance antes de ejecutarlas. No ejecutes reset,
generadores, entrenamiento ni detección contra recursos protegidos como parte de
pruebas rutinarias.

Los módulos históricos `scripts.*` y `ml.*` permanecen como wrappers para consumidores
vigentes. Las invocaciones Docker conservan esas rutas; Compose no se modifica en P02.

## Integración Docker

`Dockerfile` y `docker-compose.yml` se mantienen como referencia existente. En P02
sólo se permite su inspección estática: no ejecutar Docker o Compose, ni modificar
contenedores, montajes, redes, volúmenes o política de arranque.
