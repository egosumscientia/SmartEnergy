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

## Interfaz web de consulta

```bash
.venv/bin/uvicorn smartenergy.interfaces.api.main:app --host 127.0.0.1 --port 8000
```

Abre `http://127.0.0.1:8000/`. La API lee la base sólo al recibir solicitudes y
expone únicamente rutas GET; no inicializa esquema, escribe registros, entrena ni
persiste detecciones. Mantén el host en loopback: no hay autenticación ni
autorización para exposición en red. Antes de consultar, verifica `DATABASE_URL`,
destino y rol de sólo lectura. La ausencia de la base se informa como error de
servicio, no como datos vacíos. FastAPI/ECharts es la única presentación oficial;
Streamlit y sus dependencias exclusivas se retiraron de `requirements.txt` en P04.

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

Los módulos históricos `scripts.*` y `ml.*` se retiraron en P04. Migra cualquier
automatización externa a los comandos canónicos anteriores. Sus implementaciones
y argumentos permanecen intactos; las pruebas los invocan con efectos simulados.

## Integración Docker

`Dockerfile` y `docker-compose.yml` son referencia de integración. P04 actualiza
estáticamente sus comandos a FastAPI y operaciones canónicas; el puerto del monitor
se publica sólo en loopback. La escucha interna del contenedor permite esa
publicación local y no es el comando nativo. No se construyó ni ejecutó Docker o
Compose, ni se modificó el laboratorio, sus montajes, redes o volúmenes. La
configuración actualizada no está validada en ejecución Docker.
