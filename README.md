# SmartEnergy by makeAutomatic

Plataforma de monitoreo y análisis del consumo eléctrico concebida para instalaciones
industriales, comerciales e inmobiliarias. Su finalidad es registrar y analizar
mediciones de equipos y circuitos para conocer el consumo y orientar decisiones de
eficiencia energética. El alcance inicial es de consulta, sin control activo de cargas.

El repositorio contiene un prototipo con PostgreSQL y una interfaz web
FastAPI/ECharts local. Los ejemplos son sintéticos; la adquisición física y la
validación industrial están pendientes. El Acrel ADW210 es la referencia inicial
para el futuro contrato de mediciones, sin driver implementado. Consulta la
[definición del producto](docs/producto.md) y la [arquitectura](docs/arquitectura.md)
para distinguir capacidades previstas, implementadas y pendientes de validación.

## Documentación

- [Arquitectura implementada](docs/arquitectura.md)
- [Desarrollo y comandos](docs/desarrollo.md)
- [Alcance del producto](docs/producto.md)
- [Pruebas y evidencia](docs/pruebas.md)
- [Roadmap técnico](docs/roadmap-tecnico.md)
- [Contrato HTTP y de presentación](docs/contrato-http.md)
- [Guía de ingeniería](docs/guia-de-ingenieria.md)

## Backend y presentación

- `smartenergy/`: configuración, aplicación, análisis, infraestructura y operaciones.
- `smartenergy/interfaces/api/`: API HTTP de monitoreo de sólo lectura.
- `frontend/`: cinco vistas web servidas por la API; ECharts se distribuye localmente.
- `smartenergy/operations/`: entradas administrativas canónicas e independientes.
- Streamlit y los wrappers históricos de `app/`, `ml/` y `scripts/` se retiraron en P04.
- `tests/`: pruebas de contratos, disponibilidad, transformación, importación y UI.

ML es opcional para consultar mediciones. La ausencia o invalidez del modelo no
bloquea las demás vistas. El dashboard no habilita entrenamiento, detección
persistente ni mantenimiento.

## Configuración y ejecución local

El desarrollo se orienta al entorno nativo Debian y `.venv`. Consulta
[desarrollo](docs/desarrollo.md) antes de configurar una instancia. No se crean
tablas ni se generan datos al importar o abrir el dashboard.

La configuración de ejemplo está en [`config/env.example`](config/env.example).
Las credenciales se mantienen únicamente en un `.env` local, excluido de Git.

FastAPI/ECharts es la única interfaz oficial. Comprueba antes que `DATABASE_URL`
identifique el destino y rol autorizados para consulta. Para iniciarla localmente:

```bash
.venv/bin/uvicorn smartenergy.interfaces.api.main:app --host 127.0.0.1 --port 8000
```

La API consulta la base al recibir solicitudes; no crea tablas ni ejecuta
operaciones administrativas. No cambies el host a una dirección pública: esta
etapa no añade autenticación ni autoriza exponer el servicio en una red.

## Operaciones administrativas

Las operaciones se invocan por separado, explícitamente, con los módulos canónicos:

```text
python -m smartenergy.operations.seed_db
python -m smartenergy.operations.wait_for_db
python -m smartenergy.operations.simulate_data
python -m smartenergy.operations.stream_realtime [argumentos]
python -m smartenergy.operations.reset_db
python -m smartenergy.operations.train_model
python -m smartenergy.operations.detect_anomalies
python -m smartenergy.operations.auto_train_loop [--interval SEGUNDOS]
```

`reset_db` trunca tablas y repuebla; los generadores escriben registros; el
entrenamiento publica un artefacto y la detección persiste métricas. No los ejecutes
sin verificar el destino y la autorización. Utiliza exclusivamente las rutas
canónicas anteriores; los comandos históricos `scripts.*` y `ml.*` se retiraron.

Docker Compose se conserva como referencia de integración estática; esta etapa no
autoriza levantar, detener, reconstruir ni modificar el laboratorio protegido.

## Pruebas

Con el entorno de dependencias ya preparado, ejecuta la suite segura:

```bash
.venv/bin/python -B -m unittest discover -v
```

Las pruebas usan fixtures, mocks y temporales; no requieren PostgreSQL ni escriben
datos o modelos persistentes. Ver los límites y comandos focalizados en
[pruebas](docs/pruebas.md).
