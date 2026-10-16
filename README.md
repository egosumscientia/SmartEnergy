# SmartEnergy

Prototipo de monitoreo de mediciones energéticas con PostgreSQL y Streamlit. Los
datos de ejemplo son simulados: el producto no está validado con instrumentos ni
representa una plataforma industrial lista para operación.

## Documentación

- [Arquitectura implementada](docs/arquitectura.md)
- [Desarrollo y comandos](docs/desarrollo.md)
- [Alcance del producto](docs/producto.md)
- [Pruebas y evidencia](docs/pruebas.md)
- [Roadmap técnico](docs/roadmap-tecnico.md)
- [Guía de ingeniería](docs/guia-de-ingenieria.md)

## Backend y presentación

- `smartenergy/`: configuración, aplicación, análisis, infraestructura y operaciones.
- `app/dashboard_app.py`: las cinco vistas Streamlit de consulta.
- `app/dashboard_data.py`, `ml/` y `scripts/`: wrappers temporales de compatibilidad;
  las implementaciones canónicas están en `smartenergy/`.
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

Para abrir la interfaz de consulta con una base configurada:

```bash
streamlit run app/dashboard_app.py
```

El comando no inicializa ni pobla la base. Comprueba antes que `DATABASE_URL`
identifique el destino y rol autorizados para consulta.

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
sin verificar el destino y la autorización. Los comandos históricos `python -m
scripts...` y `python -m ml...` siguen disponibles como wrappers temporales.

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
