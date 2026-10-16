# Contrato de reestructuración técnica de SmartEnergy

Proyecto de makeAutomatic. Este documento conserva las decisiones aprobadas y los
controles de ejecución; no acredita su implementación. Complementa
[AGENTS.md](../AGENTS.md) y la [guía de ingeniería](guia-de-ingenieria.md).
La identidad y finalidad del producto y la referencia inicial ADW210 se aprobaron
en el cierre documental de P03; ver [producto](producto.md) y
[arquitectura](arquitectura.md). La implementación de adquisición, las reglas
industriales y el resto del roadmap funcional requieren autorización posterior.

La política de [Rigor de ingeniería y eficiencia de agentes](../AGENTS.md#rigor-de-ingeniería-y-eficiencia-de-agentes)
de `AGENTS.md` es obligatoria para P01–P04 y cualquier intervención correctiva
autorizada: el rigor prevalece sobre el ahorro de créditos y se permite investigar
con la profundidad necesaria, sin ampliar el alcance ni relajar las protecciones
de recursos establecidas en este contrato.

## A. Objetivo

Reorganizar SmartEnergy mediante una arquitectura modular por responsabilidades y
sustituir posteriormente Streamlit por FastAPI y Apache ECharts. Preservar el
comportamiento existente, proteger recursos persistentes y evitar sobreingeniería.
Primero se reorganiza el backend manteniendo Streamlit; después se sustituye la
presentación sin otra reorganización general del backend.

## B. Decisiones arquitectónicas aprobadas

1. Arquitectura modular pragmática por responsabilidades; no arquitectura hexagonal estricta.
2. Paquete `smartenergy/` directamente en la raíz, sin `src/` ni empaquetado adicional innecesario.
3. Separar aplicación, análisis, infraestructura y operaciones; configuración explícita y presentación delimitada.
4. Conservar temporalmente Streamlit operativo durante P02/P03; retirado en P04.
5. Sustituirlo posteriormente por FastAPI, HTML, CSS, JavaScript y Apache ECharts.
6. ML es complementario: monitoreo y visualización no requieren un modelo ni métricas históricas.
7. Diferir `domain/` hasta aprobar entidades y reglas industriales; no crear un paquete vacío.
8. Mantener los modelos ORM en infraestructura; no duplicar `Registro` y `Metrica` como entidades independientes.
9. Permitir wrappers temporales justificados, delegando en una única implementación, con condiciones de retirada.
10. Mantener documentación estructurada, proporcional y sin duplicación de contratos.
11. Validar nativamente en Debian; revisar Docker estáticamente sin operar contenedores.
12. Proteger PostgreSQL, esquema, datos, artefactos ML, bloqueos y laboratorio Docker recuperado.
13. No introducir microservicios ni frameworks frontend adicionales, como React o Angular, sin justificación y aprobación.
14. No crear directorios vacíos, interfaces abstractas, capas o generalizaciones prematuras.

## C. Arquitectura por etapas

La primera etapa describe la arquitectura implementada por P02. El árbol refleja
los módulos y wrappers existentes al cierre de P02; se conserva como referencia
histórica. La segunda etapa se implementó en P03/P04, con la evidencia indicada
en H00–H03. El árbol actual excluye `app/`, `ml/` y `scripts/`.

### Primera etapa: backend modular y Streamlit temporal (P02)

```text
SmartEnergy/
├── AGENTS.md
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .gitignore
├── .dockerignore
├── config/env.example
├── smartenergy/
│   ├── __init__.py
│   ├── config.py
│   ├── application/
│   │   ├── __init__.py
│   │   ├── monitoring.py
│   │   └── model_jobs.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── measurements.py
│   │   ├── anomalies.py
│   │   └── training.py
│   ├── infrastructure/
│   │   ├── __init__.py
│   │   ├── database.py
│   │   ├── models.py
│   │   ├── persistence.py
│   │   └── model_artifact.py
│   └── operations/
│       ├── __init__.py
│       ├── seed_db.py
│       ├── wait_for_db.py
│       ├── simulate_data.py
│       ├── stream_realtime.py
│       ├── reset_db.py
│       ├── train_model.py
│       ├── detect_anomalies.py
│       └── auto_train_loop.py
├── app/
│   ├── __init__.py
│   ├── dashboard_app.py
│   └── dashboard_data.py     # wrapper temporal de transformaciones
├── ml/                         # wrappers temporales de las dos entradas ML
├── scripts/                    # wrappers temporales de las seis entradas existentes
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_simulate.py
│   ├── test_model_artifact.py
│   ├── test_dashboard_safety.py
│   ├── test_dashboard_rendering.py
│   ├── test_import_safety.py
│   ├── test_measurements.py
│   ├── test_anomalies.py
│   ├── test_monitoring.py
│   ├── test_model_jobs.py
│   ├── test_operation_entrypoints.py
│   └── test_infrastructure.py
└── docs/
    ├── guia-de-ingenieria.md
    ├── roadmap-tecnico.md
    ├── arquitectura.md
    ├── desarrollo.md
    ├── producto.md
    └── pruebas.md
```

`application/` coordina casos de uso y recursos, sin Streamlit ni SQL directo.
`analysis/` concentra transformaciones, evaluación y entrenamiento sobre entradas
explícitas, sin persistencia. `infrastructure/` contiene SQLAlchemy, consultas,
escrituras y almacenamiento seguro del artefacto. `operations/` son entradas técnicas
explícitas; no se invocan desde monitoreo. La presentación gestiona interacción y
renderizado, sin consultas SQL ni duplicación de cálculos.

Aplicación puede utilizar análisis e infraestructura; análisis no depende de UI ni
persistencia. Infraestructura no conoce la presentación. Evitar dependencias circulares,
repositorios abstractos y servicios que sólo reenvían llamadas sin responsabilidad.
Los nombres de pruebas expresan contratos que deberán cubrirse, no obligan a duplicar
casos existentes. `.env`, `.venv` y artefactos locales siguen fuera de Git.

### Segunda etapa: sustitución de presentación

Se conserva el backend anterior, documentación y pruebas de comportamiento, añadiendo:

```text
smartenergy/interfaces/
├── __init__.py
└── api/
    ├── __init__.py
    ├── main.py
    ├── routes.py
    └── schemas.py
frontend/
├── index.html
├── css/dashboard.css
├── js/
│   ├── api.js
│   ├── dashboard.js
│   └── charts.js
└── vendor/
    ├── echarts.min.js
    └── LICENSE
tests/
├── [pruebas de backend conservadas]
├── test_monitoring_safety.py
├── test_api_monitoring.py
└── test_api_anomalies.py
```

FastAPI adapta HTTP/JSON a la aplicación, sin trasladar SQL ni cálculos a las rutas.
El frontend presenta resultados, sin duplicar agregaciones ni ML en JavaScript.
El servicio HTTP puede servir los recursos estáticos bajo el mismo origen, evitando
otro servicio. La versión y distribución concreta de ECharts se verifican antes de H02.
Las pruebas de API y presentación reemplazarán la cobertura dependiente de Streamlit
antes de retirarla; no basta con eliminar sus pruebas.

`app/` y la dependencia Streamlit sólo se retiran tras equivalencia funcional aprobada
en H03. Los wrappers se retiran únicamente sin consumidores pendientes; no se crean
anticipadamente `interfaces/`, `frontend/` ni otros directorios de la segunda etapa.

### Documentación y compatibilidad

- `AGENTS.md`: instrucciones operativas; la guía desarrolla normas de ingeniería.
- Este roadmap: alcance, secuencia, estados y controles de la reestructuración.
- `arquitectura.md`: responsabilidades y dependencias efectivamente implementadas.
- `desarrollo.md`: entorno nativo, configuración y comandos vigentes.
- `producto.md`: alcance real y límites, diferenciando futuro no implementado.
- `pruebas.md`: ejecución, evidencia y límites de validación.
- `decisiones/`: decisiones técnicas y sus motivos, referenciadas sin copiar el roadmap.
- README: orientación y comandos reales, enlazando documentación especializada.

Los wrappers conservan comandos `python -m scripts...` y `python -m ml...` cuando
existan consumidores. Sólo delegan en la entrada canónica bajo ejecución explícita.
Su retirada exige: referencias internas y subprocesos migrados, documentación y
pruebas actualizadas, consumidores externos conocidos migrados, validación de
compatibilidad y reversión disponible. Revisar en B11 y H03; si alguno permanece,
registrar consumidor, impedimento y condición pendiente, sin perpetuarlo silenciosamente.

## D. Contrato definitivo de prompts

La reestructuración se ejecutará mediante **cuatro prompts principales**:

| Prompt | Objetivo | Tareas técnicas |
|---|---|---|
| P01 | Preparación, referencia verificable y seguridad de importaciones | B00–B01 |
| P02 | Reorganización modular del backend, validación y documentación | B02–B11 |
| P03 | Contrato HTTP, API FastAPI y frontend ECharts | H00–H02 |
| P04 | Sustitución definitiva de Streamlit y cierre | H03 |

1. Las 16 tareas técnicas son controles internos del roadmap.
2. No se requiere un prompt independiente para cada tarea.
3. Cada prompt principal puede comprender varias tareas técnicas.
4. Respetar las dependencias entre tareas.
5. Cada tarea debe cumplir sus criterios de aceptación antes de continuar con la siguiente.
6. No omitir validaciones para reducir el número de prompts.
7. Ante una regresión, detenerse y reportarla.
8. No comenzar automáticamente el siguiente prompt principal.
9. Cualquier división, intervención correctiva o ampliación requiere autorización explícita.
10. Cuatro es el número planificado de prompts de ejecución, no una garantía de ausencia de intervenciones correctivas.

**Esta intervención documental no es P01.** La autorización del contrato y su
commit/push no autoriza ejecutar tareas de reestructuración.

## E. Inventario completo de tareas técnicas

Los siguientes controles conservan la secuencia y los objetivos de la propuesta
previa. Antes de ejecutar cada prompt se contrastarán rutas y contratos con el código
real; cualquier detalle no recuperable o discrepancia se declara pendiente, sin
inventarlo como decisión aprobada. Los estados sólo cambian con evidencia registrada.
Para toda tarea rigen las restricciones de B, G y H. Ninguna permite actuar sobre el
laboratorio Docker, cambiar el esquema o datos persistentes ni alterar algoritmos ML.
Una condición de reversión exige detenerse y recuperar el último punto verificado
mediante cambios controlados, preservando trabajo ajeno; no autoriza resets destructivos.

### B00 — Referencia verificable

- **Objetivo y alcance:** caracterizar comportamiento, comandos y consumidores actuales; identificar cobertura y referencia reproducible, sin reorganizar producción.
- **Dependencias:** autorización de P01 y lectura del contrato.
- **Restricciones:** sólo caracterización necesaria con mocks o fixtures aisladas; no ejecutar operaciones reales sobre datos o modelos.
- **Criterios de aceptación:** casos críticos, comandos y límites de evidencia identificados; referencia verificable sin confundir resultados históricos y nuevos.
- **Validación requerida:** pruebas nativas pertinentes; contrastar las 18 pruebas unittest reportadas y las siete funciones de prueba de modelos/simulación que unittest no descubre automáticamente. No afirmar que 25 pruebas se ejecutaron sin evidencia. Verificar cinco pestañas y cuatro escenarios mediante pruebas aisladas y revisión funcional autorizada.
- **Condición de reversión:** prueba dependiente de recursos persistentes o referencia no reproducible; retirar sólo el cambio de caracterización propio y resolver antes de avanzar.
- **Evidencia P01 (2026-10-15):** antes de B01 pasaron las 18 pruebas unittest existentes y se ejecutaron explícitamente las siete funciones no descubiertas. La referencia incluye cinco pestañas, cuatro escenarios de datos/ML, estados vacíos, filtros y poblaciones, contratos ORM y publicación/bloqueo de artefactos. Tras B01 pasaron las 23 pruebas unittest, incluidas las nuevas pruebas aisladas.
- **Estado:** COMPLETADO.

### B01 — Entradas sin ejecución accidental

- **Objetivo y alcance:** encapsular cuerpos operativos de generación batch, reset, entrenamiento, detección y dashboard en funciones y entradas explícitas.
- **Dependencias:** B00.
- **Restricciones:** añadir guardas y encapsular sin cambiar secuencias, argumentos, fórmulas ni comportamiento de ejecución explícita.
- **Criterios de aceptación:** importar no ejecuta trabajos, entrenamiento, escrituras ni renderizado; comandos existentes conservados. La eliminación de configuración implícita se completa en B10.
- **Validación requerida:** mocks que detecten accesos operativos, publicación, subprocesos y ajuste del modelo; comprobar delegación de entradas sin ejecutar trabajos reales.
- **Condición de reversión:** efectos operativos al importar o ruptura de entradas; detenerse y volver al punto B00 verificado.
- **Evidencia P01 (2026-10-15):** importación aislada de los cinco módulos prioritarios y de las entradas de flujo continuo, bootstrap y espera de BD, con bloqueos para DDL/SQL, publicación ML, ajuste, subprocesos, espera y renderizado; no se crearon archivos locales ni se modificó la configuración global de Plotly. Pruebas simuladas confirmaron que las entradas batch, reset y ML conservan sus operaciones. AppTest confirmó la ejecución normal del dashboard.
- **Estado:** COMPLETADO.

### B02 — Paquete e infraestructura

- **Objetivo y alcance:** crear paquete raíz y trasladar configuración, base, ORM y almacenamiento del artefacto desde `core/`; actualizar consumidores y pruebas.
- **Dependencias:** B01.
- **Restricciones:** sin cambios de columnas, esquema, serialización, rutas de artefactos ni implementación duplicada.
- **Criterios de aceptación:** una única `Base` y metadatos ORM; mismos contratos de `Registro`, `Metrica`, modelo, scaler y características.
- **Validación requerida:** pruebas de modelos y artefactos, bloqueo compartido/exclusivo, publicación atómica y fallos; revisar referencias antiguas y rutas sin tocar el modelo original.
- **Condición de reversión:** metadatos duplicados, incompatibilidad del artefacto o pérdida del bloqueo; revertir traslados y referencias del control.
- **Estado:** COMPLETADO.

### B03 — Transformaciones de mediciones

- **Objetivo y alcance:** extraer `app/dashboard_data.py` y transformaciones reutilizables del dashboard a `analysis/measurements.py`, incluidos suavizado y agregación existentes.
- **Dependencias:** B02.
- **Restricciones:** extracción sin optimizar ni cambiar fórmulas, límites, orden, intervalos o denominadores.
- **Criterios de aceptación:** mismos filtros, nulos, muestreo y distinción entre registros originales y puntos representados.
- **Validación requerida:** fixtures de filtros vacíos, fechas y límites, conjuntos vacíos y agregaciones; comprobar valores por defecto de interfaz.
- **Condición de reversión:** diferencia de resultados o semántica; restaurar la extracción afectada antes de avanzar.
- **Estado:** COMPLETADO.

### B04 — Evaluación ML compartida

- **Objetivo y alcance:** concentrar predicción y fórmulas realmente equivalentes de dashboard y módulos ML en `analysis/anomalies.py`.
- **Dependencias:** B03.
- **Restricciones:** no cambiar etiquetas, umbrales, redondeos ni denominadores; conservar diferencias intencionales de normalización entre presentación y entrenamiento.
- **Criterios de aceptación:** resultados y significado iguales para cada población; sin fusionar cálculos que sólo parecen equivalentes.
- **Validación requerida:** modelo/scaler falsos, positivos y negativos, ausencia de anomalías e entradas inválidas; comparación con referencia B00.
- **Condición de reversión:** diferencias estadísticas o semánticas; retirar la consolidación incorrecta.
- **Estado:** COMPLETADO.

### B05 — Consulta independiente de presentación

- **Objetivo y alcance:** extraer SQL de `app/dashboard_app.py` a `infrastructure/persistence.py` y coordinación de consulta a `application/monitoring.py`.
- **Dependencias:** B03.
- **Restricciones:** funciones concretas sin repositorios abstractos ni clases de reenvío; sin alterar consultas, filtros o seguridad.
- **Criterios de aceptación:** presentación sin SQL ni propiedad de sesiones; contenido, orden, errores y estados vacíos preservados; monitoreo sin ML.
- **Validación requerida:** sesiones simuladas y resultados de consulta; AppTest con frontera de aplicación simulada para cinco pestañas, CSV y bloqueo administrativo.
- **Condición de reversión:** cambios de contenido, orden, vacíos o permisos; recuperar consulta y coordinación anteriores.
- **Estado:** COMPLETADO.

### B06 — Trabajos ML independientes

- **Objetivo y alcance:** separar cálculo de entrenamiento en `analysis/training.py`, coordinación en `application/model_jobs.py` y escritura de métricas en persistencia.
- **Dependencias:** B04 y B05.
- **Restricciones:** sin ajustar parámetros, formato, orden de características ni entrenar o detectar sobre recursos reales.
- **Criterios de aceptación:** cálculo recibe datos explícitos; coordinación publica el artefacto; detección persistente escribe una métrica por invocación, sin dependencia de UI ni supuesta atomicidad conjunta entre archivo y PostgreSQL.
- **Validación requerida:** mocks de ajuste, carga, publicación y escritura; comprobar resultados, orden y efectos únicos.
- **Condición de reversión:** efectos duplicados o contratos de modelo/métricas alterados; restaurar límites previos verificados.
- **Estado:** COMPLETADO.

### B07 — Operaciones de esquema y disponibilidad

- **Objetivo y alcance:** trasladar `seed_db` y `wait_for_db` a operaciones, con wrappers antiguos cuando se necesiten.
- **Dependencias:** B02.
- **Restricciones:** conservar argumentos y reintentos; inicialización explícita, sin DDL real durante la validación.
- **Criterios de aceptación:** entradas equivalentes y una sola implementación; importación sin inicialización operativa.
- **Validación requerida:** mocks de metadatos, conexión y espera; verificar delegación de ambos comandos.
- **Condición de reversión:** efectos al importar o ruptura de compatibilidad; recuperar entradas anteriores.
- **Estado:** COMPLETADO.

### B08 — Generación y mantenimiento

- **Objetivo y alcance:** trasladar batch, streaming y reset a operaciones y actualizar wrappers y pruebas de simulación.
- **Dependencias:** B01 y B02.
- **Restricciones:** no cambiar distribuciones, aleatoriedad, intervalos, retención, transacciones ni alcance del mantenimiento; no fusionar generadores distintos.
- **Criterios de aceptación:** mismas operaciones sólo mediante entradas explícitas; mantenimiento inaccesible desde monitoreo.
- **Validación requerida:** mocks finitos de generador aleatorio, sesiones y subprocesos; sin generar registros persistentes ni ejecutar limpieza.
- **Condición de reversión:** cambio de distribución, retención o efectos; recuperar las entradas y lógica anteriores.
- **Estado:** COMPLETADO.

### B09 — Entradas ML y bucle existente

- **Objetivo y alcance:** trasladar entradas train/detect/auto_train_loop a operaciones, conservando wrappers y actualizando referencias `python -m` internas.
- **Dependencias:** B06.
- **Restricciones:** no ejecutar entrenamiento, detección persistente ni bucles reales; no cambiar secuencias o argumentos.
- **Criterios de aceptación:** cada entrada delega una vez; monitoreo no invoca operaciones; bucle conserva comportamiento explícito.
- **Validación requerida:** invocaciones simuladas y un ciclo controlado con espera simulada e interrupción; revisar comandos de subprocesos.
- **Condición de reversión:** bucles inesperados, llamadas duplicadas o comandos rotos; restaurar entradas anteriores.
- **Estado:** COMPLETADO.

### B10 — Configuración y ciclo de vida explícitos

- **Objetivo y alcance:** hacer explícita la carga de configuración e inicialización de recursos en configuración, base, operaciones y entrada de presentación.
- **Dependencias:** B05, B06, B07, B08 y B09.
- **Restricciones:** sin nuevos formatos, servicios, precedencias de variables ni destinos; no revelar credenciales.
- **Criterios de aceptación:** importar no carga configuración del entorno, conecta, carga modelos ni escribe; motor reutilizado por proceso y cierre definido; rutas y configuración efectiva conservadas.
- **Validación requerida:** inicialización y recursos simulados; comprobaciones de importación sin efectos y de selección explícita de configuración.
- **Condición de reversión:** destino incorrecto, configuración incompatible o fugas de recursos; restaurar composición previa verificada.
- **Estado:** COMPLETADO.

### B11 — Cierre de primera etapa

- **Objetivo y alcance:** actualizar README, documentación, plantilla sin secretos y referencias estáticas de Docker; cerrar inventario de compatibilidad.
- **Dependencias:** B10 y aceptación de todos los controles B00–B09.
- **Restricciones:** actualizar rutas/comandos sin operar Docker ni cambiar topología, puertos, volúmenes o políticas operativas. No implementar API ni frontend nuevo.
- **Criterios de aceptación:** backend modular y Streamlit operativo; comandos canónicos y wrappers documentados; deuda y consumidores pendientes explícitos.
- **Validación requerida:** pruebas nativas pertinentes, cinco pestañas y cuatro escenarios, `git diff --check`, revisión textual de Dockerfile/Compose, contexto, rutas y comandos. Comprobación funcional nativa sólo dentro de la autorización vigente.
- **Condición de reversión:** referencias rotas o regresión funcional; recuperar el último control aceptado sin tocar recursos persistentes.
- **Estado:** COMPLETADO.

### H00 — Contrato HTTP y de presentación

- **Objetivo y alcance:** especificar campos, filtros, nulos, errores, poblaciones, CSV y estados de disponibilidad ML para las capacidades existentes.
- **Dependencias:** B11 y autorización de P03.
- **Restricciones:** diseño de contrato, sin inventar indicadores o funcionalidades energéticas ni implementar API antes de aceptarlo.
- **Criterios de aceptación:** contrato revisado, ejemplos de los cuatro escenarios y correspondencia con aplicación; dependencias HTTP y versiones justificadas antes de H01.
- **Validación requerida:** contrastar contratos con resultados caracterizados y aclarar fechas, límites y errores sin depender de detalles de Streamlit.
- **Condición de reversión:** ambigüedad o pérdida de comportamiento; corregir el contrato antes de comenzar H01.
- **Evidencia P03 (2026-10-16):** se documentaron y luego cotejaron con la implementación las rutas, parámetros, límites, orden, nulabilidad, fechas, unidades no especificadas, errores y estados en [docs/contrato-http.md](contrato-http.md). La autorización del usuario permitió añadir FastAPI 0.143.2, Uvicorn 0.54.0 y HTTPX 0.28.1; el requisito de `typing_extensions` se fijó en 4.16.0 para satisfacer el entorno resuelto. Los servicios ahora distinguen indisponibilidad SQL de datos vacíos y Streamlit conserva mensajes seguros.
- **Estado:** COMPLETADO.

### H01 — API FastAPI de consulta

- **Objetivo y alcance:** implementar `interfaces/api/`, adaptación HTTP/JSON, ciclo de vida y pruebas con dependencias previamente aprobadas.
- **Dependencias:** H00.
- **Restricciones:** sin SQL en rutas, algoritmos duplicados ni endpoints administrativos; Streamlit sigue disponible.
- **Criterios de aceptación:** respuestas y errores conforme al contrato, resultados de transformaciones compartidas y sólo rutas GET.
- **Validación requerida:** cliente HTTP en memoria, aplicación simulada, errores de entrada/consulta, estados de disponibilidad y ciclo de vida; sin modificar datos.
- **Condición de reversión:** regresión o contrato incumplido; retirar el adaptador afectado conservando Streamlit y backend verificados.
- **Evidencia P03 (2026-10-16):** API implementada con consultas de sólo lectura, CSV, respuestas tipadas, errores 422/503/500 sanitizados y cierre del engine. La suite completa pasó (67 pruebas); HTTPX `ASGITransport` cubre las cuatro combinaciones de datos/modelo, filtros, CSV, fallo primario, fallo de métricas opcionales, rutas GET, frontend estático e importación/cierre. Los casos de API usan mocks y no acceden a PostgreSQL.
- **Estado:** COMPLETADO.

### H02 — Frontend y gráficas

- **Objetivo y alcance:** implementar HTML/CSS/JavaScript y ECharts consumiendo la API; recursos y licencia verificables.
- **Dependencias:** H01.
- **Restricciones:** sin frameworks adicionales, ML ni agregaciones de negocio duplicadas en JavaScript; sin retirar Streamlit.
- **Criterios de aceptación:** equivalencia de las cinco funcionalidades, filtros, vacíos, CSV y contadores; bundle local con licencia verificable; aprobación visual funcional, sin exigir igualdad de píxeles.
- **Validación requerida:** pruebas del contrato y comprobaciones de navegador, incluidos cuatro escenarios y ausencia de acciones administrativas.
- **Condición de reversión:** falta de equivalencia; conservar Streamlit como interfaz vigente y detener el avance.
- **Evidencia P03 (2026-10-16):** interfaz HTML/CSS/JavaScript responsive con cinco pestañas, filtros, tabla/CSV, estados vacíos y gráficas ECharts. ECharts 6.1.0 se incorporó desde el paquete npm oficial tras verificar su integridad SHA-512; `LICENSE` y `NOTICE` Apache-2.0 están junto al bundle y no hay CDN. La verificación en navegador con fixtures cubrió las cuatro combinaciones de disponibilidad, filtros sin coincidencias, error 503, tabla/CSV, navegación por pestañas y viewport de 390 px (sin desbordamiento horizontal ni errores JavaScript).
- **Corrección y cierre (2026-10-15):** la revisión final detectó que Tabla de
  registros no mostraba su estado vacío. Se corrigió únicamente `populateTable`
  para mostrar un mensaje sin filas y ocultarlo con datos. La nueva regresión en
  Chrome falló antes de la corrección tanto sin registros como con filtros vacíos;
  después pasó las transiciones vacío → datos → filtros sin resultados → datos.
  La suite completa pasó: 68 pruebas, incluida la regresión en navegador y la
  cobertura Streamlit. Se revisaron visualmente la tabla con datos y sin resultados.
  `pip check`, sintaxis de los tres módulos JavaScript, enlaces locales y
  `git diff --check` sin fallos. Fixtures y temporales aislados; sin PostgreSQL,
  generación persistente, entrenamiento ni operaciones Docker.
- **Estado:** COMPLETADO.

### H03 — Sustitución y retirada

- **Objetivo y alcance:** cambiar entrada nativa, documentación y configuración estática; retirar Streamlit y compatibilidad sin consumidores.
- **Dependencias:** H02, autorización de P04 y aprobación explícita del cambio de interfaz.
- **Restricciones:** eliminar dependencias de UI sólo sin consumidores; no alterar PostgreSQL, modelos ni laboratorio Docker; wrappers sujetos a las condiciones de C.
- **Criterios de aceptación:** equivalencia funcional y seguridad aceptadas; cobertura sustituta antes de retirar pruebas Streamlit; comandos y documentación coherentes.
- **Validación requerida:** pruebas de backend y HTTP, comprobación funcional de las cinco vistas y cuatro escenarios, revisión estática y aprobación manual del cierre.
- **Condición de reversión:** falla del cambio de interfaz; restaurar la entrada Streamlit en una ventana autorizada desde el punto previo, sin restauraciones destructivas de datos.
- **Evidencia P04:** FastAPI/ECharts queda como única interfaz oficial. Se retiran
  la presentación Streamlit, sus requisitos exclusivos y los wrappers `app`, `ml`
  y `scripts`; se migran sus consumidores conocidos (UI, pruebas y Compose) a
  entradas canónicas. Backend, contrato HTTP, operaciones y artefactos se conservan.
  No se identifican consumidores externos en el contexto disponible; los no
  inventariados deberán migrar sus rutas. Reversión selectiva desde `b0774ed`, sólo
  si se autoriza y sin restaurar recursos persistentes.
- **Validación:** 62 pruebas aprobadas sin omisiones tras retirar las dependencias
  antiguas del entorno; cobertura sustituta HTTP/Chrome para las cinco vistas y
  cuatro escenarios. `pip check`, sintaxis JS, enlaces y diff sin fallos; detalles
  en [pruebas](pruebas.md#evidencia-p04).
  Dockerfile/Compose actualizados y revisados estáticamente, sin operar Docker.
  PostgreSQL y artefactos reales no se usan ni modifican.
- **Aprobación manual:** el responsable confirmó expresamente una revisión
  satisfactoria de la demostración local y autorizó el cierre definitivo de H03/P04,
  el commit y el push a `origin/main`. Ver [pruebas](pruebas.md#cierre-manual-h03).
- **Estado:** COMPLETADO.

## F. Estado actual

| Prompt principal | Estado |
|---|---|
| P01 | COMPLETADO |
| P02 | COMPLETADO |
| P03 | COMPLETADO |
| P04 | COMPLETADO |

B00–B11 están **COMPLETADAS** con la evidencia registrada. H00–H02 y P03 están
**COMPLETADOS**, incluida la corrección del estado vacío de Tabla de registros.
H03 y P04 están **COMPLETADOS**, con validación automatizada, revisión visual del
agente y revisión personal satisfactoria confirmada por el responsable. El cierre
y la publicación en `origin/main` fueron autorizados expresamente. Se conserva
la evidencia histórica de P01–P03 y los pendientes independientes de adquisición
y automatizaciones externas.

| Control | Estado | Evidencia P02 (2026-10-15) |
|---|---|---|
| B00–B01 | COMPLETADOS | Evidencia P01 conservada. |
| B02 | COMPLETADO | Paquete raíz, una `Base`, modelos y artefacto trasladados; pruebas ORM y de bloqueo/publicación. |
| B03 | COMPLETADO | Transformaciones y agregaciones extraídas; fixtures de filtros, nulos, muestreo y suavizado; AppTest conserva poblaciones. |
| B04 | COMPLETADO | Predicción/evaluación compartida; positivos, negativos, denominadores y estados probados. |
| B05 | COMPLETADO | SQL y sesiones en infraestructura; monitoreo independiente de Streamlit; cinco pestañas y estados vacíos verificados con AppTest. |
| B06 | COMPLETADO | Entrenamiento/detección coordinados en aplicación; bundle y una métrica verificados con mocks. |
| B07 | COMPLETADO | Entradas de esquema/espera trasladadas; conexión, `create_all` y comandos probados sin DDL real. |
| B08 | COMPLETADO | Generación/reset/stream separados; operaciones y retención probadas con mocks, sin escrituras persistentes. |
| B09 | COMPLETADO | Comandos canónicos y wrappers históricos; parámetros y un ciclo ML controlado probados. |
| B10 | COMPLETADO | Configuración explícita, motor reutilizable y cierre; importaciones comprobadas sin leer `.env`, conectar ni cargar ML. |
| B11 | COMPLETADO | README, arquitectura, desarrollo, producto, pruebas y plantilla actualizados; enlaces, diff y Docker revisados estáticamente. |

Regresión final de P02: `.venv/bin/python -B -m unittest discover -q` — 54 pruebas
aprobadas. `git diff --check` y enlaces Markdown sin destinos rotos. Docker/Compose
se inspeccionaron sin ejecutarlos; PostgreSQL, datos, credenciales y artefactos
persistentes no se usaron ni modificaron.

## G. Contrato de validación

Preservar y demostrar, mediante evidencia acotada al cambio:

- Las cinco pestañas: Análisis general, Variables energéticas, Comparativos,
  Detección de anomalías y Tabla de registros. En la nueva UI conservar sus funciones.
- Los cuatro escenarios: sin datos/sin modelo, con datos/sin modelo, sin datos/con
  modelo y con datos/con modelo; estados vacíos claros, sin predicciones ficticias.
- Monitoreo sin ML y métricas históricas opcionales, sin bloquear presentación.
- Controles administrativos deshabilitados visualmente y en manejadores; permisos
  SQL mínimos y protección de credenciales, sin habilitar escritura desde monitoreo.
- Transformaciones, filtros vacíos, límites de muestreo, orden temporal, CSV y
  agregaciones; distinguir registros originales, agregados y muestras y sus denominadores.
- Cálculos y resultados ML, normalización intencional, características y formato
  del artefacto; publicación atómica y bloqueo coherente entre lectores y escritores.
- Integridad de datos y artefactos existentes, sin usarlos como recursos de escritura
  de pruebas. Usar fixtures, mocks y archivos temporales aislados cuando corresponda.
- Comandos existentes durante su compatibilidad y entradas sin efectos operativos al importar.

Primera etapa: pruebas pertinentes en `.venv` nativa Debian, comprobación funcional
del dashboard y revisión estática de Dockerfile, Compose, rutas y comandos. No usar
Docker CLI, levantar, detener, reconstruir ni modificar contenedores o el laboratorio
`smartenergy-f01-20261015-01`. No cambiar montajes, redes o políticas de arranque para
resolver problemas detectados por la revisión: registrar impedimento y solicitar decisión.

Segunda etapa: añadir pruebas de contratos HTTP y presentación, conservando las del
backend; revisar equivalencia antes de retirar Streamlit. No eliminar cobertura sólo
porque cambien imports o tecnología. Diferenciar pruebas unitarias, integración,
inspección estática y validación manual; registrar comando, resultado, entorno y límites.
No repetir pruebas sin cambios o incertidumbres que lo justifiquen. Si una validación
no puede ejecutarse con seguridad, declararla pendiente; no presentarla como aprobada.

No generar datos persistentes, entrenar o ejecutar detección persistente como parte
de esta migración. Las pruebas con datos de fixture no equivalen a validación
industrial; no hay validación con instrumentos reales en este contrato.

## H. Reglas de ejecución y continuidad

Antes y durante cada intervención futura:

1. Leer `AGENTS.md`.
2. Leer `docs/guia-de-ingenieria.md`.
3. Leer `docs/roadmap-tecnico.md`.
4. Verificar el estado real de Git; preservar cambios ajenos.
5. Identificar el prompt principal vigente y su autorización explícita.
6. Identificar tareas pendientes dentro de ese prompt y verificar sus dependencias.
7. Ejecutar únicamente el alcance autorizado, respetando puntos de control internos.
8. Registrar evidencia de validación y detenerse ante regresiones.
9. Actualizar estados y documentación cuando corresponda, distinguiendo terminado,
   pendiente y no verificado, sin atribuir al código capacidades previstas.
10. Esperar autorización para avanzar al siguiente prompt principal.

Los commits y push futuros requieren autorización específica. La autorización de
esta intervención sólo cubre sus tres documentos y su publicación; no se transfiere
a P01–P04. No cambiar decisiones aprobadas sin justificar un impedimento técnico
concreto y solicitar aprobación. No ampliar el alcance mediante correcciones oportunistas.

Antes de cada retirada, confirmar consumidores y vía de reversión sin descartar
trabajo ni restaurar bases de datos. Si falta detalle de la propuesta anterior,
recuperarlo o acordarlo antes de la tarea afectada, sin inventar criterios aprobados.
En H00 se concretará el contrato HTTP; antes de implementar se verificarán versiones
y dependencias necesarias. La ventana de sustitución definitiva requiere aprobación.
El acoplamiento operativo existente de Compose con inicialización/ML no se corrige
silenciosamente como traslado de rutas: cualquier cambio de política exige autorización.

## I. Pendiente técnico: adquisición y simulador

Pendiente independiente de P04; no añade prompts principales ni altera los 16
controles B00–B11 y H00–H03. Requiere una tarea posterior expresamente autorizada.

**Objetivo:** adaptar el simulador al contrato común de mediciones eléctricas,
usando inicialmente el ADW210 como referencia y preservando independencia del
fabricante. La decisión conceptual está en [arquitectura](arquitectura.md).

**Dependencias y criterios de aceptación verificables:**

- Seleccionar configuración y versión documental del ADW210; contrastar el mapa de
  registros y documentar por variable su alcance real (dispositivo, circuito o fase),
  unidades, escalas y disponibilidad. No inferir campos por circuito desde el catálogo.
- Especificar una representación común con instalación, dispositivo, circuito, fase
  o agregado, variable, valor, unidad y tiempo; definir calidad, ausencia, zona
  horaria y procedencia sintética/real/calculada/estimada con trazabilidad.
- Demostrar mediante fixtures y pruebas de contrato que simulador y un futuro
  adaptador pueden entregar esa representación sin campos dependientes de la marca.
- Probar coherencia física bajo hipótesis explícitas, continuidad temporal e
  integración de potencia y energía, con tolerancias definidas y reinicios de
  contadores documentados; no generar variables independientes sin relación física.
- Cubrir normalidad, variación de carga, arranque/parada, inactividad, desbalance y
  anomalías mediante escenarios reproducibles, incluyendo mediciones no disponibles.
- Evaluar compatibilidad con el generador, ORM, API y consumidores actuales antes
  de proponer cambios. Cualquier migración PostgreSQL, driver Modbus o intervención
  persistente tendrá alcance, validación y reversión aprobados por separado.

**Límites:** no implementado en P03; no autoriza cambiar el esquema, reemplazar el
generador, generar datos persistentes ni iniciar adquisición física. Las pruebas
sintéticas futuras no sustituirán el contraste con instrumentos reales.
