# Guía de ingeniería de SmartEnergy

Esta guía desarrolla las instrucciones operativas de [AGENTS.md](../AGENTS.md).
Describe el repositorio actual y los criterios para evolucionarlo; no aprueba un
roadmap, protocolos industriales ni una arquitectura futura.

## 1. Filosofía de ingeniería

SmartEnergy prioriza observar mediciones energéticas con claridad, trazabilidad y
seguridad. ML es complementario: su ausencia o fallo no debe impedir consultar datos.
Sin mediciones se muestran estados vacíos, no demostraciones ni valores inventados.

Elegir el cambio más pequeño que satisfaga criterios verificables. La simplicidad
se evalúa por facilidad de comprensión y mantenimiento, no sólo por número de líneas.
La recuperación del prototipo y sus pruebas sintéticas no acreditan validez eléctrica,
metrológica, predictiva ni industrial.

## 2. Organización y responsabilidades actuales

| Componente | Responsabilidad existente | Límite que conservar |
|---|---|---|
| `app/dashboard_app.py` | Entrada Streamlit, consultas actuales, cinco vistas y analítica en memoria | Consulta; no ejecutar administración desde la UI |
| `app/dashboard_data.py` | Wrapper de `smartenergy/analysis/measurements.py` (preparación, filtros, muestreo y orden de DataFrames) | Sin persistencia ni dependencia de ML |
| `smartenergy/interfaces/api/` y `frontend/` | Adaptador HTTP de monitoreo y cinco vistas web locales | Sólo lectura; sin duplicar SQL o cálculos de negocio en JavaScript |
| `smartenergy/config.py` | Configuración de entorno y carga de `.env` | Sin secretos rastreados |
| `smartenergy/infrastructure/database.py`, `smartenergy/infrastructure/models.py` | Engine, sesiones y modelos `Registro` y `Metrica` | No inicializar ni reiniciar datos automáticamente |
| `smartenergy/infrastructure/model_artifact.py` | Carga bloqueada y publicación atómica del bundle | Preservar formato y coordinación entre lectores y escritor |
| `smartenergy/analysis/`, `smartenergy/application/` | Entrenamiento opcional, detección y consultas de monitoreo | No convertirlos en requisitos de visualización |
| `smartenergy/operations/` (wrappers en `ml/` y `scripts/`) | Inicialización, espera de base, generación y mantenimiento | Examinar efectos y autorizar las escrituras |
| `tests/` | Pruebas de objetos, muestras, seguridad, renderizado y artefactos | Aislamiento y alcance explícitos |
| `requirements.txt`, Dockerfile y Compose | Dependencias e integración de referencia | No confundir definición con autorización de ejecución |

Las consultas se coordinan en aplicación y los cálculos se comparten en análisis.
No mover módulos ni proponer servicios nuevos sólo por uniformidad.
La API reutiliza los servicios y transformaciones ya existentes. Streamlit sigue
disponible; no retirarlo fuera de una etapa explícitamente aprobada.

## 3. SOLID aplicado con pragmatismo

- **Responsabilidad única:** separar una transformación de datos de su representación
  si ambas cambian por motivos distintos; no separar cada operación trivial en una clase.
- **Abierto/cerrado:** extender un contrato estable cuando exista un caso concreto,
  sin diseñar puntos de extensión para instrumentos todavía no aprobados.
- **Sustitución:** un componente alternativo debe conservar entradas, salidas, errores
  y efectos esperados; una implementación simulada no acredita equivalencia física.
- **Segregación de interfaces:** exigir sólo las operaciones que consume el llamador.
  No obligar a la consulta a disponer de entrenamiento o permisos administrativos.
- **Inversión de dependencias:** permitir sustituir I/O en pruebas mediante funciones
  o mocks existentes; introducir interfaces formales sólo si resuelven una necesidad real.

## 4. DRY, KISS, YAGNI y separación de responsabilidades

Antes de extraer lógica común, identificar qué regla se duplica y quién es responsable
de mantenerla. Dos fragmentos parecidos con contratos distintos no justifican una
abstracción genérica. No duplicar filtros, orden, definición de variables o publicación
del modelo si los helpers actuales ya resuelven la tarea.

Una lectura debe seguir siendo una lectura. No añadir generación, entrenamiento,
creación de tablas o mantenimiento al abrir una vista, refrescar o aplicar filtros.

## 5. Reutilización de funciones, clases y componentes

Buscar por comportamiento en los módulos afectados antes de implementar. Comprobar
contrato y efectos: importar `simulate_data`, `reset_db` o los módulos ML puede ejecutar
operaciones, por lo que no deben usarse como utilidades inocuas de importación.

Reutilizar `prepare_records`, `filter_records`, `sample_records`, `recent_records` y
los helpers de artefactos cuando corresponda. Extraer una función sólo con una
responsabilidad definida y evidencia de que reduce duplicación o facilita validación.

## 6. Interfaces y contratos

Para un cambio de interfaz, precisar entradas, salida, unidades conocidas, nulabilidad,
orden temporal, excepciones y efectos. Si faltan unidades o convenciones aprobadas,
registrar la incertidumbre antes de introducir conversiones o criterios industriales.

En el dashboard, distinguir registros originales filtrados de puntos muestreados,
agregados o suavizados. Cada porcentaje necesita una población y denominador claros.
No denominar «anomalías detectadas» a etiquetas originales cuando no existe predicción.
CSV y límite de tabla tienen alcances distintos y deben indicarlo.

El bundle conserva las claves `model`, `scaler` y `features`; el orden de `features`
forma parte del contrato. Un cambio incompatible requiere decisión explícita y
estrategia de compatibilidad, no regeneración silenciosa del modelo.

## 7. Errores y observabilidad

Distinguir ausencia de datos, filtros sin resultados, valores inválidos, modelo ausente,
fallo de consulta y error de predicción. No sustituir indisponibilidad por cero o «Normal».
Un fallo opcional de ML debe explicarse sin derribar las vistas de mediciones.

Capturar errores donde pueda ofrecerse una recuperación concreta; no añadir `except`
amplios que oculten el fallo. En límites opcionales de UI puede degradarse el servicio,
pero debe quedar visible la indisponibilidad. No imprimir credenciales ni volcados de
datos. Los logs de sesión Streamlit no equivalen a una auditoría persistente.

## 8. Persistencia, transacciones y acceso a datos

Confirmar destino y rol sin mostrar la URL completa. Usar sesiones acotadas y cierre
garantizado; confirmar transacciones de escritura deliberadamente y hacer rollback
ante errores. Parametrizar SQL y definir orden explícito para registros recientes.

`seed_db` usa `create_all`: crea objetos faltantes, no es un sistema de migraciones ni
verifica por sí solo la compatibilidad de un esquema existente. Justificar y autorizar
cambios de esquema por separado; nunca usar reset para resolver una incompatibilidad.

Los generadores insertan datos y pueden eliminar antiguos al superar la retención.
El entrenamiento publica un archivo y la detección agrega una métrica: no forman una
transacción atómica conjunta. No prometer esa garantía. Mantener la escritura temporal,
el reemplazo atómico y el mismo archivo de bloqueo para leer y publicar modelos.

## 9. Seguridad y configuración

El entorno cotidiano es Debian con `.venv`; se verificó Python 3.13.5 y PostgreSQL 17.
La referencia Docker utiliza Python 3.11 y PostgreSQL 16. No intercambiar runtimes ni
actualizar dependencias por iniciativa propia. Instalar sólo en el entorno autorizado.

Identificadores recuperados, no instrucciones para recrear recursos:

| Entorno | Base y acceso UI | Protección |
|---|---|---|
| Nativo | `smartenergy_dev`, UI local en `18502` | No poblar automáticamente ni tocar otras bases |
| Laboratorio | `smartenergy-f01-20261015-01`, UI local en `18501` | Conservar contenedores, red, volúmenes y datos intactos |

`DATABASE_URL` se obtiene del entorno o `.env` mediante el mecanismo existente.
Compose exige `SMARTENERGY_MONITOR_DATABASE_URL` para su dashboard. Mantener secretos
fuera de Git y restringir permisos de archivos locales; no publicar ejemplos con
credenciales reales. Un modo transaccional de lectura no sustituye privilegios SQL
mínimos. El usuario de desarrollo y el de monitoreo tienen propósitos distintos.

El archivo `.lock` debe estar disponible junto al modelo antes de montar el monitor
en lectura. El bloqueo actual usa `fcntl` en Linux/Docker; no asumir portabilidad a
otros sistemas ni cargar bundles no confiables mediante joblib.

## 10. Pruebas y criterios de aceptación

Definir resultado observable, entorno, efectos admitidos y criterio de éxito antes de
ejecutar. Elegir el nivel de evidencia necesario:

| Nivel | Evidencia | No demuestra |
|---|---|---|
| Unitario / fixtures | Contratos, filtros, bloqueos y casos límite aislados | Funcionamiento contra base o instrumento real |
| Integración | Conexión, permisos, persistencia o protocolo con recursos autorizados | Exactitud industrial |
| Manual | Comportamiento visual e interacción aprobada | Cobertura automática completa |
| Industrial | Mediciones contrastadas con instrumentos, condiciones y criterios aprobados | No se obtiene mediante datos simulados |

Para cambios del dashboard, cubrir datos/modelo presentes y ausentes de forma
independiente, métricas históricas opcionales, filtros vacíos y nulos pertinentes.
Usar mocks y temporales aislados; no poblar bases ni entrenar modelos por conveniencia.
Comprobar que los controles de administración sigan bloqueados.

Ejemplo acotado, desde la raíz y con `.venv` preparado, para esos componentes:

```bash
.venv/bin/python -B -m unittest -v tests.test_dashboard_rendering tests.test_dashboard_safety tests.test_model_artifact
git diff --check
```

Seleccionar módulos según el cambio; no ejecutar siempre todo. `test_models.py` y
`test_simulate.py` también contienen funciones de prueba libres que `unittest` no
descubre como casos: no declarar cobertura de ellas usando sólo el comando anterior.
No instalar otro runner sin necesidad. No repetir una prueba aprobada salvo nuevo
cambio, fallo o incertidumbre que lo justifique. En documentación basta validación
documental si no hay cambios ejecutables.

## 11. Refactorizaciones

Separar corrección funcional de reorganización. Explicar problema, beneficio concreto,
contratos afectados y forma de verificar equivalencia. Una refactorización local
necesaria puede estar incluida en la tarea; una reorganización de módulos o cambio
arquitectónico exige justificación y autorización. No aprovechar una corrección para
renombrar masivamente, formatear todo o introducir patrones ajenos al objetivo.

## 12. Documentación técnica y mantenimiento

Mantener una fuente principal por tema: AGENTS contiene reglas operativas; esta guía
explica su aplicación; el README debe orientar al uso. Documentar contratos, decisiones,
limitaciones y riesgos con rutas reales. Evitar copiar implementaciones o informes
transitorios que se desactualicen. Diferenciar estado comprobado de propuesta.

El README orienta a los comandos y documentos vigentes. Los ejemplos de
mantenimiento y arranque no autorizan su ejecución. La definición del producto y
la referencia inicial ADW210 aprobadas en el cierre P03 se documentan en
[producto](producto.md) y [arquitectura](arquitectura.md). Su implementación de
adquisición, los criterios metrológicos y el despliegue productivo requieren una
tarea y aprobación posteriores.

## 13. Eficiencia de agentes de IA

Seguir la secuencia de AGENTS: objetivo, inspección dirigida, reutilización, alcance,
riesgos, trabajo autorizado, verificación, documentación relevante y reporte preciso.
Agrupar búsquedas independientes y conservar lo ya aprendido; volver a leer sólo ante
cambios o dudas concretas. No invocar servicios, herramientas, instalaciones ni agentes
adicionales si la tarea no lo requiere ni lo autoriza.

Detener una ampliación no autorizada o una operación insegura, explicar el bloqueo y
continuar trabajo independiente útil si corresponde. No solicitar aprobación repetida
para pasos ya autorizados. En un trabajo documental, sustituir implementación y
pruebas de código por edición y revisión de coherencia, enlaces y alcance.

## 14. Definición de terminado

Una implementación está terminada cuando:

- Satisface los criterios acordados sin funcionalidades adicionales.
- Conserva contratos, separación de monitoreo/administración y ML opcional.
- Tiene evidencia pertinente, identificando pruebas ejecutadas, resultados y límites.
- No deja secretos, datos generados ni artefactos temporales para versionar.
- Preserva recursos existentes o deja documentada la intervención autorizada y reversible.
- Actualiza la documentación afectada, sin duplicación innecesaria.
- Presenta un diff revisado y un estado Git explicado; commit y push sólo con permiso.
- Informa pendientes, riesgos y aspectos no verificados, sin atribuir validación industrial.

Para una tarea documental, terminado significa documentos coherentes con el código,
enlaces y rutas válidos, decisiones futuras identificadas y sólo los archivos autorizados
modificados. No requiere pruebas de aplicación sin cambios ejecutables.
