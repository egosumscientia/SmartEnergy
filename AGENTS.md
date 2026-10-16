# Instrucciones para agentes de SmartEnergy

SmartEnergy es el producto de monitoreo energético industrial de makeAutomatic.
El repositorio contiene un prototipo funcional; no implica una plataforma validada
con instrumentos reales. Estas reglas aplican a tareas de código y documentación.
La guía complementaria está en [docs/guia-de-ingenieria.md](docs/guia-de-ingenieria.md).

Antes de cualquier intervención relacionada con la reestructuración, leer también
[docs/roadmap-tecnico.md](docs/roadmap-tecnico.md) y respetar el prompt principal y el alcance autorizados.

## Decisiones vigentes y límites

- El stack principal es Python, PostgreSQL, SQLAlchemy, FastAPI y Apache ECharts. El desarrollo
  cotidiano es nativo en Debian, dentro de `.venv`; Docker es integración y referencia.
- El monitoreo no depende de ML: las cinco pestañas deben existir sin modelo,
  datos ni métricas históricas, mostrando estados vacíos cuando corresponda.
- El dashboard es de consulta. No añadir acciones administrativas ni
  manejadores de escritura a la presentación o a la API. Entrenamiento, detección persistente y mantenimiento
  permanecen como operaciones independientes y controladas.
- No modificar ni destruir el laboratorio Docker `smartenergy-f01-20261015-01`.
  No poblar automáticamente `smartenergy_dev` con datos sintéticos.
- El roadmap funcional, la adquisición desde instrumentos y la arquitectura
  industrial siguen pendientes de aprobación. No convertir propuestas en decisiones.

## Secuencia de trabajo

1. Comprender el objetivo y sus criterios de aceptación verificables.
2. Inspeccionar sólo el contexto técnico necesario y las instrucciones aplicables.
3. Buscar funciones, clases y componentes reutilizables antes de crear otros.
4. Delimitar componentes, dependencias y efectos secundarios del cambio.
5. Identificar riesgos, entorno objetivo y decisiones pendientes.
6. Implementar dentro del alcance autorizado. La petición concreta autoriza ese
   trabajo; no pedir confirmación de nuevo por decisiones rutinarias reversibles.
   Solicitarla para ampliaciones, reorganizaciones o acciones de riesgo no autorizadas.
7. Verificar los criterios con pruebas pertinentes y acotadas.
8. Actualizar sólo la documentación afectada y relevante para operar o mantener.
9. Informar cambios, motivo, evidencia, pendientes y aspectos no verificados.

En tareas documentales, revisar coherencia, rutas, enlaces y alcance; no ejecutar
servicios ni pruebas de código que no aporten evidencia a la documentación.

## Rigor de ingeniería y eficiencia de agentes

El rigor de ingeniería tiene prioridad absoluta sobre el ahorro de créditos. La
jerarquía obligatoria es: corrección técnica; seguridad e integridad del producto y
sus recursos; calidad, mantenibilidad y rigor de ingeniería; eficiencia operativa y
consumo responsable de créditos. Nunca sacrificar los tres primeros por el cuarto.

- En tareas sencillas o bien documentadas, trabajar directamente: leer instrucciones
  pertinentes, reutilizar decisiones, contratos y evidencias, e inspeccionar primero
  los archivos relacionados. Evitar exploraciones generales, búsquedas circulares,
  investigaciones repetidas, abstracciones y documentación sin necesidad concreta.
  Aplicar las reglas existentes de alcance y validación; emitir reportes concisos,
  claros y verificables. Eficiencia significa eliminar desperdicio, no reducir rigor.
- Ajustar la profundidad a complejidad, incertidumbre y riesgo. Ante incertidumbre
  relevante, ampliar progresivamente hasta resolverla o identificar un impedimento
  real. Está autorizada la investigación profunda necesaria dentro del alcance:
  módulos y dependencias directas e indirectas, efectos secundarios, concurrencia,
  condiciones de carrera, contratos, compatibilidad, riesgos arquitectónicos y
  operativos, regresiones difíciles y alternativas ante decisiones importantes.
  Añadir pruebas seguras y pertinentes, incluidas las de caracterización necesarias.
- No existen límites artificiales de créditos, lecturas, pruebas ni profundidad de
  análisis. Un consumo elevado es admisible con justificación técnica real; no
  detener una investigación necesaria ni entregar resultados incompletos, inseguros
  o insuficientemente verificados para ahorrar créditos. No pedir autorización sólo
  para leer más código o ejecutar pruebas adicionales seguras dentro del alcance.
- Investigar más no autoriza implementar fuera del alcance. Solicitar autorización
  para modificar componentes ajenos, cambiar decisiones arquitectónicas aprobadas,
  instalar dependencias, alterar datos o recursos persistentes, ejecutar operaciones
  potencialmente destructivas, modificar el laboratorio Docker protegido o iniciar
  tareas posteriores. Mantener las protecciones de PostgreSQL, esquemas, registros,
  artefactos ML, configuraciones, credenciales y recursos nativos; la primera etapa
  de reestructuración sólo admite validación estática de Docker.
- Redactar futuros prompts completos, autónomos, claros, precisos, proporcionados y
  coherentes con la documentación oficial, en un único bloque sin repeticiones
  innecesarias. Cada prompt de ejecución debe definir objetivo, alcance autorizado,
  contexto documental, restricciones, criterios de aceptación, validación requerida,
  condiciones de detención, entregables y reporte final. No fragmentar artificialmente
  trabajo coherente en prompts triviales ni agrupar cambios incompatibles que
  comprometan seguridad o validación.
- Para el roadmap, conservar P01–P04 como cuatro prompts principales y B00–B11 y
  H00–H03 como sus 16 controles internos, con dependencias y criterios de aceptación;
  no convertirlos automáticamente en 16 prompts. Dedicar la profundidad necesaria
  a cada tarea compleja. Ante una regresión o un impedimento que no pueda resolverse
  dentro del alcance autorizado, detenerse y reportarlo; no iniciar automáticamente
  el siguiente prompt principal.

**Trabajar con máxima eficiencia cuando el problema sea sencillo y con toda la
profundidad necesaria cuando sea complejo. El rigor de ingeniería nunca se sacrifica
para ahorrar créditos.** Esta regla prevalece sobre cualquier instrucción de
eficiencia que pudiera interpretarse como un límite artificial de investigación,
pruebas o calidad; no modifica las restricciones de seguridad ni el alcance autorizado.

## Arquitectura, reutilización y calidad

- Respetar las decisiones aprobadas y los límites actuales: `frontend/` presenta la
  UI HTML/CSS/JavaScript con ECharts; `smartenergy/interfaces/api/` expone FastAPI;
  `smartenergy/` contiene el backend (`config`, `infrastructure`, `analysis`,
  `application`, `operations`); `tests/` verifica. Las entradas administrativas
  canónicas son `python -m smartenergy.operations.<operación>`.
  Ver [docs/arquitectura.md](docs/arquitectura.md).
- Mantener alta cohesión, bajo acoplamiento y responsabilidades concretas. Evitar
  código espagueti, dependencias circulares y lógica administrativa en la presentación.
- No reorganizar módulos sin justificación y autorización. No añadir capas, clases,
  abstracciones, dependencias o patrones sin una necesidad concreta de la tarea.
- Aplicar DRY, KISS y YAGNI con criterio: reutilizar lógica existente, evitar duplicar
  responsabilidades y preferir soluciones simples, legibles y mantenibles.
- Usar nombres descriptivos, `snake_case` en funciones y módulos y `PascalCase` en
  clases; mantener las convenciones locales y las buenas prácticas de Python.
- Conservar contratos, orden de variables, tipos y denominadores. Separar negocio,
  persistencia, presentación e integración en los cambios que realmente lo requieran.
- Gestionar errores explícitamente. No ocultar fallos relevantes ni inventar ceros,
  mediciones o predicciones para aparentar disponibilidad. Evitar efectos inesperados.
- Reutilizar `smartenergy/infrastructure/model_artifact.py` para carga y publicación del modelo; conservar
  el bloqueo coherente, el reemplazo atómico y el formato compatible del bundle.

## Seguridad y protección del entorno

- No mostrar contraseñas, tokens ni URLs completas con credenciales. Mantener `.env`,
  `.venv`, modelos, datos y temporales fuera del control de versiones; comprobarlo.
- Identificar base, rol, montajes y permisos antes de operaciones con efectos. No
  asumir que PostgreSQL local o los recursos Docker pertenecen sólo a SmartEnergy.
- Respetar la separación entre desarrollo e integración y el mínimo privilegio:
  consulta para monitoreo y permisos administrativos sólo para tareas autorizadas.
- No eliminar datos, reinicializar bases, recrear recursos ni ejecutar operaciones
  destructivas o irreversibles sin autorización explícita y reversión o respaldo viable.
- Revisar scripts antes de ejecutarlos o importarlos: algunos escriben al importarse.
  Los generadores aplican retención; `reset_db` trunca tablas. Un ejemplo del README
  no constituye autorización para ejecutar mantenimiento ni levantar Compose.

## Validación, eficiencia y control de cambios

- Elegir pruebas según el cambio. Distinguir unitarias, integración, inspección
  manual y validación industrial; no equiparar fixtures o simulación a equipos reales.
- Cubrir estados vacíos y disponibilidad independiente de datos y ML cuando cambie
  el dashboard. No afirmar que una prueba no ejecutada pasó.
- Consultar la guía para seleccionar comandos; verificar dependencias y efectos
  antes de ejecutar. No instalar herramientas sólo para satisfacer un formalismo.
- Agrupar lecturas y cambios relacionados; no releer archivos, repetir pruebas
  suficientes ni ampliar indefinidamente la validación sin evidencia nueva.
- No tocar componentes ajenos, introducir funcionalidades no solicitadas ni hacer
  refactorizaciones oportunistas. Usar herramientas y recursos proporcionales.
- Documentar contratos, riesgos y operación útil; evitar informes extensos sin valor.
  Comunicar hallazgos que exijan una decisión arquitectónica humana.
- No hacer commits ni push sin autorización explícita. Antes de publicar, revisar
  diff, secretos, archivos incluidos y estado remoto; no descartar trabajo existente.
- Entregar un estado preciso: qué terminó, qué falta y qué no se verificó. No anunciar
  preparación industrial por haber recuperado el prototipo.
