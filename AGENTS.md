# Instrucciones para agentes de SmartEnergy

SmartEnergy es el producto de monitoreo energético industrial de makeAutomatic.
El repositorio contiene un prototipo funcional; no implica una plataforma validada
con instrumentos reales. Estas reglas aplican a tareas de código y documentación.
La guía complementaria está en [docs/guia-de-ingenieria.md](docs/guia-de-ingenieria.md).

## Decisiones vigentes y límites

- El stack principal es Python, PostgreSQL, SQLAlchemy y Streamlit. El desarrollo
  cotidiano es nativo en Debian, dentro de `.venv`; Docker es integración y referencia.
- El monitoreo no depende de ML: las cinco pestañas deben existir sin modelo,
  datos ni métricas históricas, mostrando estados vacíos cuando corresponda.
- El dashboard es de consulta. Mantener deshabilitadas sus acciones administrativas,
  también en los manejadores. Entrenamiento, detección persistente y mantenimiento
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

## Arquitectura, reutilización y calidad

- Respetar las decisiones aprobadas y los límites actuales: `app/` presenta y
  transforma datos para la UI; `core/` contiene configuración, ORM y artefactos;
  `ml/` implementa ML opcional; `scripts/` contiene operaciones; `tests/` verifica.
  El dashboard aún contiene consultas existentes: no inventar una separación ya hecha.
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
- Reutilizar `core/model_artifact.py` para carga y publicación del modelo; conservar
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
