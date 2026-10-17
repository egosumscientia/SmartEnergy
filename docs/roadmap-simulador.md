# Hoja de ruta del simulador de SmartEnergy

Fecha: 2026-10-17. Rige el [contrato de fase 1](contratos/simulador-fase-1.md),
junto con [AGENTS.md](../AGENTS.md) y la [guía](guia-de-ingenieria.md).
Este roadmap desarrolla el [pendiente de adquisición y simulador](roadmap-tecnico.md#i-pendiente-técnico-adquisición-y-simulador);
no altera los cuatro prompts P01–P04 ni los 16 controles de reestructuración.

## Estado y secuencia de aprobación

La fase documental y su corrección por auditoría quedaron publicadas en
`ed2e5d8219a21e2bd29235c7140327ff5b702662`. La intervención posterior autorizó
exclusivamente la investigación S01, iniciada en `feature/simulador` con árbol
limpio. Su evidencia y decisiones pendientes están en la matriz instrumental.
No se considera aceptada una etapa futura por figurar aquí. La aprobación del
responsable debe identificar etapa, alcance, entregables, criterios y efectos
permitidos; ninguna etapa de implementación comienza sin aprobación explícita.
No hay fechas ni estimaciones de esfuerzo asignadas.

Secuencia: S01 → S02 → S03 → S04 → S05 → S06 → S07 → S08 → S09.
El núcleo inicial puede ser genérico, pero S01 conserva la investigación obligatoria
del ADW210 y el contraste documental del apartado I del roadmap técnico. S02 puede
preparar hipótesis genéricas ante pendientes, sin aprobar emulación ni dar S01
por cumplida. Todo diferimiento deberá documentarse y aprobarse según el contrato;
la emulación ejecutable de registros puede diferirse sin omitir su revisión documental.
S07 verifica físicamente el núcleo
antes de integrarlo; S06 ya incluye pruebas unitarias, no se posponen hasta S07.
Cada etapa se detiene al concluir su alcance y presenta evidencia para revisión.

| Etapa | Estado tras la investigación S01 (2026-10-17) |
|---|---|
| S01 | Investigación documental realizada; cierre formal pendiente de evidencia específica, selección de referencia y aprobación de límites/diferimientos D01–D06 |
| S02–S05 | Requisitos y criterios iniciales documentados; especificaciones detalladas pendientes |
| S06–S09 | No iniciadas; implementación, integración y aceptación no autorizadas |

La aceptación distingue tres niveles definidos en el
[contrato](contratos/simulador-fase-1.md#niveles-de-aceptación-y-límites-de-evidencia):
**nivel 1**, núcleo y reproducción (S07); **nivel 2**, contrato común y compatibilidad
(S08, inicialmente en memoria); **nivel 3**, persistencia completa y extremo a
extremo con PostgreSQL aislado y autorización expresa (alcance separado dentro de
S08). Cada cierre identifica el nivel demostrado. Fixtures o memoria no validan
persistencia; sin ensayo autorizado y aprobado, el nivel 3 permanece pendiente.
Ningún nivel acredita funcionamiento industrial ni autoriza el siguiente.

## S01 — Cerrar evidencia de instrumentos

- **Objetivo:** fijar referencias auditables para decidir qué observar sin depender
  estructuralmente de una marca.
- **Alcance:** completar la [matriz de medidores](instrumentacion/matriz-medidores.md)
  existente; ADW210 principal, SDM630-Modbus V2 y PM5310 como contraste. Resolver
  revisión/variante/firmware, configuración CT/VT y fuentes divergentes.
- **Entregables:** matriz revisada y ficha por variable con alcance, unidades,
  mapa/codificación, signos, resolución, exactitud condicionada y tiempo de
  actualización; desconocidos identificados con impacto y fuente pendiente.
- **Dependencias:** contrato de fase 1 y decisiones de arquitectura existentes.
- **Aceptación:** cada característica atribuida a un instrumento tiene fuente
  primaria y localización, distinguiendo evidencia comercial, de familia, específica
  y pendiente; cero direcciones/escalas inferidas de otra variante. ADW210 es
  referencia obligatoria aun para núcleo genérico: documentar y contrastar sus
  capacidades relevantes por variable conforme al roadmap técnico. Identificar
  el 100% de variables del perfil elegido como confirmadas documentalmente para
  la configuración o pendientes; pendientes críticos bloquean su emulación.
  No exigir un emulador ejecutable de registros para cerrar esta investigación.
  Un pendiente no equivale a un requisito cumplido: cualquier diferimiento requiere
  alcance, justificación, impacto, condición de cierre y aprobación registrados.
- **Riesgos:** manual de familia, registros no aplicables que devuelven cero,
  firmware distinto, confusión entre resolución, exactitud y sondeo.
- **Aprobación humana:** configuración de referencia y suficiencia de la evidencia;
  decidir si el primer núcleo será genérico y aprobar expresamente cualquier
  diferimiento, identificando el requisito previo afectado. La investigación
  realizada no aprueba diferimientos ni elimina dependencias documentales.

### Resultado de la investigación S01 — pendiente de revisión

Se ampliaron las [fichas instrumentales](instrumentacion/matriz-medidores.md#fichas-de-evidencia-s01-por-instrumento)
y se contrastaron capacidades/mapas por variable. A3/A4 aportan configuración
específica ADW210; A2 conserva nivel de familia. S5 permitió inspeccionar el XLS
PM5310, su aplicabilidad por modelo, cadencia del bloque básico y valores N/A.
Se documentó la discrepancia de revisión de E4 y se conservó E1 como referencia.

El [resultado por criterio](instrumentacion/matriz-medidores.md#resultado-de-s01-y-criterios-de-aceptación)
distingue evidencia disponible de pendientes. Bloquean el cierre formal la
configuración instrumental no aprobada y la falta de aprobación de suficiencia y
diferimientos. Firmware/mapa ADW210, detalles de CT/VT y convenciones no resueltas
bloquean equivalencia instrumental; no se los declara cumplidos por proponer un
núcleo genérico. Persisten las condiciones del roadmap técnico, apartado I.

**Recomendación pendiente:** revisar y aprobar o rechazar
[D01–D06](instrumentacion/matriz-medidores.md#c-decisiones-pendientes-y-recomendaciones-concretas):
referencia ADW210 3P4W con tensión común, contraste Eastron directo/PM5310 B1,
núcleo genérico sin equivalencia de marca y diferimientos acotados de decodificación
y metrología. Las reglas eléctricas y la representación siguen pendientes de
S02/S03; no se diseñaron ni implementaron. **S01 no está cerrada formalmente y S02
no está autorizada ni iniciada.** No se modificaron contrato, AGENTS, código o entornos.

## S02 — Especificar variables y modelo eléctrico mínimo

- **Objetivo:** convertir las relaciones del contrato en una especificación
  matemática revisable antes de programar.
- **Alcance:** 1P2W y 3P4W sinusoidales de consumo propuestos; por fase y total,
  P/Q/S/FP/frecuencia y energía; definir carga, signos, tensión L-N/L-L,
  agregación y respuesta a cambios. Definir separadamente signos de P/Q,
  cuadrantes, energía activa importada/exportada y reactiva inductiva/capacitiva,
  acumulación y cambios de signo, contrastando su correspondencia con las fuentes.
  Especificar temperatura auxiliar sin atribuirla
  al medidor, o declarar su indisponibilidad.
- **Entregables:** catálogo y relaciones detalladas dentro del contrato, tabla de
  nominales/dominios, tolerancias absolutas/relativas por variable, soporte temporal
  y casos analíticos de referencia. Sin crear código o una capa `domain/`.
- **Dependencias:** cierre de S01 para el perfil elegido, con evidencia ADW210 y
  cualquier diferimiento documentado y aprobado; sin atribuciones instrumentales
  que excedan la evidencia disponible.
- **Aceptación:** el 100% de variables tiene unidad, alcance, signo y nulabilidad;
  cada relación indica hipótesis; ejemplos calculados independientes para carga
  resistiva, inductiva/capacitiva, cero y desequilibrio; casos obligatorios de
  consumo capacitivo y cambio de signo de Q manteniendo P positiva, con contadores
  diferenciados. Resolver o registrar como impedimento las convenciones aún no
  verificadas; criterios de convergencia
  temporal y tolerancias numéricas aprobados antes de S06.
- **Riesgos:** imponer fórmulas equilibradas a cargas desiguales, mezclar FP verdadero
  y desplazamiento, confundir error del motor con exactitud instrumental.
- **Aprobación humana:** perfil mínimo, nominales, signos/cuadrantes/contadores y
  acumulación, método de agregación de S/FP/Q, tolerancias y cualquier ampliación
  de topología/distorsión/exportación. Definir el contador activo exportado no
  incorpora generación exportadora al mínimo.

## S03 — Diseñar contrato común y compatibilidad

- **Objetivo:** definir el límite que podrán compartir simulador y futuro dispositivo.
- **Alcance:** representación, tipos, versión, identidad, fases/agregados, unidades,
  procedencia, dimensiones independientes de disponibilidad, validez, vigencia y
  método de obtención; tiempos/ventanas, registros tardíos y duplicados. Diseñar
  linaje y artefacto de verdad física; inventariar consumidores legados.
- **Entregables:** especificación en el contrato con ejemplos de observación válida,
  ausencia, invalidez, congelación, retraso y duplicado; matriz de impacto sobre
  `Registro`, aplicación, HTTP/CSV, UI y bundle ML; propuesta de ubicación de código
  y estrategia de transición/reversión, sin migraciones ni cambios ejecutables.
  Identificar conservación o pérdida de cada campo en la proyección legada y
  diseñar el plan de aceptación por niveles, incluida la ruta de escritura,
  lectura y consumo de extremo a extremo requerida para nivel 3.
- **Dependencias:** S02 aceptada y semánticas de instrumentos relevantes de S01.
- **Aceptación:** los ejemplos distinguen evento/observación/entrega y combinaciones
  admisibles de las cuatro dimensiones de calidad, sin exponer causas conocidas
  exclusivamente por el simulador. Toda ventana identifica intervalo físico,
  transformación, instante de observación y recepción/registro según corresponda,
  y referencia al escenario/versionado; demostrarlo en ejemplos que crucen
  transiciones, permitiendo referencias eficientes por intervalo o segmento.
  Fixture conceptual de simulador y de futuro dispositivo
  representa lo mismo sin campos obligatorios de marca. UTC propuesto y tratamiento
  de históricos naive explícitos; no hay proyección ambigua a campos legados.
  Decidir qué ocurre con temperatura y etiquetas sin perder verdad de origen.
  El plan distingue nivel 2 de persistencia completa: define verificaciones por
  campo y condiciones del PostgreSQL aislado que requerirán autorización para nivel 3.
- **Riesgos:** pérdida de metadatos al proyectar, reinterpretación de históricos,
  doble escala CT/kW, fuga de etiquetas a ML, cambios incompatibles encubiertos.
- **Aprobación humana:** forma larga/ancha, nombres/tipos/unidades/versionado,
  artefacto de referencia, calidad multidimensional, soporte de ventanas,
  deduplicación/tiempo, proyección o evolución del esquema y plan por niveles.
  Aprobar diseño no autoriza ejecutar una migración ni modificar HTTP.

## S04 — Diseñar escenarios físicos y perfiles

- **Objetivo:** cubrir operaciones útiles para SmartEnergy con oráculos conocidos.
- **Alcance:** F01–F05 del contrato: reposo, estable, rampa/turno, sobrecarga,
  subtensión, desequilibrio y arranque/parada como envolventes RMS.
- **Entregables:** catálogo parametrizado en el contrato: manifiesto de escenario,
  semilla, tiempos, sujetos, límites y resultados esperados; tabla de cobertura
  con casos límite (duración cero rechazada, cambio en frontera de intervalo,
  carga cero y tramo de energía).
- **Dependencias:** S02 y S03 aceptadas.
- **Aceptación:** cada ID tiene al menos un caso normal o perturbado y su oráculo;
  nominales/umbrales/duraciones explícitos; energía consistente con perfil;
  evento físico, estado operativo y etiqueta diagnóstica diferenciados.
- **Riesgos:** escenarios vistosos sin utilidad, discontinuidades no declaradas,
  convertir parámetros de ensayo en umbrales industriales del producto.
- **Aprobación humana:** catálogo mínimo, umbrales de escenarios y utilidad para
  SmartEnergy; ampliaciones de realismo sólo con necesidad verificable.

## S05 — Diseñar instrumentación y adquisición defectuosas

- **Objetivo:** verificar errores de cadena y anomalías físicas simultáneas sin
  contaminar la referencia física.
- **Alcance:** I01–I04, A01–A02 y C01: sesgo/ruido/deriva, congelación/intermitencia,
  relación de transformación incorrecta, pérdida/demora/duplicado y caída de enlace.
- **Entregables:** perfiles de fallos con orden de composición, magnitud/duración,
  relojes, semillas separadas por capa, política de recuperación y trazabilidad;
  criterios estadísticos fijados para ruido y oráculos exactos para inyecciones.
- **Dependencias:** S03 y S04 aceptadas; datos metrológicos S01 cuando se invoque
  equivalencia con un instrumento específico.
- **Aceptación:** cada ID tiene caso reproducible; C01 conserva todas las causas
  y prueba enmascaramiento; recuentos esperados de pérdidas/duplicados explícitos;
  la verdad física no depende de activar fallos; error conocido por el ensayo
  no se confunde con diagnóstico disponible para el producto.
- **Riesgos:** sumar ruido independiente a variables ligadas sin distinguir capas,
  alterar energía física por pérdida de datos, confundir ausencia con cero o parada.
- **Aprobación humana:** perfiles/bandas estadísticas, orden de efectos y políticas
  de cola/recuperación. No se añade emulación de protocolo ni servicio de red.

## S06 — Implementar núcleo incremental aislado

- **Objetivo:** implementar únicamente los diseños aceptados con salida verificable.
- **Alcance:** primero núcleo físico determinista, luego instrumentación y adquisición;
  funciones puras y coordinación mínima conforme a S03, sin reemplazar aún los
  generadores ni escribir en PostgreSQL. Sin dependencias nuevas por defecto.
- **Entregables:** código y pruebas unitarias pertinentes, fixtures/oráculos y
  manifiestos en recursos de prueba aislados; documentación operativa afectada.
- **Dependencias:** S01–S05 cerradas para el perfil que se implemente; aprobación
  explícita de S06, ubicaciones y efectos autorizados.
- **Aceptación:** controles de reproducción, aislamiento, identidad, ausencia y
  seguridad del contrato pasan; casos analíticos cumplen tolerancias; importar
  no carga configuración, conecta o escribe. Cada incremento conserva lo anterior.
- **Riesgos:** acoplar motor a ORM, reutilizar RNG global, crear abstracciones sin
  consumidor, ejecutar generadores actuales como atajo.
- **Aprobación humana:** inicio de implementación y cualquier desviación; cambios
  de dependencias/arquitectura requieren autorización específica. Los incrementos
  internos no implican permiso de integración ni deben convertirse en prompts triviales.

## S07 — Validar coherencia física, matemática y temporal

- **Objetivo:** acreditar el alcance sintético con evidencia independiente del motor.
- **Alcance:** balances, P/Q/S/FP, energía constante y por tramos, variación del paso,
  cero, reinicios de contador, límites temporales y eventos combinados; consumo
  capacitivo y cambio de signo de Q con P positiva según S02; ventanas que
  atraviesen transiciones de operación según S03.
- **Entregables:** pruebas de aceptación física/temporal y evidencia reproducible
  con manifiesto, entorno, tolerancias, resultados y límites.
- **Dependencias:** S06 y oráculos/tolerancias aceptados de S02–S05; aprobación de etapa.
- **Aceptación:** todos los controles aplicables del contrato pasan, sin omisiones
  ocultas ni ajuste posterior de tolerancias. Fallos bloquean S08; evidencia
  numérica sólo acredita nivel 1; no acredita integración, persistencia,
  validación metrológica o industrial.
- **Riesgos:** prueba que repite implementación, aceptación circular y falsos
  positivos por aleatoriedad o error de discretización no acotado.
- **Aprobación humana:** suficiencia de evidencia, límites reconocidos y autorización
  posterior de integración; una regresión exige resolución dentro de alcance.

## S08 — Integrar progresivamente con SmartEnergy

- **Objetivo:** consumir el límite común aprobado preservando contratos y recursos.
- **Alcance:** primero nivel 2 con pruebas de aplicación/adaptación en memoria;
  después nivel 3 únicamente con autorización expresa de ensayo y entorno
  PostgreSQL aislado conforme al contrato. Conservar
  API de consulta y cinco vistas; usar la estrategia de compatibilidad de S03.
- **Entregables:** adaptadores mínimos, pruebas de contrato y de consumidores;
  evidencia separada de niveles 2 y 3, indicando pendiente/no ejecutado cuando
  corresponda; procedimiento de operación/reversión. Migraciones o HTTP nuevo son trabajos
  separados con alcance y aprobación específicos si resultan necesarios.
- **Dependencias:** S07 aceptada, diseño S03 vigente y autorización de S08.
- **Aceptación de nivel 2:** fixtures sintéticos y de dispositivo conceptual pasan el mismo
  límite; unidades/etiquetas/tiempo no se pierden silenciosamente; sin duplicación
  de energía; bundle y orden de características conservados. Si afecta monitoreo,
  verificar cinco vistas y cuatro combinaciones datos/ML, filtros vacíos, nulos,
  CSV y rechazo de escritura. Este nivel no valida persistencia ni acredita que
  `Registro` conserve el contrato común completo. Ningún acceso a recursos protegidos.
- **Aceptación de nivel 3:** recorrer escritura, lectura y consumo aprobados contra
  PostgreSQL aislado y contrastar todos los campos requeridos y linaje con el
  oráculo, incluyendo pérdidas declaradas en cualquier proyección. Para declarar
  persistencia completa no pueden quedar campos requeridos sin conservar o
  recuperar según el diseño aprobado. Mocks y memoria no sustituyen este ensayo.
  Si falta autorización o evidencia suficiente, nivel 3 queda pendiente o fallido,
  sin declarar persistencia validada ni cierre del alcance que la exija.
- **Riesgos:** proyección irreversible, contaminación de históricos, activación de
  retención existente, uso de datos sintéticos como reales o entrenamiento implícito.
- **Aprobación humana:** componentes afectados, estrategia final de compatibilidad,
  nivel de aceptación pretendido y, para nivel 3, destino, rol, permisos, datos,
  escrituras, aislamiento y reversión/limpieza autorizada. Una autorización genérica
  de integrar no habilita tocar `smartenergy_dev` ni el laboratorio protegido.

## S09 — Aceptación del mínimo y documentación operativa

- **Objetivo:** cerrar una herramienta útil con límites y operación reproducible.
- **Alcance:** revisión integral de escenarios y contratos implementados, evidencia
  de aceptación y documentación de ejecución segura, diagnóstico y limitaciones.
- **Entregables:** catálogo final con cobertura F01–C01, resultados por criterio,
  instrucciones, decisiones/desviaciones y pendientes actualizados en documentos
  existentes; declaración separada de niveles 1/2/3 y capacidades efectivamente
  verificadas. No crear informes redundantes ni declarar capacidades futuras terminadas.
- **Dependencias:** S08 aceptada y aprobación de S09.
- **Aceptación:** todos los criterios del mínimo aprobados o impedimentos declarados
  que bloqueen cierre; referencia física recuperable y reproducible; cambios limitados
  al alcance, sin secretos ni datos/artefactos locales versionados. Revisión humana
  del resultado; simulación no acredita funcionamiento con equipos reales.
  Un cierre limitado a niveles 1/2 debe declarar persistencia no validada; si el
  alcance aprobado exige nivel 3, su ausencia o fallo bloquea ese cierre.
- **Riesgos:** declarar preparación industrial o extender la herramienta a un
  producto independiente; confundir aceptación del simulador con la del detector ML.
- **Aprobación humana:** cierre del mínimo y aceptación de limitaciones. Commit,
  push y cualquier ampliación requieren autorización expresa independiente.

## Mínimo y ampliaciones

El mínimo son las topologías y magnitudes de S02, las tres capas, todos los escenarios
F01–C01, referencia verificable, determinismo y una integración acotada aprobada.
La declaración de cierre debe indicar niveles demostrados: una integración de
nivel 2 no permite afirmar validación de persistencia de nivel 3.
La investigación comparativa de tres equipos no obliga a implementar tres emuladores.
Un perfil genérico puede ser el primer núcleo si no afirma equivalencia instrumental,
conserva la referencia obligatoria ADW210 y satisface el cierre documental S01 o
registra la aprobación expresa de cualquier diferimiento que lo afecte.

Quedan fuera: driver/emulador Modbus, 3P3W completo, exportación de potencia activa, formas de onda,
armónicos/THD, demanda, tarifas y modelos térmicos detallados. Su adopción requiere
necesidad concreta, evidencia y nuevo alcance aprobado, según el contrato.
La adquisición real y arquitectura industrial conservan su aprobación pendiente.

Ante impedimentos, registrar etapa, evidencia, impacto y decisión necesaria;
no avanzar automáticamente ni ejecutar mantenimiento para desbloquear una prueba.
El responsable autoriza la próxima etapa con una petición explícita; este documento
no es una orden de ejecución continua.
