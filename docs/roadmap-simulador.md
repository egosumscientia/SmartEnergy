# Hoja de ruta del simulador de SmartEnergy

Fecha: 2026-10-17. Rige el [contrato de fase 1](contratos/simulador-fase-1.md),
junto con [AGENTS.md](../AGENTS.md) y la [guía](guia-de-ingenieria.md).
Este roadmap desarrolla el [pendiente de adquisición y simulador](roadmap-tecnico.md#i-pendiente-técnico-adquisición-y-simulador);
no altera los cuatro prompts P01–P04 ni los 16 controles de reestructuración.

## Estado y secuencia de aprobación

La fase documental y su corrección por auditoría quedaron publicadas en
`ed2e5d8219a21e2bd29235c7140327ff5b702662`. La intervención posterior autorizó
exclusivamente la investigación S01, iniciada en `feature/simulador` con árbol
limpio y publicada en `d84dc1fe48e86a14fcccf10ae183ee16f305c602`.
El 2026-10-17 el responsable aprobó D01–D06 y autorizó expresamente el cierre
documental S01 y el diseño S02 mediante «SmartEnergy — Cierre formal de S01 y
ejecución de S02». Esta intervención partió de ese commit, en `feature/simulador`
con árbol limpio. La matriz conserva la evidencia y los diferimientos aprobados.
El 2026-10-17, mediante «SmartEnergy — Cierre formal de S02 y ejecución
documental de S03», el responsable aprobó E01–E07 y autorizó el cierre formal
S02 y el diseño documental S03. Base verificada: rama `feature/simulador`, HEAD
`0bde38154ce9630c93f6cd6c38670d4111cad29e`, árbol limpio. Se modifican únicamente
contrato y roadmap, sin ejecución de código ni recursos persistentes.
El 2026-10-17, mediante «SmartEnergy — Cierre formal de S03 y diseño documental
de S04», el responsable aprueba C01–C08 (con la corrección S03.5 ya incorporada)
y cierra S03 exclusivamente como especificación documental. Autoriza sólo diseñar
S04 en contrato y roadmap. Base: `feature/simulador`, HEAD
`024d820f784aa38e321cf5970deb78a4bd838c39`, árbol limpio. Se conservan D01–D06,
E01–E07 y todos los diferimientos instrumentales; no se autorizan implementación,
servicios, bases, interfaces, ML ni integración persistente.
No se considera aceptada una etapa futura por figurar aquí. La aprobación del
responsable debe identificar etapa, alcance, entregables, criterios y efectos
permitidos; ninguna etapa de implementación comienza sin aprobación explícita.
No hay fechas ni estimaciones de esfuerzo asignadas.

Secuencia: S01 → S02 → S03 → S04 → S05 → S06 → S07 → S08 → S09.
El núcleo inicial puede ser genérico, pero S01 conserva la investigación obligatoria
del ADW210 y el contraste documental del apartado I del roadmap técnico. S02 puede
preparar el núcleo genérico con los diferimientos D02/D05 aprobados, sin aprobar
emulación ni dar requisitos instrumentales pendientes por cumplidos. Todo nuevo
diferimiento deberá documentarse y aprobarse según el contrato;
la emulación ejecutable de registros puede diferirse sin omitir su revisión documental.
S07 verifica físicamente el núcleo
antes de integrarlo; S06 ya incluye pruebas unitarias, no se posponen hasta S07.
Cada etapa se detiene al concluir su alcance y presenta evidencia para revisión.

| Etapa | Estado tras cierre documental S03 y propuesta S04 (2026-10-17) |
|---|---|
| S01 | Cerrada para el alcance documental y núcleo genérico aprobados; D01–D06 registradas, pendientes instrumentales conservados |
| S02 | Cerrada como especificación matemática aprobada E01–E07; sin implementación ni validación experimental |
| S03 | Cerrada exclusivamente como especificación documental aprobada C01–C08; sin integración ejecutada |
| S04 | Escenarios y perfiles especificados; pendiente de revisión y aprobación F-D01–F-D05; no cerrada |
| S05 | No iniciada; diseño y autorización pendientes |
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

### Resultado de S01 — cierre documental aprobado

Se ampliaron las [fichas instrumentales](instrumentacion/matriz-medidores.md#fichas-de-evidencia-s01-por-instrumento)
y se contrastaron capacidades/mapas por variable. A3/A4 aportan configuración
específica ADW210; A2 conserva nivel de familia. S5 permitió inspeccionar el XLS
PM5310, su aplicabilidad por modelo, cadencia del bloque básico y valores N/A.
Se documentó la discrepancia de revisión de E4 y se conservó E1 como referencia.

El [resultado por criterio](instrumentacion/matriz-medidores.md#resultado-de-s01-y-criterios-de-aceptación)
distingue evidencia disponible de pendientes. El responsable aceptó su suficiencia
para el núcleo genérico y aprobó referencias/límites/diferimientos D01–D06.
Firmware/mapa ADW210, detalles de CT/VT y convenciones no resueltas siguen
bloqueando equivalencia instrumental y deben resolverse antes del adaptador
específico. No se los declara cumplidos por aprobar un núcleo genérico.

**Decisiones aprobadas:**
[D01–D06](instrumentacion/matriz-medidores.md#c-decisiones-d01d06-aprobadas-y-diferimientos):
referencia ADW210 3P4W con tensión común, contraste Eastron directo/PM5310 B1,
núcleo genérico sin equivalencia de marca y diferimientos acotados de decodificación
y metrología. D04 encarga signos/energía a S02 y D06 reserva representación y
compatibilidad para S03. **S01 cerrada formalmente para ese alcance documental;
S02 autorizada sólo como diseño, sin aprobación anticipada de sus decisiones.**

**Impacto en el apartado I del roadmap técnico:** la referencia documental queda
elegida, pero la asociación firmware/mapa y detalles instrumentales de su primer
criterio siguen pendientes por D02. D05 difiere equivalencia metrológica específica;
D06 asigna representación/compatibilidad a S03. Fixtures comunes, coherencia
implementada, escenarios y pruebas de integración siguen en sus etapas futuras.
Se conserva el apartado I íntegro: aprobar diferimientos no satisface esos requisitos.

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

### Resultado de S02 — cierre formal aprobado

La [especificación matemática](contratos/simulador-fase-1.md#especificación-matemática-s02--aprobada)
contiene catálogo con dominios/nulabilidad, fuente ideal y cargas de admitancia
prescrita, relaciones fasoriales 1P2W/3P4W, agregados, neutro y límites de √3.
Aprueba S aritmética, FP con signo activo, Q inductiva/capacitiva, cuatro contadores
separados por alcance y tiempo físico relativo. Distingue integración exacta por
tramos de aproximación y error numérico; temperatura auxiliar indisponible.

Se investigaron fuentes primarias M1–M5, contrastadas con la matriz S01 sin
extender evidencia de fabricante. El contrato presenta C01–C10 obligatorios y
C11–C14 de compensación, cruce por cero, convergencia y límites, más el caso
centinela próximo a cero. Los oráculos son analíticos e independientes de un
futuro motor. Recomendación nominal: 120 V L-N y 60 Hz, con alternativas y
dominios sintéticos; no son umbrales industriales.

**Aprobación individual registrada el 2026-10-17:** E01 topologías 1P2W/3P4W,
fuente/neutro ideales y admitancia prescrita; E02 P hacia carga, Q+ inductiva/Q−
capacitiva y FP=P/S; E03 S aritmética, FP agregado P/S y Q algebraica; E04 intervalos
semiabiertos y contadores direccionales por alcance; E05 120 V L-N/60 Hz y dominios;
E06 tolerancias/convergencia de S02; E07 temperatura auxiliar indisponible.
La instrucción S02/S03 identificada al inicio aprueba estas decisiones y cierra
**S02 sólo como especificación matemática aprobada**, sin implementación ni
validación experimental. Las alternativas y límites de S02.9 se conservan.
D01–D06 y sus pendientes instrumentales mantienen su alcance; aprobar matemáticas
no demuestra equivalencia instrumental ni habilita S04 o implementación.

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

### Resultado de S03 — cierre documental aprobado

La [especificación S03](contratos/simulador-fase-1.md#6-contrato-común-y-compatibilidad--s03-aprobada)
contiene inventario verificado y matriz por componente (S03.1), representación
híbrida y catálogo de identidades/magnitudes (S03.2), cuatro dimensiones de calidad
(S03.3), tiempos/entregas/deduplicación (S03.4), ejemplos completos por envelope y
herencia explícita de campos (S03.5), artefacto físico separado y proyección legada
con pérdidas/temperatura/etiquetas (S03.6), ubicaciones futuras y aceptación por
niveles con verificación de campos en PostgreSQL aislado (S03.7).

Decisiones individuales **C01–C08 aprobadas el 2026-10-17** (S03.8): representación y
versionado; identidad/alcance; calidad/procedencia; tiempo; deduplicación;
artefacto físico; compatibilidad/ML; integración y aceptación. Cada una incluye
fundamento, alternativas, riesgos y consecuencias conservados como registro.
La instrucción S03/S04 identificada al inicio cierra S03 sólo documentalmente.
Siguen pendientes canonicalización definitiva antes de implementarla, garantías
de deduplicación concurrente, retención/recuperación de artefactos, evidencia de
unidades/zonas históricas, adaptadores, persistencia, pruebas y autorizaciones de
integración por nivel. Aprobar el contrato no demuestra esas capacidades ni
habilita cambios de esquema, HTTP o ML.

Hallazgos de compatibilidad: Registro carece de identidad/calidad/tiempos completos;
HTTP/CSV no transportan el nuevo contrato; unidades históricas sin confirmar;
temperatura null conserva honestidad, pero deja Comparativos sin puntos completos
y bloquea evaluación del modelo de cuatro variables. El ML administrativo no debe
invocarse sobre esa proyección. Sin etiquetas diagnósticas aprobadas, Sin estado
no es verdad negativa aunque la normalización histórica produzca label_anomalia=0.
Una fila multifase ambigua queda rechazada; persistir sólo Registro no acredita
nivel 3. Reversión propuesta conserva datos/modelos y ruta legada.

Verificación documental: contraste estático de consumidores reales, dimensiones de
calidad, unidades/alcances, soporte de ventanas y ejemplos, trazabilidad y pérdidas;
enlaces y referencias locales, diff y alcance exacto de dos documentos. No se
implementó código ni se ejecutaron servicios, generadores, PostgreSQL, Docker o
pruebas de aplicación. No hay validación experimental, persistente ni industrial.
**S01 y S02 cerradas para sus respectivos alcances; S03 aprobada y cerrada sólo
documentalmente.** El diseño S04 se autoriza expresamente por la instrucción
actual; no es consecuencia automática de ese cierre. Sin commit ni push.

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

### Resultado de S04 — propuesta documental pendiente de aprobación

La [propuesta de escenarios S04](contratos/simulador-fase-1.md#s04--escenarios-físicos-y-perfiles-propuestos)
conserva las familias F01–F05. S04.1 fija defaults, identidades, versiones,
condiciones iniciales y manifiesto reproducible; S04.2 define casos con U/Y
prescritas y resultados independientes; S04.3 propone condiciones sintéticas,
sin convertirlas en umbrales industriales. S04.4 incluye ventanas con transición
en inicio/final/interior y varias transiciones, cambio de signo Q, reposo y parada;
S04.5 define rechazo y tolerancias heredadas E06. S04.6 fija linaje y matriz de
cobertura; S04.7 somete **F-D01–F-D05 a aprobación**, sin cerrar S04.

Catálogo: F01-REP/EST/ESC/RAM/Q/TUR (reposo, estable resistivo/inductivo/capacitivo,
escalón, rampa, cruce reactivo y turno); F02-SOB (sobrecarga relativa a capacidad
de ensayo); F03-SUB (subtensión con Y fija e interrupción ideal); F04-CAR/TEN/COM
(desequilibrio de carga/tensión y compensación reactiva por fases); F05-ARR
(envolvente de arranque/parada); F02-F03-JUN (combinación física compatible, sin
fallos I/A). Topologías 1P2W y 3P4W, referencia 120 V L-N/60 Hz, temperatura
indisponible. No equivalen a motor dinámico ni perfiles de una instalación real.

Los oráculos reutilizan S02:C01–C14 cuando corresponden y añaden áreas e integrales
analíticas con sumas fasoriales. Se distinguen de decisiones S03:C01–C08 y F-D01–F-D05.
Se especifican contadores por alcance, ventanas y transiciones, incluida Q negativa
válida con P no negativa; condición física, operación, diagnóstico y resultado ML
permanecen separados. No hay etiquetado Anomalo automático.

Decisiones pendientes: F-D01 catálogo/defaults/manifiesto; F-D02 condiciones y
umbrales de ensayo; F-D03 ventanas/oráculos/rechazos; F-D04 identidad/calidad y
separación de verdad/diagnóstico; F-D05 cobertura/límites/aceptación futura.
Cada una registra fundamento, alternativas, consecuencias, riesgos y aprobación
necesaria. La matriz identifica pendientes de S05–S09 y mantiene D02/D05.

Verificación de esta entrega: cálculos independientes no destructivos (fracciones,
productos e integrales analíticas y fasores), coherencia con S02/S03, IDs,
enlaces/anclas, diff y alcance de dos archivos. Las secciones S02.1–S02.9 y la
matriz instrumental se conservan sin cambios. No se ejecuta simulador, servicios,
generadores, PostgreSQL, Docker, ML o pruebas de aplicación; ningún nivel de
aceptación queda demostrado por estas cuentas documentales.
**S01 cerrada con diferimientos; S02 aprobada; S03 aprobada y cerrada documentalmente;
S04 especificada, pendiente de revisión y aprobación; S05–S09 no iniciadas.**
No se crean archivos ni se hace commit/push ni se avanza a la siguiente etapa.

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
