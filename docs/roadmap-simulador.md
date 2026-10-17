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
El 2026-10-17, mediante «SmartEnergy — Cierre formal de S04 y diseño documental
de S05», el responsable aprueba individualmente F-D01–F-D05 y cierra S04 sólo como
especificación documental. Base: `feature/simulador`, HEAD
`a9ae8ef8c7039882271ef2600400db7950986576`, árbol limpio. Autoriza únicamente el
diseño documental S05 en contrato y roadmap. D02/D05 y aceptación futura siguen
pendientes; no hay autorización de implementación ni infraestructura.
El 2026-10-17, mediante «SmartEnergy — Cierre documental de S05 e implementación
controlada de S06», el responsable aprueba I-D01–I-D07 y cierra S05 sólo como
especificación documental. Base: `feature/simulador`, HEAD
`33d6a91726ea61908afb5cbe976cdc89ac65ebb6`, árbol inicialmente limpio.
Autoriza únicamente pasos internos 1–3 de S06: inspección/estructura mínima,
núcleo S02 y pruebas independientes locales. Los pasos 4–8 (escenarios,
instrumentación, adquisición, contrato/composición y pruebas integrales en memoria)
requieren nueva autorización. No habilita infraestructura ni integración.
El 2026-10-17, mediante «SmartEnergy — S06, paso 4: implementación de escenarios
físicos S04», el responsable acepta el incremento inicial y autoriza exclusivamente
el catálogo físico. Base verificada: `feature/simulador`, HEAD
`c456aa09d5278b6d3c0be7fe47d9606aaa215828`, árbol limpio. El paso 4 queda
implementado y sometido a revisión; pasos 5–8 y S07–S09 requieren autorización.
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

| Etapa | Estado tras implementación del paso 4 de S06, pendiente de revisión (2026-10-17) |
|---|---|
| S01 | Cerrada para el alcance documental y núcleo genérico aprobados; D01–D06 registradas, pendientes instrumentales conservados |
| S02 | Cerrada como especificación matemática aprobada E01–E07; núcleo implementado y probado en el incremento S06, sin validación industrial |
| S03 | Cerrada exclusivamente como especificación documental aprobada C01–C08; sin integración ejecutada |
| S04 | Especificación documental aprobada F-D01–F-D05; catálogo físico implementado y probado en S06 paso 4, pendiente de auditoría |
| S05 | Cerrada exclusivamente como especificación documental aprobada I-D01–I-D07; mecanismos sin implementar |
| S06 | Pasos 1–3 aceptados; paso 4 implementado con pruebas unitarias y sometido a revisión; pasos 5–8 pendientes de autorización |
| S07–S09 | No iniciadas; validación integral, integración y cierre no autorizados |

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

### Resultado de S04 — cierre documental aprobado

La [especificación de escenarios S04](contratos/simulador-fase-1.md#s04--escenarios-físicos-y-perfiles-aprobados)
conserva las familias F01–F05. S04.1 fija defaults, identidades, versiones,
condiciones iniciales y manifiesto reproducible; S04.2 define casos con U/Y
prescritas y resultados independientes; S04.3 propone condiciones sintéticas,
sin convertirlas en umbrales industriales. S04.4 incluye ventanas con transición
en inicio/final/interior y varias transiciones, cambio de signo Q, reposo y parada;
S04.5 define rechazo y tolerancias heredadas E06. S04.6 fija linaje y matriz de
cobertura; S04.7 registra la aprobación explícita **F-D01–F-D05 del 2026-10-17** y el
cierre únicamente documental de S04, sin acreditar pruebas ejecutadas.

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

Decisiones aprobadas expresamente el 2026-10-17: F-D01 catálogo/defaults/manifiesto;
F-D02 condiciones y umbrales de ensayo; F-D03 ventanas/oráculos/rechazos; F-D04 identidad/calidad y
separación de verdad/diagnóstico; F-D05 cobertura/límites/aceptación futura.
Cada una conserva fundamento, alternativas, consecuencias y riesgos como
registro de la aprobación humana; ésta no autoriza ejecución ni infraestructura.
La matriz identifica pendientes de aprobación/ejecución de S05–S09 y mantiene D02/D05.

Verificación de esta entrega: cálculos independientes no destructivos (fracciones,
productos e integrales analíticas y fasores), coherencia con S02/S03, IDs,
enlaces/anclas, diff y alcance de dos archivos. Las secciones S02.1–S02.9 y la
matriz instrumental se conservan sin cambios. No se ejecuta simulador, servicios,
generadores, PostgreSQL, Docker, ML o pruebas de aplicación; ningún nivel de
aceptación queda demostrado por estas cuentas documentales.
**S01 cerrada con diferimientos; S02 aprobada; S03 aprobada y cerrada documentalmente;
S04 aprobada y cerrada sólo documentalmente; S05 propuesta pendiente de aprobación;
S06–S09 no iniciadas.**
No se crean archivos ni se hace commit/push. La instrucción vigente autoriza
expresamente el diseño S05 siguiente; no es una consecuencia automática del cierre.

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

### Resultado de S05 — cierre documental aprobado

La [especificación S05](contratos/simulador-fase-1.md#s05--instrumentación-y-adquisición-aprobadas)
conserva los IDs y criterios originales: I01 ruido/error acotado, I02 sesgo/deriva/
descalibración, I03 congelación/intermitencia, I04 CT/VT incorrecto; A01 pérdida/
retraso/duplicación, A02 caída/recuperación. Fuentes primarias, fronteras conceptuales
y manifiesto están en S05.1; composición/coherencia en S05.2; dominios y evidencia
de calidad en S05.3; semillas y aceptación estadística en S05.4; adquisición y
recuperación en S05.5; matriz de casos/oráculos en S05.6.

La física S02–S04 permanece inmutable. Se propone instrumentar canales primarios
y derivar potencias coherentemente, separando errores explícitos de resultado;
secundario real, CT/VT configurado, original y canónico no son intercambiables.
Orden: transformación secundaria → ganancia/sesgo/deriva/ruido → escala configurada
→ derivados/contador instrumental → fallo de resultado declarado → cuantización
→ retención/intermitencia → normalización única → adquisición. El detalle y sus
consecuencias se fijan en S05.2, sin crear clases ni cambiar responsabilidades S03.

Se priorizan calendarios deterministas y cuentas analíticas. El perfil uniforme
propone MT19937 local, SHA-256 para semillas separadas y n=10000 con bandas de
media, segundo momento, fracción negativa y diez intervalos, riesgo total ideal de falso rechazo
≤0,001. E06 sigue siendo tolerancia numérica, no dispersión instrumental. La
calidad expresa sólo evidencia observable; repetición no prueba congelación.
Se explicitan descarte, FIFO acotada, reintentos, tardíos, duplicado y conflicto;
no se prometen durabilidad ni entrega exactamente una vez.

**C01-MAS@1 (S05.7):** F02-SOB + I03 + A01, conforme al catálogo original;
no es F02-F03-JUN ni decisión S03:C01. Sobrecarga física de 60 s, energía de 60 Wh;
12 observaciones I, seis retenidas de t=20, tres pérdidas y nueve recepciones,
todas de 10 A. Con soporte antiguo, tres recibidas son stale; con reloj aparentemente
actual puede no haber diagnóstico. Todas las causas permanecen en el artefacto
separado; no etiquetas Anomalo ni evaluación ML.

**I-D01–I-D07 aprobadas expresamente el 2026-10-17**, con recomendación, fundamento, alternativas, riesgos,
consecuencias y pendientes individuales en S05.8: modelos/parámetros; composición;
RNG/estadística; congelación/calidad; adquisición/recuperación; combinado/linaje;
cobertura/aceptación. **S05 cerrada exclusivamente como especificación documental**
por la instrucción S05/S06 identificada al inicio. Se conservan los
criterios originales de cada familia, la cobertura física previa y D02/D05.

Verificación documental de la entrega original de diseño S05: fórmulas y momentos teóricos, bandas,
recuentos de observaciones/intentos/entregas, calendarios, identidad/soporte/calidad,
correspondencia I01–I04/A01–A02/C01, referencias y enlaces/anclas, alcance exclusivo
de contrato/roadmap y git diff --check. No se ejecuta simulador, generadores,
servicios, PostgreSQL, Docker, migraciones, ML ni pruebas de aplicación. Ningún
nivel 1/2/3 queda demostrado. No hay código, archivos adicionales, commit ni push.
La aprobación posterior I-D01–I-D07 cierra esa especificación sin atribuirle
ejecución ni validación. **S01–S05 aprobadas para sus alcances documentales, con
diferimientos S01 conservados; S06 iniciada únicamente para el núcleo S02;
S07–S09 no iniciadas.** Los resultados del incremento se detallan a continuación.

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

### Resultado de S06 — núcleo matemático aislado (pasos 1–3)

Implementación iniciada con autorización S05/S06 del 2026-10-17. Se completan sólo
inspección/estructura, núcleo S02 y pruebas independientes. No se cierra S06 ni se
inician sus pasos 4–8 o S07–S09. Ubicaciones conforme a S03.7, sin reorganizar el
repositorio ni reutilizar generadores con efectos de persistencia:

| Archivo | Interfaz y responsabilidad |
|---|---|
| [`simulator_physics.py`](../smartenergy/analysis/simulator_physics.py) | `CircuitState(topology, voltages_v, admittances_s, nominal_voltage_v=120, frequency_hz=60)` valida U/Y y calcula `PhaseState`, fasores ABC, I/P/Q/S/FP, sumas aritméticas/algebraicas, neutro y tensiones L-L. Dataclasses congeladas y entradas convertidas a tuplas; temperatura None, sin observaciones/diagnósticos |
| [`simulator_energy.py`](../smartenergy/analysis/simulator_energy.py) | `PowerInterval(start_s, end_s, left, right)` declara P/Q lineales o constantes en [a,b); `right` es el límite anterior al salto final. `integrate_intervals(..., start_s=..., end_s=...)` calcula incrementos por fase/circuito, con cortes de signo y ventanas. `integrate_smooth(state_at, start_s, end_s, subintervals)` aproxima un tramo suave con trapecios. `EnergyCounters`/`CircuitEnergy` mantienen alcances separados, saldos y acumulación inmutable con `plus` |
| [`test_simulator_physics.py`](../tests/test_simulator_physics.py) | Oráculos S02:C01–C14 con referencias analíticas independientes; rechazos, reproducción, inmutabilidad y seguridad de importación/cálculo sin dependencias externas |

Sólo biblioteca estándar Python; entorno verificado `.venv`, Python 3.13.5.
No hay componentes existentes compatibles para reutilizar cálculos fasoriales o
contadores. Los módulos matemáticos se separan por responsabilidad y permanecen
independientes de configuración, aplicación, ORM y ML; no se crea paquete genérico,
servicio, DTO o adaptador. El orden L1/L2/L3 de resultados y contadores es explícito.

Invariantes: consumidor G≥0, Q con signo, S aritmética y FP=P/S (None sólo con
denominador cero); U/I/f y duración en dominios S02.6, sin saturar. Los tramos
deben ser contiguos y ordenados; se rechazan huecos, solapes y duración cero.
Una consulta a=a dentro del soporte devuelve incremento cero. Los contadores
del circuito separan signos después de sumar P/Q, nunca después de integrar fases.
Los iniciales se añaden explícitamente con `plus`, conservando el incremento para
validarlo aun con grandes valores iniciales. No hay reset/rollover implícito.

La linealidad P/Q de `PowerInterval` es una hipótesis explícita del llamador;
no se deduce de interpolar U/I. Para perfiles suaves, el llamador debe separar
discontinuidades, proporcionar el límite izquierdo final y garantizar el dominio
entre nodos; el muestreo no demuestra esas propiedades. El catálogo de escenarios
que resolverá esos perfiles sigue sin implementar. Underflow que convertiría
una carga o incremento no nulo en cero se rechaza; no se usa E06 como umbral de
ausencia. Desbordamientos fuera del dominio numérico tampoco se saturan.

Validación ejecutada, exclusivamente sobre el nuevo código:

```bash
.venv/bin/python -B -m unittest -v tests.test_simulator_physics
git diff --check
```

**23 pruebas aprobadas, cero fallos.** C01–C06: ramas, signos, equilibrio,
desequilibrio, fasores y reposo; C07–C10: integración constante/por tramos,
cambio reactivo, contadores iniciales y reanudación; C11: compensación por fases
frente a circuito; C12: áreas direccionales del cruce lineal; C13: integral
1400 Wh, aproximaciones 1400,01/1400,0025/1400,000625 y cociente de errores 4,
último error dentro de 0,0024 Wh; C14: Y fija bajo subtensión/interrupción y
reactancia pura. E06 se conserva, con controles relativos y de signo para ambos
centinelas de baja potencia, además de ventanas en fronteras, rampas, varias
transiciones y conservación del incremento con grandes iniciales.

La prueba en intérprete separado bloquea dependencias de infraestructura/ML,
configuración y operaciones, acceso a `.env`, escrituras, red, procesos, reloj
time.time y sleep; importar y calcular no altera el RNG global. Verificación
documental de enlaces locales, alcance y conservación de S02.1–S02.9/S03.1–S03.8
y matriz instrumental. Evidencia parcial del núcleo para nivel 1; no cierre del
nivel completo, integración de nivel 2, persistencia de nivel 3 ni ensayo industrial.

Pendientes: catálogo S04, mecanismos S05, contrato/composición, manifiestos,
linaje y pruebas integrales, con nueva autorización antes del paso 4. Siguen
pendientes canonicalización e integridad/retención S03 y D02/D05; no bloquean
este cálculo matemático sin artefactos persistidos. No se modificó infraestructura,
persistencia, configuración, servicios, API, frontend o modelos; no se ejecutaron
generadores legados, PostgreSQL, Docker ni suite general. Sin commit ni push.
Se detiene aquí para revisión humana.

### Resultado de S06 — catálogo físico S04 (paso 4)

Implementado exclusivamente el incremento autorizado, sin modificar las interfaces
ni los archivos del núcleo S02. Archivos nuevos:
[`simulator_scenarios.py`](../smartenergy/analysis/simulator_scenarios.py) y
[`test_simulator_scenarios.py`](../tests/test_simulator_scenarios.py).
El resultado queda sometido a auditoría independiente y aprobación humana;
no cierra S06 ni inicia S07. El apartado de pasos 1–3 conserva su estado histórico.

`SCENARIO_IDS` y `build_scenario(id)` exponen 21 definiciones inmutables con
parámetros fijos, sin overrides bajo un mismo ID@1. Incluyen los trece casos
F01-REP/EST/ESC/RAM/Q/TUR, F02-SOB, F03-SUB, F04-CAR/TEN/COM,
F05-ARR y F02-F03-JUN. Variantes: REP-24h con E0; EST inductiva,
capacitiva y 3P4W (la base EST es resistiva); SUB-interrupcion. Los sufijos
locales ESC-C09/ESC-C10 y Q-C13 identifican los controles S02 citados por S04,
conservando sus entradas y soportes. Los ejes reactivos C14 siguen cubiertos
por las pruebas S02; no se inventa un soporte temporal nuevo para ellos.

Interfaces físicas: `PhysicalSegment`, `PhysicalScenario`, `PhysicalEvent`,
`PhysicalCondition` y `sustained_condition`. Los tramos mantienen U fija e
interpolan exclusivamente Y; el control C13 utiliza Y cuadrática. Los extremos
válidos y la convexidad de la trayectoria garantizan el límite de corriente entre
nodos. `state_at(t, side="left"/"right")` distingue el límite anterior de una
frontera y su estado nuevo; `segment_at` permite consultar el estado operativo.
En t=0 se devuelve el inicial definido y en T el límite izquierdo terminal;
fuera de [0,T] se rechaza. `energy(a,b)` exige ventana positiva y devuelve
incrementos por fase/circuito; `counters_at(t)` añade E0 sin mutarlo y permite
t=0. Se reutilizan `PowerInterval`, `CircuitEnergy`, `integrate_intervals` e
`integrate_smooth`, sin otro motor eléctrico ni de integración. C13 requiere
considerar el error de discretización; N=400 por defecto por tramo recortado,
con N explícito para comprobar convergencia. No promete aditividad exacta entre
discretizaciones distintas de ese control suave.

Eventos: inicio/fin, fronteras de perfil, cambios de fuente/carga/operación y
cruce reactivo continuo. Los cambios simultáneos de fuente/carga de JUN se
agrupan en una frontera; no se crea un estado intermedio. Condiciones conocidas
por el ensayo: I_L1>15 A y U_L1N<108 V en sus casos prescritos, dispersión d_I>0,20
y d_U>0,10 en los casos resistivos F04. Exigen al menos 10 s continuos;
conservan inicio físico, inclusión del punto inicial y tiempo de confirmación.
No se aplica F02 automáticamente a la envolvente ARR ni se deriva un diagnóstico
de las condiciones o de los estados operativos.

Verificación ejecutada con el entorno existente, sin dependencias nuevas:

```bash
.venv/bin/python -B -m unittest -v tests.test_simulator_physics tests.test_simulator_scenarios
git diff --check
```

**40 pruebas aprobadas: 23 S02 conservadas y 17 nuevas, cero fallos.** Cada
identidad del catálogo tiene contraste energético independiente. Oráculos S04:
ESC=(250,0,150,75); RAM=35 Wh; Q=225 varh por dirección; TUR=(6010,0,2700,1807,5);
SOB=60 Wh; SUB=25/20 Wh; ARR=88/3 Wh y 22 varh; JUN=40 Wh.
CAR verifica I_N=10√3, TEN I_N=5/3 y tensiones L-L; COM verifica
P=3120,Q=0,S_A=3600,FP=13/15 y contadores reactivos por fase distintos del circuito.
Se verifican variantes estables, iniciales no nulos, reposo/reanudación, ventanas
S04.4, ambos lados de fronteras, continuidad ARR, Q=0, desigualdades estrictas,
9/10 s, reproducción, inmutabilidad, rechazos e invariantes por fase/agregados.
C13 contrasta 1400 Wh con errores 0,01/0,0025/0,000625 y convergencia de orden 2.

La prueba de aislamiento en intérprete separado importa y consulta todo el
catálogo bloqueando configuración, infraestructura, operaciones, aplicación,
ML, escrituras, red, procesos y reloj time.time/sleep; no altera el RNG global.
No se ejecutaron servicios, generadores, PostgreSQL, Docker, frontend, API,
migraciones ni ML. El laboratorio protegido permanece intacto. Sin commit/push.

Límites: definiciones y resultados sólo en memoria; no hay DTO de observaciones,
manifiesto/run_id, canonicalización, digests, artefactos persistidos ni mecanismo
S05. Los IDs de sujetos son los defaults S04; la composición y el linaje completo
S03 siguen pendientes. La cadencia h=1 s de futuros estados de ensayo no es un
reloj ni una adquisición implementada. S01–S05 conservan su aprobación documental
y diferimientos; pasos 5–8 requieren autorización; S07–S09 no iniciadas. Esta
prueba unitaria aporta evidencia física parcial de nivel 1; no declara aceptación
del nivel completo, integración ni validación industrial. Se detiene en el paso 4.

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
