# Simulador — Fase 1: contrato de ingeniería

Fecha: 2026-10-17. Base inspeccionada: `7656863e434e42fd14ec8719ad940fb8643e9e6d`,
rama `feature/simulador`. Esta entrega establece requisitos documentales;
su revisión y la aprobación de las etapas futuras corresponden al responsable.
No acredita implementación ni autoriza escrituras, migraciones o adquisición real.

Lecturas complementarias obligatorias: [AGENTS.md](../../AGENTS.md),
[guía de ingeniería](../guia-de-ingenieria.md), [arquitectura](../arquitectura.md),
[producto](../producto.md), [contrato HTTP](../contrato-http.md) y
[roadmap del simulador](../roadmap-simulador.md). La evidencia de instrumentos
se concentra en la [matriz](../instrumentacion/matriz-medidores.md).

## 1. Propósito, alcance y autoridad

El simulador es una herramienta interna para desarrollar y verificar persistencia,
consulta, cálculos, análisis, anomalías e interfaces de SmartEnergy. Debe permitir
contrastar cada resultado contra entradas físicas conocidas y fallos controlados.
No es un producto de simulación eléctrica independiente ni sustituye ensayos reales.

El mínimo propuesto comprende circuitos 1P2W y 3P4W sinusoidales, cargas de consumo,
series temporales de magnitudes eficaces, energía acumulada, escenarios de operación,
fallos de observación y entrega, reproducción y trazabilidad. Los requisitos de
este contrato gobiernan su diseño; el perfil concreto necesita cierre en S02/S03
del roadmap antes de implementar.

Fuera del mínimo: resolución electromagnética, formas de onda, cortocircuitos,
protecciones, electrónica de potencia, transitorios subciclo, emulador Modbus,
drivers físicos, control de cargas, facturación, certificación metrológica,
entrenamiento automático, nuevos servicios o UI administrativa. 3P3W, generación
exportadora, armónicos, demanda y modelos térmicos detallados son ampliaciones.
Sólo se admitirán con un caso concreto de SmartEnergy, evidencia, impacto,
criterios verificables, compatibilidad y autorización expresa.

La referencia ADW210 y la independencia de fabricante ya constan en arquitectura.
Este contrato no aprueba la arquitectura industrial ni reabre P01–P04 o sus
controles B00–B11/H00–H03. El nuevo roadmap desarrolla su pendiente I.

El primer núcleo puede ser genérico e independiente del fabricante; esto no
elimina al ADW210 como referencia instrumental obligatoria. S01 debe seleccionar
la referencia documental/configuración e investigar sus capacidades relevantes,
contrastando por variable alcance, unidades, escalas y disponibilidad con manual
y mapa, conforme al [roadmap técnico, apartado I](../roadmap-tecnico.md#i-pendiente-técnico-adquisición-y-simulador).
La implementación de una emulación específica de registros Modbus puede diferirse;
no debe confundirse con omitir su contraste documental. Todo diferimiento deberá
registrar qué se difiere, motivo, evidencia, impacto, dependencia de cierre y
aprobación humana. Si afecta un requisito documental previo, deberá identificarlo
expresamente y obtener aprobación de esa desviación antes de avanzar; no se
considerará cumplido. Esta corrección no aprueba ningún diferimiento ni inicia S01.

## 2. Estado real y brechas comprobadas

Inspección estática, sin importar ni ejecutar generadores:

| Componente existente | Capacidad observada | Brecha relevante |
|---|---|---|
| `smartenergy/operations/simulate_data.py` | Lote de 10.000 muestras a separación de 1 s, V/I/temperatura aleatorias, P calculada inicialmente con factor variable; etiquetas inyectadas | Usa `random` y NumPy globales sin semilla de escenario; algunas perturbaciones rompen la relación inicial de P; sin topología ni energía |
| `smartenergy/operations/stream_realtime.py` | `build_sample` devuelve `Registro`; bucle configurable | Probabilidades independientes de eventos: `anomaly_ratio` no es porcentaje final. Recalcula P con factor 0,9 y valor absoluto; no modela persistencia física entre muestras |
| Ambos generadores | `create_all`, inserción, commit y borrado por retención, por defecto 20.000 registros | No son motores puros; su ejecución afecta datos. El batch usa hora local naive; el continuo programa marcas dentro del segundo y espera por ciclo, sin garantía de cadencia uniforme |
| `smartenergy/infrastructure/models.py` | `Registro`: id, timestamp, corriente, voltaje, potencia_activa, temperatura_motor, estado; `Metrica`: resumen ML | Sin instalación/dispositivo/circuito/fase, unidad, energía, calidad, procedencia ni enlace a verdad física. `DateTime` sin `timezone=True`; no inferir UTC de registros históricos |
| `application/monitoring.py`, `infrastructure/persistence.py` | Consulta y preparación, historial ML opcional | No implementan adquisición ni normalización de instrumentos |
| `analysis/measurements.py`, `application/model_jobs.py`, `analysis/training.py` | Orden fijo de cuatro variables; `estado` origina etiqueta binaria; bundle/modelo opcional | La etiqueta histórica mezcla causas; no es verdad física multiclase. Temperatura no puede desaparecer sin evaluar consumidores y modelo |
| `interfaces/api/schemas.py`, `routes.py` y contrato HTTP | GET de monitoreo/CSV, nulos, filtros y cuatro variables | Unidades todavía no aprobadas (`chart.unit` nulo); timestamps naive se conservan sin presumir zona; sin contrato común multicanal |
| `tests/test_simulate.py` y pruebas de entradas | Tipo, rangos, etiquetas y efectos simulados | No acreditan relaciones eléctricas, integración temporal ni equivalencia metrológica |

El factor `/1000` de los generadores sugiere kW, pero no autoriza asignar unidades
a todo dato histórico. No se cambiarán esos comportamientos durante esta fase.
No se halló otro archivo de instrucciones para agentes en el árbol inspeccionado
(excluidos `.git` y `.venv`); AGENTS y la guía son compatibles con este alcance.

## 3. Modelo eléctrico que deberá respetarse

Esta sección expresa relaciones y condiciones; no prescribe ecuaciones ejecutables
ni algoritmos. Las convenciones siguientes son requisitos del diseño sintético,
no reinterpretaciones del ORM ni características de un medidor.

| Magnitud | Unidad canónica propuesta | Alcance y regla |
|---|---|---|
| Tensión eficaz | V | Identificar L1-N/L2-N/L3-N o par L1-L2/L2-L3/L3-L1; nunca «voltaje total» ambiguo |
| Corriente eficaz | A | Por conductor/fase; no sumar módulos para inventar corriente total; neutro sólo si se deriva con información suficiente |
| Potencia activa P | W | Por fase y total de circuito; positiva hacia carga, negativa para exportación en una ampliación |
| Potencia reactiva Q | var | Positiva inductiva y negativa capacitiva en perfil consumidor; declarar referencia y método |
| Potencia aparente S | VA | No negativa; método de agregado explícito, sin promediar fases |
| Factor de potencia FP | 1, adimensional | Identificar verdadero o desplazamiento y convención de signo; indisponible si S es cero |
| Frecuencia | Hz | Del suministro/referencia declarada, no suma de frecuencias por fase |
| Energía activa/reactiva | Wh / varh | Distinguir activa importada/exportada y reactiva inductiva/capacitiva; correspondencia con cuadrantes y acumulación pendientes de S02 |
| Temperatura de motor | °C, auxiliar | Sólo con perfil auxiliar declarado; no inferirla de un analizador ni inventarla para completar ML |

Los adaptadores futuros convertirán kW/kWh a unidades canónicas exactamente una
vez, registrando unidad original y método. El perfil legado requerirá autorización
y mapeo independiente; no sustituye al contrato común.

S02 deberá especificar por separado los signos de P y Q, la convención de
cuadrantes y su referencia de flujo, los contadores de energía activa importada
y exportada y los de energía reactiva inductiva y capacitiva. Deberá definir
acumulación, tratamiento de cambios de signo y, si corresponde, separación por
cuadrante. No equiparar exportación activa con comportamiento capacitivo:
una carga consumidora puede mantener P positiva y presentar Q negativa bajo
la convención propuesta. La generación con exportación activa sigue fuera del
mínimo; especificar su contador no autoriza añadir ese escenario.

La correspondencia entre esos conceptos y los registros denominados
importados/exportados, directos/inversos o de cuatro cuadrantes depende de la
fuente y configuración. S02 deberá contrastarla con las fuentes de la matriz,
incluidas las convenciones distintas de FP descritas en S1. No se fija aquí una
codificación universal ni una regla definitiva de acumulación reactiva; cualquier
ambigüedad documental quedará pendiente de investigación y aprobación en S02.

Reglas físicas obligatorias:

- En monofásico, S depende de tensión y corriente eficaces; P depende también del
  FP. En régimen sinusoidal el FP de desplazamiento corresponde al desfase y
  P/Q/S deben formar un triángulo de potencia coherente. Los módulos de P y Q no
  pueden superar S en ese perfil; valores de FP deben estar entre −1 y 1.
- En 3P4W sinusoidal calcular y conservar cada fase. P y Q totales son las sumas
  algebraicas de fases bajo la misma referencia temporal. El mínimo propondrá
  S total como suma de S de fases, identificando el método; no imponer al total
  el triángulo de potencias si hay desfases diferentes entre fases.
- Las simplificaciones con raíz de tres y conversión fija entre tensión L-L y
  L-N sólo son válidas bajo equilibrio y geometría de fases declarados. En
  desequilibrio no se reutilizan sin verificar hipótesis; las tensiones entre
  líneas proceden de relaciones fasoriales, no de sumas de magnitudes RMS.
- Bajo distorsión, FP verdadero y desplazamiento dejan de ser equivalentes;
  no extender el triángulo sinusoidal o la definición de Q sin un método aprobado.
  La semántica PQS y las convenciones distintas de FP se documentan también en
  [S1](../instrumentacion/matriz-medidores.md#fuentes-primarias-consultadas).
- La energía representa acumulación de potencia en el tiempo transcurrido, con
  conversión de segundos a horas y tratamiento del signo. Los contadores
  direccionales acumulativos no decrecen salvo reinicio/rollover registrado;
  distinguirlos de un eventual saldo neto con signo, si S02 lo aprueba. Una pérdida de
  adquisición no detiene la energía física ni autoriza rellenar el hueco con cero.
- Definir si una potencia es instantánea o promedio de intervalo y el soporte
  temporal usado para integrar. Un promedio aritmético de muestras irregulares no
  equivale automáticamente a potencia media temporal.
- Sobrecarga se define respecto de una capacidad nominal del escenario, no de un
  umbral industrial universal. Subtensión y desequilibrio requieren referencia,
  definición y duración. Una carga puede responder manteniendo impedancia,
  corriente o potencia: elegir y declarar el modelo antes de perturbar la tensión.

Perfiles mínimos: reposo, carga estable resistiva/inductiva, rampa, turno con
arranque/parada y carga desigual por fase. Arranque será una envolvente temporal
de magnitudes RMS con duración declarada; no se presentará como onda de arranque
real. La temperatura auxiliar y el estado operativo no deben confundirse con
la etiqueta de anomalía. Nominales, frecuencia 50/60 Hz, potencias, escalas de tiempo
y límites se parametrizarán y aprobarán en S02/S04; no se deducen de los promedios
aleatorios actuales ni se heredan automáticamente de la ficha ADW210.

## 4. Tres capas y referencia verificable

| Capa | Responsabilidad | Referencia conservada |
|---|---|---|
| Realidad física | Evolución de suministro/cargas y eventos eléctricos | Valores verdaderos, estado de carga, parámetros, evento, inicio/fin, circuito y fases afectados |
| Instrumentación | Transformación CT/VT, ventana de medida, sesgo, ruido, deriva, resolución, saturación y fallo de sensor | Valor observado, instante/intervalo de medida, configuración verdadera y configurada, transformaciones y fallos aplicados |
| Adquisición | Muestreo/sondeo, comunicación, demora, pérdida, duplicación, normalización y entrega | Identidad original, intento de entrega, instante de recepción/registro, resultado y causa de ausencia/rechazo |

Flujo lógico: física → instrumentación → adquisición → normalización → aplicación
y persistencia → API → frontend. Son responsabilidades, no autorización para crear
paquetes, capas abstractas o servicios nuevos. La ubicación de código se resolverá
en S03 respetando la arquitectura existente.

Cada ejecución tendrá un manifiesto versionado y cada observación un identificador
estable enlazado a su referencia física, incluso si se entrega varias veces.
El registro de entregas debe permitir reconstruir una relación uno-a-varios y las
pérdidas: un valor perdido no tendrá fila registrada pero sí evidencia de su pérdida.
Un fallo sólo modifica su capa y las posteriores, nunca la verdad física almacenada.
En escenarios combinados se conservarán todos los eventos y su precedencia.

Toda observación derivada de una ventana debe identificar el intervalo físico
de referencia, la operación o transformación aplicada, el instante de observación,
el instante de recepción o registro según corresponda y una referencia verificable
al escenario físico original y su versión. S03 definirá la representación y la
semántica de los límites temporales. Se permiten referencias por intervalo o
segmento que permitan reconstruir la relación; no se exige un enlace por cada
muestra física. Una referencia a un único instante no basta si el valor depende
de una ventana. Las pruebas deberán incluir ventanas que atraviesen transiciones
de operación y verificar el soporte físico y la transformación contra el oráculo
aprobado, además de los tiempos de observación y entrega.

La verdad física y las causas de inyección se conservarán en un artefacto de prueba
separado del conjunto de características del detector. No filtrar etiquetas de
verdad a las entradas de ML. Una anomalía inyectada puede ser invisible en la
observación: evaluar por separado presencia física, observabilidad y detección.

## 5. Catálogo mínimo de escenarios

| ID | Capa | Escenario y resultado verificable |
|---|---|---|
| F01 | Física | Normal estable/reposo y variación de carga; balances y energía coherentes, sin evento físico anómalo |
| F02 | Física | Sobrecarga acotada por magnitud/duración; evento respecto de nominal aprobado |
| F03 | Física | Subtensión; respuesta de carga conforme al modelo elegido y energía coherente |
| F04 | Física | Desequilibrio de tensión y/o carga; fases y definición identificables |
| F05 | Física | Arranque/parada; continuidad y transiciones previstas; no todo arranque es anomalía |
| I01 | Instrumentación | Ruido y error acotados; física invariable, distribución y semilla registradas |
| I02 | Instrumentación | Sesgo, deriva y descalibración; error evoluciona según perfil, sin modificar físico |
| I03 | Instrumentación | Lectura congelada o sensor intermitente; física/energía continúan y se conserva edad real del dato |
| I04 | Instrumentación | Relación CT/VT configurada incorrectamente frente a la real; distinguir primario/secundario y evitar conversión doble |
| A01 | Adquisición | Pérdida, retraso y duplicación; recuentos y linaje exactos; no avanzar tiempo físico por velocidad de entrega |
| A02 | Adquisición | Caída/restablecimiento de comunicación; sin muestras ficticias durante caída; recuperación y cola explícitas |
| C01 | Combinada | Sobrecarga F02 con lectura congelada I03 y pérdida A01; conservar causas y efecto de enmascaramiento |

Cada escenario debe declarar parámetros, unidades, intervalo, capa, sujetos,
semillas, observables esperados y criterio de éxito. Ruido no significa anomalía
eléctrica; fallo de comunicación no significa máquina parada. La política de
etiquetado será multicausal; una eventual proyección a `Normal`/`Anomalo` necesita
regla explícita y aprobación, sin colapsar la referencia original.

## 6. Contrato común y compatibilidad

El futuro contrato común debe identificar instalación, dispositivo, circuito,
topología, fase/par de conductores o agregado, variable, valor, unidad y tiempo;
incluir versión, fuente sintética/real, naturaleza medida/calculada/estimada,
método, calidad y procedencia. Debe separar identidad de observación y de entrega,
tiempo de ocurrencia/medida/recepción/registro y duración de la ventana. Los nombres
de campos, tipos y representación larga/ancha quedan por aprobar en S03.

La calidad y la procedencia deben representar dimensiones independientes:

- **Disponibilidad:** distinguir dato presente, ausente o no disponible por
  capacidad/topología, con motivo cuando se conozca.
- **Validez:** indicar si el valor satisface las reglas aplicables o si no hay
  evidencia suficiente para determinarlo.
- **Vigencia:** distinguir actualidad u obsolescencia según la referencia temporal
  y el criterio que se apruebe, incluyendo vigencia desconocida.
- **Método de obtención:** medido, calculado o estimado, con procedencia y método.

Estas dimensiones pueden coexistir: un valor estimado puede ser válido y obsoleto;
uno medido puede ser inválido. Ausencia no equivale a invalidez ni implica por sí
sola un método de obtención. S03 definirá combinaciones admisibles, representación
de lo desconocido y ejemplos; no se eligen aquí clases ni enums definitivos.
La calidad observable por SmartEnergy sólo puede basarse en información disponible
en su límite de adquisición. Un error puede ser desconocido para el sistema:
las causas físicas y etiquetas de inyección conocidas exclusivamente por el
simulador van en la referencia de prueba, no se atribuyen como diagnóstico
automático al futuro dispositivo. Valores no finitos no deben entrar como
mediciones válidas ni serializarse como NaN/Infinity. Cero válido se conserva;
ausencia o invalidez se expresa sin cero sustituto y con motivo.

Evolución propuesta, pendiente de S03/S08:

1. Diseñar y verificar el contrato en memoria/artefactos aislados sin SQLAlchemy
   en el motor físico y sin I/O al importar. No duplicar entidades ORM.
2. Mantener inicialmente los generadores y consumidores existentes. Si se necesita
   una proyección legada, declarar qué circuito/fase representa, conversiones,
   pérdida de metadatos, temperatura ausente y política de etiquetas; rechazar
   proyecciones ambiguas. Nunca copiar un total a cada fase.
3. Reutilizar normalización/consultas existentes sólo donde sus contratos sean
   compatibles; `prepare_records` normaliza el formato histórico, no el de un
   instrumento. Definir un límite común para simulador y futuro adaptador físico
   sin implementar Modbus en esta iniciativa mínima.
4. Evaluar cada consumidor: ORM, consultas, cuatro variables/orden ML, bundle,
   CSV, filtros, unidades, timestamps y UI. Una migración de almacenamiento o
   cambio HTTP exige diseño, autorización y reversión propios; no usar `reset_db`
   ni `create_all` para resolver incompatibilidades.

No se eligió aún un esquema nuevo ni un formato de archivo. El monitoreo seguirá
siendo sólo lectura, con cinco vistas independientes de datos y ML; ejecución de
simulación y persistencia serán operaciones administrativas separadas. No alterar
`model_artifact.py` ni publicar modelos por esta tarea.

### Niveles de aceptación y límites de evidencia

| Nivel | Evidencia necesaria y resultado que demuestra | Lo que no demuestra |
|---|---|---|
| 1 — Núcleo | Pruebas aisladas de relaciones físicas, tiempo, escenarios, linaje y reproducción; cierre en S07 | Integración con consumidores actuales, persistencia PostgreSQL o funcionamiento con instrumentos |
| 2 — Contrato común y compatibilidad | Pruebas del límite común y consumidores afectados, con proyección legada y pérdidas declaradas; cierre de integración en memoria en S08 | Conservación completa del contrato común en PostgreSQL ni validación persistente de extremo a extremo |
| 3 — Persistencia completa y extremo a extremo | Escritura, lectura y consumo por la ruta de aplicación aprobada usando PostgreSQL aislado; contraste de valores, identidades, unidades, tiempos, calidad y procedencia contra la referencia, sin pérdidas silenciosas | Exactitud metrológica, adquisición real ni preparación industrial |

Fixtures y pruebas exclusivamente en memoria pueden acreditar niveles 1/2, nunca
validar persistencia ni sustituir el nivel 3. S03 deberá identificar qué campos
conserva o pierde una proyección a `Registro`; esa proyección no acredita por sí
sola persistencia completa. Cualquier evolución necesaria del almacenamiento o
HTTP requiere diseño y autorización separados, sin decidir aquí su estructura.

Antes del nivel 3 se requiere autorización expresa del ensayo y del entorno
PostgreSQL aislado: destino, rol, permisos, datos de prueba, escrituras permitidas,
procedimiento de aislamiento y reversión o limpieza autorizada. No usar recursos
protegidos ni ejecutar mantenimiento implícito. S03 definirá la ruta de extremo
a extremo y los criterios por campo; S08 sólo podrá ejecutarla con esa autorización.
Si el nivel 3 no se autoriza o no pasa, se reportará pendiente o fallido, sin
declarar validada la persistencia. Todo cierre deberá identificar el nivel y alcance
realmente aceptados; la aceptación de un nivel no habilita automáticamente el siguiente.

## 7. Reproducibilidad, tiempo y aceptación cuantificable

El manifiesto futuro registrará versión de escenario/motor/contrato, parámetros,
semilla y algoritmo aleatorio, entorno numérico relevante, instante inicial,
cadencia, duración, topología, perfiles, eventos y política de integración.
Separar secuencias aleatorias por capa evitará que activar una pérdida cambie la
física. El tiempo simulado no dependerá de `now()` ni de esperas del reloj real.
Propuesta: UTC con offset explícito y límites de intervalo inequívocos; aprobación
en S03, sin reinterpretar timestamps naive históricos.

Los siguientes son **criterios del simulador**, no exactitudes de instrumentos.
S02/S03 deben fijar tolerancias numéricas absolutas y relativas, por variable y
dominio, antes de implementar; no se aceptan tolerancias arbitrarias posteriores
para hacer pasar pruebas. Sin esa tabla no se inicia S06.

| Control | Criterio de aceptación futuro |
|---|---|
| Reproducción | Dos ejecuciones de igual manifiesto en el entorno fijado: 100% de identidades, tiempos, eventos y valores iguales; cambios de versión se declaran, no se promete igualdad entre versiones |
| Aislamiento | Física idéntica al activar/desactivar fallos I/A, para el 100% de muestras físicas |
| Identidad/linaje | 100% de observaciones y entregas enlazadas a su soporte físico; incluir ventanas que crucen transiciones, operación y tiempos; cero duplicados de identidad física involuntarios; cada pérdida/duplicado inyectado contabilizado |
| Tiempo | Cero inversiones del tiempo físico por serie; pasos y límites cumplen el manifiesto; retrasos sólo alteran entrega y quedan registrados |
| Física | 100% de casos válidos cumplen relaciones por fase y agregados dentro de tolerancias aprobadas; incluir cero, FP unitario, reactiva inductiva/capacitiva y desequilibrio |
| Energía | Casos de potencia constante y por tramos, carga consumidora capacitiva y cambio de signo de Q con P positiva contra oráculo analítico independiente, según convenciones y tolerancias aprobadas en S02; cambio de paso converge bajo el criterio aprobado; cero doble conteo por duplicación |
| Errores | Fallos deterministas: cantidades y duraciones exactas. Ruido: media, dispersión y límites contrastados con semillas/tamaño de muestra y bandas fijadas antes del ensayo; no pruebas aleatorias sin control |
| Calidad y ausencia | 100% de casos sin capacidad, inválidos y pérdidas preservan su diferencia; probar dimensiones independientes y combinaciones aprobadas de disponibilidad, validez, vigencia y método, sin revelar causas exclusivas de la referencia; cero valores artificiales para completar datos |
| Cobertura | Al menos un escenario aceptado por cada ID F01–C01, y pruebas de casos límite de tiempo/contador; todas las causas combinadas preservadas |
| Seguridad | Importación y generación pura con cero conexión, DDL, escritura persistente o carga ML; pruebas con mocks/temporales aislados |
| Integración, nivel 2 | Paridad de fixtures sintéticos y fixture de futuro dispositivo en el límite común; todos los contratos legados afectados y cuatro estados datos/ML verificados si cambia su integración; no acredita persistencia |
| Persistencia, nivel 3 | PostgreSQL aislado y expresamente autorizado; recorrido de escritura/lectura/consumo aprobado en S03, contrastando todos los campos requeridos y trazabilidad; sin evidencia de este ensayo no declarar persistencia validada |

Reinicio/rollover de contador se representará como evento explícito y nuevo tramo;
no como consumo negativo. Orden de llegada puede ser distinto del orden de medida.
La deduplicación y política de registros tardíos son decisiones de S03, no garantías
de la persistencia actual. Pruebas físicas y temporales deben usar oráculos
independientes, evitando copiar el mismo cálculo de producción en la prueba.

## 8. Gobierno obligatorio y condiciones de detención

- Antes de cada trabajo leer AGENTS, este contrato y el roadmap; verificar que la
  rama activa sea la autorizada para esa intervención. Para la corrección documental
  actual es obligatoria `feature/simulador`. Si no coincide, detenerse sin cambiarla.
  Revisar estado Git y preservar cambios ajenos; no hacer commit/push sin permiso.
- Inspeccionar código y consumidores reales, reutilizar componentes compatibles
  y fundamentar decisiones en fuentes primarias. Diferenciar hecho verificado,
  requisito de diseño y propuesta pendiente. No atribuir a SmartEnergy funciones
  del catálogo de un instrumento.
- Obtener aprobación explícita antes de cada etapa de implementación; el cierre de
  una etapa no habilita la siguiente. La autorización presente sólo cubre documentación.
- No introducir dependencias, migraciones, reorganización arquitectónica o cambios
  de API sin autorización específica. La tarea documental actual prohíbe ejecutar
  generadores y operar PostgreSQL o Docker. Las futuras pruebas del núcleo usarán
  aislamiento; cualquier ensayo de persistencia requerirá la autorización de nivel 3.
  No ejecutar mantenimiento, entrenamiento, detección persistente ni operaciones
  destructivas como atajo de validación.
- Preservar `.env`, `.venv`, credenciales, esquemas, datos, artefactos y bloqueos ML;
  no poblar `smartenergy_dev` ni tocar `smartenergy-f01-20261015-01` o sus recursos.
- Registrar decisiones y desviaciones en el apartado correspondiente de estos
  documentos: motivo, evidencia, impacto, autorización, validación y reversión.
  Una desviación propuesta no suspende el contrato vigente.
- Detener el componente afectado ante contradicción de fuentes, ambigüedad física
  que impida aceptar resultados, regresión o intervención fuera de alcance.
  Informar impedimento y continuar sólo trabajo independiente autorizado.

## 9. Decisiones abiertas y cierre de esta fase

Requieren aprobación humana: perfil mínimo y topologías; nominales y umbrales de
escenarios; método de S total/FP/Q y tolerancias; variante/firmware de referencia;
representación y versionado del contrato común; semántica temporal y deduplicación;
artefacto de referencia; proyección legada/temperatura/etiquetas; estrategia de
integración, nivel de aceptación pretendido y cualquier modificación de esquema,
HTTP o recursos persistentes. S01 deberá cerrar la evidencia relevante del ADW210
y someter cualquier diferimiento a la regla del apartado 1. S02 cerrará signos,
cuadrantes, contadores y acumulación; S03, dimensiones de calidad, soporte temporal,
compatibilidad y plan de aceptación por nivel. Los pendientes de instrumentos
están en la matriz; no justifican inventar capacidades para el núcleo genérico.

Entrega documental: contrato, roadmap, matriz y adición mínima en AGENTS.
Validación requerida en esta fase: correspondencia estática con código, coherencia
de fuentes/alcance, enlaces locales y diff. No se requieren servicios ni pruebas
de aplicación. No se implementa el simulador ni se inicia la etapa siguiente.
