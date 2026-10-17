# Matriz de instrumentos de referencia

Investigación documental: 2026-10-17. Complementa el
[contrato del simulador](../contratos/simulador-fase-1.md).
No constituye selección de compra, homologación ni prueba con equipos.
ADW210 conserva su condición de referencia inicial aprobada; los otros dos
instrumentos contrastan semánticas y evitan acoplar el modelo a una marca.

## Fuentes primarias consultadas

| ID | Documento y localización | Identificación y alcance de la revisión |
|---|---|---|
| A1 | [Ficha ADW210, Acrel](https://www.acrelpower.com/adw210-ac-multi-channels-iot-smart-energy-meter-product/) | Sin revisión indicada; capacidades generales y parámetros técnicos |
| A2 | [ADW2XX series rail type multiloop power instrument, Installation and Operation Instruction](https://v6-file.globalso.com/upload/p/2089/file/2026-06/adw-2xx-manual.pdf) | N.º 342, V1.9; enlazado por A1; portada, parámetros y tablas de comunicación, pp. impresas 2–3 y 20–26 |
| E1 | [SDM630-Modbus V2 User Manual](https://www.eastrongroup.com/eastrongroup/2024/08/21/eastronsdm630modbusv2usermanual2020v1.4.pdf) | V1.4, ©2020; configuración, magnitudes y especificaciones, pp. PDF 4–7, 11 y 17–18 |
| E2 | [SDM630Modbus Smart Meter Modbus Protocol Implementation](https://www.eastroneurope.com/images/uploads/products/protocol/SDM630_MODBUS_Protocol.pdf) | V1.8; tablas de entrada y codificación, pp. PDF 1–5; compatibilidad con variante/firmware por confirmar |
| E3 | [Ficha SDM630-Modbus V2, Eastron](https://www.eastrongroup.com/product/three-phase-multifunction-energy-meter/rs485-100a-din-rail-three-phase-multifunction-energy-meter.html) | Sin revisión indicada; conexión directa y comunicación |
| S1 | [PowerLogic PM5300 series User manual](https://www.productinfo.schneider-electric.com/pm5300/5be97f3b347bdf0001d99c87/PM5300%20User%20Manual/English/EAV15107-EN11.pdf) | EAV15107-EN11, ©2023; modelo seleccionado PM5310; sistemas, configuración y semántica PQS/FP, pp. 19–20, 34 y 90–94 |
| S2 | [PM5300: Features and options](https://productinfo.se.com/pm5300/5be97f3b347bdf0001d99c87/PM5300%20User%20Manual/English/R_MeterOverview_FeaturesOptionsPM5300_0000082140.xml) | Manual web, tabla específica de modelos; no transferir prestaciones Ethernet del PM5320 al PM5310 |
| S3 | [PM5300: Device specifications](https://productinfo.se.com/pm5300/5be97f3b347bdf0001d99c87/PM5300%20User%20Manual/English/R_Specifications_PM5300_0000085537.xml) | Manual web; condiciones de exactitud y entradas CT |
| S4 | [Mapa de registros PM5100/PM5300, FAQ FA283821](https://www.se.com/br/pt/faqs/FA283821/) | Modificada 2025-04-25; identifica paquetes de firmware y anexo `PM51xx_PM53xx_PMC Register List_v2038_v2066_R01.xls`; contenido del XLS no verificado en esta fase |

Se enlazan los originales; no se redistribuyen manuales. A2 restringe expresamente
su reproducción. Los títulos extraídos automáticamente de algunos PDF son erróneos:
se usan sus portadas, no el título del buscador. La fecha del directorio de descarga
no equivale a la fecha de edición.

## Capacidades documentadas, nivel de evidencia y límites

Cada afirmación de la tabla identifica fuente y nivel de evidencia:

| Marca | Significado y límite |
|---|---|
| COM | Información de ficha comercial del fabricante, sin confirmación adicional de configuración |
| FAM | Información de manual/mapa de familia; no confirma por sí sola aplicabilidad a un modelo, variante o firmware concreto |
| ESP | Confirmación documental para el modelo y configuración expresamente indicados por la fuente; no equivale a ensayo con equipo real ni a confirmación de toda configuración posible |
| PEN | Información o aplicabilidad pendiente de verificación; no implica ausencia de capacidad |

ESP sólo se usa donde la fuente identifica el modelo/configuración o una columna
específica del modelo. E1 documenta SDM630-Modbus V2 de conexión directa 100 A,
con las topologías indicadas; no confirma otro firmware ni SDM630MCT. E2 conserva
nivel FAM hasta confirmar su correspondencia con esa variante. En S2 se distinguen
las filas específicas de PM5310 de las características generales de PM5300.
Las condiciones de S3 siguen siendo de familia hasta cerrar la configuración.
No hay confirmación experimental con instrumentos en esta fase.

| Aspecto | Acrel ADW210 | Eastron SDM630-Modbus V2 | Schneider PowerLogic PM5310 |
|---|---|---|---|
| Topología | COM: hasta 4 circuitos trifásicos o 12 monofásicos; conexiones 3P3W/3P4W [A1] | ESP: 1P2W, 3P3W y 3P4W [E1, «Introduction» y «Supply System»] | FAM: monofásico y trifásico [S1, «Power system»]; PEN: configuración elegida |
| Tensión y corriente | FAM: CH1 y bloques CH2–CH4 con V por fase/entre líneas e I por fase [A2, «Power parameter of loop 1», pp. 20–23]; PEN: aplicabilidad ADW210; no implica tensiones independientes por circuito | ESP: V/I por fase, L-N no aplicable en 3P3W [E1, «Voltage and Current», «Voltage and Current Inputs»]; FAM: mapa [E2, §1.2.1] | FAM: I por fase/neutro y V L-N/L-L [S2, «Instantaneous rms values»]; PEN: disponibilidad en cableado elegido |
| P, Q, S y FP | FAM: por fase/total de circuito [A2, pp. 20–23]; PEN: confirmación en variante elegida | FAM: potencias y FP por fase/totales [E2, §1.2.1]; PEN: correspondencia con variante/firmware V2 elegido | FAM: P/Q/S y FP verdadero/desplazamiento por fase y total [S2, «Instantaneous rms values»] |
| Energía y frecuencia | FAM: energías activa/reactiva directa/inversa por fase/circuito y frecuencia [A2, pp. 20–23]; PEN: configuración concreta | ESP: energías activa/reactiva importada/exportada, totales y frecuencia [E1, «Energy Measurements», «Frequency and Power Factor and Demand»] | FAM: energías activa/reactiva/aparente, recibida/entregada, neta/absoluta y frecuencia [S2, «Energy values», «Instantaneous rms values»] |
| CT y escala | COM: CT externos 5/100/400/600 A [A1]; PEN: configuración efectiva | ESP: conexión directa 100 A [E1, «Introduction»]; COM: corroboración [E3]; no atribuirle CT configurable de SDM630MCT | FAM: secundario CT 1/5 A [S3, «Current inputs»]; primario/secundario configurables [S1, p. 34]; PEN: valores elegidos |
| Exactitud declarada | COM: V/I clase 0,5; energía activa clase 1, reactiva clase 2; frecuencia 0,05 Hz [A1] | ESP: V 0,5% del máximo de rango; I 0,5% nominal; P/Q/S ±1% del máximo de rango; FP 0,01; energía activa clase 1/B y reactiva clase 2 [E1, «Accuracy», p. PDF 18] | FAM: energía activa clase 0,5S, reactiva clase 2; V L-N/I/P/S clase 0,5, Q clase 2, FP ±0,005 [S3, «Measurement accuracy»]; PEN: condiciones de firmware/corriente aplicables |
| Calidad eléctrica | COM: armónicos 2–31 [A1]; FAM: THD/desequilibrio [A2, «Harmonic data», pp. 23–24]; PEN: configuración | ESP: THD V/I [E1, «Voltage and Current Inputs»]; validez según conexión | ESP: THD/thd/TDD y armónicos impares hasta 31 en columna PM5310 [S2, «Features and options»]; FAM: desequilibrio [S2, «Instantaneous rms values»] |
| Comunicación | COM: RS-485 Modbus RTU [A1]; FAM: mapa mixto float/enteros [A2, §6.1]; PEN: variante/firmware | ESP: RS-485 Modbus RTU [E1, «Communication»]; FAM: IEEE754 de 32 bits en dos registros y FC04 [E2, §1.1–1.2]; PEN: mapa aplicable | ESP: puerto serie Modbus en columna PM5310 [S2, «Features and options»]; PEN: mapa por firmware identificado pero no inspeccionado [S4] |
| Tiempo y resolución | PEN: actualización y resolución por magnitud; un float no determina resolución metrológica | ESP: escalón típico 1 s a 50 Hz [E1, «Accuracy»]; no es período de sondeo. PEN: cadencia/resolución por magnitud | FAM: 64 muestras/ciclo [S3, «Measurement accuracy»]; no es refresco de registros. PEN: cadencia/resolución por magnitud |

Las clases no son un ruido porcentual universal ni incluyen automáticamente el CT,
el cableado o la cadena completa. No se usarán como límites de aceptación del motor
físico. E3 anuncia respuesta de comunicación menor de 100 ms: no contradice por sí
sola el tiempo de respuesta metrológica de E1; son magnitudes distintas.

## Evidencia de mapas y pendientes de cierre

- **ADW210:** A2, p. impresa 21, presenta CH1 P/Q/S y FP por fase/total,
  energía en kWh/kvarh y tipos `float`; pp. 22–23 extienden bloques a CH2–CH4.
  Ejemplo documental: `13C–13D` identifica energía activa positiva total CH1.
  No es una instrucción de lectura aprobada. A2 es un manual de familia:
  confirmar aplicabilidad a la variante ADW210, firmware, opciones y referencia
  de tensión compartida. Revisar orden de palabras, escalas y consistencia de
  tablas antes de cualquier decodificador.
- **SDM630:** E2 distingue referencia `30001` y dirección de protocolo `0x0000`
  para V L1-N, en V; marca su invalidez en 3P3W. Algunas variables no aplicables
  retornan cero: el normalizador debe usar la configuración para distinguirlas
  de un cero medido. E2 no debe aplicarse automáticamente a todas las variantes.
- **PM5310:** S4 identifica el anexo oficial, pero la vista web consultada no
  expuso un enlace al XLS que permitiera inspeccionarlo con la herramienta usada.
  Direcciones, escalas, tipos y disponibilidad por firmware permanecen pendientes.
  S1 advierte convenciones y codificaciones distintas de FP: no interpretar todo
  registro de FP como valor normalizado directo.

Antes de emular un medidor se requiere una ficha por variable: documento/revisión,
modelo/firmware, topología, alcance, unidad, dirección/base, tipo, orden de palabras,
escala, signo, valor inválido, resolución, tiempo de actualización y transformación
CT/VT. Cada campo debe conservar su nivel COM/FAM/ESP/PEN y localización documental;
un dato FAM no pasa a ESP sin evidencia de aplicabilidad. No completar campos
desconocidos por analogía con otro modelo. S01 debe investigar obligatoriamente
las capacidades relevantes del ADW210 aun si el núcleo inicial es genérico.
Diferir emulación ejecutable no elimina el contraste documental exigido por el
roadmap técnico; cualquier diferimiento y su impacto requieren registro y aprobación
según el contrato. Esta reclasificación no cierra S01 ni confirma nuevos equipos.

## Consecuencias para SmartEnergy

**Requisitos de diseño, no prestaciones verificadas del producto:** usar un núcleo
independiente de la marca; separar L-N de L-L, fases de totales, FP de desplazamiento
de FP verdadero y energía importada de exportada. El mínimo del simulador no exige
emular registros ni errores propios de un modelo. La exactitud documental sirve
para futuros perfiles de instrumentación con condiciones explícitas.

**Pendiente de investigación:** revisión completa de mapas compatibles, semántica
de agregación de S/FP y Q con distorsión, límites/resolución/rollover de contadores,
cadencia por magnitud, simultaneidad de lecturas y presupuesto de incertidumbre
de la cadena CT/medidor. THD, demanda y armónicos se reservan como ampliaciones;
su existencia en un instrumento no obliga a incorporarlos ahora al simulador.
