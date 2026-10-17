# Matriz de instrumentos de referencia

Investigación documental: 2026-10-17. Complementa el
[contrato del simulador](../contratos/simulador-fase-1.md).
No constituye selección de compra, homologación ni prueba con equipos.
ADW210 conserva su condición de referencia inicial aprobada; los otros dos
instrumentos contrastan semánticas y evitan acoplar el modelo a una marca.

S01 investigada desde `ed2e5d8219a21e2bd29235c7140327ff5b702662`, en
`feature/simulador`, con árbol inicialmente limpio. **Investigación documental
realizada; cierre formal pendiente de evidencia y aprobación**, según los resultados
y decisiones al final de este documento. No se ha validado ningún instrumento real.

## Fuentes primarias consultadas

| ID | Documento y localización | Identificación y alcance de la revisión |
|---|---|---|
| A1 | [Ficha ADW210, Acrel](https://www.acrelpower.com/adw210-ac-multi-channels-iot-smart-energy-meter-product/) | Sin revisión indicada; capacidades generales y parámetros técnicos |
| A2 | [ADW2XX series rail type multiloop power instrument, Installation and Operation Instruction](https://v6-file.globalso.com/upload/p/2089/file/2026-06/adw-2xx-manual.pdf) | N.º 342, V1.9; enlazado por A1; portada, parámetros y tablas de comunicación, pp. impresas 2–3 y 20–26 |
| A3 | [ADW210+CT Instruction Manual, 3-phase 3P4W](https://www.acrel-electric.ke/uploads/file/adw210-wiring-installation-instruction-manual-3p4w.pdf) | Acrel, Aaron Shi; sin revisión/fecha declarada; pp. 1–2 CT y conectores, pp. 3–4 tensión/corriente y polaridad, p. 5 RS-485; revisión visual de p. 4 |
| A4 | [ADW210+CT Instruction Manual, 3-phase 3P3W](https://www.acrel-electric.ke/uploads/file/adw210-wiring-installation-instruction-manual-3p3w.pdf) | Acrel, Aaron; sin revisión/fecha declarada; pp. 3–5, configuración alternativa; ambas guías enlazadas en [Installation Manual](https://www.acrel-electric.ke/smart-energy-meter-installation-manual/) |
| E1 | [SDM630-Modbus V2 User Manual](https://www.eastrongroup.com/eastrongroup/2024/08/21/eastronsdm630modbusv2usermanual2020v1.4.pdf) | V1.4, ©2020; configuración, magnitudes y especificaciones, pp. PDF 4–7, 11 y 17–18 |
| E2 | [SDM630Modbus Smart Meter Modbus Protocol Implementation](https://www.eastroneurope.com/images/uploads/products/protocol/SDM630_MODBUS_Protocol.pdf) | V1.8; tablas de entrada y codificación, pp. PDF 1–5; compatibilidad con variante/firmware por confirmar |
| E3 | [Ficha SDM630-Modbus V2, Eastron](https://www.eastrongroup.com/product/three-phase-multifunction-energy-meter/rs485-100a-din-rail-three-phase-multifunction-energy-meter.html) | Sin revisión indicada; conexión directa y comunicación |
| E4 | [SDM630-Modbus V2, portada 2023 V1.5](https://www.eastrongroup.com/eastrongroup/2024/12/25/eastronsdm630modbusv2usermanual2023v1.5.pdf) | Eastron; encabezados internos V1.4; pp. PDF 17–19 especificaciones y condiciones; conserva discrepancia de edición, no se asume firmware V1.5 |
| S1 | [PowerLogic PM5300 series User manual](https://www.productinfo.schneider-electric.com/pm5300/5be97f3b347bdf0001d99c87/PM5300%20User%20Manual/English/EAV15107-EN11.pdf) | EAV15107-EN11, ©2023; modelo seleccionado PM5310; sistemas, configuración y semántica PQS/FP, pp. 19–20, 34 y 90–94 |
| S2 | [PM5300: Features and options](https://productinfo.se.com/pm5300/5be97f3b347bdf0001d99c87/PM5300%20User%20Manual/English/R_MeterOverview_FeaturesOptionsPM5300_0000082140.xml) | Manual web, tabla específica de modelos; no transferir prestaciones Ethernet del PM5320 al PM5310 |
| S3 | [PM5300: Device specifications](https://productinfo.se.com/pm5300/5be97f3b347bdf0001d99c87/PM5300%20User%20Manual/English/R_Specifications_PM5300_0000085537.xml) | Manual web; condiciones de exactitud y entradas CT |
| S4 | [Mapa de registros PM5100/PM5300, FAQ FA283821](https://www.se.com/br/pt/faqs/FA283821/) | Modificada 2025-04-25; localiza el XLS y el paquete específico inspeccionado en S5 |
| S5 | [PM5110_PM5310_PM5330_v2.2.4_HW, Schneider](https://www.se.com/br/pt/download/document/PM5110_PM5310_PM5330_v2.2.4_HW/) | Publicación 2024-05-14, versión 2.2.4; [ZIP oficial](https://download.schneider-electric.com/files?p_Doc_Ref=PM5110_PM5310_PM5330_v2.2.4_HW). Inspeccionados `PM51xx_PM53xx_PMC Register List_v2038_v2066_R01.xls` y `PM5110_PM5310_PM5330 v2.2.4.8 FW Rev History.pdf`, p. 1. No ejecutados los archivos `.fw` |

Se enlazan los originales; no se redistribuyen manuales. A2 restringe expresamente
su reproducción. Los títulos extraídos automáticamente de algunos PDF son erróneos:
se usan sus portadas, no el título del buscador. La fecha del directorio de descarga
no equivale a la fecha de edición.

Todos los enlaces de esta tabla se consultaron en S01 el 2026-10-17. Se usaron
copias temporales fuera del repositorio para A2/A3 y S5, sin redistribución.
El XLS se convirtió temporalmente para leer celdas conservando coordenadas;
las referencias siguientes corresponden a sus hojas originales. No se instalaron
dependencias ni se ejecutó firmware. Identidad de los originales descargados:

| Fuente | SHA-256 |
|---|---|
| A2, PDF | `b6587bdee481e935373b116779e67d23684fc56a3dd89e27118ae20072b2cf59` |
| A3, PDF | `b843ac95153689851b5ae61600c2c609caa96b4c61cc00cd13185c511e036958` |
| S5, ZIP | `48b2d7706ff2e5ef4be664c622dcf75673e05b4eaf3e41f59dd6cafc0552e951` |
| S5, XLS original | `489ff0fb00e7939d064f7337bd8b67855922114fbabf180492e72393ec1e1d3a` |

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
| Topología | ESP: 3P4W y 3P3W, hasta cuatro grupos trifásicos [A3/A4]; COM: 12 monofásicos [A1]; PEN: detalle de configuración 1P2W | ESP: 1P2W, 3P3W y 3P4W [E1, «Introduction» y «Supply System»] | FAM: monofásico y trifásico [S1, «Power system»]; PEN: configuración elegida |
| Tensión y corriente | ESP: un juego común UA/UB/UC/UN y CH1–CH4 de corriente [A3, pp. 3–4]; FAM: V repetida e I por fase en bloques [A2, pp. 20–23]; PEN: aplicabilidad exacta del mapa | ESP: V/I por fase, L-N no aplicable en 3P3W [E1, «Voltage and Current», «Voltage and Current Inputs»]; FAM: mapa [E2, §1.2.1] | FAM: I por fase/neutro y V L-N/L-L [S2, «Instantaneous rms values»]; ESP: disponibilidad de registros PM5310 condicionada al cableado [S5, filas 244–263] |
| P, Q, S y FP | FAM: por fase/total de circuito [A2, pp. 20–23]; PEN: confirmación en variante elegida | FAM: potencias y FP por fase/totales [E2, §1.2.1]; PEN: correspondencia con variante/firmware V2 elegido | FAM: P/Q/S y FP verdadero/desplazamiento por fase y total [S2, «Instantaneous rms values»] |
| Energía y frecuencia | FAM: energías activa/reactiva directa/inversa por fase/circuito y frecuencia [A2, pp. 20–23]; PEN: configuración concreta | ESP: energías activa/reactiva importada/exportada, totales y frecuencia [E1, «Energy Measurements», «Frequency and Power Factor and Demand»] | FAM: energías activa/reactiva/aparente, recibida/entregada, neta/absoluta y frecuencia [S2, «Energy values», «Instantaneous rms values»] |
| CT y escala | COM: CT externos 5/100/400/600 A [A1]; PEN: configuración efectiva | ESP: conexión directa 100 A [E1, «Introduction»]; COM: corroboración [E3]; no atribuirle CT configurable de SDM630MCT | FAM: secundario CT 1/5 A [S3, «Current inputs»]; primario/secundario configurables [S1, p. 34]; PEN: valores elegidos |
| Exactitud declarada | COM: V/I clase 0,5; energía activa clase 1, reactiva clase 2; frecuencia 0,05 Hz [A1] | ESP: V 0,5% del máximo de rango; I 0,5% nominal; P/Q/S ±1% del máximo de rango; FP 0,01; energía activa clase 1/B y reactiva clase 2 [E1, «Accuracy», p. PDF 18] | FAM: energía activa clase 0,5S, reactiva clase 2; V L-N/I/P/S clase 0,5, Q clase 2, FP ±0,005 [S3, «Measurement accuracy»]; PEN: condiciones de firmware/corriente aplicables |
| Calidad eléctrica | COM: armónicos 2–31 [A1]; FAM: THD/desequilibrio [A2, «Harmonic data», pp. 23–24]; PEN: configuración | ESP: THD V/I [E1, «Voltage and Current Inputs»]; validez según conexión | ESP: THD/thd/TDD y armónicos impares hasta 31 en columna PM5310 [S2, «Features and options»]; FAM: desequilibrio [S2, «Instantaneous rms values»] |
| Comunicación | ESP: RS-485 en ADW210 [A3, p. 5]; COM: Modbus RTU [A1]; FAM: mapa float/enteros [A2, §6.1]; PEN: variante/firmware | ESP: RS-485 Modbus RTU [E1, «Communication»]; FAM: IEEE754 de 32 bits en dos registros y FC04 [E2, §1.1–1.2]; PEN: mapa aplicable | ESP: puerto serie Modbus en columna PM5310 [S2]; mapa con columna PM5310 inspeccionado [S5]; PEN: orden/base para un futuro adaptador |
| Tiempo y resolución | PEN: actualización y resolución por magnitud; un float no determina resolución metrológica | ESP: escalón típico 1 s a 50 Hz [E1, «Accuracy»]; no es período de sondeo. PEN: cadencia/resolución por magnitud | ESP: bloque básico «1s Metering (50/60 Cycles)» [S5, fila 242]; FAM: 64 muestras/ciclo [S3]. PEN: simultaneidad y resolución metrológica por magnitud |

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
- **PM5310:** la limitación anterior de acceso al XLS quedó resuelta mediante S5.
  Se verificaron hoja `Register List`, columna F de aplicabilidad PM5310, y hoja
  `Data Types`. Esto confirma contenido documental, no lecturas de hardware.
  Las diferencias de unidades, FP y tipos se detallan a continuación.

Antes de emular un medidor se requiere una ficha por variable: documento/revisión,
modelo/firmware, topología, alcance, unidad, dirección/base, tipo, orden de palabras,
escala, signo, valor inválido, resolución, tiempo de actualización y transformación
CT/VT. Cada campo debe conservar su nivel COM/FAM/ESP/PEN y localización documental;
un dato FAM no pasa a ESP sin evidencia de aplicabilidad. No completar campos
desconocidos por analogía con otro modelo. S01 debe investigar obligatoriamente
las capacidades relevantes del ADW210 aun si el núcleo inicial es genérico.
Diferir emulación ejecutable no elimina el contraste documental exigido por el
roadmap técnico; cualquier diferimiento y su impacto requieren registro y aprobación
según el contrato. La investigación no cierra formalmente S01 ni confirma equipos reales.

## Fichas de evidencia S01 por instrumento

### Acrel ADW210: configuración física y límites

**ESP, A3/A4:** cuatro grupos de corriente trifásica; A3 identifica CT
AKH-0.66/K de tres elementos y su polaridad. Su diagrama conecta los circuitos
a un mismo suministro de tensión. **Inferencia física respaldada por el cableado:**
esas tensiones son compartidas; los doce canales de corriente no son doce entradas
de tensión independientes. No extrapolar esto a circuitos alimentados desde barras
con tensiones diferentes. La configuración monofásica anunciada en A1 sigue COM:
no se obtuvo una guía específica que cierre su asignación de canales/fases.

**FAM, A2, §2.2 y §5.4.3:** opciones CT 5/100/400/600 A y relación primaria por
circuito. En §6.1 aparecen primario V `0x13`, primarios I CH1–CH4 `0x14–0x17`,
secundario I `0x18` y nominal secundario V `0x19`. Son campos de configuración,
no una autorización de escritura. Su interpretación no permite tratar todas las
opciones como CT convencional de secundario 5 A: confirmar accesorio y relación
efectiva. No se conoce el firmware al que corresponde cada variante de A2.

Las siguientes referencias son **FAM**, tomadas de A2 §6.1, pp. impresas 20–23;
direcciones hexadecimales de palabra, no referencias decimales 3xxxx/4xxxx.
La tabla agrupa variables y no es un mapa ejecutable ni copia completa del manual.

| Magnitud / alcance CH1 | Palabras iniciales relevantes | Unidad / representación |
|---|---|---|
| V A/B/C; V AB/BC/CA | `0x100/102/104`; `0x106/108/10A` | V; float, dos palabras por valor |
| Frecuencia; I A/B/C | `0x10C`; `0x10E/110/112` | Hz; A; float, dos palabras |
| P A/B/C y total | `0x116/118/11A/11C` | W; float, dos palabras |
| Q A/B/C y total | `0x11E/120/122/124` | var; float, dos palabras |
| S A/B/C y total | `0x126/128/12A/12C` | VA; float, dos palabras |
| FP A/B/C y total | `0x12E/130/132/134` | Adimensional; float, dos palabras |
| Energía activa positiva / inversa total | `0x13C`; `0x144` | kWh; float, dos palabras; también hay filas por fase |
| Energía reactiva directa / inversa total | `0x14C`; `0x154` | kvarh; float, dos palabras; también por fase |

CH2–CH4 tienen bloques documentados; no se infiere que su disponibilidad real sea
idéntica sin accesorios/configuración. **PEN:** orden de palabras, escala primaria
efectiva, resolución metrológica, sentinelas, actualización, sincronía y significado
de los contadores reactivos por cuadrante. La declaración de cuatro cuadrantes en
A2 §3.1 no resuelve por sí sola esas reglas. A1/A2 declaran clases de exactitud;
no se confirmó un presupuesto completo CT/medidor o todas sus condiciones de ensayo.

**Discrepancias que impiden cerrar una emulación:** A2 §5.4, p. 11, intercambia las
descripciones 3P4L/3P3L respecto de §5.4.3, p. 14; no corregir registros por intuición.
Los nominales secundarios de V de §5.4 y `0x19` tampoco ofrecen una especificación
inequívoca. Recomendación: mantener estas operaciones fuera del núcleo genérico y
resolverlas con documentación compatible o confirmación del fabricante antes de
configurar un adaptador. No se contactó al fabricante ni se eligió hardware real.

### Eastron SDM630-Modbus V2: contraste documental

**ESP, E1:** un sistema medido por dispositivo, 1P2W/3P3W/3P4W, conexión directa
100 A; no es un equipo de cuatro circuitos como ADW210. La ilustración de firmware
en «Start-up Screens» es sólo un ejemplo, no una versión aplicable confirmada.
E4 corrobora capacidades, pero su portada V1.5 y encabezados V1.4 impiden usar el
nombre del archivo como prueba de revisión uniforme. No se sustituye E1 silenciosamente.

**FAM, E2 §1.1–1.2 y §3.7:** FC04, dos palabras IEEE754, unidades directas sin
factor entero adicional; orden más significativo primero por defecto, configurable.
Mapa de contraste, pendiente de asociación inequívoca con el firmware V2 elegido:

| Magnitud | Referencia / dirección inicial de protocolo | Unidad |
|---|---|---|
| V L1-N; I L1 | `30001 / 0x0000`; `30007 / 0x0006` | V; A |
| P/Q/S total | `30053 / 0x0034`; `30061 / 0x003C`; `30057 / 0x0038` | W; var; VA |
| FP total; frecuencia | `30063 / 0x003E`; `30071 / 0x0046` | 1; Hz |
| Energía activa importada/exportada | `30073 / 0x0048`; `30075 / 0x004A` | kWh |
| Energía reactiva importada/exportada | `30077 / 0x004C`; `30079 / 0x004E` | kvarh |

E2 §1.2.1 distingue validez por topología: valores marcados no aplicables devuelven
cero. Su nota 1 usa el signo del FP para dirección de corriente; nota 3 define
energía total como suma importación + exportación, no saldo neto. No trasladar
el signo de FP a inductivo/capacitivo ni confundir validez de valores por fase
en 3P3W con la de los totales. La correspondencia reactiva/cuadrantes sigue PEN.

**ESP, E1 «Accuracy» y «Reference Conditions»:** las exactitudes de la tabla anterior
dependen de referencia sinusoidal y condiciones nominales; se declara frecuencia
±0,2% de frecuencia central, no de cualquier lectura. El escalón típico de 1 s
no demuestra refresco de cada registro ni simultaneidad. La energía visible con
dos decimales y el float Modbus no demuestran resolución interna o rollover.
**PEN:** cadencia por variable, resolución efectiva, sentinelas no finitos y CT/VT
externos no previstos para esta variante directa. No añadir un CT configurable por
analogía con SDM630MCT.

### Schneider PowerLogic PM5310: mapa y firmware contrastados

**ESP documental, S5:** el historial identifica hardware B1, firmware 2.2.4,
revisión de 2024-03-27. No es una selección aprobada ni prueba de un equipo
instalado. La rama 2.xx no es intercambiable con hardware de firmware 1.xx.
La corrección de Q durante exportación y las adiciones de registros FP que describe
el historial justifican conservar firmware y tipo de registro, no sólo el modelo.

El XLS incluido tiene columna F para PM5310. En las filas siguientes F=Y;
los números son la columna K «Register», sin convertirlos aquí a offsets de trama.
La cadencia del bloque básico está identificada en B242 como 1 s / 50–60 ciclos;
no se generaliza a energía, alarmas o lectura atómica multirregistro.

| Magnitud / alcance | Register; fila de `Register List` | Unidad / tipo / palabras |
|---|---|---|
| I A; V A-N | 3000, fila 244; 3028, fila 260 | A; V; FLOAT32 / 2 |
| P A; P total | 3054, fila 275; 3060, fila 278 | kW; FLOAT32 / 2 |
| Q total; S total | 3068, fila 282; 3076, fila 286 | kvar; kVA; FLOAT32 / 2 |
| FP total cuatro cuadrantes | 3084, fila 291 | `4Q_FP_PF` / 2; no es FP directo entre −1 y 1 |
| FP total IEC; frecuencia | 3192, fila 305; 3110, fila 297 | 1; Hz; FLOAT32 / 2 |
| Energía activa entregada/recibida | 3204/3208, filas 312–313 | Wh; INT64 / 4 |
| Energía reactiva entregada/recibida | 3220/3224, filas 316–317 | varh; INT64 / 4 |

S5 `Data Types` D8 declara N/A FLOAT32 `0xFFC00000`; D7, N/A INT64
`0x8000000000000000`. No registrarlos como mediciones válidas. `Register List`
R260 condiciona V A-N a su conexión. Hay también contadores float en kWh/kvarh
(filas 208–215): elegir un tipo/unidad expresamente, no intercambiarlos por nombre.
INT64 con unidad Wh no certifica por sí solo resolución metrológica de 1 Wh.

**FAM/ESP según S1/S2:** un sistema eléctrico por medidor, conexiones monofásicas
y trifásicas, CT nominal 1/5 A y VT configurable. S5 filas 155–159 identifica sus
campos; el rango/formato efectivo debe reconciliarse con hardware/firmware y con
las mejoras de relaciones CT/VT de su historial antes de programar configuración.
S3 condiciona energía clase 0,5S/2 a firmware y corriente nominal; para secundario
1 A indica I > 0,15 A. No extender exactitud a toda la cadena de medida.

S1 pp. 90–94 distingue FP verdadero/desplazamiento y convenciones IEC/IEEE; los
registros cuatro cuadrantes pueden superar módulo 1 como codificación, no como FP
físico. **PEN:** asociación completa de configuración a registro, orden/base de
trama, resolución por magnitud, reglas de rollover/reinicio y agregación Q/S con
distorsión. El rango representable del mapa no sustituye el rango de exactitud
eléctrica de S3. No se considera el ZIP la versión más reciente sin verificarlo
en una futura tarea de adquisición.

## Consecuencias para SmartEnergy

### A. Requisitos respaldados por evidencia instrumental

- **Magnitudes comunes:** V, I, P, Q, S, FP, frecuencia y energías activa/reactiva
  están documentadas en A2, E1/E2 y S1/S5, con diferentes alcances y unidades.
  Forman una base documental para S02, sin afirmar que cada modelo entrega todo
  en cualquier configuración. La disponibilidad por fase/total debe acompañar al valor.
- **Identidad y alcance:** A3 documenta varios grupos de corriente con tensión
  común; E1 y S1 documentan un sistema por dispositivo. «Dispositivo», «circuito»,
  «fase» y «referencia de tensión» no son identificadores intercambiables.
- **Interpretación:** los mapas exigen topología, unidad, tipo y codificación.
  P está en W en A2/E2 y en kW en S5; hay acumuladores en kWh float y Wh INT64.
  E2 distingue total absoluto de saldo neto; S5 ofrece ambos. No sumar contadores
  ya totales ni tratar energía acumulada como potencia instantánea.
- **Calidad observable:** cero no aplicable de E2 y N/A de S5 requieren interpretar
  capacidad/topología y tipo; la falta de sentinela documentado en A2 no autoriza
  inventar uno. La marca temporal de recepción no demuestra el instante de medida.
- **Convenciones:** las notas E2 y las variantes FP de S1/S5 prueban que el signo
  del FP no permite deducir universalmente inductivo/capacitivo. La correspondencia
  de energía reactiva por cuadrante debe cerrarse en S02 con la referencia elegida.

Las fuentes describen entradas de tensión/corriente y resultados del medidor; no
demuestran que P/Q/S/FP o corriente de neutro sean canales de sensor independientes.
Como clasificación de trabajo, V/I son entradas físicas y potencias/FP/energía son
resultados derivados; el método exacto de cada instrumento sólo se atribuirá cuando
esté documentado. Esta inferencia no constituye una implementación de S02.

### B. Recomendaciones de arquitectura, todavía no aprobadas

Mantener el motor genérico y perfiles de instrumentación separados. Conservar
la procedencia del valor observado y de toda conversión/cálculo; no derivar Q o S
de V/I/P sin hipótesis aprobadas ni llamar «medido» a un cálculo de SmartEnergy.
Permitir que un circuito haga referencia al suministro compartido en vez de
inventar sensores de tensión adicionales. S03 deberá decidir su representación,
sin añadir ahora clases, entidades, migraciones o un esquema definitivo.

Mantener separadas ventana física, actualización interna, sondeo y recepción.
La cadencia del motor debe ser parámetro de ensayo, no una promesa de frecuencia
de todos los medidores. Ruido, deriva, congelación y fallos podrán diseñarse como
perfiles genéricos; reproducir clases de exactitud de marca exige además condiciones
y cadena de incertidumbre. THD/armónicos, demanda, 3P3W y exportación activa
conservan sus límites actuales; su presencia en un medidor no amplía el simulador.

La inspección del repositorio confirma que `Registro` no conserva estos metadatos
y que el HTTP sigue ofreciendo cuatro variables. S01 no modifica ese contrato ni
decide cómo persistir el nuevo modelo. Esas diferencias se trasladan a S03 junto
con la compatibilidad de temperatura, etiquetas, unidades y tiempos históricos.

### C. Decisiones pendientes y recomendaciones concretas

Todas las filas están **pendientes de aprobación**. No constituyen una autorización
para avanzar a S02 ni una selección de compra. El orden indica prioridad de cierre.

| ID / decisión | Recomendación y justificación | Alternativa / condición de cierre |
|---|---|---|
| D01 — Referencia física ADW210 | Adoptar como referencia documental 3P4W, suministro común y capacidad CH1–CH4 con CT AKH-0.66/K; proponer opción 100 A de A2 como punto de contraste, sin atribuirle firmware confirmado. A3 aporta cableado específico y evita tratar circuitos como entradas de tensión independientes | Usar menos grupos o CT distinto si el caso real lo exige; confirmar accesorio/SKU y compatibilidad documental antes de afirmar configuración completa. No usar 12 monofásicos como emulación confirmada mientras falte su guía |
| D02 — Núcleo genérico y límites de S01 | Aceptar la evidencia física relevante y permitir que S02 estudie un núcleo genérico, conservando PEN para firmware/decodificación ADW210 y compatibilidad E2–V2. No hace falta implementar un emulador para estudiar magnitudes comunes | Diferimiento propuesto, aún no aprobado: cerrar asociación exacta firmware/mapa, orden/base/sentinelas y discrepancias de configuración antes de cualquier perfil específico o adaptador. Si se exige equivalencia instrumental desde el inicio, esos pendientes bloquean el cierre de S01 y se requiere evidencia del fabricante |
| D03 — Referencias de contraste | Conservar E1 como manual V2 directo 100 A y S5 como paquete documental PM5310 HW B1/FW 2.2.4; registrar E4 como contraste con edición inconsistente. Son referencias reproducibles, no afirmaciones de última versión o equipo instalado | Cambiar de revisión sólo con comparación explícita y configuración identificada; una versión de software ilustrativa no confirma el firmware de un equipo |
| D04 — Signos y contadores a resolver en S02 | Definir semántica canónica explícita y preservar la denominación/codificación original en futuras conversiones. Separar FP verdadero/desplazamiento y energía direccional/neta/absoluta evita pérdidas de significado | Adoptar una convención concreta de un fabricante acoplaría el núcleo; si se elige, documentar transformación y casos por cuadrante. La interpretación exacta de contadores reactivos ADW210/Eastron bloquea su equivalencia, no la investigación genérica de S02 |
| D05 — Exactitud, resolución y tiempo | Diseñar después un perfil genérico parametrizable, sin prometer cadencia o exactitud de marca. Mantener tolerancias matemáticas del motor separadas de incertidumbre CT/medidor. El 1 s de S5 sólo acredita el bloque indicado | Un perfil metrológico específico requiere presupuesto de incertidumbre, resolución, simultaneidad y refresco por magnitud; diferimiento propuesto hasta autorizar ese perfil, nunca sustituir desconocidos por ceros o cifras inferidas de floats |
| D06 — Traslado a S03 y cierre formal | Exigir en S03 identidad/alcance, calidad, unidades y codificación, referencia temporal y procedencia, con plan de conservación por nivel de aceptación. Someter ahora la suficiencia de S01 y D01–D05 a revisión humana | Si no se aprueban límites/diferimientos, mantener S01 abierta con los impedimentos identificados. La alternativa a una proyección legada con pérdidas es una evolución separadamente aprobada; S01 no decide migraciones ni valida persistencia |

Los diferimientos propuestos identifican alcance y condición de cierre; todos
conservan estado pendiente. Antes de aprobarlos, el responsable debe registrar
la decisión y su efecto sobre el requisito previo del apartado I del roadmap
técnico. No se marca cumplido un requisito que permanece PEN ni se elimina la
investigación obligatoria ADW210 por usar un núcleo genérico.

## Resultado de S01 y criterios de aceptación

| Pregunta de aceptación | Evidencia y resultado |
|---|---|
| 1. ADW210 confirmado / pendiente | A3/A4 confirman cableado trifásico y grupos; A2 documenta magnitudes/mapa de familia. PEN: firmware, aplicación exacta del mapa, configuración monofásica y semánticas señaladas; D01/D02 pendientes |
| 2. Base independiente de fabricante | Catálogo común y diferencias de unidades contrastados en las fichas. Se puede proponer su estudio en S02; no se aprobaron ecuaciones, nominales o tolerancias |
| 3. Diferencias para contrato común | Alcance, referencia de tensión, FP, energía, CT/VT, calidad, codificación y tiempos identificados en A/B; representación definitiva pendiente de S03 |
| 4. Mapa relevante verificable | Tablas acotadas con localización y nivel; XLS PM5310 inspeccionado. No se certifica funcionamiento de un decodificador ni disponibilidad real de todos los registros |
| 5. Incertidumbres diferibles | D02/D05 proponen diferir detalles de emulación, firmware, refresco y metrología específica; impacto y condición de cierre documentados, aprobación pendiente |
| 6. Incertidumbres bloqueantes | Falta de configuración aprobada y de aprobación de diferimientos bloquea cierre formal/inicio S02. Discrepancias ADW210 y compatibilidad de firmware bloquean emulación/adquisición; signos/contadores/tolerancias deben resolverse en S02 antes de implementar |
| 7. Decisiones previas a S02 | Revisar suficiencia de evidencia, referencia D01/D03 y límites D02/D05; aceptar los asuntos que D04/D06 trasladan a S02/S03 sin darlos por resueltos. No hay aprobación registrada |

**Estado:** investigación realizada y entregable disponible para revisión; **S01 no
cerrada formalmente**. No hubo pruebas con hardware, consultas Modbus, ejecución de
generadores, PostgreSQL o Docker. La imposibilidad inicial de capturar algunos PDF
en el navegador se resolvió con inspección local temporal; el acceso al XLS también
se resolvió. Persisten incertidumbres documentales de contenido y aprobaciones,
no una supuesta verificación física. No se inicia automáticamente S02.
