# Simulador — Fase 1: contrato de ingeniería

Fecha: 2026-10-17. Base de la entrega documental inicial: `7656863e434e42fd14ec8719ad940fb8643e9e6d`,
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
considerará cumplido.

**Actualización S01/S02 (2026-10-17):** sobre el commit
`d84dc1fe48e86a14fcccf10ae183ee16f305c602`, el responsable aprobó D01–D06 y
autorizó el cierre documental S01 para el núcleo genérico y el diseño S02, según
la instrucción «SmartEnergy — Cierre formal de S01 y ejecución de S02».
La [matriz, decisiones D01–D06](../instrumentacion/matriz-medidores.md#c-decisiones-d01d06-aprobadas-y-diferimientos)
registra configuración, alcance, impacto sobre el apartado I y condiciones de
cierre de los pendientes. La evidencia instrumental diferida no está cumplida.
**Actualización S02/S03 (2026-10-17):** el responsable aprobó E01–E07 mediante
«SmartEnergy — Cierre formal de S02 y ejecución documental de S03», sobre
`0bde38154ce9630c93f6cd6c38670d4111cad29e`, en `feature/simulador` con árbol
limpio. **S02 queda cerrada como especificación matemática aprobada**, sin
implementación ni validación experimental. Se mantienen D01–D06, los límites
del núcleo y los diferimientos instrumentales S01. La misma instrucción autoriza
únicamente el diseño documental S03 de este contrato y del roadmap: **S03
especificada, pendiente de revisión y aprobación; S04 no iniciada**. No autoriza
implementación, migraciones, servicios, commit ni push.


**Actualización S03/S04 (2026-10-17):** mediante «SmartEnergy — Cierre formal de
S03 y diseño documental de S04», el responsable aprueba expresamente C01–C08 y
cierra **S03 exclusivamente como especificación documental aprobada**, incluida
la corrección puntual de S03.5. Base verificada: `feature/simulador`, HEAD
`024d820f784aa38e321cf5970deb78a4bd838c39`, árbol limpio. Esta instrucción autoriza
el diseño documental S04 en este contrato y el roadmap; **S04 queda propuesta,
pendiente de revisión y aprobación F-D01–F-D05; S05–S09 no iniciadas**.
Los estados de entregas anteriores son históricos; el último registro de
aprobación y el apartado 9 determinan el estado vigente; no se alteran las
matemáticas S02.1–S02.9.

La aprobación C01–C08 dejó pendiente la canonicalización definitiva; la posterior
consolidación D1–D3 de S06 paso 7 aprueba la política en S03.4. La consolidación
D2.1–D2.2 aprueba sus dominios numérico y temporal, todavía sin implementar.
Siguen pendientes unicidad y
deduplicación concurrente demostrables, políticas de
retención/recuperación de artefactos, evidencia de unidades y zonas históricas,
adaptadores, almacenamiento/HTTP, pruebas y autorizaciones de integración por
nivel. No habilita cambios en bases, servicios, interfaces, modelos ML ni recursos
persistentes; tampoco acredita capacidades ejecutadas o validación industrial.

**Actualización S04/S05 (2026-10-17):** la instrucción «SmartEnergy — Cierre formal
de S04 y diseño documental de S05» aprueba F-D01–F-D05 y cierra S04 exclusivamente
como especificación documental (S04.7). Base verificada: `feature/simulador`, HEAD
`a9ae8ef8c7039882271ef2600400db7950986576`, árbol limpio. Autoriza sólo modificar
este contrato y el roadmap para diseñar S05. **S05 especificada, pendiente de
revisión y aprobación I-D01–I-D07; S06–S09 no iniciadas.** No autoriza implementación,
servicios, infraestructura, PostgreSQL, Docker, ML, commit ni push.

**Actualización S05/S06 (2026-10-17):** mediante «SmartEnergy — Cierre documental
de S05 e implementación controlada de S06», el responsable aprueba expresamente
I-D01–I-D07 y cierra **S05 exclusivamente como especificación documental aprobada**.
Base verificada: `feature/simulador`, HEAD
`33d6a91726ea61908afb5cbe976cdc89ac65ebb6`, árbol inicialmente limpio.
Autoriza sólo los pasos internos 1–3 de S06: inspección/estructura mínima, núcleo
matemático S02 y pruebas locales independientes. No autoriza aún escenarios S04,
instrumentación, adquisición, contrato común ejecutable ni composición; tampoco
aplicación, servicios, API, frontend, PostgreSQL, Docker o ML. **S06 iniciada
únicamente para ese incremento; S07–S09 no iniciadas.** Sin commit ni push.

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

Esta sección expresa relaciones y condiciones matemáticas; no implementa
algoritmos. Las convenciones siguientes son requisitos del diseño sintético,
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
| Energía activa/reactiva | Wh / varh | Distinguir activa importada/exportada y reactiva inductiva/capacitiva; especificación S02 aprobada detallada abajo |
| Temperatura de motor | °C, auxiliar | Sólo con perfil auxiliar declarado; no inferirla de un analizador ni inventarla para completar ML |

Los adaptadores futuros convertirán kW/kWh a unidades canónicas exactamente una
vez, registrando unidad original y método. El perfil legado requerirá autorización
y mapeo independiente; no sustituye al contrato común.

S02 especifica por separado los signos de P y Q, la convención de
cuadrantes y su referencia de flujo, los contadores de energía activa importada
y exportada y los de energía reactiva inductiva y capacitiva. Define
acumulación, tratamiento de cambios de signo y, si corresponde, separación por
cuadrante. No equiparar exportación activa con comportamiento capacitivo:
una carga consumidora puede mantener P positiva y presentar Q negativa bajo
la convención propuesta. La generación con exportación activa sigue fuera del
mínimo; especificar su contador no autoriza añadir ese escenario.

La correspondencia entre esos conceptos y los registros denominados
importados/exportados, directos/inversos o de cuatro cuadrantes depende de la
fuente y configuración. S02 la contrasta con las fuentes de la matriz,
incluidas las convenciones distintas de FP descritas en S1. No se fija aquí una
codificación universal; la especificación S02 siguiente resuelve la semántica genérica.
La equivalencia de registros no verificados permanece diferida según D02/D04.

Reglas físicas obligatorias:

- En monofásico, S depende de tensión y corriente eficaces; P depende también del
  FP. En régimen sinusoidal el FP de desplazamiento corresponde al desfase y
  P/Q/S deben formar un triángulo de potencia coherente. Los módulos de P y Q no
  pueden superar S en ese perfil; valores de FP deben estar entre −1 y 1.
- En 3P4W sinusoidal calcular y conservar cada fase. P y Q totales son las sumas
  algebraicas de fases bajo la misma referencia temporal. El mínimo adopta
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

### Especificación matemática S02 — aprobada

Todo este apartado, incluidas sus decisiones E01–E07, es especificación matemática
aprobada por la instrucción S02/S03 registrada en el apartado 1. No es capacidad
implementada ni convención instrumental verificada. Se conserva el razonamiento
de las recomendaciones originales y sus alternativas; su aprobación no amplía
el mínimo ni resuelve los pendientes instrumentales D02/D05.

#### S02.1 — Hipótesis y fuentes

Se recomienda un suministro ideal rígido, neutro ideal y cargas independientes
entre fase y neutro: una rama en 1P2W y tres en 3P4W. Sin pérdidas de conductores,
impedancia de fuente, desplazamiento del neutro, acoplamiento entre cargas ni
cargas entre líneas. Se admite desequilibrio de magnitudes de tensión y de carga;
los ángulos de tensión de referencia son 0°, −120°, +120°, secuencia ABC.
No se resuelve una red eléctrica ni un motor electromecánico.

En cada instante de evolución lenta se usa un estado sinusoidal fundamental
cuasiestacionario a frecuencia común. Las potencias son medias activas/reactivas
del estado sobre ciclos ideales, no potencia instantánea subciclo. Las transiciones
son cambios de envolvente, sin inferir transitorios electromagnéticos. Frecuencia
variable no genera por sí sola una respuesta dinámica de motor o de carga.

Fuentes primarias consultadas el 2026-10-17; las deducciones y elecciones del
núcleo se distinguen de las capacidades de fabricante:

| ID | Fuente y localización | Evidencia utilizada y límite |
|---|---|---|
| M1 | Schneider, Electrical Installation Guide: [Definition of Power Factor](https://www.electrical-installation.org/enwiki/Definition_of_Power_Factor), definición y cociente P/S; revisión web 2026-08-05 | Base de FP; no fija por sí sola un agregado trifásico desequilibrado |
| M2 | Misma guía: [Definition of reactive power](https://www.electrical-installation.org/enwiki/Definition_of_reactive_power), diagrama vectorial y fórmulas de circuitos; revisión web 2026-08-05 | Relaciones sinusoidales por rama; aquí las fórmulas con raíz de tres se limitan a equilibrio exacto |
| M3 | Schneider, [MicroLogic Active, Power Calculation Algorithm](https://productinfo.se.com/micrologic_active_ug/doca0265-micrologic-active-user-guide/English/DOCA0265_MicroLogicActUserGuide_0000988795.xml/%24/MicAct_E_PwrCalcAlgo_0001002184), DOCA0265EN-08, apartados Reactive Power/Apparent Power | Documenta alternativas aritmética/vectorial: exige identificar método. No se importa su algoritmo con armónicos ni se atribuye al PM5310/ADW210 |
| M4 | Enel Colombia, [ET911 Armarios para medidores](https://likinormas.enelcol.com.co/especificaciones/especificaciones-tecnicas-locales-btmt/et911-armarios-para-medidores), alcance y características eléctricas | Referencia local 120/208 V y 60 Hz; no demuestra tensión universal colombiana ni certifica el perfil de ensayo |
| M5 | Enel Colombia, [Generalidades 7.4.3](https://likinormas.enelcol.com.co/normas/acometidas-y-medidores/medidores-de-energia-electrica/generalidades-743-medidores-utilizados-por-enel-colombia), tabla de tensión de medidores | Alternativa documental 277/480 V; no selección de instalación |

S1 de la matriz identifica FP IEC/IEEE y energía por cuadrantes; E1/E2 y A2
documentan denominaciones de energía sin acreditar una equivalencia canónica
universal. Sus incertidumbres específicas siguen vigentes (D02/D04). Las ecuaciones
fasoriales y las integrales siguientes son deducciones para el modelo ideal;
los nominales, tolerancias y oráculos son propuestas propias, no datos de M1–M5.

#### S02.2 — Catálogo, obtención y nulabilidad

Notación: k es A en 1P2W o A/B/C en 3P4W; j²=−1; «fasorial» distingue el fasor
RMS de su módulo y conjugado se indica explícitamente. Los valores son reales finitos salvo los
fasores auxiliares. Los límites de ensayo de S02.6 se añaden a estos dominios
físicos. «Indisponible» no prescribe aún un tipo o campo de S03.

| Magnitud / símbolo | Unidad | Alcance, signo y dominio físico | Obtención e hipótesis; indisponibilidad |
|---|---|---|---|
| Tensión eficaz U_k | V | Fase-neutro; ≥0 | Módulo del fasor de suministro prescrito; disponible en el modelo ideal aun con carga cero |
| Ángulo de tensión θ_k | rad | Fase respecto de A; referencia común | Parámetro fasorial, no canal de medidor prometido; ángulo físico indefinido si U_k=0, aunque se conserve una referencia de cálculo |
| Tensión eficaz U_kl | V | Par de fases distinto; ≥0 | Módulo de diferencia fasorial; no aplica en 1P2W fase-neutro |
| Admitancia Y_k=G_k+jB_k | S (siemens) | Rama; G_k≥0, B_k real, ambos finitos | Entrada conceptual de carga; B<0 inductiva y B>0 capacitiva. No confundir unidad S con símbolo de potencia aparente |
| Corriente eficaz I_k | A | Conductor de fase; ≥0 | Módulo de Y_k·U_k fasorial; orientación positiva hacia carga; no hay corriente total escalar |
| Corriente eficaz de neutro I_N | A | Conductor neutro de circuito; ≥0 | Suma fasorial con orientación declarada en S02.3; indisponible si sólo se conocen módulos sin ángulos suficientes |
| Desfase φ_k | rad | Fase; [−π/2,π/2] en consumidor | Diferencia de ángulos U/I; atan2(Q_k,P_k) si S_k>0; indefinido con S_k=0 |
| Potencia activa P_k, P_T | W | Fase y total de circuito; ≥0 en mínimo | Parte real de potencia compleja; suma algebraica para total. Valor negativo rechaza perfil consumidor |
| Potencia reactiva Q_k, Q_T | var | Fase y total; real con signo | Parte imaginaria y suma algebraica; + inductiva, − capacitiva para consumo; cero admite compensación |
| Potencia aparente S_k, S_A | VA | Fase y total aritmético; ≥0 | Producto RMS por fase y suma de módulos; método explícito, no módulo de suma compleja |
| FP_k, FP_A | 1 | Fase y total aritmético; [0,1] en mínimo | P/S del mismo alcance; indefinido únicamente cuando su S=0. FP no codifica inductivo/capacitivo |
| Frecuencia f | Hz | Suministro; >0 | Parámetro común del estado sinusoidal; referencia ideal incluso con tensión cero, sin afirmar que un instrumento pueda medirla en un apagón |
| Activa importada E_P+, exportada E_P− | Wh | Fase o circuito explícito; ≥0 | Acumuladores direccionales S02.5; definidos con inicio conocido. Exportada permanece en su valor inicial en el mínimo |
| Reactiva inductiva E_Q+, capacitiva E_Q− | varh | Fase o circuito explícito; ≥0 | Acumuladores separados por signo de Q según S02.5; definidos con inicio conocido |
| Saldos E_Pnet, E_Qnet | Wh, varh | Mismo alcance; con signo | Diferencia de contadores respectivos; derivados, no contadores monótonos |
| Temperatura auxiliar T_motor | °C | Motor; real finito si existe perfil auxiliar | Indisponible en S02: no hay modelo térmico ni perfil auxiliar aprobado. No se deduce de U/I/P, no se rellena para satisfacer ML |

En este diseño las entradas completas determinan todos los resultados eléctricos;
un parámetro requerido ausente/inválido impide evaluar ese estado, no genera ceros.
No se modelan aquí datos perdidos ni calidad de observaciones. Energía de un
histórico sin inicialización conocida no se inventa; el ensayo siempre declara E(t0).
Los nombres de esta tabla son notación matemática, no esquema del contrato común.

#### S02.3 — Relaciones por fase y agregación

Con U_k e I_k como módulos RMS y φ_k=θ_k−arg(I_k fasorial):

```text
U_k fasorial = U_k exp(j θ_k)
I_k fasorial = Y_k · U_k fasorial
C_k = U_k fasorial · conjugado(I_k fasorial) = P_k + j Q_k
P_k = U_k I_k cos(φ_k) = G_k U_k²
Q_k = U_k I_k sin(φ_k) = −B_k U_k²
S_k = U_k I_k = sqrt(P_k² + Q_k²)
FP_k = P_k/S_k = cos(φ_k), sólo si S_k > 0
```

Se recomienda **admitancia prescrita por tramos**, constante dentro de cada tramo
estable y con envolvente declarada si varía lentamente. Resuelve tensión cero
sin singularidad: U=0 ⇒ I=P=Q=S=0 para Y finita. Una caída de tensión con Y fija
reduce I proporcionalmente a U y P/Q proporcionalmente a U². No se recortan
independientemente P, Q o I para cumplir límites: un conjunto inconsistente se
rechaza. Para construir un caso con U>0 y P/Q conocidos, Y=(P−jQ)/U².
La carga de potencia constante requeriría otra respuesta ante subtensión y
singularidades; no se recomienda añadirla al mínimo. G=0 permite comprobar
reactancia pura (FP=0 si S>0), sin confundirla con carga cero.

Para el conjunto de fases presentes, simultáneas y en la misma referencia:

```text
P_T = Σ P_k; Q_T = Σ Q_k; C_T = P_T + j Q_T
S_A = Σ S_k; FP_A = P_T/S_A si S_A>0
S_V = |C_T| = sqrt(P_T² + Q_T²) ≤ S_A
```

Se recomienda S_A por ser aditivo, transparente y permitir conservar cada rama.
S_V es una alternativa vectorial distinta; no se sustituye silenciosamente.
La igualdad S_V=S_A exige potencias complejas no nulas con el mismo argumento;
ni tensiones equilibradas ni ausencia de armónicos bastan. El agregado aritmético
no caracteriza por sí solo pérdidas por desequilibrio ni equivale a una potencia
aparente efectiva normativa. No se afirma cumplimiento de IEEE 1459.

FP verdadero se entiende por fase como potencia activa media dividida por
producto RMS; FP de desplazamiento es coseno del desfase fundamental. Coinciden
por fase en este régimen sinusoidal. FP_A es el cociente con agregado aritmético:
no hay un único desfase de corriente trifásica cuyo coseno lo represente en
general. Aun sin armónicos, no se impone FP_A=cos(atan2(Q_T,P_T)). Un fabricante
puede usar otra definición total: la conversión requiere conocer su método.

```text
U_kl fasorial = U_k fasorial − U_l fasorial
U_kl² = U_k² + U_l² − 2 U_k U_l cos(θ_k−θ_l)
```

Sólo con iguales módulos U y separación 120° se obtiene U_LL=√3 U_LN.
Sólo con corrientes de igual módulo y desfase de carga común se permite además
S_A=3 U_LN I=√3 U_LL I, P_T=S_A cosφ y Q_T=S_A sinφ. En desequilibrio se usan
las ecuaciones por fase, incluso si el suministro sigue equilibrado.

Tomando todos los conductores positivos hacia la carga, Kirchhoff da
I_N fasorial=−Σ I_k fasorial e I_N=|Σ I_k fasorial|. En 1P2W I_N=I_A.
Se excluyen fugas y otros caminos de retorno; módulos sin desfases no bastan
para el neutro. Ejemplo de tensión desigual: U_A=120 V, U_B=100 V y ángulo 120°
dan U_AB=√36400=190,787840283 V; no √3·120 ni 120+100.

La referencia compartida ADW210 significa que varios circuitos pueden usar el
mismo conjunto de fasores de tensión del suministro, con corrientes/cargas propias.
No se duplican ni suman esas tensiones como magnitudes independientes de cada CT.
Es una asociación física del perfil documental D01, no requisito universal de
todo instrumento ni diseño de identidad/representación S03.

#### S02.4 — Flujo, signos y cuadrantes

Referencia pasiva en terminales: corriente de fase hacia la carga y tensión
fase-neutro. S siempre es no negativa. Se recomienda FP=P/S con signo de P;
en el mínimo FP≥0 tanto en consumo inductivo como capacitivo. La naturaleza
reactiva se conserva mediante Q/φ, sin sobrecargar el signo de FP.

| Cuadrante geométrico | P | Q | Significado respecto de la referencia | Estado |
|---|---|---|---|---|
| I | >0 | >0 | Consumo activo, reactiva inductiva | Incluido |
| IV | >0 | <0 | Consumo activo, reactiva capacitiva | Incluido; no es exportación activa |
| II | <0 | >0 | Exportación activa, Q positiva | Fuera del mínimo; no etiquetar carga inductiva por analogía automática |
| III | <0 | <0 | Exportación activa, Q negativa | Fuera del mínimo |

Ejes no pertenecen a un cuadrante abierto: P>0,Q=0 es resistivo neto; P=0,Q≠0
es reactivo puro y FP=0; P=Q=0 puede ser carga cero por fase o compensación en
un agregado. FP total sólo es indefinido si S_A=0. Un Q_T=0 no demuestra ausencia
de corriente reactiva en cada fase. No hay código de cuadrante universal de registros.

S1 diferencia signos de FP IEC/IEEE según flujo y adelanto/atraso; no copiar
esos signos al cociente canónico. A2/E1/E2 pueden denominar contadores reactivos
«importados/exportados»: no equipararlos sin evidencia con E_Q+/E_Q−, ni con
E_P+/E_P−. Una futura conversión conservará semántica original y método verificado;
lo pendiente sigue bloqueando equivalencia instrumental, no esta propuesta física.

#### S02.5 — Tiempo físico e integración

t es tiempo físico relativo en segundos, no timestamp de adquisición. Partición
t0<t1<…<tn, intervalos [ti,ti+1); el estado nuevo rige en ti y el punto final no
añade duración. Integrar intervalos adyacentes una sola vez equivale al intervalo
completo; no redondear cada incremento a la resolución de un medidor.
Duración negativa o tramo de duración cero se rechazan. Se permite consultar el
estado inicial t0 sin avanzar energía. No se eligen eventos, relojes de entrega,
zona horaria, almacenamiento ni representación S03.

Para cada alcance a (fase k o circuito T), con contadores iniciales no negativos
declarados, cero por defecto en un ensayo nuevo:

```text
E_P+(t) = E_P+(t0) + (1/3600) ∫[t0,t] max(P_a(τ),0) dτ
E_P−(t) = E_P−(t0) + (1/3600) ∫[t0,t] max(−P_a(τ),0) dτ
E_Q+(t) = E_Q+(t0) + (1/3600) ∫[t0,t] max(Q_a(τ),0) dτ
E_Q−(t) = E_Q−(t0) + (1/3600) ∫[t0,t] max(−Q_a(τ),0) dτ
E_Pnet = E_P+ − E_P−; E_Qnet = E_Q+ − E_Q−
```

E_Q+ es acumulación inductiva y E_Q− capacitiva **en el perfil consumidor**;
la extensión a exportación exigiría revisar esa denominación y separación por
cuadrantes. E_P− define energía activa saliente, pero no habilita escenarios
P<0. Sus incrementos son cero en el mínimo. Los cuatro contadores no decrecen
y una inversión de Q incrementa el contador opuesto sin borrar el anterior.
En reposo todos se conservan; el saldo neto puede disminuir sin que disminuya
ningún contador. Reinicio/rollover no es parte de la física ideal: ensayo nuevo
o reanudación con los mismos valores iniciales explícitos, nunca reset silencioso.

**Orden de agregación recomendado:** para contadores de circuito, sumar primero
P_k/Q_k simultáneos y después separar signos e integrar. Para contadores de fase,
aplicar la misma regla a cada fase. No sumar contadores direccionales de fase
para sustituir los del circuito: max(ΣQ_k,0) puede diferir de Σmax(Q_k,0).
Los incrementos netos sí son aditivos. Con Q_A=+720, Q_B=−720 y Q_C=0 constantes
durante una hora, el circuito suma cero a ambos contadores reactivos, mientras
las fases acumulan 720 varh inductivos y 720 varh capacitivos. Son alcances distintos.

Para P/Q constantes por tramo, ΔE=P·Δt/3600 con separación de signo es exacta.
Dividir en cada discontinuidad conocida; una muestra al final del tramo no
representa automáticamente su media. Para P/Q lineales por tramo, trapecios son
exactos para energía neta, pero los contadores separados exigen partir también
en cada cruce por cero. Para perfiles suaves no lineales se propone trapecio
compuesto sobre potencias del estado físico, con refinamiento y oráculo analítico;
interpolar U/I por separado no implica P lineal. La energía física es independiente
de la cadencia de lectura instrumental y de datos perdidos o duplicados.

Convergencia propuesta: comparar pasos h,h/2,h/4, manteniendo fronteras/cruces,
contra la integral independiente de cada contador. En C13, cuadrático sobre red
uniforme, se exige orden 2 (cociente de errores cercano a 4; banda propuesta
3,5–4,5 antes del piso de redondeo). Para otro perfil suave, justificar su orden
analíticamente: cancelación del término dominante puede dar otro cociente.
Debe cumplirse además el presupuesto de error S02.7. Casos constantes/lineales
exactos se comprueban por exactitud, sin exigir cociente de errores 0/0. Convergencia entre dos
aproximaciones sin oráculo no demuestra corrección.

#### S02.6 — Perfil nominal y dominios de ensayo propuestos

Se recomienda 60 Hz y U_n=120 V fase-neutro. En equilibrio U_LL=120√3 V,
aproximadamente 207,846096908 V, denominado comercialmente 208/120 V (M4).
No usar 208 exactos y 120 exactos simultáneamente en una identidad de equilibrio.
Alternativa: U_n=277 V, U_LL=277√3≈479,778073697 V, denominada 480/277 V (M5).
Otra instalación puede declarar 220 V fase-neutro y dar 381,051177665 V entre
líneas; no confundirla con 220 V entre líneas. Elegir 120 simplifica oráculos y
dispone de referencia colombiana; no representa todas las plantas industriales.

| Parámetro/magnitud | Nominal o dominio de ensayo | Condición |
|---|---|---|
| Topología | 1P2W fase-neutro; 3P4W estrella con neutro ideal | Sin cargas L-L, 3P3W ni neutro flotante |
| U_k | U_n=120 V; 0…1,2 U_n, incluida interrupción ideal | Alternativas requieren declarar U_n y recalcular límites derivados |
| θ_A/B/C | 0, −2π/3, +2π/3 rad | En mínimo el desequilibrio de tensión cambia módulos, no separación angular |
| f | 60 Hz nominal; 50…65 Hz de ensayo | Rango sintético, no tolerancia de red. Y declarada para cada estado; no se deriva respuesta RLC a cambios de f |
| I_k | Referencia 10 A; dominio 0…100 A | Límite de ensayo propio; no capacidad homologada CT/medidor ni umbral de sobrecarga |
| P_k, Q_k, S_k | P_k≥0, Q_k con signo; S_k≤U_k·100 A | Para U_n=120: S_k≤14400 VA y P_k²+Q_k²=S_k²; los límites no son independientes |
| Totales | S_A≤43200 VA en 3P4W; P_T≥0 y P_T²+Q_T²≤S_A² | En 1P2W total igual a su única fase |
| FP_k, FP_A | [0,1] si denominador>0 | Nominal de ejemplos 0,8 o 1; incluir reactancia pura y cercanía a cero |
| I_N | 0…ΣI_k, cota 300 A en 3P4W | Cota matemática conservadora, no dimensionamiento de conductor |
| Duración T | 0<T≤86400 s; consulta inicial permitida | Un millón de subintervalos como máximo por ensayo de precisión propuesto; no restricción universal del producto |
| Contadores | Inicio finito ≥0; ΔE direccional ≥0 | Incremento por circuito ≤1036800 Wh o varh en 24 h para U_n=120; fases ≤345600. No límite de vida útil ni modelo de rollover |

G≥0 y B finita se eligen de forma que toda la trayectoria cumpla I≤100 A y
los límites derivados; el caso fuera de dominio se rechaza, no se satura.
Sobrecarga, subtensión, duración de arranque y sus umbrales diagnósticos quedan
para S04. Los casos de S02 son oráculos matemáticos, no su catálogo de escenarios.

#### S02.7 — Tolerancias propuestas y casos próximos a cero

Para un resultado x y referencia independiente x_ref, exigir
|x−x_ref|≤a_x+r_x|x_ref| en la unidad canónica. Se recomienda aritmética binaria
de 64 bits para el futuro núcleo; estos presupuestos permiten error de representación
y operaciones, no error de modelo ni incertidumbre instrumental. Se deben probar
con el entorno elegido antes de aceptar implementación; no ajustarlos para ocultar fallos.

| Variable o resultado | a_x absoluto | r_x relativo |
|---|---|---|
| U, I (incluido neutro) | 1e−9 V o A | 1e−9 |
| f, ángulos definidos | 1e−9 Hz o rad | 1e−9 |
| P, Q, S (fase y total) | 1e−6 W, var o VA | 1e−9 |
| FP definido | 1e−12 | 1e−9 |
| Contadores/saldos con integración analítica, constante o lineal | 1e−6 Wh o varh | 1e−9 |

Justificación de escala: en binario de 64 bits, ε≈2,22e−16; para potencias de
hasta 43200 VA el redondeo elemental es del orden de 1e−11 VA, muy inferior al
presupuesto absoluto. Para hasta 10^6 sumandos positivos, la cota clásica de
acumulación Nε/(1−Nε) es aproximadamente 2,22e−10 relativa, por debajo de 1e−9.
Esto orienta el presupuesto, no prueba la estabilidad de cualquier algoritmo:
cancelaciones, resta de grandes acumuladores y productos requieren los oráculos
y controles por incremento. Evaluar energía añadida independientemente del gran
valor inicial; una comparación relativa sólo del contador final podría ocultarla.

Para integrales suaves aproximadas añadir un presupuesto de discretización
separado: ≤1e−3 Wh/varh + 1e−6·E_abs_ref, donde E_abs_ref es la integral del
valor absoluto de la potencia relevante, en horas. En saldos con cancelación
usar esa escala, no el saldo cercano a cero. Verificar además cada contador
positivo por separado. La cota de redondeo no sustituye la de discretización.

Protecciones contra tolerancias excesivas:

- Para referencia no nula próxima a cero, exigir también error relativo ≤1e−6
  y signo correcto. Es un requisito adicional al de la tabla, no una alternativa
  que permita aceptar cero. Para FP se conserva su cota propia y el dominio.
- No declarar S=0 porque S sea menor que a_S: cero físico requiere U=0 o Y=0
  en la rama (o todas las ramas en el agregado). S>0 arbitrariamente pequeño
  conserva FP definido; no recortar a cero. En casos diminutos usar entradas
  escaladas y referencia analítica con precisión suficiente; si underflow impide
  el cálculo, rechazar el caso como fuera del dominio numérico, no inventar un valor.
- Para referencia cero, aplicar cota absoluta más invariantes estructurales:
  reposo explícito produce I/P/Q/S=0 y FP indefinido; contadores constantes y
  E_P− sin incremento son exactos. Para cancelación fasorial, sí admitir residuo
  numérico acotado; no afirmar una corriente negativa o FP fuera de [0,1].
- Comparar ecuaciones mediante residuos con unidades: por ejemplo
  S_k−hypot(P_k,Q_k) en VA, no P²+Q²−S² con una tolerancia en W.
  Verificar cada fase antes del agregado y cada incremento antes del saldo.
  La tolerancia nunca cambia disponibilidad, dominio o monotonía.
- Caso centinela: U=120 V, S=1e−6 VA, P=8e−7 W, Q=6e−7 var,
  I=(1e−6/120) A y FP=0,8. Aunque a_P sea mayor que P, devolver P=0 falla
  por el criterio relativo adicional. Añadir la versión capacitiva Q=−6e−7.

Resolución instrumental (cuantización), incertidumbre CT/medidor y tiempos de
refresco/sondeo son asuntos diferentes, diferidos por D05 hacia el perfil S05
o adaptador específico. Ninguna clase de exactitud justifica estas tolerancias.
Los decimales impresos abajo son de lectura; los valores radicales, fracciones
e integrales son la referencia, sin redondeo previo al comparar.

#### S02.8 — Oráculos analíticos independientes

Todos parten de senoides fundamentales, suministro ideal y contadores iniciales
cero salvo indicación. Potencias en W/var/VA y energía en Wh/varh. Cada entrada
P/Q especificada determina Y=(P−jQ)/U², sin generar variables independientes.

| Caso | Entradas y operaciones | Resultado esperado |
|---|---|---|
| C01 — Monofásica resistiva | U=120, I=10, φ=0; Y=1/12 S; S=120·10 | P=1200, Q=0, S=1200, FP=1, I_N=10 A |
| C02 — Monofásica inductiva | U=120, I=10, cosφ=4/5, sinφ=3/5; Y=1/15−j/20 S | P=960, Q=+720, S=1200, FP=0,8; φ=+36,869897646° |
| C03 — Monofásica capacitiva | C02 con sinφ=−3/5 e Y=1/15+j/20 S | P=960, Q=−720, S=1200, FP=0,8; cuadrante IV, sin incremento activo exportado |
| C04 — Trifásica equilibrada | U_k=120; θ=0,−120,+120°; cada rama como C02; sumar tres fases | P_T=2880, Q_T=2160, S_A=S_V=3600, FP_A=0,8; U_LL=120√3; I_N=0 |
| C05 — Trifásica desequilibrada | Mismas tensiones, resistencias con I_A=10,I_B=20,I_C=0; Y_A=1/12,Y_B=1/6,Y_C=0 S | P_k=(1200,2400,0), P_T=3600, Q_T=0, S_A=3600, FP_A=1; suma de corrientes=−j10√3 A ⇒ I_N=10√3≈17,320508076 A; FP_C indefinido |
| C06 — Carga cero | U_k=120, Y_k=0; I=P=Q=S=0; mantener estado 600 s | FP por fase/total indefinido; I_N=0, f de referencia=60 Hz; todos los contadores conservan su inicio |
| C07 — Potencia constante | Una rama U=120, P=1200,Q=900, T=600 s; S=1500, I=12,5, FP=0,8 | ΔE_P+=1200·600/3600=200; ΔE_Q+=900·600/3600=150; otros incrementos cero |
| C08 — Potencia por tramos | [0,600): P=1200,Q=900; [600,900): P=600,Q=−900; U=120 | E_P+=200+50=250; E_Q+=150; E_Q−=75; E_Qnet=75; E_P−=0. La frontera 600 s no añade otro incremento |
| C09 — Q cambia de signo con P>0 | U=120,P=1200 constante; Q=+900 en [0,1800), −900 en [1800,3600) | E_P+=1200, E_P−=0; E_Q+=450, E_Q−=450, saldo reactivo=0. Ningún contador disminuye |
| C10 — Acumuladores inicializados | Mismo perfil C09; inicio (E_P+,E_P−,E_Q+,E_Q−)=(100,7,20,30) | Final (1300,7,470,480); saldos P=1293, Q=−10. Reanudar con esos finales y carga cero no los modifica |

Controles adicionales necesarios para distinguir alternativas:

- **C11, compensación entre fases:** tensiones equilibradas de 120 V; ramas
  A/B como C02/C03 y C resistiva de C01. P_T=960+960+1200=3120;
  Q_T=720−720=0; S_A=3600; S_V=3120; FP_A=13/15≈0,866666667.
  El cociente P_T/S_V=1 no es FP_A. En una hora E_P+=3120 y ambos incrementos
  reactivos del circuito son cero; por fase A/B acumulan 720 varh en sentidos
  distintos. El agregado no oculta sus corrientes de fase.
- **C12, cruce lineal:** U=120, P=1200 y Q(t)=900(1−2t/3600), 0≤t≤3600.
  Partir en t=1800: cada triángulo da (900·1800/2)/3600=225 varh.
  E_Q+=E_Q−=225, E_Qnet=0, E_P+=1200. Integrar sólo el Q neto perdería ambos
  contadores. Y(t)=(1200−jQ(t))/14400 es finita y consumidora.
- **C13, convergencia no lineal:** U=120, Q=0,
  P(t)=1200+600(t/3600)², T=3600; Y(t)=P(t)/14400.
  Integral exacta E_P+=1200+600/3=1400 Wh. Trapecio compuesto con N intervalos
  uniformes produce 1400+100/N²: N=100 → 1400,01; N=200 → 1400,0025;
  N=400 → 1400,000625 Wh. Errores 0,01/0,0025/0,000625, cociente 4;
  N=400 cumple presupuesto 0,001+0,0014=0,0024 Wh. No usar tolerancia de
  redondeo para justificar error de discretización.
- **C14, subtensión y ejes:** con Y=1/12 S fija, U de 120 a 60 V reduce
  I de 10 a 5 A y P de 1200 a 300 W. Con U=0 da reposo eléctrico y FP
  indefinido. Con U=120 e Y=−j/12 S: I=10, P=0,Q=1200,S=1200,FP=0;
  Y=+j/12 invierte Q, conservando P=0. No hay exportación activa.

Estos resultados proceden de productos, sumas fasoriales, áreas e integrales
analíticas, no de un simulador ni de datos del producto. La futura prueba deberá
conservar esos oráculos independientes de la implementación.

#### S02.9 — Decisiones técnicas aprobadas y límites

| ID | Decisión aprobada el 2026-10-17 | Consecuencia / alternativa |
|---|---|---|
| E01 | 1P2W L-N y 3P4W estrella, fuente/neutro ideales, Y prescrita y equilibrio angular ABC | Admite desequilibrio de módulos/cargas; neutro flotante, cargas L-L, potencia constante y dinámica RLC requerirían más diseño |
| E02 | P hacia carga; Q+ inductiva/Q− capacitiva; FP=P/S sin signo reactivo | Preserva consumo capacitivo; cualquier codificación de fabricante precisa transformación específica verificada |
| E03 | S_A=ΣS_k, FP_A=P_T/S_A y Q_T=ΣQ_k | Transparente para el mínimo; no representa potencia efectiva ni FP normativo de toda red desequilibrada |
| E04 | Separación de contadores por signo después de agregar por alcance; intervalos semiabiertos e integración de S02.5 | Conserva monotonía y evita doble conteo; sumar contadores direccionales de fase es otra magnitud |
| E05 | 120 V L-N, 60 Hz, 10 A de referencia y dominios S02.6 | Hipótesis de ensayo; alternativas 277/480 u otra tensión requieren nominal explícito, sin pretensión universal |
| E06 | Tolerancias S02.7, guardas próximas a cero y convergencia C13 | Aceptación previa a S06; precisión y paso deberán demostrar cumplimiento sin aumentar tolerancias a posteriori |
| E07 | Temperatura auxiliar indisponible y límites expresos del modelo ideal | Compatibilidad con las cuatro variables legadas y su nulabilidad se decide en S03; no proyectar ahora a Registro ni ML |

Riesgos conocidos: usar el cociente vectorial como FP_A, perder compensaciones
entre fases al acumular, aplicar √3 bajo desequilibrio, tratar envolventes como
transitorios reales, inferir equivalencia instrumental del núcleo ideal o convertir
dominios de ensayo en umbrales industriales. S03 conserva identidad, calidad,
procedencia, unidades representadas, tiempo de observación y compatibilidad.
La presente notación no migra las unidades/timestamps históricos ni amplía el ORM.
**S02 cerrada como especificación matemática aprobada E01–E07; sin implementación
ni validación experimental. S03 se especifica en el apartado 6, pendiente de aprobación.**

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
al escenario físico original y su versión. La especificación S03 del apartado 6 define
la representación y la semántica de los límites temporales, aprobadas con C04.
Se permiten referencias por intervalo o segmento que permitan reconstruir la relación; no se exige un enlace por cada
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

### S04 — Escenarios físicos y perfiles aprobados

**Especificación documental aprobada F-D01–F-D05 el 2026-10-17**, sin
implementación ni aceptación experimental. La investigación se limita a derivar
casos del modelo aprobado
S02 y del contrato S03; no requiere nuevas atribuciones instrumentales ni normas
externas. F01–F05 conservan exactamente sus significados de la tabla anterior.
Los sufijos siguientes son casos de esas familias; `S02:Cxx` identifica oráculos
S02.8 y `S03:Cxx` decisiones S03.8, nunca escenarios físicos nuevos.

#### S04.1 — Configuración común y manifiesto reproducible

Cada fila de S04.2 es una especificación completa al combinarse con estos defaults
y sus reemplazos explícitos. ID y versión de escenario son `<ID>@1`; cada variante
usa `<ID>-<variante>@1` (p. ej. F01-EST-capacitiva@1, F01-REP-24h@1), con sus
reemplazos expandidos en el manifiesto. No se reutiliza un ID para entradas distintas; modificar
entradas o semántica exige nueva versión. No se sortean valores: seed=null y
algoritmo aleatorio=null. Una repetición conserva entradas, reloj simulado y
resultados, independientemente de su velocidad real de ejecución.

| Elemento | Valor o regla común propuesta |
|---|---|
| Identidad | installation_id=plant-a, device_id=sim-a, circuit_id=c1, voltage_reference_id=bus-a; L1/L2/L3 equivalen a A/B/C de S02; 1P2W usa sólo L1-N, 3P4W usa las tres fases y neutro ideal |
| Configuración | `s04-1p@1` o `s04-3p@1` según topología; cada manifiesto resuelve configuración, unidades y perfiles exactos de la fila, sin asumir ADW210 implementado |
| Referencia | U_n=120 V L-N, f=60 Hz; 3P4W ángulos 0,−2π/3,+2π/3 rad. U_LL=120√3 sólo en equilibrio; no usar 208 exactos. Referencia sintética, no planta universal |
| Tiempo | Origen 2026-10-17T12:00:00Z, t en s, soporte [0,T); transiciones rigen en su inicio. Consultar estado final en T no añade energía. Cadencia de estados de ensayo h=1 s con todas las fronteras añadidas; no es cadencia de un medidor |
| Entradas | U_k(t) en V y Y_k(t)=G_k+jB_k en S; únicamente fuente y admitancia prescritas, con dominios S02.6 para toda la trayectoria, no sólo extremos muestreados. P/Q de referencia en la tabla sirven para especificar Y, no para un controlador de potencia constante |
| Iniciales | E0=(E_P+,E_P−,E_Q+,E_Q−)=(0,0,0,0) por cada alcance salvo variante explícita; época `run_id/epoch-0`; eventos iniciales con estado definido, sin tramo anterior implícito |
| Salidas físicas | U/I/P/Q/S/FP por fase, P_T/Q_T/S_A/FP_A, I_N, tensiones L-L donde aplican, f y cuatro contadores por fase/circuito. Totales y fases nunca se suman juntos. Temperatura unsupported/null; no modelo térmico |
| Manifiesto | ID/versión, run_id, versión de contrato 1.0.0, motor futuro y entorno numérico (pendientes hasta S06), configuration_ref, perfiles/unidades, duración/transiciones, E0/época, h/fronteras, métodos de integración/observación, semillas nulas y referencias/digests del artefacto S03.6 |
| Terminación | Fin físico T, sin dependencia de now/sleep. Rechazar antes de producir segmentos aceptados si las entradas no cumplen S04.5. No activar escrituras/retención por terminar un ensayo |

#### S04.2 — Casos físicos y resultados independientes

Oráculo común deducido de S02: I_k=U_k|Y_k|, P_k=U_k²G_k,
Q_k=−U_k²B_k, S_k=U_k²|Y_k|; agregados S02.3 y neutro por suma fasorial.
En 1P2W I_N=I_L1 y total=única fase, conservando IDs distintos. Para cada tramo,
ΔE_P+=∫P/3600, ΔE_P−=0, ΔE_Q±=∫max(±Q,0)/3600; sumar Q de fases **antes** de
separar signos para contadores de circuito. Los resultados de la tabla son exactos
como fracciones/radicales/integrales, salvo decimales marcados ≈. Energías en Wh/varh;
P/Q/S en W/var/VA. Todos los casos mantienen f=60 y temperatura indisponible.

| ID / propósito / topología y T | Entradas, transiciones y estado operativo de ensayo | Oráculo de magnitudes e incrementos energéticos |
|---|---|---|
| F01-REP — reposo, 1P2W, T=600 | U=120, Y=0 todo [0,600); estado reposo | I/P/Q/S/I_N=0 exactos; FP indefinido, no cero; cuatro incrementos cero. Variante prolongada T=86400, E0=(100,7,20,30): final idéntico a E0; no reinicio ni pérdida de energía previa. S02:C06/C10 |
| F01-EST — estable, 1P2W, T=600 | U=120; tres ejecuciones separadas: Y=1/12 resistiva, Y=1/15−j/20 inductiva, Y=1/15+j/20 capacitiva; estado operación | I=10 y S=1200 en todas; (P,Q,FP)=(1200,0,1), (960,720,4/5), (960,−720,4/5). ΔE respectivos (200,0,0,0), (160,0,120,0), (160,0,0,120). Variante 3P4W equilibrada con Y inductiva en cada rama: P_T=2880,Q_T=2160,S_A=3600,FP_A=4/5,I_N=0; ΔE circuito=(480,0,360,0), cada fase=(160,0,120,0). S02:C01–C04 |
| F01-ESC — escalón y cambio de signo Q, 1P2W, T=900 | U=120; Y=(1200−j900)/14400 en [0,600), Y=(600+j900)/14400 en [600,900); operación en ambos tramos, evento de carga t=600 | Primer tramo P=1200,Q=900,S=1500,I=12,5,FP=4/5; segundo P=600,Q=−900,S=300√13≈1081,665382639,I=(5/2)√13≈9,013878189,FP=2/√13. ΔE=(250,0,150,75), saldo Q=75. Consumo capacitivo no es exportación activa. S02:C07/C08 |
| F01-RAM — rampa de admitancia, 1P2W, T=120 | U=120; Y=(600+10t)/14400 en [0,60), Y=1/12 en [60,120); operación/rampa→estable | P=S=600+10t,Q=0,I=5+t/12 en rampa; después P=S=1200,I=10; FP=1. Áreas de potencia 900·60 y 1200·60 ⇒ ΔE_P+=15+20=35, otros cero. Y continua en t=60. Oráculo por integral de recta, S02.5 |
| F01-Q — cruce reactivo, 1P2W, T=3600 | U=120; Y=(1200−j900(1−2t/3600))/14400; operación constante en P con cambio inductivo→capacitivo en t=1800 | P=1200,Q=900(1−2t/3600),S=√(1200²+Q²),I=S/120,FP=1200/S; en cruce FP=1; ΔE=(1200,0,225,225), saldo Q=0. E0 alternativo de S02:C10 sólo se aplica al perfil C09, no a esta rampa. Reutiliza S02:C12; control escalonado alternativo S02:C09/C10 y no lineal S02:C13 sin cambiar sus entradas/resultados |
| F01-TUR — turno por tramos, 1P2W, T=28800 | U=120; Y=0 [0,3600); Y=(1200−j900)/14400 [3600,14400); Y=0 [14400,18000); Y=(1200+j900)/14400 [18000,25200); Y=a(1200+j900)/14400, a=1−(t−25200)/60, [25200,25260); Y=0 [25260,28800). Estados reposo/operación/reposo/operación/parada/reposo | Tramos operativos P=1200,Q=±900,S=1500,I=12,5,FP=4/5; durante parada todas las potencias y corriente se escalan por a, FP=4/5 mientras a>0. Reposo tiene FP indefinido. ΔE_P+=3600+2400+10=6010; ΔE_Q+=2700; ΔE_Q−=1800+7,5=1807,5; exportada cero. No todo cambio de turno es anomalía |
| F02-SOB — sobrecarga acotada, 1P2W, T=120 | U=120; Y=1/12 [0,30) y [90,120), Y=1/6 [30,90). Operación; condición de ensayo respecto de I_cap=15 A y duración mínima 10 s (S04.3) | I=10→20→10, P=S=1200→2400→1200,Q=0,FP=1. Condición I>15 durante exactamente 60 s; ΔE=(60,0,0,0). 20 A cumple dominio ≤100 A; sobrecarga es relativa a capacidad de ensayo, no fuera del dominio matemático |
| F03-SUB — respuesta a subtensión, 1P2W, T=120 | Y=1/12 fija; U=120 [0,30) y [90,120), U=60 [30,90); operación con fuente reducida | I=10→5→10; P=S=1200→300→1200; Q=0,FP=1; condición U<108 V durante 60 s. ΔE=(25,0,0,0). Variante interrupción ideal U=0 sólo en tramo central: I/P/Q/S=0,FP indefinido,f ideal=60; ΔE_P+=20. S02:C14; no suponer potencia constante ni frecuencia medible en apagón |
| F04-CAR — desequilibrio de carga, 3P4W, T=3600 | U_k=120; Y_A=1/12,Y_B=1/6,Y_C=0, operación con rama C en reposo | I=(10,20,0),P=(1200,2400,0),Q_k=0; P_T=S_A=3600,FP_A=1,FP_C indefinido; I_N=10√3≈17,320508076; ΔE_P+ circuito=3600, fases=(1200,2400,0); demás incrementos cero. S02:C05 |
| F04-TEN — desequilibrio de tensión, 3P4W, T=3600 | U=(120,100,120), ángulos ABC sin cambio; Y_k=1/12; operación | I=(10,25/3,10),P=S=(1200,2500/3,1200); Q=0,FP_k=FP_A=1; P_T=S_A=9700/3≈3233,333333333; I_N=5/3 A; U_AB=U_BC=√36400≈190,787840283,U_CA=120√3. ΔE_P+ circuito=9700/3 y fases iguales a sus P; otros cero. Fórmulas fasoriales S02.3, no √3·120 para todos los pares |
| F04-COM — compensación reactiva desigual, 3P4W, T=3600 | U_k=120; Y_A=1/15−j/20,Y_B=1/15+j/20,Y_C=1/12; operación | P=(960,960,1200),Q=(720,−720,0),I_k=10,S_k=1200; P_T=3120,Q_T=0,S_A=3600,FP_A=13/15; I_N=6√3−2≈8,392304845 A (suma fasorial (3√3−1)+j(√3−9) A). ΔE_P+ circuito=3120,ΔE_Q± circuito=0; fase A acumula 720 inductivos y B 720 capacitivos. No afirmar igualdad de corrientes fasoriales por igualdad de módulos. S02:C11 |
| F05-ARR — envolvente de arranque/parada, 1P2W, T=120 | U=120,Y=a(1/15−j/20). a=t/5 [0,10); 2 [10,20); 2−(t−20)/10 [20,30); 1 [30,90); 1−(t−90)/10 [90,100); 0 [100,120). Estados arranque [0,30),operación [30,90),parada [90,100),reposo [100,120) | I=10a,P=960a,Q=720a,S=1200a; FP=4/5 si a>0, indefinido si a=0; I máximo=20 A. ∫a dt=10+20+15+60+5=110 s ⇒ ΔE_P+=88/3≈29,333333333,ΔE_Q+=22; otros cero. Continuidad de Y en todas las fronteras, inicio con Y=0; no transitorio electromagnético ni motor dinámico |
| F02-F03-JUN — combinación física, 1P2W, T=120 | U=120,Y=1/12 antes de 30 y desde 90; U=60,Y=1/3 en [30,90). Operación; eventos simultáneos de fuente y carga, evaluados como un nuevo estado completo | I=10→20→10,P=S=1200 todo el ensayo,Q=0,FP=1. Coexisten U<108 e I>15 durante 60 s, ΔE_P+=40, demás cero. La misma P no demuestra normalidad ni identifica la causa. No es C01 combinado de fallos, que sigue reservado a S05 |

Los perfiles con potencia usada para definir Y lo hacen a la tensión de referencia
indicada. Sólo F03-SUB varía U con Y fija; en el caso conjunto U/Y cambian según
su tabla, sin realimentación. No exigir que una carga de admitancia mantenga P
cuando baje U. Todas las variantes citadas de S02:C09/C10/C13/C14 conservan sus
soportes y entradas originales, con el contexto común; no amplían el modelo.

#### S04.3 — Condiciones de ensayo y límites de interpretación

Condiciones exclusivamente sintéticas aprobadas mediante F-D02:

- F02-SOB: capacidad I_cap=15 A; condición sobrecarga si I_L1>15 A de forma
  continua al menos 10 s. Umbral estricto: exactamente 15 A no cumple; 20 A
  durante 9 s no alcanza duración, 10 s sí. La condición física comienza al
  cruzar el umbral, pero su duración sólo se confirma tras observar el intervalo.
- F03-SUB: U_n=120 V y umbral 0,9U_n=108 V; condición subtensión si U_L1N<108 V
  durante al menos 10 s. Exactamente 108 V no cumple. Ni 108 V ni 15 A son
  límites del dominio S02; no se recorta ni invalida la física por cruzarlos.
- F04: distinguir perfil de tensión y de carga, sin atribuir índice normativo.
  Para los casos resistivos se propone dispersión d_U=(max U_k−min U_k)/120 y
  d_I=(max I_k−min I_k)/10; condición d_U>0,10 o d_I>0,20 durante al menos 10 s.
  F04-CAR da d_I=2 y d_U=0; F04-TEN da d_U=1/6 y d_I=1/6, por lo que sólo el
  criterio de tensión cruza su umbral. F04-COM es asimetría de admitancias/desfases,
  aunque d_U=d_I=0: la cobertura física no depende de un detector por módulos.
  No llamarlos componentes de secuencia negativa ni criterio industrial de desbalance.
- F05-ARR tiene duración de arranque 30 s y parada 10 s, pico a=2; se verifica
  su envolvente de ensayo, sin inferir rotor, par, inercia ni corriente subciclo.
  Sus estados operativos no implican automáticamente F02 ni etiqueta diagnóstica.

Evento físico (p. ej. cambio de U), estado operativo (arranque/reposo), condición
física (umbral de ensayo cumplido), etiqueta diagnóstica y predicción ML son
objetos distintos. No rellenar `estado=Anomalo` por estos escenarios. La condición
conocida por el oráculo y su confirmación temporal no se publican como diagnóstico
del producto; el detector puede no observarla y aquí no se evalúa ML. Una
observación ideal de sobrecarga o subtensión dentro de los dominios S02 puede
ser válida; cruzar el umbral de condición física no implica invalidez del dato.

#### S04.4 — Ventanas ideales y fronteras temporales

Pruebas futuras con soporte explícito y linaje a **todos** los segmentos intersectados.
Para W=[a,b), Δt=b−a>0: P_media=∫_W P/Δt, Q_media=∫_W Q/Δt. Usar áreas
analíticas, dividir en transiciones y cruces de Q. La observación ideal de potencia
media no es la del último estado; tampoco max(Q_media,0)·Δt reproduce necesariamente
un contador direccional. Los contadores leídos al final conservan acumulación desde
su época y soporte; los incrementos de ventana se calculan como diferencias con
oráculo, sin sumar lecturas acumuladas. No introducir una nueva definición de S/FP:
las salidas de estado siguen S02; medias RMS/S/FP de ventanas no se inventan por
promediar estados. Las ventanas aquí especificadas son medias de P/Q y contraste
de contadores; no prometen un algoritmo instrumental de ventana para otras magnitudes.

| Frontera / caso | Soporte y resultado independiente |
|---|---|
| Transición en inicio, F01-ESC | [600,610): sólo estado nuevo, P_media=600,Q_media=−900; ΔE=(5/3,0,0,5/2) |
| Transición en final, F01-ESC | [590,600): sólo estado anterior, P_media=1200,Q_media=900; ΔE=(10/3,0,5/2,0). El estado en t=600 no añade duración |
| Una transición interior, F01-ESC | [590,610): P_media=900,Q_media=0; ΔE=(5,0,5/2,5/2). Linaje a ambos tramos; Q_media=0 no borra las dos acumulaciones |
| Varias transiciones, F01-TUR | [14390,18010): 10 s inductivos, 3600 s de reposo, 10 s capacitivos; P_media=1200/181≈6,629834254 W,Q_media=0; ΔE=(20/3,0,5/2,5/2). Incluir eventos t=14400 y 18000; no asignar los 3620 s al último estado |
| Cruce continuo de Q, F01-Q | [0,3600): P_media=1200,Q_media=0; separar t=1800 para E_Q±=225. Q negativa válida con P positiva; no invalidarla ni sumar a energía activa exportada |
| Arranque/parada, F05-ARR | [0,10): a_media=1,P_media=960,Q_media=720; [95,105): ∫a dt=5/4 s, P_media=120,Q_media=90,ΔE_P+=1/3,ΔE_Q+=1/4; linaje a parada y reposo; FP de estado es indefinido desde t=100 |
| Reposo prolongado | F01-REP [0,86400): P_media=Q_media=0; incrementos cero exactos y contadores iniciales conservados. No interpretar cero válido como lectura congelada |

Un evento en frontera tiene instante puntual e IDs de sujetos; las ventanas que
terminan allí excluyen el estado posterior, sin excluir el evento del historial
de verdad. Estados finales y transiciones simultáneas se consultan de forma
coherente, sin estados intermedios de duración positiva no prescritos.

#### S04.5 — Rechazo, precisión y aceptación futura

Rechazar duración T≤0 o T>86400; ventanas de duración cero/negativa, fuera del
soporte de ejecución, con extremos no finitos o sin referencia temporal; tiempos
de transición desordenados/fuera del soporte; perfiles de carga incompletos,
contradictorios para el mismo sujeto/instante o no finitos. Eventos simultáneos en
sujetos/campos distintos son admisibles como F02-F03-JUN: el manifiesto agrupa sus
cambios en una única frontera. Un fin de rampa en T no crea un tramo de duración cero.
Rechazar U/G/B, frecuencia, I resultante o contadores iniciales fuera de S02.6;
G<0 no se admite en el mínimo, Q<0 sí. Rechazar 3P3W, cargas L-L, ángulos de fuente
fuera del perfil aprobado, neutro flotante, exportación activa y modelo térmico.
Una consulta de estado inicial con duración de integración nula sigue permitida
por S02.5; no equivale a un escenario/ventana de duración cero.

Todos los casos aceptados deben cumplir S02.7 sin cambiar tolerancias: U/I/f y
ángulos, P/Q/S/FP, contadores y saldos con sus unidades y guardas cerca de cero.
Comparar cada fase, luego agregado, cada incremento, luego contador final/saldo;
identidad, nulabilidad, tiempos de frontera, ausencia de E_P− añadido, monotonía
y reposo estructural exactos. Para estados discontinuos evaluar ambos lados,
no promediar la verdad física. Las rampas lineales en Y con U fija dan P/Q
lineales y energía por áreas exactas; partir siempre en cruces de Q.
Como control adicional suave reutilizar íntegro S02:C13: E_ref=1400 Wh,
N=100/200/400 da errores 0,01/0,0025/0,000625 Wh, cociente 4; el último cumple
presupuesto de discretización 0,0024 Wh además de redondeo. h=3600/N sustituye
la cadencia común en ese control. No confundir esa aproximación con los resultados
analíticos de la tabla ni validar contra el mismo integrador del futuro código.

#### S04.6 — Artefactos, identidad y cobertura

Cada ejecución futura producirá estados/segmentos físicos con parámetros y
resultados, eventos de carga/fuente/operación y ventanas ideales con method_ref,
soporte [a,b), measured_at=b y lineage_ref a segmentos/versiones. Manifest/run_id
resuelve escenario ID/versión y configuración; IDs estables de segmento/evento/
observación derivan del manifiesto y orden lógico, nunca de now(). La igualdad de
manifiesto en el entorno fijado deberá reproducir identidades/valores; no se
promete igualdad entre versiones del motor. Los digests/serialización definitivos
siguen pendientes de canonicalización C01/C05, sin inventar hoy garantías binarias.

Como en S03, observaciones idealmente calculadas llevan origin=synthetic,
method=calculated, availability=present y validity=valid sólo si pasan sus reglas;
freshness=current requiere evaluación explícita en el ensayo (recepción ideal en b,
max_age_s=5, política test-age@1), no sólo conocer el oráculo. Con referencia temporal
insuficiente freshness=unknown. En un punto usar measured_at=t y soporte point, event_at sólo si hay un evento
físico identificable en ese instante;
ventanas usan interval; una media con varios eventos lleva event_at=null y el
linaje resuelve sus eventos. Temperatura unsupported, FP con S=0 not_applicable,
ambos null/not_evaluated/not_applicable como S03.3, sin completar ML. Frecuencia
ideal a 60 Hz durante interrupción es parámetro calculado, no lectura de un medidor.
Tensión y frecuencia del bus compartido se emiten una vez por referencia; potencias,
corrientes y contadores identifican circuito/fase o total, con unidad canónica.

El artefacto físico S03.6 guarda causas/condiciones y transiciones; el conjunto
consumible sólo magnitudes, soporte y procedencia observables, sin etiquetas de
verdad ni causas inyectadas en features. S04 no introduce ruido, sesgo, congelación,
pérdidas, retrasos o transporte real: la recepción ideal es convención de fixture
futuro, recorded_at=null; no diseño de fallos S05 ni persistencia. No se crean los
artefactos en esta entrega.

| Caso / familia | Topología / fenómeno y variables verificadas | Oráculos asociados | Fronteras / rechazo específico | Nivel futuro / límite |
|---|---|---|---|---|
| F01-REP / F01 | 1P2W, cero, FP nulo, cuatro contadores | S02:C06/C10 | Inicio/fin, reposo 24 h; rechazar reinicio tácito | 1; no congelación ni retención |
| F01-EST / F01 | 1P2W y variante 3P4W; P/Q/S/FP,I_N,energía resistiva/inductiva/capacitiva | S02:C01–C04/C07 | Ventanas constantes; rechazar Q negativa tratada como inválida | 1; sinusoidal ideal |
| F01-ESC / F01 | 1P2W; escalón P/Q, contadores separados | S02:C08/C09/C10 como controles | Inicio/final/interior de ventana; rechazar doble conteo | 1; sin transitorio subciclo |
| F01-RAM, F01-Q, F01-TUR / F01 | 1P2W; rampas/turno/signo reactivo/reposo, continuidad Y, energía | S02.5,C12/C13 | Cruce por cero, varias transiciones; rechazar ventanas vacías o asignación al estado final | 1; no dinámica de motor |
| F02-SOB / F02 | 1P2W; I/P/S, duración y nominal de ensayo | S02:C01 y áreas | Fronteras 30/90, umbral/duración exactos; rechazar I>100 A | 1; no umbral industrial |
| F03-SUB / F03 | 1P2W; U/I/P/FP, Y fija, interrupción | S02:C14 | 30/90 y U=0; rechazar conservación artificial de P | 1; no frecuencia instrumental en apagón |
| F04-CAR,TEN,COM / F04 | 3P4W; fases/totales, U_LL,I_N,S_A,FP_A y compensaciones | S02:C05/C11 y fasores S02.3 | Estado inicial/ventanas constantes; rechazar √3 indebida y suma de contadores de fase como total | 1; índices por módulos no detectan todo desfase |
| F05-ARR / F05 | 1P2W; envolvente I/P/Q/S, estados y energía | Áreas S04.2, S02.5 | 0/10/20/30/90/100/120; rechazar parámetros que rompan continuidad declarada o dominios | 1; no simulación electromagnética |
| F02-F03-JUN / F02+F03 | 1P2W; carga/fuente simultáneas, I/U y energía | S02.3/5 y áreas | 30/90, cambios atómicos de estado; rechazar perfiles contradictorios | 1; no C01 combinado I/A |

Las ventanas/identidades de estos casos aportarán fixtures al nivel 2, que exige
validación independiente del contrato/consumidores en S08. Ningún cálculo documental
acredita ya nivel 1, 2 o 3. Cobertura física F01–F05 especificada, no ejecutada:
I01–I04/A01–A02/C01 y estadísticas/perfiles instrumentales se proponen en S05;
quedan pendientes su aprobación,
implementación/pruebas unitarias (S06), aceptación con tolerancias y reproducción
(S07), compatibilidad ejecutada/persistencia autorizada (S08) y cierre operativo
(S09). Reinicio/rollover y cercanía a cero conservan los controles S02/S03, sin
simularlos como fenómenos nuevos S04. D02/D05 siguen bloqueando equivalencia de
instrumentos específicos; no bloquean estos perfiles genéricos ni se dan por cumplidos.

#### S04.7 — Decisiones aprobadas y cierre documental

**Aprobación humana explícita registrada el 2026-10-17:** el responsable aprueba
individualmente F-D01–F-D05 mediante «SmartEnergy — Cierre formal de S04 y diseño
documental de S05», sobre HEAD `a9ae8ef8c7039882271ef2600400db7950986576`,
rama `feature/simulador`, árbol inicialmente limpio. **S04 cerrada únicamente
como especificación documental aprobada.** La tabla conserva recomendaciones,
alternativas y riesgos como fundamento histórico de las decisiones aprobadas.

- F-D01: catálogo físico F01–F05, parámetros, versiones y manifiesto reproducible.
- F-D02: capacidades, umbrales y duraciones exclusivamente sintéticos.
- F-D03: ventanas, transiciones, oráculos independientes y condiciones de rechazo.
- F-D04: separación de verdad física, eventos, condiciones, diagnóstico y ML.
- F-D05: cobertura y aceptación futura, sin acreditar pruebas ejecutadas.

Se conservan límites S02/S03, D02/D05, implementación y aceptación pendientes.
La instrucción autoriza sólo el diseño documental S05; no ejecución de código,
infraestructura ni inicio de S06. Aprobar S04 por sí solo no autoriza otra etapa.

| ID | Recomendación y fundamento | Alternativas | Consecuencias / riesgos / aprobación requerida |
|---|---|---|---|
| F-D01 | Aprobar catálogo S04.2 y defaults/manifiesto S04.1: casos deterministas de U/Y, ambas topologías y oráculos analíticos | Series aleatorias o sólo estables | Determinismo permite contrastar fronteras, no acredita realismo de planta. Aprobar IDs/versiones/parámetros/duraciones y suficiencia de cobertura F01–F05 |
| F-D02 | Aprobar capacidades/umbrales/duraciones de S04.3 sólo como condiciones de ensayo; distinguir dispersión por módulos de desfases | No etiquetar condiciones o usar índices normativos tras investigación separada | Riesgo de interpretar nominal como protección industrial; sin diagnóstico legado. Aprobar 15 A, 108 V, 10 s y definiciones d_U/d_I, más envolvente 30/10 s, sin alterar dominios E05 |
| F-D03 | Aprobar ventanas de P/Q, contadores y fronteras de S04.4 con áreas independientes y tolerancias E06 | Sólo puntos finales; usar integración del propio motor como esperado | Alternativas ocultan transiciones/compensaciones. Aprobar soportes y criterios/rechazos, manteniendo pendientes métodos instrumentales de otras ventanas |
| F-D04 | Aprobar estados/eventos/condiciones separados de diagnóstico/ML y artefactos/linaje S04.6 conforme a S03 | Etiqueta binaria automática; verdad mezclada en observaciones | Riesgo de fuga de causas y evaluación ficticia; requiere trazabilidad de todos los segmentos. Aprobar separación y calidad ideal sin adelantar fallos S05 |
| F-D05 | Aprobar suficiencia documental y plan de aceptación física futuro nivel 1, fixtures posteriores nivel 2, sin acreditar persistencia | Exigir integración inmediata o dar pruebas por aprobadas con estas cuentas | Las cuentas no validan código; mantener D02/D05, pendientes de S03 y autorizaciones de S05–S09. Aprobar límites/pendientes y cierre documental S04 sólo tras revisión humana |

### S05 — Instrumentación y adquisición aprobadas

**Especificación documental aprobada I-D01–I-D07 el 2026-10-17**, conforme a la
instrucción S05/S06 del apartado 1; S05 cerrada sólo documentalmente. Se conservan
las recomendaciones y alternativas como fundamento del diseño aprobado. Este
apartado no implementa instrumentación, adquisición ni almacenamiento. I01 es ruido/error
acotado; I02 sesgo/deriva/descalibración; I03 congelación/intermitencia; I04
relación CT/VT incorrecta; A01 pérdida/retraso/duplicación; A02 caída/recuperación.
C01 conserva exactamente F02 + I03 + A01 del catálogo mínimo. No se confunde con
F02-F03-JUN, oráculo S02:C01, ni decisión S03:C01.

#### S05.1 — Evidencia, fronteras y manifiesto

Fuentes primarias consultadas el 2026-10-17:

| Referencia | Uso y límite |
|---|---|
| [Python 3.13, random](https://docs.python.org/3.13/library/random.html), generador y reproducibilidad | MT19937, instancias con estado separado y random() de 53 bits; no demuestra independencia estadística entre semillas ni exactitud instrumental |
| [Python 3.13, hashlib](https://docs.python.org/3.13/library/hashlib.html), SHA-256 | Derivación determinista de semillas; no es certificación de subflujos independientes |
| [NIST, Uniform Distribution](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3662.htm), media y desviación | Uniforme acotada: media (A+B)/2, varianza (B−A)²/12; no acredita una distribución de ruido real |
| [Hoeffding, 1963, Probability Inequalities for Sums of Bounded Random Variables](https://www.cs.rpi.edu/academics/courses/spring06/random/hoefding.pdf), JASA 58, 13–30 | Cotas para sumas independientes acotadas; aplicación bilateral y unión de controles como deducción de ensayo, sin asumir que un PRNG sea IID verdadero |
| Matriz S01, apartados A/B y D02/D05 | V/I como entradas y P/Q/S/FP/energía como resultados derivados es una clasificación conceptual. Cadencia, resolución, incertidumbre y CT/VT específicos siguen pendientes |

Las fórmulas, bandas conservadoras y políticas siguientes son deducciones y
propuestas de ensayo propias. No se atribuyen a ADW210, SDM630 ni PM5310. Se
prefiere uniforme acotada a gaussiana: permite un límite absoluto sin recortes
que cambien la distribución. Normal truncada sería una alternativa con densidad,
normalización y aceptación propias; no se incluye en el mínimo propuesto.

Fronteras conceptuales, sin cambiar el DTO S03:

| Frontera | Entrada → salida conceptual | Evidencia separada del producto |
|---|---|---|
| Física → instrumentación | Consulta inmutable de estados/segmentos y soporte → resultado instrumental con ID lógico, valor original, método y soporte declarado | Segmentos verdaderos y soporte real, parámetros y causa de error |
| Instrumentación → adquisición | Observación S03 de origen inmutable → intentos con observation_key, ordinal y tiempo simulado | Resultado previo/posterior a cada transformación; lectura retenida y reloj real/declarado |
| Adquisición → disponible | Intentos → cero/una/varias entregas S03, recepción/evaluación separadas | Calendario completo, descartes, intentos fallidos y causas no observables |
| Disponible → admisión futura | Validación y deduplicación conceptual → accepted/duplicate/conflict o rechazo | Correspondencias, evaluación temporal y payload conflictivo conservados; sin SQL ni garantía transaccional ejecutada |

Una pérdida no borra la observación producida en el artefacto ni el evento físico.
La salida disponible contiene sólo lo recibido, incluidos avisos con evidencia
propia si el ensayo los habilita; el producto no recibe el calendario oculto.
Admisión futura no equivale a recorded_at: éste sigue null en todos los ejemplos.
La normalización es responsabilidad de frontera del adaptador (apartado 4);
se especifica aquí después de instrumentar el valor original y antes de exponer
su forma canónica. Puede materializarse al recibir, sin alterar su identidad ni
hacer depender los errores instrumentales del orden de llegada.

Manifiesto S04.1 ampliado **sólo en el artefacto de ensayo**: caso/versión S05,
referencia física ID@versión, run_id, configuración, sujetos, canales y unidades;
perfiles y orden de operaciones; relaciones CT/VT reales/configuradas y lugar de
escala; cadencia/ventanas, reloj real y declarado; calendarios de producción,
retención y entrega, política de edad/cola/reintentos; algoritmo/versión/semillas,
repetición, orden de sorteos y entorno; resultados independientes y tolerancias.
ID de caso `<familia>-<sufijo>@1`; cambios de parámetros exigen variante/versionado.
Los perfiles I/A no forman parte del digest de entradas físicas: comparar física
con/sin ellos por soporte, no por run_id o digest del manifiesto completo.
Estas reglas no sustituyen la política canónica D2 de S03.4, aprobada posteriormente
en la consolidación documental de S06 paso 7. Sus precisiones D2.1–D2.2 están
también aprobadas documentalmente; no hay canonicalización ejecutada.

#### S05.2 — Transformaciones, orden y coherencia

Orden recomendado para el perfil de canales primarios:

```text
verdad física inmutable
→ secundario ideal CT/VT (si está declarado)
→ ganancia/sesgo/deriva de sensor y ruido en unidad secundaria
→ relación configurada para volver a primario indicado
→ cálculo instrumental de magnitudes derivadas y acumulación instrumental
→ error de canal de resultado, sólo si se declara ese perfil distinto
→ cuantización de resultados → retención/congelación/intermitencia
→ original con unidad explícita → normalización canónica una sola vez
→ adquisición/recepción/evaluación → observaciones disponibles
```

Sin CT/VT, secundario=primario y relación=1. Para canal x:
`z_sec=(1+b+d(t))*x_sec+a+η(t)`; indicado `z=R_cfg*z_sec`, con
`x_sec=x_true/R_real`. b/d son adimensionales, a/η usan la unidad secundaria;
un offset especificado en primario se convierte por R_cfg explícitamente.
No intercambiar offset y relación: cambia su magnitud. Error de configuración
es R_cfg/R_real distinto de 1, no deriva del sensor. Error de resultado es una
transformación posterior a la derivación (p. ej. sesgo sólo en P); puede romper
coherencia con I/Q/S. No es otra forma de ruido en el CT.

En el perfil primario sólo se alteran módulos U/I con ángulos físicos preservados;
no se simula error de fase CT, saturación ni respuesta espectral. Con ganancias
positivas g_V/g_I por fase, sin offsets ni ruido: U'=g_V U, I'=g_I I,
P'=g_V g_I P, Q'=g_V g_I Q, S'=g_V g_I S, FP'=FP. f no cambia; neutro y
U_LL se recalculan fasorialmente sólo con información suficiente. Una tensión
compartida se perturba una vez por bus, con el mismo error para sus circuitos;
cada CT conserva su canal. No sortear P/Q/S/FP por separado.

Con offsets/ruido en módulos y ángulos conocidos, derivar P'/Q'/S' con los módulos
alterados y el mismo desfase; a carga cero no inventar un ángulo de corriente:
rechazar ese perfil en el manifiesto o usar error explícito de resultado.
Totales se calculan de fases instrumentadas simultáneas: P/Q algebraicos,
S aritmética y FP=P/S. No sustituir S por hypot(P_T,Q_T). El signo Q se conserva
con ganancias positivas; el offset de Q de un perfil de resultado sí puede
cruzarlo, sin cambiar el cuadrante físico. Q negativa sigue admisible.

Contadores instrumentales son un estado separado: integrar P'/Q' por alcance,
separando signos después de agregar como S02.5; nunca reescribir E verdadera.
En ganancia constante e iniciales cero, E'=g_V*g_I*E para cada contador afectado.
Para ganancia variable integrar el producto: no multiplicar E(t) por ganancia
final. Cuantizar la lectura acumulada, sin redondear cada incremento interno.
Una congelación de pantalla no detiene el acumulador interno de este perfil;
si se retiene el canal de contador, la siguiente lectura puede saltar al valor
interno actual. Época e iniciales son explícitos, sin reset implícito. Un sesgo
sólo de P reportada no afecta energía salvo que el perfil declare esa ruta.

**Aclaración de implementación S06 paso 5 (2026-10-17):** las innovaciones I01
están definidas por índice de captura, sin trayectoria temporal entre capturas.
El responsable decidió mantener pendiente, con rechazo explícito, la integración
energética de una ruta con ruido hasta especificar esa trayectoria. No se adopta
interpolación ni retención implícita de muestras. Los contadores deterministas y
las capturas con ruido permanecen dentro del alcance; I-D01–I-D07 no cambian.

Cuantización opcional genérica `q*round_ties_to_even(z/q)`, q>0 en unidad de
salida, |error|≤q/2; q=null la desactiva. No representa resolución de fabricante.
Normalizar después preserva el escalón original (q kW equivale a 1000q W).
Retener después de cuantizar reproduce el registro realmente leído; retener antes
del ruido produciría lecturas variables y sería otro modelo. La normalización
conserva original, unidad y método/versionado, incluido CT/VT ya aplicado.

Resultados inconsistentes por fallo de canal, cuantización o soportes distintos
se conservan sin reparar silenciosamente. Un chequeo con entradas simultáneas,
método y presupuesto de cuantización conocidos puede marcar validity=invalid
con evidencia de range_violation; sin evidencia suficiente, unknown. No imponer
identidades de S02 a toda salida defectuosa ni ajustar su valor para aprobarla.
E06 mide precisión de cálculo contra la transformación esperada; dispersión η,
resolución q y exactitud del instrumento son conceptos distintos.

#### S05.3 — Catálogo instrumental y dominios

Todos los tiempos se expresan en s relativos al origen S04.1; perfiles activos en
[a,b), 0≤a<b≤T. Parámetros finitos; las cotas siguientes son de ensayo genérico.
Física conserva S02.6. Salidas erróneas fuera de dominio se conservan como inválidas
si son finitas; no se saturan ni se cambian los límites físicos para admitirlas.

| ID | Mecanismo, parámetros y dominio propuesto | Temporalidad, dependencias y rechazo |
|---|---|---|
| I01 | η uniforme en [−a_n,a_n], a_n≥0 en V/A/Hz o unidad de canal de resultado declarado; amplitud relativa opcional 0≤r≤0,1, a_n=r*x_ref con x_ref fijo en manifiesto | Una innovación por índice lógico de captura; ruido blanco de ensayo, sin correlación temporal prometida en hardware. a_n=0 exacto sin sorteo. Primarios propagan error a derivados; canal de resultado independiente exige declarar inconsistencia posible. Rechazar amplitud negativa/no finita, canal/unidad ambiguos, o módulos negativos en perfil que exige no negatividad; no recortar ni volver a sortear |
| I02 | Ganancia 1+b, b∈[−0,5;0,5]; offset a∈[−0,1*x_ref;0,1*x_ref] en unidad del canal y x_ref>0 declarado; deriva d(t)=v*min(max(t−t_d,0),L), v en s⁻¹, L>0 s, abs(v)*L≤0,1 | Continua desde t_d, luego meseta; 1+b+d>0 obligatorio. No reinicia al cambiar la carga. Activación/desactivación abrupta debe declararse como evento instrumental; origen t_d no es recepción. Alternativa de paseo aleatorio diferida: no necesaria para cobertura y requeriría paso, límites y estadística propios |
| I03 | Captura retenida identificada; intervalo de congelación [a,b); máscara determinista de intermitencia por índices; sin probabilidad por defecto | Congelar grupo/canal explícito, no todas las variables por accidente. Cada nueva captura puede tener ID propio enlazado al retenido; reentregar la misma captura mantiene ID. Rechazar retención sin lectura anterior y soporte real no recuperable en artefacto; fuera de intervalo volver a medición actual, sin interpolación |
| I04 | R_Ireal/R_Vreal y R_Icfg/R_Vcfg positivos, adimensionales, cada uno en [1;1000]; cociente configurado/real explícito, lugar de escala device o adapter mutuamente excluyente | Configuración constante por tramo versionado; no cambia instalación real. Relación 1 representa conexión directa genérica, no CT físico universal. Doble escala es fixture negativo deliberado; rechazar configuración accidental que escale en ambos lugares, unidad secundaria desconocida o denominador cero |

Para I01, r=0,1 corresponde al 10% de x_ref. En canales de resultado, el error
absoluto puede aplicarse sin propagar a otras magnitudes si
se declara que es un fallo de ese resultado. Para cada perfil, `method_ref` resuelve
la transformación conocida por el productor; no publica parámetros secretos de
inyección. El acceso al artefacto está separado. origin=synthetic y method=calculated
para estos fixtures; no atribuir measured a una cadena sintética por su parecido.

I03 distingue tres políticas de evidencia, todas conservando el soporte real oculto:

- **Soporte antiguo conocido:** repetir valor/soporte/measured_at de la captura
  original, nuevo ID si es nueva captura; evaluar edad en recepción. Puede ser
  present/valid/stale con age_exceeded; no demuestra causa de congelación.
- **Reloj aparentemente actual:** el dispositivo declara punto/medida nuevos con
  el valor viejo. Sin diagnóstico público, el producto puede evaluar valid/current
  respecto de ese reloj; el oráculo guarda la discrepancia y el soporte real antiguo.
  No inventar advertencia ni diagnóstico de congelación desde la verdad oculta.
- **Sin evidencia temporal suficiente:** soporte unknown, measured_at=null,
  clock unknown y freshness=unknown; el artefacto sí conoce la captura retenida.
  Si falta evidencia de validez, validity=unknown, sin convertir repetición en fallo.

Intermitencia puede producir silencio (ninguna observación) o un aviso de sensor
con identidad propia y valor null, según evidencia pública configurada. No mezclar
con caída de enlace A02. Un timeout del gateway es otra observación missing con
soporte unknown, acquisition_timeout y evidencia de timeout, como S03.5; no es
la medición perdida. Repetición legítima F01-EST/REP es control negativo obligatorio.

#### S05.4 — Aleatoriedad y aceptación estadística propuesta

Se propone MT19937 mediante instancias locales Random de Python 3.13, sin RNG
global ni dependencias nuevas. Cada flujo usa seed entero obtenido por SHA-256
de una lista JSON de strings UTF-8, ensure_ascii=true, separadores coma/dos puntos
sin espacios: `["s05-rng-v1",master,run_id,layer,device,channel,profile,replicate]`.
master es decimal canónico de entero no negativo <2^64; run_id y demás strings
son los del manifiesto expandido, layer=instrumentation/acquisition; digest completo
como entero big-endian. No usar hash() del proceso ni hora actual. Seeds derivados,
versión exacta de Python y regla de transformación se registran en el artefacto.
Una nueva ejecución estadística cambia replicate; reproducción conserva la tupla.
La física determinista S04 no consume RNG.

Dentro de cada flujo, orden creciente por índice de producción preasignado;
una llamada random() por captura elegible, η=a_n*(2u−1). Generar la innovación
aunque una congelación posterior la oculte o A01 pierda el dato; así activar A no
cambia I. No compartir flujo entre capas/dispositivos/canales ni con reintentos.
Cambiar lista de capturas cambia el ensayo/versionado. Dos ejecuciones iguales
reproducen valores/IDs/calendarios en entorno fijado; no prometer igualdad de
funciones auxiliares entre versiones Python ni independencia demostrada por seeds.

Para errores comunes, asignar una fuente latente identificada a los canales que
la comparten: ruido de bus se reutiliza por todos sus circuitos, no se vuelve a
sortear. Para dos canales correlacionados opcionales, usar η_k=a_k*(c*u_c+(1−c)*u_k),
u en [−1,1], c∈[0,1]. Su varianza es a_k²*(c²+(1−c)²)/3; no es uniforme general
ni conserva a_k²/3. El mínimo acepta sólo c=0 (flujos distintos) o c=1 (común);
valores intermedios requieren perfil y aceptación propios. No añadir ruido a los
derivados que ya reciben error de sus primarios.

Ensayo I01-UNI: F01-EST resistiva, T=600, n=10000 puntos t_j=600j/n,
j=0…9999, ruido sólo I con a_n=0,1 A; x_ref=10 A, U=120, φ=0. master="20261017",
run_id="s05-i01-uni", layer="instrumentation", device="sim-a", channel="c1/L1/I",
profile="I01-UNI@1", replicate="0". I'∈[9,9;10,1] A, P'=S'=120I', Q'=0, FP'=1.
No integrar estas muestras instrumentales para sustituir la energía física 200 Wh.
El ensayo estadístico no requiere consumir ni transportar los 10000 puntos.

Oráculo independiente: uniforme continua de media cero, E[η²]=a_n²/3 y
Pr(η<0)=1/2. Evaluar residuos I'−10, nunca sólo un histograma de I'. Con α=0,001,
L=ln(26/α), exigir simultáneamente:

- |media(η)| ≤ a_n*sqrt(2L/n) ≈ 0,0045090690 A.
- |media(η²)−a_n²/3| ≤ a_n²*sqrt(L/(2n)) ≈ 0,0002254535 A².
- |fracción(η<0)−1/2| ≤ sqrt(L/(2n)) ≈ 0,0225453452.
- Para cada uno de diez intervalos iguales de [−a_n,a_n], frecuencia relativa
  dentro de 0,1±sqrt(L/(2n)); intervalos semiabiertos salvo el último cerrado.
  Comparar las diez frecuencias, sin ajustar bordes a las muestras.
- Todas las muestras finitas y |η|≤a_n; duplicación deliberada de canales c=1
  exige igualdad exacta de innovaciones y relaciones derivadas dentro de E06.

Bandas deducidas de la cota de Hoeffding para variables independientes acotadas:
Pr(|media−esperanza|≥e)≤2exp(−2ne²/(B−A)²); aplicar a η, η², al indicador η<0 y
a los diez indicadores de intervalo, y sumar los trece riesgos por unión (cada
control tiene riesgo ≤α/13). Riesgo total de falso rechazo ≤0,001 bajo
hipótesis IID ideal; no es garantía matemática de MT19937. La malla de 53 bits
introduce correcciones de momentos despreciables aquí; margen numérico adicional
máximo 1e−12 en la unidad de cada estadístico, fijado antes del ensayo. Validar
transformación con u=0,1/4,1/2,3/4 (resultados exactos −a_n,−a_n/2,0,a_n/2), no
usar la misma función de producción como esperado. Una sola seed que pase no
certifica la distribución; el plan combina esos controles, momentos teóricos y
diez frecuencias y bandas predefinidas. Si falla, investigar sin cambiar seed,
bandas o n para aprobar.
Pruebas de independencia multicanal o gaussianas no quedan acreditadas ni forman
parte de este perfil mínimo; requerirán aceptación adicional si se incorporan.

#### S05.5 — Adquisición, calendarios y recuperación

Cada observación conserva observation_key, contenido, soporte y measured_at.
Intento tiene ordinal propio y tiempo de envío en artefacto; delivery_id identifica
cada recepción. Un intento fallido sin recepción no crea entrega S03. Retraso
δ≥0 finito en s; caída [a,b), producción no se detiene. Ordenar recepción por
(received_at, ordinal de intento) para empates, sin cambiar el orden físico.
No enviar antes de completar la observación. Calendario de ensayo puede vaciar
cola después de T con horizonte explícito, sin añadir consumo después de T.

A01 usa conjuntos de índices perdidos, retrasos por índice y lista de reentregas;
I/A se componen sin cambiar IDs de origen. Alternativa probabilística acotada:
Bernoulli con p∈[0,1], perder si u<p por observación, flujo acquisition separado;
E[pérdidas]=np, no recuento exacto prometido. Un perfil aleatorio futuro fijará n,
seed y banda binomial antes del ensayo; S05 propone aceptar el mínimo con
calendarios deterministas, sin añadir ese perfil a la cobertura obligatoria.
Retrasos no negativos o una permutación realizable de recepciones modelan fuera
de orden; no cambiar tiempos declarados para simularlo. No sumar contadores
acumulados ni calcular energía por número de entregas.

A02 propone políticas alternativas explícitas, no simultáneas:

| Política de ensayo | Cola, intentos y recuperación | Límite |
|---|---|---|
| Descartar durante caída (default) | Un intento por captura; sin cola ni reintento; al restablecer recibir sólo nuevas observaciones | Perdidas definitivamente al horizonte; no backfill ni ceros |
| Recuperar FIFO acotada | Capacidad B entero 1…10000, retención H>0 s; al llenarse descartar la más antigua y registrar motivo; expirar cuando edad>H | Retener payload/ID inmutables; no nuevo soporte al recuperar; cola sólo de ensayo en memoria futura |
| Reintentar | Máximo K entero 1…10 intentos totales por observación, calendario fijo por caso, sin jitter por defecto; exceder K descarta | Reintento idéntico mantiene ID y nuevo intento/entrega; si el primer envío llegó, puede duplicarse |

Capacidad, H, K, calendario y horizonte de drenaje son obligatorios si hay cola.
No llamar recuperación a inventar mediciones del período caído. Contabilidad por
horizonte: N_producidas=N_únicas_recibidas+N_nunca_recibidas+N_pendientes;
N_intentos=N_recibidos+N_fallidos+N_en_vuelo. Al cerrar fixtures no quedan pendientes
ni en vuelo. Recibidos=accepted+duplicate+conflict en los casos finitos válidos;
rechazos malformados se cuentan aparte si el ensayo negativo los incorpora.

La llegada vieja se conserva en historial; el «último» lógico se selecciona por
soporte conocido y política versionada, no por última recepción. Si soporte es
unknown no reemplaza silenciosamente el último conocido: permanece serie/evaluación
sin orden físico demostrado. Política de edad s05-age@1: recepción menos fin de
soporte (o punto)≤5 s implica current, >5 stale; reloj/soporte insuficientes unknown.
Para contadores usar tiempo de lectura conocido y preservar época/acumulación.
No reescribir freshness de origen ni incluir evaluación del receptor en dedup.
Conflicto se conserva para resolución, sin overwrite. S03 sigue rigiendo la
canonicalización; no prometer unicidad concurrente, durabilidad, entrega exactamente
una vez ni comportamiento de Modbus/redes reales a partir de estas políticas.

#### S05.6 — Matriz de casos y oráculos independientes

Defaults heredados explícitamente: origen/identidades/topologías/unidades/E0 de
S04.1; física ID@1 sin cambios; escenarios de duración completa S04 salvo soporte
de consulta indicado. Casos deterministas seed/algoritmo=null, sin otros fallos,
normalización identidad y q=null; perfiles activos en [0,T) salvo reemplazo.
contract_version=1.0.0, producer_version=s05-design@1. clock usa basis=simulated,
uncertainty_s=0 y resolution_s=1, excepto I01-UNI con resolution_s=0,06. Cada serie de la tabla es una magnitud/alcance
S03 independiente. Capturas puntuales con measured_at=t_j y soporte point; entrega
inmediata salvo calendario reemplazado. Para counters, soporte acumulado desde
época hasta lectura, no punto sin historia. Calidad por defecto present/valid/
current/calculated, origin synthetic, no_diagnostic y evidencia de configuración/
validación pública de dominio. La verificación de coherencia entre canales sólo
cambia validity cuando está habilitada y cuenta con la evidencia de S05.2; assessment en recepción con s05-age@1. FP/temperatura nulos
conservan S03.3. Conteos corresponden **sólo a la serie emitida indicada**, no a
cada variable calculada para contrastar la frontera instrumental.

Cada observación usa ID `<caso>/j`, entregas `<caso>/j/attempt-k`; claves incluyen
plant-a/sim-a. scenario_ref identifica el caso y versión; el manifiesto resuelve
física original. lineage_ref=`<run_id>/<caso>/j` resuelve soporte real, segmentos,
transformaciones y captura retenida. run_id de casos deterministas=`s05-<caso>`;
provenance.configuration_ref y method_ref resuelven configuración y método
públicos versionados; el perfil expandido de inyección sólo se resuelve dentro
del artefacto separado, nunca desde características ML. original conserva
valor/unidad antes de normalizar. source_observation_ids enlaza sólo
fuentes disponibles; no crea observaciones físicas para alimentar al producto.
Referencias son conceptuales, no archivos creados. Intentos y causas permanecen
fuera del DTO. Resultado esperado se obtiene por fracciones/áreas/tablas externas
al motor; igualdad exacta de IDs/tiempos/conteos/nulls, E06 para aritmética de
transformación en su unidad, q/2 sólo para error de cuantización declarado.

| Caso / cobertura | Física, parámetros y calendario | Resultado independiente, entregas y rechazo |
|---|---|---|
| I01-AC / I01 | F01-EST resistiva; emitir I en t=0,1,2,3; innovaciones prescritas −0,1;−0,05;0;+0,05 A (u de S05.4), sin PRNG | I'=9,9;9,95;10;10,05; P'=1188;1194;1200;1206, Q'=0,FP'=1. 4 producidas/intentos/accepted, cero pérdidas/duplicados/conflictos. Física I=10,E_P+(600)=200. Rechazar ruido adicional independiente en P |
| I01-UNI / I01 | F01-EST y seed/n/soportes exactos S05.4; entrega ideal de las n lecturas si se verifica contrato | 10000 producidas/intentos/accepted, cero pérdidas/duplicados/conflictos; criterio estadístico S05.4 sobre innovaciones, derivación P=120I. Energía física 200. No dar por ejecutado ni aceptar por una realización visual |
| I02-GAN / I02 | F01-EST inductiva; emitir I en t=0,60,120; b_I=0,1, b_V=0, a=d=η=0 | I'=11,P'=1056,Q'=792,S'=1320,FP'=0,8; U=120,f=60. E instrumental final (176,0,132,0), física (160,0,120,0). 3/3/3, cero P/D/C. Variante offset sólo resultado P: a=12 W produce 972 W, resto ideal, contador ideal si su ruta no usa ese resultado; no «reparar» S |
| I02-DER / I02 | F01-EST resistiva; t_d=0,L=100,v=0,001 s⁻¹ en ganancia I; emitir I en t=0,50,100,150 | I'=10;10,5;11;11, P'=1200;1260;1320;1320; meseta hasta 600. E instrumental=(integral 0…100 de 1200+1,2t + 500*1320)/3600=655/3 Wh; física 200. 4/4/4, cero P/D/C. Variante deriva de resultado P conserva ruta de energía declarada; rechazar ganancia final aplicada a toda E |
| I02-QNT / I02 | Control de composición/descalibración, F01-EST resistiva, una P en t=0; sesgo sólo resultado a=5 W, q=10 W antes de salida original | Previo 1205 W, ties-to-even→1200 W; original=1,2 kW si perfil de unidad lo declara, canónico=1200 W una vez. Física 1200. 1/1/1, cero P/D/C; presupuesto error canal 5 W separado de E06; q negativo/cero se rechaza |
| I03-RET / I03 | F01-RAM; emitir I en t=0,10,20,30; capturar normalmente hasta 10; retener lectura t=10 en [20,40) | I=5;35/6;35/6;35/6, P instrumental retenida=700 desde 10; física I(20)=20/3,I(30)=15/2. IDs 0…3 distintos; soportes 0,10,10,10. 4/4/4, cero P/D/C; a 20/30 stale por edades 10/20. Variantes reloj actual: valid/current sin diagnóstico; tiempo insuficiente: unknown temporal (y validity unknown si no verificable). Oráculo conserva soporte 10 en todas |
| I03-INT / I03 | F01-EST resistiva; capturas previstas t=0…5; silencio de sensor j=2,3,4; sin avisos ni timeout público | Sólo IDs 0,1,5: 3 producidas/intentos/accepted; 3 capturas sin resultado, cero pérdidas de transporte/D/C. Física sigue I=10,E(600)=200. Variante aviso explícito por captura ausente: 3 missing/null/not_evaluated/not_applicable/unknown con evidencia propia y 3 valores, 6 entregas; no convertir silencio en missing automáticamente |
| I03-EST / I03 control negativo | F01-EST resistiva t=0…5, sin fallo; control adicional F01-REP con mismos tiempos | 6 IDs y soportes distintos, igual I=10 o 0; 6/6/6, cero P/D/C. No diagnóstico de congelación por igualdad; en REP FP not_applicable y counters constantes |
| I04-CT / I04 | F01-EST inductiva; R_Ireal=20,R_Icfg=40, R_Vreal=R_Vcfg=1; emitir I en t=0,60 | I secundario=0,5 A; primario indicado=20 A; P'=1920,Q'=1440,S'=2400,FP'=0,8,U=120,f=60. E instrumental=(320,0,240,0), física (160,0,120,0). 2/2/2, cero P/D/C. Conservar secundario en artefacto y original indicado 20 A; no aplicar otra vez CT al normalizar |
| I04-VT / I04 | F01-EST inductiva; R_Vreal=2,R_Vcfg=3, CT=1; una U en t=0 | Secundario=60 V, indicado=180 V, I=10,P'=1440,Q'=1080,S'=1800,FP'=0,8,f=60. 1 producida/intento/accepted, cero P/D/C; salida U fuera de dominio sintético observable validity=invalid con range_violation, física U=120 intacta. No limitarla a 144 V |
| I04-DOB / I04 negativo | F01-EST inductiva; CT real/configurado=20, lectura ya primaria I=10; fixture incorrecto fuerza otra escala ×20 | Candidato I=200 A rechazado como válido por dominio; conservar finito/evidencia, 1 producida/intento/accepted con validity=invalid, sin modificar físico ni agregar ganancia a f. Configuración de doble escala accidental se rechaza antes del ensayo; fixture negativo explícito permite contrastar esa detección, cero P/D/C |
| A01-CAL / A01 | F01-RAM, emitir sólo P puntual j=0…5, t=j, P=600+10j; perder j2, recibir j1 en 7, demás en j; duplicar j3 en 8, idéntico origen | Producción 6, intentos 7, fallidos 1, recepciones 6: orden IDs 0,3,4,5,1,3; 5 accepted+1 duplicate, 1 nunca recibida, cero conflictos/pendientes. P recibida 600,630,640,650,610,630; j1 stale (edad6), j3 duplicada edad5 current en evaluación separada. Último lógico j5. E física(120)=35 Wh, no depende de entregas |
| A01-CON / A01 límite de conflicto | A01-CAL más candidato negativo con ID j4 y P=641 en t=9 frente a original 640, mismo soporte/origen restante | 8 intentos, 7 recepciones=5 accepted+1 duplicate+1 conflict; 1 fallido/nunca recibida. Candidato no es nueva medición ni fallo de contenido añadido a A01: fixture adverso para probar regla S03. Conservar ambos contenidos, último lógico sigue j5; no overwrite. j0/j5 con distinto ID son legítimas aunque un control estable les dé igual valor |
| A02-DES / A02 | F01-RAM, P j=0…5, caída [1,4); default descartar, sin timeout | 6 producidas/intentos, fallidos3 (1/2/3), recibidas 0/4/5: 3 accepted, 3 nunca recibidas, D/C=0, sin missing ficticio. Retorno en 4 no recupera hueco ni reinicia física |
| A02-FIFO / A02 | Mismos 6 puntos/caída; B=3,H=5,K=2; fallan intentos iniciales 1/2/3, reintentar éstos en 6/7/8; horizonte cierre 8 inclusivo | 9 intentos, fallidos3, recibidos6: 0,4,5,1,2,3; 6 accepted, cero pérdidas/D/C/pendientes; edades recuperadas 5, current en umbral inclusivo. Último lógico j5 aunque j3 llegue al final. Variante B=2 descarta j1, reintenta j2 en 6/j3 en7: 8 intentos, recibidos5, fallidos3, 1 nunca recibida, edades4/current; H=3 en esa variante expira ambos antes de reintentar: 6 intentos, 3 recibidos,3 nunca recibidas |
| C01-MAS / C01 | F02-SOB con I03 y A01; calendario y resultados S05.7 | 12 capturas/observaciones I; 12 intentos, 9 accepted y 3 fallidos/nunca recibidas; D/C=0. Física sobrecarga de 60 s y energía de 60 Wh; todas las I recibidas=10 A; la causa no se deduce de esa serie |

P/D/C en las filas significa pérdidas de transporte/duplicados/conflictos, no
potencia P. 3/3/3 u otra terna sin nombres significa producidas/intentos/accepted.
Cuando no se emiten contadores, sus valores citados son verificación de frontera
y artefacto, no observaciones adicionales contadas. Para I02-GAN variante de
offset se emite P en los tres tiempos en lugar de I; I02-DER conserva el mismo
calendario si se prueba su variante. Cada variante es ejecución independiente.
Toda fila debe rechazar alteración de física, unidad original perdida, escala
doble involuntaria, IDs/tiempos/conteos diferentes, calidad inventada por la causa
oculta, falta de linaje o comparación exclusiva contra el propio motor.

Controles adicionales de frontera obligatorios: perfiles sobre F01-EST capacitiva
con ganancia I=1,1 (P'=1056,Q'=−792,S'=1320,FP'=0,8;
E'=(176,0,0,132), sin E_P− añadido), sobre F04-COM 3P4W con ganancia común
I=1,1 (P_T'=3432,Q_T'=0,S_A'=3960,FP_A'=13/15; E_Q± del circuito=0, no suma de
fases), ventanas S04.4 [590,610) de F01-ESC con ganancia común 1,1
(P_media'=990,Q_media'=0, ΔE'=(11/2,0,11/4,11/4)), congelación al inicio/fin
semiabierto, atraso exactamente 5/>5 s, dos valores iguales con IDs diferentes,
δ negativo, identidad conflictiva, retención sin antecedente, parámetros no
finitos/unidades desconocidas y entrega anterior a measured_at. Por cada control
puntual emitir una observación de la magnitud probada (una entrega, cero P/D/C);
para ventana emitir una P_media al final, contador e incrementos sólo en artefacto.
Los controles de rechazo de manifiesto producen cero observaciones/intentos.
No ampliar a ventanas RMS/FP de método desconocido ni atribuir simultaneidad real.

#### S05.7 — C01 combinado y enmascaramiento

C01-MAS@1 usa íntegro F02-SOB@1, T=120: I=10 en [0,30) y [90,120), I=20 en
[30,90), U=120,Q=0, P=S=120I; evento sobrecarga de 60 s respecto de 15 A/10 s. Energía
verdadera=(1200*60+2400*60)/3600=60 Wh. Emitir sólo I L1 cada 10 s, t_j=10j,
j=0…11; mantener en artefacto las demás variables y los contadores verdaderos.

I03 retiene la captura t=20 (10 A, soporte point en 20) para las seis capturas nuevas
t=30,40,50,60,70,80; termina en 90, que mide estado actual de 10 A. Cada captura tiene
ID nuevo con linaje a 20; no es reentrega duplicada. A01 pierde exactamente las
observaciones de 40,60,80, sin reintentos ni aviso/timeout observable. Recepción
inmediata de las otras nueve: 0,10,20,30,50,70,90,100,110. No se modifica el
calendario físico, el acumulador ni el evento al perderlas.

| Frontera | Resultado independiente |
|---|---|
| Física | Seis puntos en 30…80 dan 20 A, condición continua de 60 s; counters finales (60,0,0,0), eventos 30/90 intactos |
| Instrumentación | 12 resultados I=10 A; soportes reales/declarados 0,10,20,20,20,20,20,20,20,90,100,110 en política de soporte conocido |
| Adquisición | 12 intentos, 3 fallidos y 9 recibidos/accepted; cero duplicate/conflict; ninguna medición missing inventada |
| Producto con soporte conocido | I=10 en las 9 recibidas; en 30/50/70 edades 10/30/50 ⇒ stale con age_exceeded. Otras 6 current. La edad evidencia antigüedad, no sobrecarga ni causa de congelación |
| Variante reloj aparentemente actual | Mismos valores/entregas, soportes declarados t_j; nueve valid/current sin diagnóstico público; oráculo conserva la retención real. No afirmar ausencia de sobrecarga ni detectar automáticamente congelación |

El artefacto registra F02, I03 y A01, intervalos, captura retenida y pérdidas; cada
observación tiene linaje resoluble sin exponer las causas al producto/ML. Congelar
oculta el aumento de I y perder tres lecturas reduce además evidencia temporal.
Desde las nueve lecturas de I no puede reconstruirse la energía de 60 Wh ni concluir
parada/normalidad; con edad conocida sólo puede observarse dato viejo. Una lectura
de contador independiente podría aportar evidencia adicional, pero no se emite en
este caso. No crear estado=Anomalo ni evaluar detector. Se verifica física idéntica
con/sin I03/A01 y observabilidad separada de presencia de condición física.

#### S05.8 — Decisiones aprobadas y aceptación futura

**I-D01–I-D07 aprobadas expresamente el 2026-10-17** mediante la instrucción
S05/S06 del apartado 1, sobre HEAD `33d6a91726ea61908afb5cbe976cdc89ac65ebb6`.
Se aprueban individualmente modelos/dominios (I-D01), composición/energía/escala
(I-D02), RNG y estadística (I-D03), retención/calidad (I-D04), adquisición y
recuperación (I-D05), combinado y linaje (I-D06), cobertura/aceptación (I-D07).
La tabla conserva fundamentos, alternativas, riesgos y pendientes. **S05 cerrada
exclusivamente como especificación documental**; la aprobación no acredita
pruebas instrumentales ni de adquisición. La autorización separada de esa misma
instrucción inicia únicamente el núcleo S02 y sus pruebas en S06, sin servicios,
infraestructura o persistencia, y se detiene antes de los pasos internos 4–8.

| ID | Recomendación y fundamento | Alternativas | Riesgos, consecuencias de aprobar y pendientes no resueltos |
|---|---|---|---|
| I-D01 | Modelos/dominios S05.2/3: primarios coherentes y fallos de resultado explícitos; CT/VT real/configurado separado | Errores independientes en toda variable; perfiles específicos de fabricante | Independencia indiscriminada falsea causas. Aprobar transformaciones/unidades/dominios; no resuelve D02/D05, error angular/saturación ni equivalencia metrológica |
| I-D02 | Orden S05.2, energía instrumental separada y normalización única; conserva semántica del original | Ruido después de retención; escala en ambos extremos; corregir datos para cerrar balances | Cambiar orden cambia oráculo. Aprobar orden/rutas/cuantización; no decide clases/módulos, mapa ni precisión de almacenamiento |
| I-D03 | MT19937 local, semillas/orden S05.4 y uniforme con bandas α=0,001; determinismo como base | Philox con dependencia/versionado evaluados; gaussiana con aceptación propia; sólo determinismo | Seeds distintas no demuestran independencia. Aprobar flujo reproducible y riesgo estadístico; no acredita distribución ni pruebas ejecutadas o realismo del ruido |
| I-D04 | I03 con soporte antiguo, reloj declarado o desconocido; calidad sólo por evidencia S03 | Marcar automáticamente frozen por repetición o por conocimiento del ensayo | Riesgo de ocultamiento con reloj actual. Aprobar controles y nulls/edad; no resuelve diagnóstico real ni garantiza detectar congelación |
| I-D05 | Calendarios A01/A02, default descartar y variantes FIFO/reintentos acotados; conflicto sin overwrite | Cola ilimitada; recuperación implícita; promesa exactly once | Pérdidas/backlog son visibles en oráculo. Aprobar políticas/contabilidad/último lógico; no resuelve protocolo, durabilidad, dedup concurrente ni canonicalización final |
| I-D06 | C01-MAS conserva F02+I03+A01, tres fronteras y causas ocultas separadas | Otro combinado como ampliación; etiquetado binario por causa | Puede enmascarar evento real. Aprobar trazabilidad/enmascaramiento; no aprueba Anomalo, evaluación ML ni energía inferida desde huecos |
| I-D07 | Cobertura S05.6/7 y oráculos exactos/estadísticos independientes; plan por tres niveles | Sólo comparar motor consigo mismo; integrar ahora para validar documentos | Cuentas documentales no validan código. Aprobar suficiencia del diseño; siguen S06/S07, contratos/compatibilidad S08 y nivel 3 con autorización y entorno propios, más cierre S09 |

Aceptación futura: nivel 1 contrasta física S02/S04, aislamiento y reproducción,
transformaciones/calendarios contra las cuentas independientes de esta sección;
nivel 2 valida DTO/calidad/identidad/dedup/proyección en memoria conforme S03;
nivel 3 exige escritura, lectura y consumo completos en PostgreSQL aislado, con
permiso expreso, diseño aprobado y reversión propia. S05 no acredita ninguno.
I-D01–I-D07 quedan aprobadas; implementar sus mecanismos requiere la autorización
del incremento correspondiente de S06. Perfiles de marca,
incertidumbre CT/medidor, tiempos/resolución/sentinelas, emulación y hardware siguen
pendientes D02/D05; tampoco se resuelven unidades/zonas históricas, compatibilidad
ML con temperatura nula, retención de artefactos ni integración persistente.

## 6. Contrato común y compatibilidad — S03 aprobada

**Estado:** especificación documental aprobada C01–C08 y cerrada por la instrucción
S03/S04 registrada en el apartado 1; sin implementación ni pruebas de integración.
Inspección exclusivamente estática sobre el HEAD registrado en el apartado 1;
no se importó código, abrió una conexión ni ejecutó operaciones. Los ejemplos
son datos conceptuales de contrato, no mediciones ni fixtures ejecutados. Los
campos y ubicaciones propuestos todavía no existen en SmartEnergy.

### S03.1 — Inventario verificado y matriz de compatibilidad

Rutas relativas a la raíz del repositorio. Esta tabla identifica hechos del código
y el impacto de la propuesta, sin atribuir unidades a históricos desconocidos.

| Componente / evidencia | Contrato existente verificado | Impacto, pérdida y transición propuesta |
|---|---|---|
| `smartenergy/infrastructure/models.py`, `Registro` | Tabla `registros`: `id` Integer PK/index; `timestamp` DateTime sin zona, default UTC aware en Python; `corriente`, `voltaje`, `potencia_activa`, `temperatura_motor` Float y `estado` String, todos sin `nullable=False` | Faltan todas las identidades eléctricas, unidad, Q/S/FP/f/energía, calidad y linaje. El default no acredita UTC histórico ni conserva zona en SQL. Una fila no representa un circuito multifase sin selección explícita. Conservar tabla; ningún DDL aprobado |
| Mismo archivo, `Metrica` | `id`, `total`, `anomalias`, `porcentaje`, `estado`, `fecha` DateTime naive; resumen ML | No es almacén de observaciones ni referencia física; no usarlo para guardar calidad o escenario |
| `infrastructure/persistence.py` | `load_records`: siete columnas de Registro, orden timestamp/id asc; últimas métricas por id desc; entrenamiento lee cuatro variables y estado, detección añade id | Nuevos campos no atraviesan estas consultas. Necesitará ruta común autorizada y selección de serie; usar el SQL actual sólo para la proyección identificada. Orden de llegada no equivale a orden físico |
| `analysis/measurements.py` | `ENERGY_COLUMNS = [corriente, voltaje, potencia_activa, temperatura_motor]`; fechas inválidas→NaT, números inválidos/no finitos→NaN; estado nulo→Sin estado, prefijo `an`→Anomalo, cualquier otro→Normal; etiqueta binaria | No pasar estados de calidad por `estado`: `invalid` terminaría como Normal. Preparación pierde la causa del nulo y no valida instrumentos. Filtra fechas inclusivas/estados, ordena, muestrea, suaviza y promedia ventanas; estas transformaciones no conservan por sí solas linaje ni son integración de energía |
| `application/monitoring.py` | Reutiliza consultas y `prepare_records`; datos vacíos→None; métricas/modelo opcionales; valida features como lista no vacía/subconjunto de ENERGY_COLUMNS | No ingesta común ni validación eléctrica. Validación de nombres del bundle no demuestra compatibilidad de unidades, distribución o temperatura |
| `interfaces/api/routes.py`, `schemas.py` | GET `/api/v1/monitoring` y `/api/v1/monitoring/records.csv`; cuatro variables numéricas nullable, unidad de gráfica null; filtros de fecha/estado, muestra y tabla; 422/503/500 | HTTP no transporta identidad/alcance/calidad/procedencia. Mantener contrato v1; extensión de lectura separada necesita aprobación. No añadir escritura a la API. CSV tiene id,timestamp,cuatro variables,estado,label_anomalia; exporta originales filtrados asc, no sólo tabla reciente desc |
| `frontend/js/dashboard.js`, `charts.js`, `api.js` e `index.html` | Cinco vistas comparten la respuesta de monitoreo; ejes/captions «unidad no especificada»; sin selector de circuito/fase/calidad | Análisis general: serie y cobertura ML; Variables energéticas: serie/histograma; Comparativos: exige cuatro valores completos tanto en frontend como histogramas backend; Detección: evaluación y predicción; Tabla: filas/CSV. Con temperatura null, Comparativos queda vacío y el modelo de cuatro features no evalúa; otras mediciones y tabla siguen consultables. Las cinco vistas deben permanecer accesibles |
| `analysis/training.py`, `anomalies.py`, `application/model_jobs.py` | Orden ML fijo ENERGY_COLUMNS; StandardScaler + IsolationForest; etiqueta exacta Anomalo afecta contaminación/evaluación; detect persiste Metrica | No cambiar orden ni sustituir temperatura por cero. Entrenamiento/detección administrativos no tienen la guarda HTTP de features incompletas: no invocarlos sobre esta proyección. Etiqueta sintética no prueba anomalía detectable. Nuevo modelo sin temperatura requeriría autorización/dataset/validación propios |
| `infrastructure/model_artifact.py` | Bundle joblib `model`, `scaler`, `features`; lectura con flock compartido, escritor FileLock, temporal/fsync/os.replace y lock estable | No incluye versión eléctrica/unidades. No reutilizarlo por coincidencia de nombres; mantener formato, archivo y bloqueo. No entrenar ni publicar en S03 |
| `operations/simulate_data.py`, `stream_realtime.py` | Batch 10.000 a 1 s; continuo `build_sample`→Registro, recalcula P=abs(V·I·0,9/1000); RNG global sin escenario. Ambos `create_all`, insert/commit y DELETE por retención (20.000 por defecto); batch hora local naive | No son núcleo físico puro ni adaptadores del nuevo contrato. No ejecutarlos para validar S03. `/1000` sugiere kW sólo en ese productor, no demuestra unidad de cualquier histórico. Mantener operaciones separadas; sustitución futura explícita y reversible |

Unidades propuestas: V, A, W, var, VA, `1`, Hz, Wh, varh y `degC` auxiliar.
La tabla S02.2 gobierna dominios, signos y alcances. Ángulos/admitancias son
parámetros de verdad física; no se prometen como canales instrumentales del mínimo.
Conversión instrumental sólo con unidad, escala y referencia primaria/secundaria
verificadas: kW×1000→W, kWh×1000→Wh, kvar×1000→var, kvarh×1000→varh.
CT/VT se aplica una sola vez y queda registrado. No convertir FP de fabricante
sin su convención; método desconocido impide declarar equivalencia canónica.

### S03.2 — Representación, versión e identidad

Recomendación **híbrida**: observaciones largas (una magnitud/alcance/soporte por
observación), catálogo versionado de identidades y entregas separadas; vistas anchas
sólo como proyecciones explícitas para consumidores actuales. JSON es el formato
de intercambio conceptual; JSONL por tipo de registro es la propuesta de artefacto
de ensayo, sin crear archivos ahora. No es un esquema SQL ni un endpoint nuevo.

Versión de diseño `1.0.0`, aprobada con C01; aún no implementada. Cambio mayor para semántica, unidad,
identidad, obligatoriedad o enum incompatible; menor para campos opcionales que el
lector pueda ignorar; parche para aclaraciones sin cambio observable. Rechazar mayor
desconocida y enums desconocidos en campos críticos, conservar payload rechazado en
cuarentena autorizada; no degradarlos a Normal. `unknown` explícito expresa falta
de evidencia, no un comodín para datos mal formados. El catálogo de magnitudes y
métodos lleva versión junto al contrato; alias de marca pertenecen al adaptador.

| Campo / objeto | Tipo y obligación propuesta | Regla |
|---|---|---|
| `contract_version` | string obligatorio en cada registro | Versión exacta de semántica |
| `context` | objeto obligatorio: `installation_id`, `device_id`, `circuit_id`, `voltage_reference_id`, `topology`, `scope` | IDs string no vacíos, estables, sin dirección de red/credenciales; referencias resueltas contra catálogo versionado |
| Identidad de instalación/dispositivo | installation_id/device_id no nulos | Un dispositivo sintético también se identifica; cambio de configuración lleva versión, no reutilización ambigua de identidad |
| Circuito/referencia de tensión | circuit_id y voltage_reference_id string o null, claves siempre presentes | Circuito obligatorio para corriente/potencias/contadores; null en tensión/frecuencia del suministro común. voltage_reference_id identifica el suministro compartido; no sustituye circuito |
| `topology` | `1P2W` o `3P4W` | Perfil consumidor S02; otros perfiles requieren versión/alcance aprobados |
| `scope` | objeto `{kind, conductors}` | kind=`phase`, `circuit_total`, `voltage_pair` o `supply`; conductors lista ordenada L1,L2,L3,N según alcance. L1/L2/L3 corresponden a A/B/C de S02; phase para magnitudes de rama, current_rms de N admite phase con [N]; voltage_pair tiene dos conductores; total/supply tienen lista vacía |
| `observation_id` | string obligatorio, único dentro de instalación y device_id | Identifica resultado lógico, estable en reentregas; no depende de recepción |
| `quantity`, `unit`, `value` | strings de catálogo, número finito o null | No NaN/Infinity; unidad de catálogo obligatoria incluso con null |
| `quality` | objeto obligatorio | Cuatro dimensiones de S03.3, con evidencia observable |
| `time` | objeto obligatorio | Soporte y conocimiento temporal S03.4 |
| `provenance` | objeto obligatorio | origin=`synthetic` o `instrumental`; `producer_version`, `configuration_ref`, `method_ref`, `source_observation_ids`, `original`, `scenario_ref` y `lineage_ref` (nullable según reglas siguientes) |

Catálogo mínimo de `quantity`: `voltage_rms`, `current_rms`, `active_power`,
`reactive_power`, `apparent_power`, `power_factor`, `frequency`,
`active_energy_import`, `active_energy_export`, `reactive_energy_inductive`,
`reactive_energy_capacitive`, `active_energy_net`, `reactive_energy_net`,
`motor_temperature`. Un contador exige `counter_epoch` string en procedencia;
su acumulación identifica inicio y soporte hasta el punto leído. Reinicio/rollover
es evento y nueva época, sin resta negativa interpretada como consumo.
Métodos identifican RMS, promedio temporal P/Q, suma aritmética S, cociente FP y
contador direccional; no equiparar promedio de RMS o FP con RMS/FP de una ventana.
`method_ref=unknown` sólo permite guardar un resultado de semántica incompleta
con validez unknown, excluido de cálculos que requieran la equivalencia no probada.

Catálogo conceptual: instalación `plant-a`; dispositivos `sim-a` y `meter-a`;
referencia `bus-a` (L1-N/L2-N/L3-N); circuitos `c1`, `c2` enlazados a `bus-a`,
y asociación de cada circuito a sus canales de corriente. Una tensión de bus-a se
observa una vez como voltage_pair con circuit_id=null. Ambos circuitos la referencian;
no crear dos sensores ni sumar valores repetidos en bloques de registros. Dos
instrumentos independientes sobre el mismo bus conservan device_id y observaciones
distintos. Un total de circuito nunca es total de instalación ni fase. En 1P2W
el total coincide numéricamente con su fase, pero conserva identidad de alcance;
ningún consumidor suma total y fases ni mezcla métodos/ventanas incompatibles.

`original` conserva valor/unidad/código originales o null si no aplican; conversiones
registran fórmula, versión y CT/VT ya aplicado en configuration_ref/method_ref.
`source_observation_ids` enlaza cálculos disponibles al producto, lista vacía si
no existen. Para sintéticos scenario_ref={id,version,run_id} y lineage_ref son
obligatorios y resuelven a soporte en el artefacto separado; para instrumentos
pueden ser null. No exigir registro Modbus, firmware, marca ni escenario ficticio
a un instrumento conceptual; sus metadatos opcionales no condicionan el núcleo.

### S03.3 — Calidad, procedencia y desconocidos

| Dimensión | Valores permitidos | Semántica |
|---|---|---|
| `availability` | present, missing, not_applicable, unsupported, unknown | present contiene número; missing indica observación esperada sin valor; not_applicable es ajena a topología/definición; unsupported corresponde a capacidad no ofrecida; unknown no permite decidir |
| `validity` | valid, invalid, unknown, not_evaluated | valid satisface reglas documentadas; invalid tiene evidencia de rechazo; unknown evidencia insuficiente; not_evaluated no hubo valor que evaluar |
| `freshness` | current, stale, unknown, not_applicable | Se evalúa en recepción usando soporte declarado conocido, límite de edad versionado y evidencia de reloj; no usa verdad privada del simulador ni deriva de validez o método |
| `method` | measured, calculated, estimated, unknown | measured: obtenido por cadena de medida; calculated: derivación explícita, incluida física sintética; estimated: inferencia/sustitución declarada; unknown: no documentado |

quality añade `reason` string de catálogo o null y `evidence_ref` string o null.
Se proponen razones públicas `acquisition_timeout`, `undefined_zero_denominator`,
`topology_not_applicable`, `capability_not_supported`, `range_violation`,
`age_exceeded`, `time_unknown`, `no_diagnostic`. No incluir nombres de inyección.
La evidencia puede ser configuración, chequeo o estado reportado por dispositivo.
Origen synthetic/instrumental es independiente de method: una lectura instrumental
puede ser calculada; la salida física ideal es synthetic/calculated.

**D1 — Clasificación del método, aprobada documentalmente el 2026-10-17**
mediante «SmartEnergy — S06, paso 7: consolidación documental D1–D3»:
`quality.method` describe cómo se obtuvo el resultado; `provenance.method_ref`
identifica el procedimiento documentado. El perfil sintético S06 conserva
`quality.method="calculated"`, conforme a esta sección y S05.6. No se reclasifican
automáticamente mediciones ni contadores como `measured`. Los futuros adaptadores
instrumentales clasificarán cada resultado según el proceso documentado, no sólo
por el origen del dispositivo. Esta decisión no modifica la implementación actual.

Combinaciones admisibles y reglas de exclusión:

- present exige value finito y validity valid/invalid/unknown; cero es un número
  presente válido cuando cumple reglas. Un valor inválido finito se conserva para
  trazabilidad, pero se excluye de gráficas de medición válida, integración y ML.
- missing/not_applicable/unsupported/unknown exigen value=null y validity
  not_evaluated; freshness=not_applicable salvo missing con soporte anterior
  conocido, que admite stale. method puede conocerse aunque falte valor; si no,
  unknown. No deducir método de la ausencia.
- present admite current/stale/unknown; freshness=not_applicable sólo para una
  magnitud sin criterio temporal aplicable expresamente documentado. valid/stale,
  invalid/current y estimated/valid/stale son combinaciones admisibles.
- current/stale exige política de edad, tiempo de evaluación y evidencia temporal
  suficiente. Sin ello freshness=unknown. No usar «recibido ahora» como evidencia
  de que la medición es actual. Una evaluación posterior conserva su tiempo y
  política sin reescribir la observación original.
- FP con S=0: null, not_applicable/not_evaluated/not_applicable, razón
  undefined_zero_denominator; no cero ni error instrumental. Temperatura E07:
  null, unsupported/not_evaluated/not_applicable, método unknown.

Congelación no se infiere por repetir un valor: puede ser carga estable. Si se
conoce el soporte original, se conserva y se evalúa edad; si un dispositivo avanza
su reloj con un valor congelado sin diagnóstico público, validity/freshness sólo
reflejan lo que se conoce, pudiendo seguir valid/current. La causa verdadera y
el soporte físico oculto permanecen en el oráculo. Retraso y duplicación son
propiedades de entrega, no magnitudes ni anomalías eléctricas. Valores no finitos
rechazan la observación numérica y quedan como evidencia cruda de rechazo;
un aviso de ausencia separado no debe convertirlos en cero ni en medición válida.

### S03.4 — Tiempo, entregas y deduplicación

Cada clave temporal existe aunque sea null por desconocimiento. Referencia UTC
con sufijo Z para tiempos conocidos; conservar origen y transformación. No
atribuir precisión de microsegundos a un dispositivo que sólo reporta segundos.

| Campo de `time` | Semántica propuesta |
|---|---|
| `event_at` | Instante del evento físico identificable; null si se trata de una ventana sin un único evento o no se conoce. No equivale al tiempo de recepción |
| `support` | `{kind,start,end}`; kind=point (start=end), interval (start<end, [start,end)), unknown (ambos null). Ventana física que el emisor puede conocer, no un período de sondeo supuesto |
| `measured_at` | Momento en que la cadena completó la medición; null si desconocido. En sintéticos deriva de reloj simulado; no es datetime.now |
| `clock` | `{basis,resolution_s,uncertainty_s}`; basis=simulated/device/gateway/unknown; resolución e incertidumbre no conocidas son null, nunca cero supuesto |
| `assessment` | `{at,max_age_s,policy_ref}`; criterio y momento de evaluar freshness, nulls si no se evalúa |

Para simulación existe origen UTC y tiempo relativo del manifiesto, independientes
de velocidad de ejecución y reloj real. La transformación de ventana es method_ref;
lineage_ref resuelve intervalos/segmentos físicos y su versión. Si la ventana cruza
una transición se conservan todos los segmentos, nunca se atribuye al estado final.
En instrumentos sin reloj ni ventana conocida: soporte unknown, measured_at=null,
clock.basis=unknown; el gateway conserva recepción. El sondeo sólo acota la entrega,
no inventa evento/ventana de medición. Un reloj incoherente se marca desconocido
o inválido con evidencia; no se corrige silenciosamente para ordenar.

Registro separado de entrega: contract_version, delivery_id, observation_key
(installation_id,device_id,observation_id), received_at, recorded_at, status,
duplicate_of y transport_ref. delivery_id distingue intentos; received_at es tiempo
del receptor, recorded_at instante de admisión al almacenamiento, null en nivel 2.
status=accepted/duplicate/conflict/rejected; duplicate_of es delivery_id o null.
Pérdidas antes de recepción no tienen entrega aceptada ni fila de medición; el
oráculo conserva intento y pérdida. Un timeout conocido por gateway puede emitir
observación missing de identidad propia; no constituye recepción del valor perdido.

Deduplicar por observation_key y contenido canónico inmutable: repetición idéntica
con nueva entrega conserva una observación y varias entregas; mismo ID con contenido
diferente es conflict, preservado para resolución, nunca overwrite. El contenido
incluye versión, identidad, valor, unidad, calidad de origen, tiempo y procedencia;
recepción/registro y evaluación de freshness del receptor (incluidas la razón y
evidencia de edad producidas por esa evaluación) se almacenan en entrega o evaluación
enlazada, fuera de la clave y comparación de contenido de origen. El envelope
combina origen y evaluación para lectura: no sobrescribe la calidad original.
Al canonicalizar se excluyen time.assessment y los campos de calidad producidos
por el receptor; se conservan las razones/evidencias de validez de origen.
La política aprobada es `smartenergy-jcs-nfc@1.0.0`, definida en D2 más abajo;
sus precisiones D2.1–D2.2 están aprobadas, sin autorización de implementación.
No deduplicar sólo por timestamp/valor. Para dispositivos sin ID/secuencia estable,
el adaptador asigna ID por transacción capturada; reintentos de esa transacción
reutilizan ID. Sondeos distintos con el mismo valor no prueban duplicación del
origen: declarar garantía limitada al gateway, sin inventar «exactly once».

Aceptar retrasos y fuera de orden conservando ambos órdenes (soporte y recepción),
sin sobrescribir último valor ni integrar otra vez un contador. Llegada tardía
no cambia tiempo físico. Cada consumidor elige ventana y política de recálculo
versionadas; no alimentar alertas actuales con un dato viejo sin marcarlo.

**Históricos naive:** conservar valor literal y zona desconocida en la ruta legada;
no añadir Z ni usar zona de host/navegador como prueba. No mezclarlos en una serie
UTC canónica hasta aprobar, por lote/origen, zona IANA, reglas de horario estacional,
ambigüedades y evidencia. Guardar original, regla, versión y resultado; ambiguos
se aíslan. No hacer backfill ni reescribir PostgreSQL en S03. Timestamps conscientes
se normalizan a UTC conservando offset original; precisión desconocida sigue null.

#### D2 — Política canónica aprobada: smartenergy-jcs-nfc@1.0.0

**Aprobación arquitectónica documental del 2026-10-17**, mediante la instrucción
de consolidación D1–D3; no acredita implementación ni autoriza ejecutarla.
Base: [RFC 8785 (JCS)](https://www.rfc-editor.org/rfc/rfc8785.html), secciones
3.1–3.2, con las normalizaciones complementarias siguientes **antes de JCS**.
NFC y normalización temporal son reglas de SmartEnergy: JCS por sí mismo no
normaliza Unicode ni interpreta fechas. Después del preprocesamiento se aplican
la serialización JCS, su orden de propiedades por unidades UTF-16 y salida UTF-8.

- **D2.1 — Números:** JCS conforme a RFC 8785, IEEE 754 binary64 y serialización
  ECMAScript. Admitir números internos `float` si son finitos y compatibles con
  binary64. Para todo número JSON externo, conservar el token original antes de
  convertirlo, obtener su valor decimal exacto, convertirlo a binary64 y rechazar
  desbordamientos, resultados no finitos y subdesbordamientos no nulos a cero.
  Obtener después la serialización numérica JCS del binary64 resultante.
  Admitir un entero JSON externo si su conversión a binary64 es matemáticamente
  exacta **o** si su valor decimal coincide exactamente con el valor decimal de
  esa serialización JCS. Admitir un decimal JSON externo únicamente cuando su
  valor decimal coincida exactamente con el valor decimal de la serialización JCS.
  Comparar valores decimales mediante aritmética decimal exacta, sin tolerancias.
  Esta equivalencia decimal no exige representación binaria exacta: `0.1` puede
  admitirse aunque no sea una fracción binaria exacta. Las diferencias sintácticas
  de notación científica, ceros finales o formatos decimales equivalentes se
  admiten sólo si satisfacen la regla correspondiente al token. La política
  garantiza la readmisión de la representación JCS producida desde una entrada
  admitida, pero no admite todas las formas numéricas matemáticamente equivalentes.
  Rechazar los tokens que no satisfagan las condiciones anteriores. `1` y `1.0` son
  equivalentes; `-0` y `0` también. Mantener separados valor declarado,
  representación binary64 y bytes canónicos. Un `float` interno no permite
  recuperar la precisión declarada de un token externo ya perdido; no omitir
  por ello el control de entrada externa. No convertir números a strings, cambiar
  unidades, redondear metrológicamente ni aplicar tolerancias E06. El dominio
  contractual S03.2 sigue vigente, sin añadir límites físicos.
- **D2.2 — Tiempo:** entradas ISO 8601 extendidas con fecha y hora completas,
  segundos obligatorios y zona explícita `Z` o desplazamiento `±HH:MM`. Años
  0001–9999; desplazamientos `±HH:MM` con `00 ≤ HH ≤ 14` y `00 ≤ MM ≤ 59`;
  si `HH = 14`, obligatoriamente `MM = 00`. `+14:00` y `-14:00` son válidos;
  `+14:01` y `-14:01` son inválidos. Fracción opcional de uno a
  seis dígitos. Rechazar más de seis dígitos incluso si terminan en cero, tiempos
  naive, fechas incompletas o inexistentes, segundos intercalares y `24:00:00`.
  Rechazar conversiones a UTC fuera del rango representable. Salida canónica:
  UTC con sufijo `Z`, máximo microsegundos, eliminando ceros fraccionarios finales
  y la fracción si queda vacía. Instantes equivalentes producen idénticos bytes.
  No inferir tiempos desconocidos; conservar `null` donde el contrato lo permita.
  La normalización es para la comparación: no modificar tiempos originales ni
  evaluaciones del receptor. Conservar originales según S03.4, sin convertir
  históricos naive por suposición ni interpretar referencias opacas como fechas.
- **D2.3 — Unicode:** aplicar NFC a claves y valores de texto antes de JCS. Rechazar
  claves duplicadas, colisiones de claves tras NFC y secuencias Unicode inválidas,
  incluidos sustitutos aislados. No alterar mayúsculas, espacios ni puntuación.
- **D2.4 — Contenido:** incluir `contract_version`, identidad lógica, `context`, `quantity`,
  `value`, `unit`, calidad de origen, tiempos de origen y procedencia completa,
  incluidas todas las referencias y `counter_epoch` cuando corresponda. La identidad
  sigue siendo `(installation_id, device_id, observation_id)`; recepción no la forma.
  Excluir entrega/recepción, registro del receptor, `time.assessment` y evaluaciones
  temporales con sus razones/evidencias generadas por el receptor. Conservar las
  razones/evidencias de origen, aunque una evaluación posterior añada otras.
- **D2.5 — Estructura e igualdad:** conservar orden de arreglos y distinguir `null`, cero,
  `false`, cadena vacía y propiedad ausente. La igualdad se decide comparando todos
  los bytes canónicos, sin digest obligatorio. Canonicalizar no convierte un payload
  incompatible en una observación admisible.

**D2.1–D2.2 aprobadas como precisiones arquitectónicas, todavía sin implementar**,
mediante «SmartEnergy — S06, paso 7: consolidación documental D2.1–D2.2», sobre
`feature/simulador`, HEAD `fe7c4b8`, árbol limpio. Sustituyen los pendientes de
dominio numérico y formatos/rangos temporales de la consolidación D1–D3.
RFC 8785 no garantiza conservar exactamente cualquier número JSON. La corrección
documental D2.1 aprobada mediante «SmartEnergy — S06, paso 7: corrección documental
D2.1» añade la coincidencia decimal JCS como segunda condición de admisión para
enteros externos y garantiza la readmisión de la representación JCS; sigue sin
implementar canonicalización ni admisión.
No cambia la política `smartenergy-jcs-nfc@1.0.0`, S03.2, D1, D2.3–D2.5 ni D3.

**Vectores documentales de aceptación futura (no ejecutados).** Los fragmentos
suponen el resto de una observación válida idéntico. «Distinto» significa bytes
distintos, no rechazo; la admisibilidad contractual se verifica por separado.

| Área | Vector positivo / resultado | Vector negativo o diferencia exigida |
|---|---|---|
| Números externos equivalentes | `1` / `1.0` → `1`; `-0` / `0` → `0`; `1e3` / `1000` / `1000.00` → `1000` | `0.10000000000000001`: rechazo; su binary64 se serializa como `0.1`, decimal diferente del declarado |
| Decimal externo | `0.1` / `0.10` / `1e-1` → `0.1`, equivalencia decimal exacta con la salida JCS | No exigir que `0.1` tenga representación binaria exacta ni usar E06 para aceptar el vector anterior |
| Enteros externos | `9007199254740992` y `9007199254740994`: conversiones exactas, cada uno conserva su valor | `9007199254740993`: rechazo; conversión inexacta y decimal distinto de la salida JCS `9007199254740992`; no imponer como límite global el mayor entero consecutivo exacto |
| Readmisión JCS | `295147905179352825856`: aceptar por conversión binaria exacta, salida JCS `295147905179352830000`; este último entero se acepta al reingresar por coincidencia decimal JCS | `295147905179352825856.0`: rechazar, su valor decimal difiere del decimal JCS `295147905179352830000`; no se garantiza admisión de todas las formas matemáticamente equivalentes |
| Magnitudes externas extremas | `1.7976931348623157e308` → `1.7976931348623157e+308`; `5e-324` → `5e-324` | `1e309`: desbordamiento; `1e-400`: subdesbordamiento no nulo a cero; `4.9e-324`: su salida `5e-324` cambia el decimal declarado; todos rechazados |
| Números internos | `float` binary64 finito: admisible como valor interno, sin atribuirle un token externo | NaN/Infinity: rechazo; haber convertido antes el token externo no elimina el control de precisión declarada |
| Tiempo equivalente | `2026-10-17T07:00:00.125000-05:00` / `2026-10-17T12:00:00.125Z` → segundo formato; `.000000Z` → `Z` | Fracciones `.1234567` y `.1234560`: rechazo por longitud, incluso con cero final |
| Cambio de fecha | `2026-10-17T00:30:00+01:00` / `2026-10-16T23:30:00Z` → segundo formato | No conservar la fecha local como fecha UTC ni sustituir por recepción |
| Límites UTC | `0001-01-01T14:00:00+14:00` → `0001-01-01T00:00:00Z`; `9999-12-31T09:59:59.999999-14:00` → `9999-12-31T23:59:59.999999Z` | `0001-01-01T00:00:00+14:00` y `9999-12-31T23:59:59-14:00`: UTC fuera de rango; años 0000/10000 y offsets ±14:01: rechazo |
| Formato temporal | `null` se conserva sólo donde esté permitido; no equivale a fecha conocida | Rechazar `2026-10-17`, `2026-10-17T12:00Z`, `2026-10-17T12:00:00`, `2026-02-30T12:00:00Z`, `2016-12-31T23:59:60Z` y `2026-10-17T24:00:00Z` |
| Unicode | `"e\u0301"` / `"\u00e9"` → mismo texto NFC | Claves `"e\u0301"` y `"\u00e9"` en un objeto: rechazo por colisión; `"\uDEAD"`: rechazo; `"A"` / `"a"`: distintos |
| Propiedades | `{"b":2,"a":1}` / `{"a":1,"b":2}` → `{"a":1,"b":2}` | Clave duplicada antes de NFC: rechazo |
| Arreglos | `["L1","N"]` / `["L1","N"]`: iguales | `["L1","N"]` / `["N","L1"]`: distintos; no reordenar para aparentar igualdad |
| Calidad | Mismo origen, evaluaciones del receptor current/stale: mismo contenido canónico de origen | Cambio de validity, reason o evidence_ref de origen: contenido distinto |
| Procedencia | Mismos datos y referencias con propiedades en otro orden: iguales | Cambiar method_ref, original, lineage_ref o counter_epoch: distinto; no eliminar procedencia al comparar |
| Ausentes y tipos | `{"x":null}` / `{"x":null}`: iguales | `{}`, `{"x":null}`, `{"x":0}`, `{"x":false}`, `{"x":""}`: distintos; omitir una clave obligatoria causa rechazo contractual |

El identificador versiona esta política complementaria, separado de
`contract_version`. Cambios futuros de equivalencia o dominio exigen evaluar
compatibilidad y aprobar su versionado conforme a S03.2; no reinterpretar datos
existentes silenciosamente. No se cambia aquí la versión del DTO ni su código.

#### D3 — Admisión y deduplicación aprobadas en memoria

**Aprobación arquitectónica documental del 2026-10-17**, por la misma instrucción.
Identidad, contenido de origen, recepción y evaluación son conceptos separados.
Estas reglas especifican comportamiento futuro; no diseñan persistencia,
concurrencia, transacciones ni integración.

- **D3.1 — Duplicados:** misma identidad y contenido canónico que la primera
  aceptación → `duplicate`. `duplicate_of` referencia siempre el `delivery_id`
  de esa primera recepción aceptada; no construir cadenas de duplicados.
- **D3.2 — Conflictos:** misma identidad y contenido canónico diferente →
  `conflict`. Conservar íntegramente ambas recepciones, sin sobrescribir la primera
  aceptación. Comparar recepciones posteriores contra esa primera aceptación,
  no contra candidatos conflictivos. Excluir conflictos del último lógico.
- **D3.3 — Último lógico:** política `smartenergy-latest@1.0.0`, sólo observaciones
  `accepted`. Priorizar `support.end` conocido del soporte declarado, nunca verdad
  privada ni hora de recepción. Conservar todos los candidatos empatados, sin
  desempate arbitrario. Un soporte desconocido no desplaza uno conocido ni adquiere
  orden físico demostrado; conservar sus candidatos sin inventar tiempos.
- **D3.4 — Rechazo:** rechazar estructura, identidad, unidad o contenido incompatible
  con el contrato autorizado, o incumplimiento de D2. Conservar motivo y referencia
  de recepción. `validity="invalid"`, stale o unknown temporal no implican rechazo
  automático; no confundir invalidez observable del valor con payload inadmisible.
- **D3.5 — Orden:** procesar en orden de llegada y aceptar la primera recepción
  válida aunque sea tardía. Un rechazo no reserva identidad. Clasificar las
  posteriores respecto de la primera aceptada. No modificar origen ni evaluación
  temporal. Para las mismas entradas y políticas, garantizar determinismo en memoria.

Los empates de llegada del transporte conservan el ordinal de intento de S05.5;
esto no desempata soportes físicos en la selección del último lógico. Una nueva
evaluación temporal por sí sola no cambia el contenido canónico ni crea conflicto.
`recorded_at` permanece null en nivel 2; no se promete unicidad concurrente ni
durabilidad. A01-CON y C01-MAS siguen pendientes de ejecución completa con sus
oráculos S05.6/S05.7, sin cambios de calendario, conteos ni aceptación física.
Manifiestos definitivos y reservas anteriores mantienen su estado. D1–D3 no
autorizan implementar, cerrar S06, iniciar paso 8 ni S07–S09.

### S03.5 — Ejemplos completos del límite compartido

El siguiente envelope incluye todas las claves obligatorias de una observación
y su entrega en nivel 2. IDs de referencia se resuelven conceptualmente en el
catálogo y manifiesto indicados; no se pretende que esos archivos existan hoy.
La configuración `cfg-a@1` define 1P2W, c1→bus-a; `methods@1/mean_active_power`
integra P y divide por duración. run-1 usa origen 2026-10-17T12:00:00Z.

```json
{
  "observation": {
    "contract_version": "1.0.0",
    "observation_id": "o1",
    "context": {
      "installation_id": "plant-a", "device_id": "sim-a",
      "circuit_id": "c1", "voltage_reference_id": "bus-a",
      "topology": "1P2W", "scope": {"kind": "phase", "conductors": ["L1"]}
    },
    "quantity": "active_power", "value": 900, "unit": "W",
    "quality": {
      "availability": "present", "validity": "valid", "freshness": "current",
      "method": "calculated", "reason": null, "evidence_ref": "cfg-a@1"
    },
    "time": {
      "event_at": null,
      "support": {"kind": "interval", "start": "2026-10-17T12:00:00Z", "end": "2026-10-17T12:00:02Z"},
      "measured_at": "2026-10-17T12:00:02Z",
      "clock": {"basis": "simulated", "resolution_s": 1, "uncertainty_s": 0},
      "assessment": {"at": "2026-10-17T12:00:02Z", "max_age_s": 5, "policy_ref": "test-age@1"}
    },
    "provenance": {
      "origin": "synthetic", "producer_version": "ideal-design@1",
      "configuration_ref": "cfg-a@1", "method_ref": "methods@1/mean_active_power",
      "source_observation_ids": [], "original": null,
      "scenario_ref": {"id": "window-change", "version": "1", "run_id": "run-1"},
      "lineage_ref": "run-1/window-0-2"
    }
  },
  "delivery": {
    "contract_version": "1.0.0", "delivery_id": "d1",
    "observation_key": ["plant-a", "sim-a", "o1"],
    "received_at": "2026-10-17T12:00:02Z", "recorded_at": null,
    "status": "accepted", "duplicate_of": null, "transport_ref": null
  }
}
```

Para evitar repetir el envelope, la tabla define **instancias completas por herencia
explícita**: partir siempre de o1/d1, reemplazar todos los campos indicados (rutas
con punto), conservar todos los demás, y asignar observation_key según el ID/device
resultante. `time.assessment` y freshness son evaluación del receptor, separables
del contenido de origen para deduplicar. No son patches de una API existente.
Todas las instancias sintéticas del ejemplo apuntan al mismo escenario; su artefacto
contiene los linajes indicados, incluidos los fallos, sin exponer sus causas al producto.

| Caso | Reemplazos completos respecto del envelope y resultado |
|---|---|
| Ideal, ventana con cambio de carga | o1/d1 sin reemplazos. U=120 V; P=1200 W en [0,1), P=600 W en [1,2); Y=1/12 y 1/24 S resistivas. Media=900 W; incremento de energía=0,5 Wh, no 600×2/3600. lineage window-0-2 resuelve ambos segmentos y evento t=1 |
| Error instrumental no diagnosticado | observation_id=o2, delivery_id=d2, value=909; provenance.producer_version=instrument-design@1, method_ref=methods@1/instrument_window, lineage_ref=run-1/instrument-0-2; quality.reason=no_diagnostic. validity=valid expresa reglas observables satisfechas, no exactitud garantizada. Sesgo +1% sólo en oráculo; P física permanece 900 W |
| Pérdida de adquisición observable como timeout | observation_id=o3, delivery_id=d3; value=null; quality={availability:missing,validity:not_evaluated,freshness:not_applicable,method:unknown,reason:acquisition_timeout,evidence_ref:gateway-timeout-1}; time.support={kind:unknown,start:null,end:null}, measured_at=null, clock={basis:unknown,resolution_s:null,uncertainty_s:null}, assessment={at:null,max_age_s:null,policy_ref:null}; provenance.method_ref=methods@1/timeout, lineage_ref=run-1/timeout-1. d3 es entrega del aviso, no de o1; el intento perdido sólo está en oráculo. Sin timeout observable tampoco existe o3 |
| Instrumento real conceptual, sin reloj | observation_id=r1, delivery_id=dr1, context.device_id=meter-a, value=900; quality={availability:present,validity:unknown,freshness:unknown,method:measured,reason:time_unknown,evidence_ref:cfg-real@1}; time.support={kind:unknown,start:null,end:null}, measured_at=null, clock={basis:unknown,resolution_s:null,uncertainty_s:null}, assessment={at:null,max_age_s:null,policy_ref:null}; provenance={origin:instrumental,producer_version:adapter-concept@1,configuration_ref:cfg-real@1,method_ref:unknown,source_observation_ids:[],original:{value:0.9,unit:kW},scenario_ref:null,lineage_ref:null}. cfg-real@1 define c1/bus-a y conversión kW×1000; no atribuye mapa ni marca real |
| No aplicable por topología | observation_id=o4, delivery_id=d4; context.circuit_id=null, scope={kind:voltage_pair,conductors:[L1,L2]}; quantity=voltage_rms, unit=V, value=null; quality={availability:not_applicable,validity:not_evaluated,freshness:not_applicable,method:unknown,reason:topology_not_applicable,evidence_ref:cfg-a@1}; provenance.method_ref=methods@1/applicability, lineage_ref=run-1/na-1. Mantiene topología 1P2W; los conductores solicitados identifican el canal no aplicable, no una conexión existente |
| Congelada con edad conocida | observation_id=o5, delivery_id=d5; time.assessment.at=2026-10-17T12:00:12Z, delivery.received_at=2026-10-17T12:00:12Z; quality.freshness=stale, reason=age_exceeded; provenance.lineage_ref=run-1/frozen-1. Conserva soporte [0,2), measured_at=12:00:02Z y valor 900; es una nueva captura con linaje a la lectura retenida, no otra ventana física. Causa de congelación sólo en oráculo |
| Retrasada | observation_id=o6, delivery_id=d6; time.assessment.at=2026-10-17T12:00:20Z, delivery.received_at=2026-10-17T12:00:20Z; quality.freshness=stale, reason=age_exceeded; provenance.lineage_ref=run-1/delayed-1. Soporte/medición siguen en [0,2); puede llegar después de observaciones con soporte posterior |
| Reentrega duplicada de la retrasada | Instanciar exactamente o6 del caso anterior; delivery_id=d7, received_at=2026-10-17T12:00:21Z, status=duplicate, duplicate_of=d6; observation_key=[plant-a,sim-a,o6]. Mantener evaluación original; una evaluación a t=21 se enlazaría aparte. Una observación, dos entregas, cero energía adicional |
| Cero e invalidez finita | Dos instancias: o7/d8 con quantity=reactive_power, unit=var, value=0, provenance.method_ref=methods@1/mean_reactive_power y provenance.lineage_ref=run-1/zero-1 conserva present/valid (carga resistiva); o8/d9 con quantity=active_power, unit=W, value=−10, quality.availability=present, quality.validity=invalid, quality.reason=range_violation, quality.evidence_ref=consumer-domain@1, provenance.method_ref=methods@1/mean_active_power y provenance.lineage_ref=run-1/invalid-1. Conserva el alcance de fase L1 de o1 y el valor finito rechazado: P=−10 W incumple P_k≥0 del perfil consumidor aprobado en S02.2; no se convierte a null sin conservar evidencia ni a Normal |

En o8/d9, `consumer-domain@1` identifica la comprobación de P_k≥0 de
[S02.2](#s022--catálogo-obtención-y-nulabilidad) para la fase L1 del perfil
consumidor; se conservan el valor, el motivo, la evidencia y el linaje sin usarlo
como medición válida. Cada instancia parte de o1/d1: o8/d9 no hereda la magnitud
reactiva de o7/d8. **Q negativa es válida en consumo capacitivo**, incluso con P
positiva, conforme a S02.2/S02.4; el signo negativo por sí solo no determina
invalidez. La validez depende del dominio de la magnitud, su alcance y las
restricciones aplicables.

Segunda ventana de contraste: parada en t=1, P=1200 W en [0,1), P=0 en [1,2).
La misma forma que o1 produce media 600 W y ΔE=1/3 Wh; conserva ambos segmentos
y el evento de parada en el oráculo. Una ventana que termina en t=1 sólo incluye
el tramo anterior; la que empieza allí usa el nuevo estado. No asignar etiqueta
Anomalo por una transición de operación. Ventanas RMS/FP necesitan su método
propio, no la media de los valores de estado usada aquí para P.

### S03.6 — Verdad física y proyección legada

Artefacto de ensayo separado propuesto: manifiesto JSON versionado con referencias
a registros JSONL de estados/segmentos físicos, eventos, ventanas instrumentales e
intentos de adquisición. Incluye run_id, versiones de motor/escenario/contrato,
semillas por capa, parámetros/unidades/topología, origen UTC, paso, integración,
contadores iniciales, entorno numérico y digest de cada parte. lineage_ref resuelve
segmentos y transformación por ventana; intentos relacionan observación y entregas,
incluso si no llega ninguna. Incluye física verdadera, errores aplicados y causas
multicausales; no es el bundle ML ni columna estado. Acceso de ensayo separado del
conjunto de observaciones del producto; referencias opacas no autorizan a ML a
leer verdad o causas. Eliminar un artefacto rompe trazabilidad: retención/integridad
requieren política aprobada antes de persistir; no borrar ni crear estos recursos ahora.

Proyección legada sólo **optativa y con pérdidas declaradas**, recomendada inicialmente
en memoria. Configuración versionada selecciona instalación, dispositivo, circuito,
fase L1 y referencia L1-N de una misma topología, soporte/método compatibles,
conversiones y política de edad. En 3P4W una fase puede proyectarse con su P de fase;
P total no se combina con I de L1 y V L1-N como si fueran monofásicos. No existe
I total RMS escalar ni voltaje total que permita una fila total genérica. Un
multicircuito se separa antes de proyectar; rechazar mezcla o selección implícita.
No interpolar ni alinear ventanas por cercanía sin regla aprobada; faltas quedan nulas.

| Destino legado | Mapeo y pérdida declarada |
|---|---|
| id | ID local de fila, no observation_id; tabla de correspondencias a todas las observaciones fuente y versión de proyección fuera de Registro |
| timestamp | Fin de soporte conocido bajo política declarada; pierde inicio, evento, medida, recepción y registro. Si UTC→naive fuera imprescindible, sólo en dataset aislado explícitamente UTC con correspondencia; nunca mezclar con históricos sin zona |
| corriente / voltaje | current_rms fase L1 y voltage_rms L1-N del bus asociado en A/V; el vínculo compartido y la identidad se pierden en la fila. Rechazar total o pareja de tensión equivocada |
| potencia_activa | P de la misma fase en W canónicos; a perfil legado kW sólo con perfil aprobado W/1000 y procedencia conservada. No inferir kW global del nombre. El HTTP actual no puede anunciar unidad, requiere revisión separada para uso nuevo inequívoco |
| temperatura_motor | null por E07; jamás rellenar con constante, ruido o estimación ficticia. La fila SQL nullable no implica compatibilidad ML ni Comparativos |
| estado / label_anomalia | Sin etiqueta diagnóstica aprobada: estado=null→Sin estado en prepare_records. Hoy label_anomalia=0 aun en Sin estado: significa «no coincide con Anomalo», no verdad negativa. No evaluar métricas de clasificación con desconocidos; la ruta actual no ofrece esa exclusión, por lo que se bloquea su uso ML. Reglas futuras multicausales requieren aprobación |
| Sin destino | Q, S, FP, f, energía/época, todas las identidades/alcances/unidades/versiones, calidad/procedencia, ventanas, entregas, escenario y linaje: no conservados por Registro ni CSV/HTTP actuales |

Para validez invalid o freshness unknown/stale, la proyección por defecto no admite
el número como medición utilizable; devuelve null y registra el motivo/correspondencia
fuera de la fila. No confundir esta pérdida deliberada con el almacén canónico completo.
No agregar enums de calidad a estado ni etiquetar una pérdida como parada/anomalía.
La proyección aislada permite inspeccionar adaptaciones en nivel 2; no se conectará
al dashboard vigente ni PostgreSQL sin aprobación específica y revisión de unidades.

Transición recomendada: mantener generadores/datos/bundle y HTTP actuales;
validar primero contrato y proyección en memoria; aprobar después almacenamiento
completo aditivo y lectura común, conservando ruta legada seleccionable. No dual-write
implícito ni reentrenamiento. Reversión: volver a seleccionar ruta legada y retirar
adaptador nuevo, conservando observaciones, correspondencias y artefactos de ensayo;
no restaurar/truncar bases ni sobrescribir modelo. Migración/backfill, copia de
seguridad, atomicidad y limpieza tendrán diseño y autorización propios.

### S03.7 — Límites futuros e integración por niveles

Propuesta de ubicación futura, sin crear paquetes/archivos: funciones puras físicas,
ventanas y validación del contrato en `smartenergy/analysis/` (módulos concretos
sólo al aprobar S06); coordinación en `application/`; normalización instrumental
sin I/O mezclado con física; ORM y repositorio en `infrastructure/models.py` y
`persistence.py`, reutilizando una única Base. Entradas explícitas bajo
`operations/`; consulta en API/frontend sólo tras aprobar su evolución. El objeto
de intercambio es un DTO, no una segunda entidad Registro ni un repositorio genérico.
No crear domain/, interfaces abstractas, driver Modbus o servicio de adquisición.

| Nivel | Plan de aceptación propuesto; todavía no ejecutado |
|---|---|
| 1, S06/S07 | Núcleo puro, casos S02 y escenarios aprobados después en S04/S05; identidad/reproducción exactas, tolerancias físicas E06, independencia de capas y linaje de ventanas. Sin ORM/PostgreSQL. Pruebas unitarias en S06 y cierre físico S07 |
| 2, S08 inicial | Fixtures de ideal/error/pérdida y dispositivo conceptual usan el mismo validador/DTO en memoria; matriz de combinaciones de calidad, N/A, cero, inválido, timestamps desconocidos, ventana cruzada, duplicado/conflicto/tardío. Probar selección de fase, referencia compartida y rechazo de proyección ambigua; comprobar cada pérdida y correspondencia. No afirmar persistencia. Si se modifica monitoreo en tarea autorizada, verificar cinco vistas y cuatro estados datos/ML, filtros, nulos, CSV y sólo lectura |
| 3, S08 alcance separado | Operación de ingesta autorizada→validación común→aplicación→repositorio canónico completo→PostgreSQL aislado→repositorio de lectura→aplicación→API de consulta aprobada→consumidor de UI/CSV o fixture de consumo autorizado. Comparar contra entrada/oráculo, no sólo conteos. Si HTTP v1 sigue sin conservar campos, necesita ruta de lectura común aprobada; una proyección no completa el nivel 3 |

Antes de nivel 3: aprobar diseño de almacenamiento/HTTP, destino aislado identificado,
rol de escritura acotado y rol de lectura, permisos, datos, escrituras, respaldo,
ensayo de fallos y reversión/limpieza. Nunca smartenergy_dev ni laboratorio protegido.
No usar create_all/reset para migrar. Persistencia recomendada aditiva conservará
observación inmutable y entregas relacionadas con unicidad transaccional por clave;
el DDL, tipos/índices y estrategia concurrente se diseñarán y aprobarán por separado.

| Grupo requerido | Comprobación de ida y vuelta en nivel 3 |
|---|---|
| Versión/identidades | Igualdad exacta de versión, IDs, configuración, topología, scope/conductores; referencias resolubles; dos circuitos comparten una referencia sin duplicar sensor; total/fase separados |
| Magnitud/valor/unidad | Igualdad de quantity/unit y nulls; números finitos conservados según precisión de almacenamiento aprobada antes del ensayo, sin conversiones repetidas; contrastar W/kW y Wh/kWh. Comparación física E06 separada de fidelidad de serialización |
| Calidad/procedencia | Cuatro dimensiones, razones/evidencias, método/original/conversiones, productor y enlaces exactos; no fuga de causas del oráculo a UI/ML |
| Tiempo/contadores | Todos los instantes, soporte/límites, resolución/incertidumbre y evaluación conservados, nulls siguen null; época/inicio de contador recuperables; historial naive no convertido por defecto |
| Entregas/linaje | IDs, recepción/registro, duplicados/conflictos y orden fuera de secuencia recuperables; una observación ante reintentos/concurrencia, ninguna overwrite de conflicto; contabilidad de pérdidas contra oráculo separado; no duplicar energía |
| Consumo y reversión | Consulta recupera todos los campos requeridos, referencias y artefactos con integridad verificable; consumidores conservan alcances y explicitan proyección. Simular fallo de escritura/reintento sin registros parciales aceptados; volver a ruta legada sin modificar datos/modelo protegidos |

Ningún campo requerido puede quedar sólo en logs efímeros, memoria del proceso o
un archivo de correspondencia no recuperable. Si el diseño usa artefactos externos,
el almacenamiento conservará referencia/digest y se verificará su recuperación y
retención conjunta. Si no se autoriza esa solución o la lectura completa, nivel 3
queda pendiente; la aceptación de niveles 1/2 no la sustituye.

### S03.8 — Decisiones integrales aprobadas y límites

**C01–C08 aprobadas expresamente el 2026-10-17** por la instrucción S03/S04 del
apartado 1: representación híbrida/catalogación/versionado; identidad/alcances y
tensión compartida; calidad/procedencia; UTC/soporte/históricos; identidad de
observaciones/entregas/deduplicación; verdad física/linaje; proyección legada con
pérdidas y restricciones ML; integración y aceptación por niveles. Se conservan
los fundamentos, alternativas y riesgos de la tabla como registro de la decisión.
S03 queda cerrada sólo documentalmente. La aprobación no resuelve los trabajos
pendientes enumerados en el apartado 1 ni autoriza implementación, migración,
pruebas persistentes o arquitectura industrial. S04 se autoriza únicamente para
el diseño documental mediante la misma instrucción, no por efecto automático de C08.

| ID / asunto | Recomendación y fundamento | Alternativas | Riesgos y consecuencias |
|---|---|---|---|
| C01 Representación/versionado | Híbrida, observaciones largas + catálogo + entregas, JSON/JSONL y versión semántica; evita columnas por fabricante y conserva ventanas diferentes | Ancha por circuito; sólo larga sin vistas | Larga requiere pivot explícito; ancha fuerza sincronía/nulls. Aprobar catálogo/tipos/versión antes de S06, fijar canonicalización |
| C02 Identidad/alcance | IDs estables, scope explícito y bus compartido, configuración versionada; respeta D01 y consumidores | Duplicar tensión por circuito con alias; ID sólo de dispositivo | Duplicación aparenta sensores; IDs débiles mezclan total/fase. Obliga a resolver referencias y seleccionar serie |
| C03 Calidad/procedencia | Cuatro dimensiones independientes y unknown explícito; causas sólo observables en producto | Un quality_flag global; inferir causa del escenario | Flag único pierde estados; causa inyectada filtra verdad. Exige validar combinaciones y separar evidencia/oráculo |
| C04 Tiempo | UTC explícito, soporte semiabierto, relojes/resolución conocidos o null; preservar naive histórico | Usar recepción como medida; asumir zona local histórica | Alternativas inventan edad/ventana. Exige registrar incertidumbre y aprobación por lote para backfill |
| C05 Deduplicación | ID lógico estable + entrega por intento + conflicto sin overwrite; dedup transaccional futura | Timestamp/valor; hash sin ID; ignorar duplicados | Hash/valor confunde lecturas iguales legítimas; ID gateway limita garantía de origen. Exige unicidad/concurrencia y política de tardíos antes de integrar |
| C06 Verdad física | Artefacto versionado separado con manifiesto/segmentos/eventos/intentos/digests y linaje por ventana | Meter verdad en Registro; conservar sólo semilla | Primera filtra etiquetas; semilla sola no permite auditar versiones/resultados. Exige integridad y retención del artefacto |
| C07 Legado/ML | Proyección optativa seleccionada por fase, pérdidas explícitas, temperatura null y Sin estado sin verdad binaria; almacenamiento completo aditivo futuro | Reemplazar Registro ahora; fila multifase total; imputar temperatura | Alternativas requieren migración o falsean datos. Modelo actual no se acredita compatible y Comparativos puede quedar vacío; no usar ML administrativo sobre proyección incompleta |
| C08 Integración/aceptación | Primero núcleo, luego memoria, después PostgreSQL completo aislado con autorización propia | Integración directa al legado; detenerse en memoria | Legado pierde metadatos; memoria no prueba persistencia. Obliga a evidencia por campo y evolución de lectura aprobada para nivel 3; por sí sola no autoriza S04 ni S06/S08 |

### Niveles de aceptación y límites de evidencia

| Nivel | Evidencia necesaria y resultado que demuestra | Lo que no demuestra |
|---|---|---|
| 1 — Núcleo | Pruebas aisladas de relaciones físicas, tiempo, escenarios, linaje y reproducción; cierre en S07 | Integración con consumidores actuales, persistencia PostgreSQL o funcionamiento con instrumentos |
| 2 — Contrato común y compatibilidad | Pruebas del límite común y consumidores afectados, con proyección legada y pérdidas declaradas; cierre de integración en memoria en S08 | Conservación completa del contrato común en PostgreSQL ni validación persistente de extremo a extremo |
| 3 — Persistencia completa y extremo a extremo | Escritura, lectura y consumo por la ruta de aplicación aprobada usando PostgreSQL aislado; contraste de valores, identidades, unidades, tiempos, calidad y procedencia contra la referencia, sin pérdidas silenciosas | Exactitud metrológica, adquisición real ni preparación industrial |

Fixtures y pruebas exclusivamente en memoria pueden acreditar niveles 1/2, nunca
validar persistencia ni sustituir el nivel 3. S03.6 identifica qué campos
conserva o pierde una proyección a `Registro`; esa proyección no acredita por sí
sola persistencia completa. Cualquier evolución necesaria del almacenamiento o
HTTP requiere diseño y autorización separados, sin decidir aquí su estructura.

Antes del nivel 3 se requiere autorización expresa del ensayo y del entorno
PostgreSQL aislado: destino, rol, permisos, datos de prueba, escrituras permitidas,
procedimiento de aislamiento y reversión o limpieza autorizada. No usar recursos
protegidos ni ejecutar mantenimiento implícito. S03.7 propone la ruta de extremo
a extremo y los criterios por campo; S08 sólo podrá ejecutarla con diseño aprobado
y esa autorización.
Si el nivel 3 no se autoriza o no pasa, se reportará pendiente o fallido, sin
declarar validada la persistencia. Todo cierre deberá identificar el nivel y alcance
realmente aceptados; la aceptación de un nivel no habilita automáticamente el siguiente.

## 7. Reproducibilidad, tiempo y aceptación cuantificable

El manifiesto futuro registrará versión de escenario/motor/contrato, parámetros,
semilla y algoritmo aleatorio, entorno numérico relevante, instante inicial,
cadencia, duración, topología, perfiles, eventos y política de integración.
Separar secuencias aleatorias por capa evitará que activar una pérdida cambie la
física. El tiempo simulado no dependerá de `now()` ni de esperas del reloj real.
UTC con offset explícito y límites de intervalo inequívocos, aprobados con C04
en S03, sin reinterpretar timestamps naive históricos.

Los siguientes son **criterios del simulador**, no exactitudes de instrumentos.
S02.7 fija tolerancias numéricas absolutas y relativas, por variable y
dominio, aprobadas con E06 antes de implementar; no se aceptan tolerancias arbitrarias posteriores
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
  rama activa sea la autorizada para esa intervención. Para la intervención documental
  actual es obligatoria `feature/simulador`. Si no coincide, detenerse sin cambiarla.
  Revisar estado Git y preservar cambios ajenos; no hacer commit/push sin permiso.
- Inspeccionar código y consumidores reales, reutilizar componentes compatibles
  y fundamentar decisiones en fuentes primarias. Diferenciar hecho verificado,
  requisito de diseño y propuesta pendiente. No atribuir a SmartEnergy funciones
  del catálogo de un instrumento.
- Obtener aprobación explícita antes de cada etapa de implementación; el cierre de
  una etapa no habilita la siguiente. La autorización S05/S06 sólo cubre cierre
  documental S05 y pasos internos 1–3 de S06; detenerse después de sus pruebas.
- No introducir dependencias, migraciones, reorganización arquitectónica o cambios
  de API sin autorización específica. La intervención S05/S06 prohíbe ejecutar
  generadores y operar PostgreSQL o Docker. Las pruebas autorizadas del núcleo
  son locales y aisladas; cualquier ensayo de persistencia requerirá autorización de nivel 3.
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

D01–D06 mantienen el cierre documental S01 con diferimientos; E01–E07 la
especificación matemática aprobada S02; C01–C08 el contrato común aprobado S03.
F-D01–F-D05 e I-D01–I-D07 fueron aprobadas expresamente el 2026-10-17:
**S04 y S05 cerradas sólo como especificaciones documentales aprobadas**.
**Estado tras consolidación documental D2.1–D2.2 de S06 paso 7 (2026-10-17):**
pasos 1–5 aprobados; paso 6 pendiente de auditoría; paso 7 parcialmente implementado,
pendiente de auditoría independiente y de implementación de admisión. D1 en S03.3
y D2/D3 en S03.4 están aprobadas como decisiones arquitectónicas, incluidas las
precisiones numéricas y temporales D2.1–D2.2, todavía sin implementar. Paso 8 no
autorizado; S07–S09 no iniciadas. Los estados históricos
anteriores no sustituyen este registro vigente. No se aprueban umbrales industriales,
etiquetas ML, equivalencia instrumental ni nuevas convenciones físicas.

Se conservan S02.1–S02.9, S03.1–S03.8, la matriz instrumental y los diferimientos
D02/D05. El incremento implementado y su evidencia están en el
[resultado S06 del roadmap](../roadmap-simulador.md#resultado-de-s06--contratocomposición-parcial-paso-7).
Hay escenarios, instrumentación, adquisición y composición DTO parcial en memoria;
no hay canonicalización, admisión ni manifiestos definitivos o artefactos
persistentes implementados. Integridad/retención S03 conservan sus reservas.
A01-CON y C01-MAS siguen pendientes de ejecución completa. La evidencia previa
del núcleo y de los incrementos aislados no cierra S06 ni S07, ni acredita
el nivel 1 completo, integración (nivel 2), persistencia
(nivel 3), equivalencia instrumental o preparación industrial.
No se ejecutan servicios, generadores legados, PostgreSQL, Docker, migraciones,
ML ni pruebas generales de aplicación. No se hace commit ni push. La intervención
se detiene tras esta consolidación documental. Implementar D1–D3 o iniciar trabajo
posterior requiere autorización expresa; esta aprobación no la concede.
