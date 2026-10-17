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
al escenario físico original y su versión. La propuesta S03 del apartado 6 define
la representación y la semántica de los límites temporales, pendiente de aprobación. Se permiten referencias por intervalo o
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

## 6. Contrato común y compatibilidad — propuesta S03

**Estado:** diseño recomendado, pendiente de revisión y aprobación humana C01–C08.
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

Versión candidata `1.0.0`, todavía no aprobada. Cambio mayor para semántica, unidad,
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
| `freshness` | current, stale, unknown, not_applicable | Se evalúa en recepción usando soporte real, límite de edad versionado y reloj conocido; no deriva de validez ni método |
| `method` | measured, calculated, estimated, unknown | measured: obtenido por cadena de medida; calculated: derivación explícita, incluida física sintética; estimated: inferencia/sustitución declarada; unknown: no documentado |

quality añade `reason` string de catálogo o null y `evidence_ref` string o null.
Se proponen razones públicas `acquisition_timeout`, `undefined_zero_denominator`,
`topology_not_applicable`, `capability_not_supported`, `range_violation`,
`age_exceeded`, `time_unknown`, `no_diagnostic`. No incluir nombres de inyección.
La evidencia puede ser configuración, chequeo o estado reportado por dispositivo.
Origen synthetic/instrumental es independiente de method: una lectura instrumental
puede ser calculada; la salida física ideal es synthetic/calculated.

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
Canonicalización futura: objetos por claves ordenadas, números finitos normalizados
sin redondeo metrológico y UTF-8; fijar algoritmo/versión antes de implementar.
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

### S03.8 — Decisiones integrales pendientes de aprobación

Todas las recomendaciones C01–C08 están **pendientes**; no constituyen aprobación
de arquitectura industrial, implementación ni ejecución de etapas siguientes.

| ID / asunto | Recomendación y fundamento | Alternativas | Riesgos y consecuencias |
|---|---|---|---|
| C01 Representación/versionado | Híbrida, observaciones largas + catálogo + entregas, JSON/JSONL y versión semántica; evita columnas por fabricante y conserva ventanas diferentes | Ancha por circuito; sólo larga sin vistas | Larga requiere pivot explícito; ancha fuerza sincronía/nulls. Aprobar catálogo/tipos/versión antes de S06, fijar canonicalización |
| C02 Identidad/alcance | IDs estables, scope explícito y bus compartido, configuración versionada; respeta D01 y consumidores | Duplicar tensión por circuito con alias; ID sólo de dispositivo | Duplicación aparenta sensores; IDs débiles mezclan total/fase. Obliga a resolver referencias y seleccionar serie |
| C03 Calidad/procedencia | Cuatro dimensiones independientes y unknown explícito; causas sólo observables en producto | Un quality_flag global; inferir causa del escenario | Flag único pierde estados; causa inyectada filtra verdad. Exige validar combinaciones y separar evidencia/oráculo |
| C04 Tiempo | UTC explícito, soporte semiabierto, relojes/resolución conocidos o null; preservar naive histórico | Usar recepción como medida; asumir zona local histórica | Alternativas inventan edad/ventana. Exige registrar incertidumbre y aprobación por lote para backfill |
| C05 Deduplicación | ID lógico estable + entrega por intento + conflicto sin overwrite; dedup transaccional futura | Timestamp/valor; hash sin ID; ignorar duplicados | Hash/valor confunde lecturas iguales legítimas; ID gateway limita garantía de origen. Exige unicidad/concurrencia y política de tardíos antes de integrar |
| C06 Verdad física | Artefacto versionado separado con manifiesto/segmentos/eventos/intentos/digests y linaje por ventana | Meter verdad en Registro; conservar sólo semilla | Primera filtra etiquetas; semilla sola no permite auditar versiones/resultados. Exige integridad y retención del artefacto |
| C07 Legado/ML | Proyección optativa seleccionada por fase, pérdidas explícitas, temperatura null y Sin estado sin verdad binaria; almacenamiento completo aditivo futuro | Reemplazar Registro ahora; fila multifase total; imputar temperatura | Alternativas requieren migración o falsean datos. Modelo actual no se acredita compatible y Comparativos puede quedar vacío; no usar ML administrativo sobre proyección incompleta |
| C08 Integración/aceptación | Primero núcleo, luego memoria, después PostgreSQL completo aislado con autorización propia | Integración directa al legado; detenerse en memoria | Legado pierde metadatos; memoria no prueba persistencia. Obliga a evidencia por campo y evolución de lectura aprobada para nivel 3; no autoriza S04 ni S06/S08 |

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
Propuesta: UTC con offset explícito y límites de intervalo inequívocos; aprobación
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

E01–E07 están aprobadas y S02 cerrada exclusivamente como especificación
matemática, según la instrucción registrada en el apartado 1. D01–D06 mantienen
el cierre documental S01 y sus diferimientos, sin aprobar firmware/mapa ni
equivalencia instrumental. **S03 queda especificada, pendiente de revisión y
aprobación C01–C08; S04 no iniciada.** Siguen pendientes umbrales/escenarios,
perfiles de fallos, metrología específica, implementación y todos los niveles de
aceptación ejecutable. Ninguna migración, cambio HTTP ni recurso persistente
queda autorizado por la propuesta S03.

Esta intervención modifica únicamente este contrato y el roadmap del simulador.
La matriz S01 se consulta sin modificar sus evidencias históricas. Verificación:
contraste estático con ORM, repositorio, análisis, aplicación, HTTP, frontend,
ML y generadores; revisión de identidad/unidades/calidad/tiempo, ejemplos,
pérdidas declaradas y criterios por nivel; enlaces locales y `git diff --check`.
No se ejecutan servicios, generadores, pruebas de aplicación, PostgreSQL ni Docker.
Sin código implementado, ensayos experimentales, commit ni push. Los resultados
documentales no acreditan compatibilidad ejecutada ni preparación industrial.
