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
considerará cumplido.

**Actualización S01/S02 (2026-10-17):** sobre el commit
`d84dc1fe48e86a14fcccf10ae183ee16f305c602`, el responsable aprobó D01–D06 y
autorizó el cierre documental S01 para el núcleo genérico y el diseño S02, según
la instrucción «SmartEnergy — Cierre formal de S01 y ejecución de S02».
La [matriz, decisiones D01–D06](../instrumentacion/matriz-medidores.md#c-decisiones-d01d06-aprobadas-y-diferimientos)
registra configuración, alcance, impacto sobre el apartado I y condiciones de
cierre de los pendientes. La evidencia instrumental diferida no está cumplida.
La especificación siguiente es una **propuesta S02 pendiente de aprobación**;
no autoriza S03 ni implementación, commit o push.

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
| Energía activa/reactiva | Wh / varh | Distinguir activa importada/exportada y reactiva inductiva/capacitiva; propuesta S02 detallada abajo, pendiente de aprobación |
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
codificación universal; la propuesta S02 siguiente resuelve la semántica genérica.
La equivalencia de registros no verificados permanece diferida según D02/D04.

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

### Propuesta matemática S02 — pendiente de aprobación

Todo este apartado, incluidas sus decisiones E01–E07, es diseño propuesto, no
capacidad implementada ni convención instrumental verificada. La aprobación de
D04 encarga resolver estos asuntos; no aprueba automáticamente sus soluciones.

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

#### S02.9 — Decisiones técnicas pendientes y límites

| ID | Recomendación sometida a aprobación | Consecuencia / alternativa |
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
**S02 investigada y especificada; pendiente de revisión y aprobación humana de
E01–E07. S03 no iniciada.**

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

Requieren aprobación humana las propuestas E01–E07 de S02: perfil y topologías;
nominales de ensayo; signos, contadores y método S/FP/Q; tolerancias e integración;
temperatura y límites. D01–D06 ya están aprobadas para el cierre documental S01,
sin aprobar firmware/mapa ni equivalencia instrumental. También siguen pendientes
umbrales de escenarios; confirmación instrumental de variante/firmware;
representación y versionado del contrato común; semántica temporal y deduplicación;
artefacto de referencia; proyección legada/temperatura/etiquetas; estrategia de
integración, nivel de aceptación pretendido y cualquier modificación de esquema,
HTTP o recursos persistentes. S01 queda cerrada sólo para el alcance aprobado;
sus pendientes instrumentales mantienen las condiciones D02/D05 y la regla del
apartado 1. S02 somete a aprobación signos,
cuadrantes, contadores y acumulación; S03, dimensiones de calidad, soporte temporal,
compatibilidad y plan de aceptación por nivel. Los pendientes de instrumentos
están en la matriz; no justifican inventar capacidades para el núcleo genérico.

La entrega documental inicial incluyó contrato, roadmap, matriz y adición mínima
en AGENTS. Esta intervención S01/S02 sólo actualiza los tres primeros documentos.
Validación requerida en esta fase: correspondencia estática con código, coherencia
de fuentes/alcance, enlaces locales y diff. No se requieren servicios ni pruebas
de aplicación. No se implementa el simulador ni se inicia la etapa siguiente.
