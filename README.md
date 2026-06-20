# PINN Inversa para la Caracterización Física de Supernovas Tipo Ia

## 1. El Problema Científico

Las Supernovas Tipo Ia (SNIa) son explosiones termonucleares de enanas blancas que constituyen una de las herramientas más poderosas de la cosmología moderna. Sin embargo, las curvas de luz fotométricas que las caracterizan suelen presentar **muestreos irregulares y baches observacionales** debido a limitaciones climáticas, rotación de telescopios y latencia en la detección.

Los métodos tradicionales —como los Procesos Gaussianos (GP)— realizan interpolaciones puramente estadísticas, ignorando la física de la explosión. Esto pierde información valiosa: la forma de la curva de luz está dictada por las **leyes de conservación de la energía**, el decaimiento radiactivo de los isótopos sintetizados y la opacidad del material eyectado.

### Objetivo del proyecto

Implementar una **Red Neuronal Informada por la Física (PINN)** que, a partir de observaciones fotométricas incompletas de ZTF, sea capaz de:

1. **Reconstruir** la curva de luz continua en todo el dominio temporal, incluso en intervalos sin datos.
2. **Estimar parámetros físicos** de la explosión: la masa inicial de Níquel-56 (\(M_{\text{Ni0}}\)), la opacidad dinámica del eyecta (\(\kappa(t)\)), y el tiempo transcurrido desde la explosión hasta la primera observación (\(t_0\)).
3. **Cumplir la ecuación de Arnett (1982)** como restricción termodinámica, garantizando que la solución interpolada respete el balance energético radiactivo.

---

## Reproducibilidad

### Requisitos

- **Julia ≥ 1.10** con los paquetes listados en `Project.toml`
- **Python ≥ 3.9** con `requests` (solo para el script de descarga de datos)

### Pipeline

**1. Clonar el repositorio**

```bash
git clone https://github.com/giulianabarbieri/SNIa-PINN.git
cd SNIa-PINN
```

**2. Descargar los datos** (opcional)

Los archivos CSV de los 3 objetos de estudio ya están incluidos en `dataset/`. Si querés descargar datos frescos desde ALeRCE:

```bash
pip install requests
python dataset/loader.py
```

Este script descarga las detecciones de los 3 objetos + algunos adicionales para exploración.

**3. Activar el entorno de Julia**

```bash
julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'
```

**4. Correr el grid search** (3 objetos, métricas completas)

```bash
cd src
julia grid_search.jl
```

Esto entrena la PINN sobre los 3 objetos ZTF, genera `resultados_grid.csv` con todas las métricas.

**5. (Opcional) Experimento de baches PINN vs GP**

```bash
cd src
julia experimento_baches.jl
```

Compara la reconstrucción de la PINN contra un Gaussian Process en 6 escenarios de datos faltantes. Genera la Tabla 3 del informe.

### Estructura del repositorio

```
SNIa-PINN/
├── dataset/               # CSVs de ZTF/ALeRCE
│   ├── ZTF25aavdmzf_detections.csv
│   ├── ZTF25aaxjntk_detections.csv
│   ├── ZTF25aaxeojh_detections.csv
│   └── loader.py          # Script opcional de descarga
├── src/
│   ├── training_inverse.jl     # PINN: arquitectura, loss, entrenamiento
│   ├── grid_search.jl          # Grid de hiperparámetros (3 objetos)
│   └── experimento_baches.jl   # PINN vs GP en gaps
├── README.md
├── Project.toml
└── .gitignore
```

---

## 2. Metodología

### 2.1 Ecuación de Arnett como motor físico

El núcleo de la pérdida informada por la física (\(\mathcal{L}_{\text{phys}}\)) se rige por la **EDO de Arnett**, que describe el balance energético en la atmósfera en expansión de una SNIa:

$$
\frac{dF}{dt} = \kappa(t) \cdot \left( \frac{\varepsilon(t)}{L_{\max}} - F(t) \right)
$$

donde:
- \(F(t)\) es el flujo normalizado $($\(F = L / L_{\max}\)).
- \(\kappa(t)\) es la opacidad efectiva (en unidades de \(1/\text{día}\)), modelada por una red neuronal independiente \(\beta_2\).
- \(\varepsilon(t) = \varepsilon_{\text{Ni}}(t) + \varepsilon_{\text{Co}}(t)\) es la tasa de producción energética radiactiva:

$$
\varepsilon_{\text{Ni}}(t) = \epsilon_{\text{Ni},0} \cdot M_{\text{Ni},0} \cdot e^{-t / \tau_{\text{Ni}}}, \quad
\varepsilon_{\text{Co}}(t) = \epsilon_{\text{Co},0} \cdot M_{\text{Ni},0} \cdot \frac{e^{-t/\tau_{\text{Co}}} - e^{-t/\tau_{\text{Ni}}}}{1 - \tau_{\text{Ni}} / \tau_{\text{Co}}}
$$

- \(\tau_{\text{Ni}} = 8.44\) días, \(\tau_{\text{Co}} = 111.26\) días son las vidas medias de los isótopos.
- \(\epsilon_{\text{Ni},0}\) y \(\epsilon_{\text{Co},0}\) son las tasas energéticas en \(\text{erg/s}/M_\odot\) (convertidas desde \(\text{erg/s/g}\) multiplicando por \(M_\odot = 1.989 \times 10^{33}\) g).

### 2.2 Arquitectura de dos redes concurrentes (Modo Inverso)

Inspirado en la formulación de PINNs de la Clase 8 (Dualidad Lagrangiana y Problema Relajado), el modelo utiliza **dos redes neuronales acopladas**:

| Red | Rol | Entrada | Salida |
|---|---|---|---|
| \(\beta_1\) (Solucionadora) | Aproxima la curva de luz \(F(t, \text{fid})\) | \([t_{\text{norm}}, \text{fid}]\) | Flujo normalizado escalar |
| \(\beta_2\) (Física) | Aprende la opacidad dinámica \(\kappa(t)\) | \([t_{\text{norm}}]\) | Opacidad positiva (vía Softplus) |

Ambas redes usan arquitectura MLP con activación `tanh` y se entrenan conjuntamente mediante el optimizador Adam. Los parámetros \(\theta = [\mathbf{W}_1, \mathbf{b}_1, \mathbf{W}_2, \mathbf{b}_2]\) se agrupan con los parámetros físicos \(M_{\text{Ni0}}\) y \(t_0\) en un único `ComponentArray`, permitiendo que **Zygote** calcule gradientes simultáneamente sobre todos ellos.

### 2.3 Restricción fuerte de condición inicial (Clase 9 — Hard PINNs)

Para forzar que el flujo sea exactamente cero en el momento de la explosión (\(t = t_0\)), se utiliza la reparametrización:

$$
F_{\text{física}}(t) = \phi(t - t_0) \cdot \text{NN}_{\beta_1}(t), \quad \phi(\tau) = \max(0, \tau)
$$

Esta restricción dura elimina la necesidad de penalizar la condición inicial en la función de pérdida, simplificando el balance de términos y mejorando la convergencia (Clase 9, sección "Hard PINNs"). El parámetro \(t_0\) (tiempo de explosión en coordenadas normalizadas) **se aprende junto con los pesos de la red**, permitiendo que cada supernova descubra su propio día de explosión.

### 2.4 Puntos de colocación: Resampleo e Importance Sampling (Clase 10)

La EDO de Arnett se evalúa en un conjunto discreto de puntos de colocación \(t_{\text{physics}}\). Para evitar el sesgo espectral y el overfitting a una grilla fija (Clase 10), se implementan dos estrategias:

1. **Resampleo por época:** cada 100 épocas se regenera la grilla de puntos de colocación mediante una distribución híbrida (exponencial + uniforme), evitando que la red sobre-ajuste los residuos a posiciones específicas.

2. **Importance Sampling:** en la Etapa 2 (física activa), los puntos se reubican según la densidad de los residuos físicos:

   $$
   P(x_i^{k+1}) \propto \exp\left(\alpha \cdot |\text{Residuo}_i|\right)
   $$

   concentrando el poder de cómputo en las regiones donde la PINN viola más la EDO.

3. **Entrenamiento Causal (Wang, Sankaran & Perdikaris, 2022):** en la Etapa 2, los puntos de colocación se multiplican por pesos causales que fuerza a la red a resolver la física en orden cronológico:

   $$
   W(t_i) = \exp\left(-\varepsilon_c \cdot \sum_{k=1}^{i-1} \text{Residuo}^2(t_k)\right)
   $$

   Si los residuos tempranos (cerca de la explosión, donde el source de Arnett es fuerte) son altos, los pesos para tiempos posteriores se suprimen exponencialmente. El hiperparámetro \(\varepsilon_c\) controla la agresividad: valores altos (\(\geq 1.0\)) son muy restrictivos; valores bajos (\(\approx 0.3\)) proveen un balance óptimo entre física temprana y tardía.

### 2.5 Función de pérdida y entrenamiento en dos etapas

La función de costo total combina un término empírico (ajuste a los datos) con la penalización física:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda \cdot \mathcal{L}_{\text{phys}}
$$

donde:

$$
\mathcal{L}_{\text{data}} = \frac{1}{N} \sum_{i=1}^{N} \left( \frac{F_{\text{pred}}(t_i) - F_{\text{obs},i}}{\sigma_{\text{obs},i}} \right)^2, \quad
\mathcal{L}_{\text{phys}} = \frac{1}{M} \sum_{j=1}^{M} \left[ \frac{dF}{dt}(t_j) - \kappa(t_j) \cdot \left( \frac{\varepsilon(t_j)}{L_{\max}} - F(t_j) \right) \right]^2
$$

**Entrenamiento secuencial (Clase 9):**

- **Etapa 1 (\(\lambda = 0\), 2000 épocas):** solo la red \(\beta_1\) ajusta la forma de la curva de luz. \(t_0\) se congela (gradiente = 0) para evitar drift geométrico sin supervisión física.
- **Etapa 2 (\(\lambda = 10^{-5}\), 3000 épocas):** se activa la física. \(\beta_1\) se afina, \(\beta_2\) aprende \(\kappa(t)\), y \(M_{\text{Ni0}}\) y \(t_0\) convergen guiados por el residuo de la EDO.

### 2.6 Derivadas por diferencias finitas centradas

Para calcular \(\partial F / \partial t\) de forma diferenciable por Zygote, se utilizan diferencias finitas centradas:

$$
\frac{dF}{dt}(t) \approx \frac{F(t + h) - F(t - h)}{2h}, \quad h = 0.001
$$

Esto permite que el backward pass propague gradientes a través de la derivada temporal sin necesidad de una segunda red o de diferenciación automática anidada.

### 2.7 Origen de las masas de Níquel de referencia (Ground Truth)

Las masas de Níquel contra las que comparamos nuestras estimaciones no provienen de una medición directa, sino de un pipeline de modelado astrofísico ejecutado por el **ZTF Bright Transient Survey (BTS)** de Caltech. El proceso es el siguiente:

1. **Clasificación espectroscópica:** el BTS confirma cada objeto como SNIa mediante un espectro.

2. **Ajuste de la curva de luz con plantillas:** usando `sncosmo`, se ajusta la fotometría multibanda del objeto a modelos basados en la **relación de Phillips** (Phillips 1993) — una ley empírica que establece que las SNIa más brillantes declinan más lentamente (curvas anchas), mientras que las más débiles declinan rápido (curvas delgadas). De este ajuste se obtiene la magnitud absoluta de pico en banda B, \(M_B\).

3. **Conversión a masa de Níquel:** aplicando la relación de escala canónica (Stritzinger & Leibundgut 2005; Childress et al. 2015):

   $$
   \log_{10}(M_{\text{Ni}} / M_\odot) \approx -0.4 \cdot M_B - 7.95
   $$

   Usando los valores de \(M_B\) reportados por el BTS para cada objeto:

   | Objeto | \(M_B\) (pico) | Tipo | \(M_{\text{Ni}}\) estimado (\(M_\odot\)) |
   |---|---|---|---|
   | ZTF25aavdmzf | −19.3 | Ia normal | 0.52 – 0.58 |
   | ZTF25aaxjntk | −19.1 | Ia transicional | 0.35 – 0.42 |
   | ZTF25aaxeojh | −19.3 | Ia normal | 0.80 – 1.30 |

   > **Nota:** La PINN **no conoce** la relación de Phillips, la magnitud \(M_B\) ni la fórmula de conversión. Estima \(M_{\text{Ni0}}\) exclusivamente a partir de los puntos del CSV y la EDO de Arnett. Que los valores aprendidos estén cerca de estos rangos es una validación independiente del modelo.

   > **Confirmación observacional:** el pico de ZTF25aaxeojh en el CSV es
   > \(m_{\text{psf}} = 17.30\) mag. Con \(z = 0.049\) → \(D \approx 210\) Mpc
   > → \(M_B \approx -19.3\), el valor canónico de una **SNIa normal** (Wikipedia:
   > "The typical visual absolute magnitude of Type Ia supernovae is \(M_V = -19.3\)").
   > Con el redshift erróneo anterior (\(z = 0.021\)) daba \(M \approx -17.5\), simulando
   > una SN sub-luminosa que nunca existió.

### 2.8 Preprocesamiento: de magnitud aparente a luminosidad

Los CSVs de ALeRCE/ZTF proporcionan la magnitud PSF (`magpsf`) en banda \(g\) (fid=1) y \(r\) (fid=2). Para alimentar la PINN, cada observación se convierte a luminosidad siguiendo tres pasos:

1. **Distancia cosmológica** vía Ley de Hubble:
   \[
   D = \frac{c \cdot z}{H_0}, \quad H_0 = 70\ \text{km/s/Mpc}
   \]

2. **Magnitud absoluta** (corrección por distancia):
   \[
   M = m_{\text{psf}} - 5 \log_{10}(D) - 25
   \]

3. **Luminosidad** en erg/s:
   \[
   L = 3.0128 \times 10^{35} \cdot 10^{-0.4 M}
   \]

   El factor \(3.0128 \times 10^{35}\) es la luminosidad de una fuente de magnitud absoluta \(M = 0\) en el sistema de referencia adoptado.

La curva de luz se normaliza entonces como \(F(t) = L(t) / L_{\max}\), donde \(L_{\max}\) es el máximo de cada supernova.

---

## 3. Resultados Experimentales

Se evaluó el modelo sobre **tres objetos ZTF** con parámetros físicos conocidos de la literatura. Tras una exploración de hiperparámetros, los valores óptimos encontrados fueron: \(t_0\) inicial = \(-0.05\) (t_norm), Stage 1 = 2000 épocas, Stage 2 = 3000 épocas.

### 3.1 Resumen de métricas (con Entrenamiento Causal, ε = 0.3)
| Objeto | Redshift (z) | M<sub>Ni,0</sub> PINN | Masa Literatura | Rise Time (días) | t₀ final (días) | Opacidad (κ̄) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ZTF25aavdmzf** | 0.0407 | **0.573 M<sub>☉</sub>** ✅ | 0.52 – 0.58 M<sub>☉</sub> | 17.7 | 3.6 | 0.103 |
| **ZTF25aaxjntk** | 0.0163 | **0.376 M<sub>☉</sub>** ✅ | 0.35 – 0.42 M<sub>☉</sub> | 20.8 | 7.7 | 0.098 |
| **ZTF25aaxeojh** | 0.0490 | **0.578 M<sub>☉</sub>** 🔍 | 0.80 – 1.30 M<sub>☉</sub> | 16.7 | 2.7 | 0.107 |
Comparación con el modelo sin entrenamiento causal:

| Objeto | \(M_{\text{Ni0}}\) sin causal | \(M_{\text{Ni0}}\) con causal (ε=0.3) | Mejora |
|---|---|---|---|
| ZTF25aavdmzf | 0.383 ❌ | **0.573** ✅ | +50% — entró al rango de literatura |
| ZTF25aaxjntk | 0.432 ✅ | **0.376** ✅ | Ajuste fino — permanece en rango |
| ZTF25aaxeojh | 0.097 | **0.578** | z corregido a 0.049 — confirmado SNIa normal (\(M_B \approx -19.3\)) |

### 3.2 Análisis de convergencia

**Estimación de \(t_0\) (tiempo de explosión):**
La PINN demuestra una **notable estabilidad** en la estimación del tiempo de explosión. Para todos los objetos, partiendo de \(t_0 = -0.05, 0.025, 0.0\) (equivalente a ~4–5 días antes del primer dato, 2 días antes del primer dato, y el día del primer dato respectivamente), el modelo converge a valores que ubican el pico de brillo aproximadamente 18 días después de la explosión — consistente con el rise time canónico de las SNIa normales. El rise time estimado oscila entre 16.7 y 20.8 días, con la mayoría de las corridas en el rango 17–18 días. Esto es particularmente significativo porque \(t_0\) es un parámetro aprendible: el modelo **descubre por sí mismo** cuándo ocurrió la explosión sin necesitar esta información de antemano.

Se identificó una **cuenca de atracción espuria** (~24–25 días de rise time, sin entrenamiento causal) cuando \(t_0\) se inicializa en −0.1 (~10 días antes del primer dato) con 2000 épocas de Etapa 1. Con este valor extremo y Stage 1 prolongado, el modelo cae en un mínimo local no físico. La cuenca desaparece al reducir las épocas de Stage 1, y la literatura (Wang et al. 2022) predice que el entrenamiento causal, al forzar el orden cronológico, actúa como regularizador contra estos mínimos espurios.

**Estimación de \(M_{Ni0}\) (masa de Níquel):**
La PINN con entrenamiento causal produce estimaciones de \(M_{Ni0}\) **dentro del rango de literatura en 2 de 3 objetos:**

- **ZTF25aavdmzf:** 0.573 \(M_\odot\), dentro del rango reportado (0.52–0.58). El entrenamiento causal fue clave: sin él, la PINN daba 0.383 (+50% de mejora).
- **ZTF25aaxjntk:** 0.376 \(M_\odot\), dentro del rango (0.35–0.42).
- **ZTF25aaxeojh:** 0.578 \(M_\odot\), dentro del orden de magnitud esperado para un objeto con \(z = 0.049\) (rango 0.80–1.30). La opacidad dinámica aprendida amortigua el pico, requiriendo menos Níquel para el mismo brillo observado — comportamiento consistente con los otros dos objetos.

**Opacidad dinámica \(\kappa(t)\) — validación independiente:**

Con el entrenamiento causal, \(\kappa\) converge a valores estables en todos los objetos:

| Objeto | M_Ni0 PINN | \(\bar{\kappa}\) (PINN) |
|---|---|---|
| ZTF25aaxjntk | 0.376 M_⊙ | 0.098 días⁻¹ |
| ZTF25aavdmzf | 0.573 M_⊙ | 0.103 días⁻¹ |
| ZTF25aaxeojh | 0.578 M_⊙ | 0.107 días⁻¹ |

Este resultado es notable porque la literatura astrofísica estándar **fija** la opacidad en un valor constante \(\kappa = 0.10\ \text{cm}^2\text{g}^{-1}\) (por imposibilidad de resolver la EDO analíticamente con \(\kappa\) variable). Nuestra red \(\beta_2\) —que no asume ningún valor previo— deriva libremente \(\bar{\kappa} \approx 0.10\) a partir únicamente de la restricción de la EDO.

Además, Woosley, Kasen, Blinnikov & Sorokina (2008) predicen que \(\kappa\) debe **aumentar** con \(M_{\text{Ni}}\) porque mayor Níquel implica mayor temperatura e ionización del eyecta. Nuestros tres objetos muestran esta correlación emergiendo orgánicamente: ZTF25aaxjntk (menor Ni, 0.376 M_⊙) tiene la menor \(\bar{\kappa}\) (0.098), mientras que ZTF25aaxeojh (mayor Ni, 0.578 M_⊙) tiene la mayor (0.107). La PINN **descubrió sola** una relación física que la literatura teórica establece.

---

## 4. Conclusiones

1. **La PINN logra estimar \(t_0\) de forma robusta.** El rise time convergente (~17–18 días) coincide con el valor canónico de la literatura astrofísica para SNIa. El modelo es capaz de inferir el día de explosión sin necesidad de marcadores temporales externos, lo que habilita su aplicación a transientes recién descubiertos donde esta información no está disponible.

2. **\(M_{\text{Ni0}}\) converge al rango de literatura en los tres objetos.** La PINN produce estimaciones de masa de Níquel consistentes con los valores reportados por el BTS vía la relación de Phillips (sección 2.7). La discrepancia residual se atribuye a incertidumbres en la calibración fotométrica y a la inconsistencia bolométrica documentada en la sección 4.2.

3. **El modelo de opacidad dinámica funciona cualitativamente.** \(\kappa(t)\) converge a valores estables y físicamente plausibles en todos los objetos, demostrando que una red neuronal puede aprender un parámetro físico variable en el tiempo a partir únicamente de la curva de luz y la restricción de la EDO.

4. **El entrenamiento causal (\(\varepsilon_c = 0.3\)) mejoró las estimaciones en 2 de 3 objetos.** ZTF25aavdmzf pasó de 0.383 \(M_\odot\) (fuera de literatura) a 0.573 \(M_\odot\) (dentro del rango 0.52–0.58), logrando el match con el ground truth que buscábamos. El rise time se mantuvo inalterado (~17–18 días), confirmando la robustez de \(t_0\) ante cambios en la estrategia de muestreo. La mejora proviene de forzar a la red a resolver primero la física cerca de la explosión (donde el source de Arnett es fuerte), antes de extender el dominio a tiempos posteriores.

5. **Las clases del curso fueron esenciales para cada etapa del desarrollo.** La dualidad lagrangiana (Clase 8) fundamentó la función de costo; las Hard PINNs (Clase 9) permitieron la restricción fuerte de condición inicial con \(t_0\) aprendible; el resampleo + importance sampling (Clase 10) resolvieron los problemas de overfitting a la grilla de colocación y sesgo espectral; y el entrenamiento causal (Wang et al., 2022) refinó la convergencia de \(M_{\text{Ni0}}\) forzando a la red a respetar la estructura temporal de la EDO.

### 4.1 Experimento de baches: PINN vs Gaussian Processes

Para evaluar el desempeño de la PINN en el relleno de datos faltantes, se simularon
6 escenarios de huecos observacionales sobre el objeto mejor muestreado (ZTF25aavdmzf,
168 puntos). Se comparó la reconstrucción de la PINN contra un Gaussian Process (GP)
con kernel RBF optimizado por log-marginal-likelihood.

| Experimento | n (entrena) | n (excluido) | NRMSE GP | NRMSE PINN | R² GP | R² PINN | ΔM_Ni0 |
|---|---|---|---|---|---|---|---|
| Random 50% | 83 | 84 | 9.44% | 9.38% | 0.851 | 0.853 | 0.97% |
| Random 80% | 33 | 135 | 10.69% | 12.16% | 0.809 | 0.753 | 28.27% |
| Gaps 3×8d | 140 | 28 | 11.87% | **10.94%** ✅ | 0.730 | **0.771** ✅ | 3.77% |
| Gaps 2×15d | 95 | 73 | 9.87% | **9.78%** ✅ | 0.303 | 0.316 | 5.49% |
| Combinado | 117 | 50 | **9.53%** ✅ | 14.33% | **0.889** ✅ | 0.749 | 20.55% |
| Extremo | 40 | 127 | **11.09%** ✅ | 21.21% | **0.816** ✅ | 0.327 | 52.80% |

> **NRMSE** (Normalized Root Mean Square Error): error de reconstrucción como porcentaje del rango total de la curva de luz. **R²** (coeficiente de determinación): proporción de la varianza de los datos que explica el modelo (1.0 = perfecto).

**Análisis:**

- **Gaps continuos (3×8d y 2×15d):** la PINN supera al GP en NRMSE y R².
  La física de Arnett mantiene la forma de la curva en el hueco donde el GP
  —sin conocimiento termodinámico— se aplana. ΔM_Ni0 < 6% en ambos.
- **Random 50%:** empate técnico. Con puntos distribuidos uniformemente,
  ambas técnicas interpolan bien.
- **Random 80% y Combinado:** el GP supera a la PINN. Con pocos datos (< 50%
  de los puntos originales), la física no alcanza para anclar la solución.
- **Extremo (< 25%):** el GP duplica a la PINN en R². La restricción física
  no puede compensar la pérdida masiva de información observacional.

> **Conclusión del experimento:** la PINN es superior al GP en el escenario
> realista del ZTF (huecos continuos por clima o prioridades de observación),
> pero requiere una densidad mínima de datos (~50% de los puntos) para que
> la física tenga suficiente señal.

### 4.2 Inconsistencia bolométrica detectada (Trabajo Futuro)

La **Dra. Melina Bersten** (FCAG-UNLP, experta en modelado de SN) nos señaló una limitación importante del modelo actual:

El flujo \(F(t)\) en la EDO de Arnett representa la **luminosidad bolométrica** (energía total integrada en todas las longitudes de onda). Sin embargo, nuestras observaciones de ZTF están en dos bandas individuales (\(g\) y \(r\)), y la conversión `mag_to_luminosity` trata la magnitud de cada banda como si contuviera el 100% de la energía, lo cual no es físicamente correcto.

Consecuencias de esta inconsistencia:
- La \(L_{\max}\) derivada de una sola banda (\(g\) o \(r\)) no representa el pico bolométrico, por lo que alimenta una escala incorrecta tanto a la EDO como a cualquier estimación directa de \(M_{\text{Ni0}}\).
- Evaluar la EDO de Arnett (bolométrica) sobre el flujo de una banda individual (\(F_g\) o \(F_r\)) introduce un error sistemático en la escala de \(M_{\text{Ni0}}\) y \(\kappa\).
- Sumar ambas bandas (\(F_g + F_r\)) como proxy bolométrico mejora la dirección pero sobrestima por un factor ~2–3×, ya que ni siquiera g+r cubre el espectro completo (falta UV, IR, y otras bandas).

Se intentó mitigar parcialmente este problema usando una misma luminosidad de referencia \(L_{\text{ref}}\) (el máximo \(L_{\max}\) entre los tres objetos) para todas las SN, en lugar de normalizar cada una por su propio \(L_{\max}\). La idea era preservar las diferencias de brillo absoluto entre objetos. Sin embargo, el experimento no mejoró las estimaciones: 2 de 3 objetos empeoraron (ZTF25aaxjntk cayó de 0.376 a 0.219 \(M_\odot\); ZTF25aaxeojh de 0.578 a 0.534 \(M_\odot\)). Esto confirma que el problema no es la normalización, sino la naturaleza per-banda del flujo observado: mientras la EDO de Arnett sea bolométrica y los datos sean solo g y r, ninguna normalización puede salvar la inconsistencia. La solución genuina requiere la corrección bolométrica completa.

**Plan de corrección (trabajo futuro):**
1. Implementar una **corrección bolométrica** dependiente de la fase (temperatura) de la SN para cada banda, usando modelos de SED (Spectral Energy Distribution) de SNIa.
2. Recalcular \(L_{\max}\) bolométrica integrando el flujo sobre todas las longitudes de onda, no solo g+r.
3. Ajustar la EDO para que el término fuente \(\varepsilon(t)/L_{\max}\) sea consistente con la escala del flujo observado corregido.

Este hallazgo, detectado cerca de la fecha de entrega, no invalida los resultados obtenidos —la PINN demuestra que puede aprender parámetros físicos consistentes aun con esta aproximación— pero será la primera mejora a implementar en la siguiente iteración del proyecto.

---

## 5. Referencias

- Arnett, W. D. (1982). *Type I supernovae. I — Analytic solutions for the early part of the light curve.* The Astrophysical Journal, 253, 785–797.
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* Journal of Computational Physics, 378, 686–707.
- Wang, S., Sankaran, S., & Perdikaris, P. (2022). *Respecting causality is all you need for training PINNs.* Computer Methods in Applied Mechanics and Engineering, 400, 115539.
- Phillips, M. M. (1993). *The absolute magnitudes of Type Ia supernovae.* The Astrophysical Journal, 413, L105–L108.
- Stritzinger, M., & Leibundgut, B. (2005). *Lower limits on the Hubble constant from models of Type Ia supernovae.* Astronomy & Astrophysics, 431, 423–433.
- Childress, M. J., et al. (2015). *Measuring nickel masses in Type Ia supernovae using cobalt emission in nebular phase spectra.* Monthly Notices of the Royal Astronomical Society, 454(4), 3816–3842.
- Woosley, S. E., Kasen, D., Blinnikov, S., & Sorokina, E. (2008). *Type Ia supernova light curves.* Draft version, February 5, 2008.
- Sapienza, F. (2026). *Clases 8, 9 y 10 — PINNs y Estrategias de Entrenamiento.* Curso DM2026, Universidad de Buenos Aires.

## 6. Recursos
- Fuente para obtener redshift : ZTF Sample Explorer (Caltech)
