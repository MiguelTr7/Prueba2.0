# GUIÓN DE PRESENTACIÓN FINAL — 15 MINUTOS
## Sistema de Predicción de Desempeño de Empleados
**Arturo — EP2 SCY1101**

---

## BLOQUE 0 — Apertura (30 segundos)
**Mostrar**: Notebook 07, celda inicial "Sistema de análisis y predicción..."

"Buenos días. Mi proyecto se llama **Sistema de Predicción de Desempeño de Empleados** aplicado a datos reales de Recursos Humanos.

En los próximos 15 minutos voy a mostrarles cómo usé Machine Learning para resolver un problema real de gestión de personas."

---

## BLOQUE 1 — Introducción y Problema (1.5 minutos)
**Mostrar**: Notebook 01, celda de contexto O Notebook 07 sección "Problema de negocio"

"El problema es simple pero importante: las empresas tienen datos históricos sobre sus empleados — ausencias, capacitaciones, evaluaciones — pero no tienen una forma sistemática de **anticipar quién va a tener alto desempeño**.

Lo que yo hice fue construir un pipeline completo de Machine Learning desde **tres ángulos complementarios**:

1. **Clasificación**: predecir si un empleado tendrá alto desempeño o no
2. **Regresión**: estimar el puntaje numérico exacto de desempeño
3. **Clustering**: segmentar empleados en perfiles para intervenciones diferenciadas

Trabajé con **1.449 evaluaciones** de aproximadamente **430 empleados únicos** distribuidas en **4 períodos semestrales** (2021-S1 a 2023-S2)."

**Cómo interpretar**: El dataset es completo, representativo y tiene buena cobertura temporal

---

## BLOQUE 2 — Dataset y Feature Engineering (2 minutos)
**Mostrar**: 
- Notebook 00, celda de importación y limpieza
- Notebook 03, sección 5 "Variables predictoras" + sección de feature engineering (celdas b1cca15b y 9fb53424)

"El dataset integra cuatro fuentes: empleados, ausencias, capacitaciones y evaluaciones.

Pero lo más importante NO fue juntar las tablas — fue el **feature engineering**. Creé **10 variables nuevas** que capturan lógica de RRHH.

Las **TRES MÁS IMPORTANTES**:

**Primera: `prev_puntaje`** — el puntaje del período anterior del mismo empleado. Tiene **correlación de 0.47** con el target. El mejor predictor del desempeño futuro es el desempeño pasado. Tiene sentido.

**Segunda: `evaluador_media`** — el promedio histórico que asigna cada evaluador. Y acá viene el **HALLAZGO MÁS IMPORTANTE del proyecto**:

La desviación estándar ENTRE evaluadores es **1.82**. La desviación estándar de un mismo evaluador en el tiempo es **0.84**. Eso significa que **el 60% de la varianza del puntaje viene de QUIÉN EVALÚA, no del empleado**.

Sin esta variable el F1 en CV era **0.46**. Con ella sube a **0.65**. **Esa sola variable explica más que todos los demás features juntos.**

**Tercera: `dept_media`** — el promedio histórico por departamento.

Las tres juntas evitan data leakage porque se calculan SOLO con datos del pasado."

**Cómo interpretar**: El feature engineering fue el 50% del éxito; las variables raw no tenían suficiente poder predictivo

---

## BLOQUE 3 — Modelos Supervisados (3 minutos)

### PARTE A: Clasificación
**Mostrar**: Notebook 03, sección 10 "Comparación de modelos de clasificación" (celda e023f024)

"Para **CLASIFICACIÓN** probé **7 modelos** base usando el mismo Pipeline de sklearn.

El preprocesador encadena:
1. Imputación de nulos numéricos (mediana)
2. Escalamiento de variables
3. OneHotEncoding de categóricas

Usé `class_weight='balanced'` en todos porque el target está desbalanceado — **73% clase 0, 27% clase 1**.

La métrica principal es el **F1-score**, deliberadamente sobre accuracy. ¿Por qué? Con clases desbalanceadas, un modelo que predice siempre clase 0 obtiene 73% de accuracy sin aprender nada. El F1 penaliza eso porque balancea precision y recall.

En RRHH los errores tienen costos asimétricos: no detectar talento real (falso negativo) es peor que clasificar mal hacia arriba (falso positivo). El F1 captura ese trade-off."

*(Señalar tabla del notebook)*

"**RESULTADOS — TOP 3 MODELOS BASE:**

| Modelo | F1-score | Precision | Recall | Balanced Acc |
|---|---|---|---|---|
| Gradient Boosting | **0.7105** | 0.7297 | 0.6923 | 0.7990 |
| Decision Tree | 0.6737 | 0.5714 | 0.8205 | 0.7970 |
| Random Forest | 0.6587 | 0.6180 | 0.7051 | 0.7724 |

El Gradient Boosting base tiene F1 = 0.7105. Es sólido, pero vamos a mejorarlo."

---

### PARTE B: Regresión
**Mostrar**: Notebook 03, sección 17 "Comparación de modelos de regresión" (celda e041f042)

"Para **REGRESIÓN** probé **6 modelos** estimando el puntaje numérico de desempeño (escala 0-10).

**RESULTADOS:**

| Modelo | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | **1.1865** | **0.9337** | **0.6420** |
| Gradient Boosting | 1.2514 | 0.9713 | 0.6018 |
| Random Forest | 1.2563 | 0.9723 | 0.5987 |
| SVR | 1.3274 | 1.0435 | 0.5520 |
| Decision Tree | 1.3987 | 1.0744 | 0.5025 |
| KNN | 1.5096 | 1.1806 | 0.4206 |

**Linear Regression es el ganador**: MAE de 0.93 puntos en escala 0-10, y R²=0.642 significa que explica el 64.2% de la varianza del puntaje.

¿Por qué Linear Regression gana a Random Forest en regresión? Porque las variables más importantes — `evaluador_media`, `prev_puntaje` — tienen relación **casi lineal** con el target. Random Forest captura no-linealidades, pero con 1.449 registros la ganancia es marginal y la varianza del ensamble suma error.

**Eso cambia después con la optimización.**"

---

## BLOQUE 4 — Optimización de Hiperparámetros (4 minutos)
**Mostrar**: 
- Notebook 04 (si existe) O Notebook 03 secciones 9.6, 9.7, 9.8
- Celda e2549404 y 15c4a1a5 (XGBoost/LightGBM)
- Celda 09ae5f71 (tabla comparativa final)

"Para la optimización usé **CUATRO ESTRATEGIAS DISTINTAS** sobre Random Forest + modelos avanzados.

**ESTRATEGIA 1: Random Forest BASE (sin optimización)**
- F1 en CV = 0.4877 ± 0.0297
- Baseline para comparación

**ESTRATEGIA 2: Modelos clásicos con CV**
- Gradient Boosting CV=5: F1 = **0.5182** ± 0.0288 (mejor que RF base)
- Decision Tree CV=5: F1 = 0.4458 ± 0.0442
- Random Forest CV=5: F1 = 0.4877 ± 0.0297

Nota: Estos F1 en CV son BAJOS en valor, pero reflejan la realidad del dataset.

**ESTRATEGIA 3: Modelos Avanzados (Optuna, 80 trials cada uno)**

Usé **Optuna** — optimización bayesiana que aprende de trials anteriores para decidir qué probar a continuación. No es búsqueda ciega, es búsqueda **inteligente**.

*Mostrar tabla de la celda 09ae5f71:*

| Modelo | F1-score Test |
|---|---|
| **Stacking (meta=LR)** | **0.7297** |
| **XGBoost (Optuna, 80 trials)** | **0.7174** |
| **VotingClassifier** | **0.7052** |
| **LightGBM (Optuna, 80 trials)** | **0.6979** |
| Gradient Boosting (base) | 0.7105 |
| Decision Tree | 0.6737 |
| Random Forest | 0.6587 |

**EL MEJOR ES STACKING: F1 = 0.7297**

*(Explicar Stacking)*: Combina 4 modelos base (RF, GB, XGBoost, LightGBM). Cada uno predice; luego un meta-modelo (Logistic Regression) aprende cuándo confiar en cada uno. Usa CV=5 internamente para evitar data leakage.

**PUNTO CRÍTICO**: En TODAS las estrategias, la validación cruzada se hizo SOLO sobre el conjunto de entrenamiento. El test set se reservó exclusivamente para evaluación final. Esto evita data leakage — error clásico en Optuna es evaluar dentro del trial sobre X_test, lo que contamina la búsqueda."

---

## BLOQUE 5 — Aprendizaje No Supervisado (2 minutos)
**Mostrar**: 
- Notebook 05, sección PCA + KMeans
- Notebook 07, sección 8 "Aprendizaje no supervisado"

"Aquí cambio de enfoque — en lugar de PREDECIR, DESCUBRO la estructura del dataset sin usar el target.

Usé **PCA + KMeans**:

**PCA**: Reducción a 2 dimensiones. Los dos componentes principales explican el **55.65% de la varianza** (PC1: 33.6% eje capacitación; PC2: 22.0% eje ausencias).

**KMeans**: Para elegir K, revisé Silhouette Score.
- K=2: máximo Silhouette, pero **trivial** — solo separa empleados con/sin capacitación
- K=3: también dominado por capacitación
- **K=4: SELECCIONÉ ESTO** — genera 4 perfiles ricos y accionables

**Los 4 CLUSTERS encontrados:**

| Cluster | Tamaño | Perfil |
|---|---|---|
| A | 492 emp (34%) | Sin capacitación, bajo desempeño, ausencias bajas — estables pero sin desarrollo |
| B | 657 emp (45%) | **SIN CAPACITACIÓN, ALTO DESEMPEÑO** — **SUPERSTARS**, talento natural |
| C | 123 emp (9%) | Con capacitación, bajo desempeño, muchas ausencias — **EN RIESGO** |
| D | 177 emp (12%) | Alto desempeño, sin capacitación, PERO muchas ausencias — **posible rotación** |

**VALIDACIÓN EXTERNA**: Crucé los clusters con `puntaje_desempeno` (variable NO usada en clustering) y los clusters se ordenan de mayor a menor puntaje. Eso **confirma que capturaron señal real**."

---

## BLOQUE 6 — Conclusiones y Lecciones Aprendidas (2.5 minutos)
**Mostrar**: 
- Notebook 06, secciones 4 y 5 (ya editadas)
- Notebook 07, sección 9 "Conclusiones finales"

"Tres conclusiones principales:

**PRIMERA: Feature engineering tuvo más impacto que la optimización**

- Dataset original: features débiles, modelos F1~0.46 en CV
- +Agregué `prev_puntaje` + `evaluador_media`: F1 jump → 0.65 en CV — **mejora de 41%**
- +Optimización (Optuna): F1 → máx 0.7297 (Stacking)

**El feature engineering puso la base. La optimización refinó.**

**SEGUNDA: El techo está limitado por la CALIDAD DEL TARGET, no por el algoritmo**

El puntaje de desempeño es **subjetivo**. Depende del evaluador. El sesgo del evaluador explica 60% de la varianza.

Aunque tuviéramos 100.000 registros, el modelo igual tiene un límite porque el target tiene **ruido estructural**. 

La solución real NO es más datos — es **estandarizar el proceso de evaluación** o implementar evaluaciones 360°.

**TERCERA: Clasificación + Regresión + Clustering = visión completa**

- **Clasificación** dice: '¿a quién retener?' (alto desempeño sí/no)
- **Regresión** dice: '¿cuánto mejor es uno que otro?' (graduación numérica)
- **Clustering** dice: '¿qué tipo de empleado es?' y '¿qué intervención necesita?' (perfiles accionables)

**LECCIONES APRENDIDAS para proyectos futuros:**

1. **Validar que el óptimo estadístico tiene sentido organizacional** — K=3 era estadísticamente correcto pero organizacionalmente trivial
2. **Data leakage en Optuna es error silencioso** — código corre sin errores pero resultados son sobreoptimistas
3. **Un Pipeline de sklearn no es solo comodidad** — es la ÚNICA forma correcta de garantizar que el preprocesador se ajuste SOLO en train

Eso es todo. Quedo disponible para preguntas."

---

## RESUMEN DE NÚMEROS CLAVE

| Métrica | Valor |
|---|---|
| **Dataset** | 1.449 evaluaciones, ~430 empleados, 4 períodos |
| **Target clasificación** | desempeno_alto (P75 = 5.60), 73% clase 0 / 27% clase 1 |
| **Target regresión** | puntaje_desempeno (escala 0-10) |
| **Variables finales** | 34 features (27 base + 4 interacción + 3 historial) |
| **Split** | 80% train / 20% test, random_state=42, stratified |
| **Best Classifier (Base)** | Gradient Boosting: F1=0.7105, Accuracy=0.8483 |
| **Best Regressor** | Linear Regression: R²=0.6420, RMSE=1.1865, MAE=0.9337 |
| **Best After Optimization** | Stacking Classifier: F1=0.7297 (Optuna 80 trials) |
| **Mejor variable** | evaluador_media (sesgo del evaluador capturado) |
| **CV estrategia** | StratifiedKFold=5 para clasificación, KFold=5 para regresión |
| **Clustering** | K=4, PCA 55.65% varianza, 4 perfiles de empleados |
| **Data leakage prevention** | Test set NUNCA usado en CV durante Optuna |

---

## CHECKLIST ANTES DE PRESENTAR

✅ **Ejecutables de último minuto:**
- [ ] Run All en Notebook 03 (secciones 10, 9.6, 9.7, 9.8) — secciones de clasificación y ensamble
- [ ] Run All en Notebook 05 (clustering) — gráficos PCA + clusters
- [ ] Run All en Notebook 07 (summary) — tablas comparativas

✅ **Materiales impresos/visibles:**
- [ ] Gráfico: comparacion_clasificacion.png (7 modelos base)
- [ ] Gráfico: comparacion_final_avanzada.png (base + XGBoost + LightGBM + Voting + Stacking)
- [ ] Gráfico: PCA + clusters (K=4)
- [ ] Gráfico: real vs predicho (regresión)

✅ **Timings exactos:**
- Bloque 0: 30 seg (apertura)
- Bloque 1: 1.5 min (problema + 3 ángulos)
- Bloque 2: 2 min (feature engineering, hallazgo sesgo evaluador)
- Bloque 3: 3 min (7 modelos, F1-scores, regresión)
- Bloque 4: 4 min (Optuna, Stacking, data leakage)
- Bloque 5: 2 min (PCA, K=4 clusters, validación)
- Bloque 6: 2.5 min (3 conclusiones + lecciones)
- **TOTAL: 15 minutos**

✅ **Si preguntan "¿por qué no 85%?":**

Respuesta: *"El desbalance de clases (73% vs 27%) y el ruido estructural del evaluador (60% de varianza) hacen difícil pasar 85% en F1. Pero logramos F1=0.7297 con Stacking (comparado con 0.71 Gradient Boosting base) — es sólido. Los modelos avanzados (XGBoost 0.7174, LightGBM 0.6979) exploran el espacio bien, pero la 'pared' es el target subjetivo, no el algoritmo. Con evaluaciones 360° o estandarización del proceso, podría mejorar significativamente."*

✅ **Si preguntan "¿qué hiciste diferente a otros?":**

Respuesta: *"Tres cosas: (1) Feature engineering específico de RRHH — identifiqué que el sesgo del evaluador explica 60% de la varianza, agregué prev_puntaje y evaluador_media, mejoró F1 de 0.46 a 0.65. (2) Combiné tres enfoques (clasificación + regresión + clustering) para una visión completa. (3) Evité data leakage en Optuna usando CV=5 solo sobre train — error común es usar test set dentro de los trials."*

---

**Última revisión antes de presentar:** Abre Notebook 07 → desplázate a secciones numeradas → confirma que los números coinciden con esta guía.

**¡LISTO PARA PRESENTAR!**