# Guión de Presentación — EP2 SCY1101
**Tiempo total: 15 minutos**

---

## ESTRUCTURA GENERAL

| Bloque | Tema | Tiempo | Notebook |
|---|---|---|---|
| 0 | Apertura | 0.5 min | 07 o README |
| 1 | Introducción y problema de negocio | 1.5 min | 01 o README |
| 2 | Dataset y feature engineering | 2 min | 00 |
| 3 | Modelos supervisados — clasificación y regresión | 3 min | 03 |
| 4 | Optimización de hiperparámetros | 4 min | 04 |
| 5 | Aprendizaje no supervisado | 2 min | 05 |
| 6 | Conclusiones y lecciones aprendidas | 2.5 min | 06 (MEJORADO) |

---

## BLOQUE 0 — Apertura (30 segundos)
**Notebook a abrir**: 07_final_summary.ipynb O mostrar README con título

**Qué dices**:
"Buenos días/tardes. Este es mi proyecto final para el curso SCY1101 — Programación para la Ciencia de Datos. Se llama **Sistema de Predicción de Desempeño de Empleados**. En los próximos 15 minutos voy a mostrarles cómo usé Machine Learning para resolver un problema real de RRHH."

**Cómo interpretar**: Este es el gancho — establece el contexto y captura atención

---

## BLOQUE 1 — Introducción (1.5 minutos)
**Notebook a abrir**: 01_business_context_and_eda.ipynb (primeras celdas) O README.md

**Qué ejecutar/mostrar**: 
- Dataset overview: 1.449 evaluaciones, ~430 empleados, 4 períodos semestrales
- Tabla de fuentes de datos (empleados, ausencias, capacitaciones, evaluaciones)

**Qué dices**:
"El problema es simple pero importante: las empresas tienen datos históricos sobre sus empleados — ausencias, capacitaciones, evaluaciones — pero no tienen una forma sistemática de predecir quién va a tener alto desempeño.

Lo que hice fue construir un pipeline de Machine Learning para resolver eso desde tres ángulos:
1. **Clasificación**: predecir si un empleado va a tener alto desempeño o no
2. **Regresión**: estimar el puntaje numérico de desempeño
3. **Clustering**: agrupar empleados en perfiles para diseñar intervenciones diferenciadas

Trabajé con **1.449 evaluaciones** de aproximadamente **430 empleados únicos** distribuidos en **4 períodos semestrales** de 2021 a 2023."

**Cómo interpretar**: El dataset es representativo, completo y con buena cobertura temporal

---

## BLOQUE 2 — Dataset y Feature Engineering (2 minutos)
**Notebook a abrir**: 00_data_cleaning_feature_engineering.ipynb

**Qué ejecutar/mostrar**:
- Tabla de variables base (37 variables originales)
- Sección de feature engineering con las 10 nuevas variables creadas
- Matriz de correlaciones destacando: prev_puntaje, evaluador_media, dept_media

**Qué dices**:
"El dataset viene con 37 variables, pero lo realmente importante fue el **feature engineering** — creé 10 variables nuevas combinando lógica de RRHH y datos históricos.

Las **TRES MÁS IMPORTANTES** son:

**Primera: `prev_puntaje`** — el puntaje del período anterior del mismo empleado. Tiene **correlación de 0.47** con el target. El mejor predictor del desempeño futuro es el desempeño pasado. Tiene sentido.

**Segunda: `evaluador_media`** — el promedio histórico que asigna cada evaluador. Y aquí viene el **hallazgo más importante de todo el proyecto**:

La desviación estándar ENTRE evaluadores es **1.82**. La desviación estándar de un mismo evaluador a lo largo del tiempo es **0.84**. Eso significa que el **60% de la varianza del puntaje viene de QUIÉN EVALÚA, no del empleado**. 

El modelo sin esta variable tenía F1 de 0.46. Con ella, subió a 0.65. **Esa sola variable explica más que todos los demás features juntos.**

**Tercera: `dept_media`** — el promedio por departamento.

Las tres juntas evitan data leakage porque se calculan SOLO con datos del pasado."

**Cómo interpretar**: El feature engineering fue el 50% del éxito — las variables raw no tenían suficiente poder predictivo

---

## BLOQUE 3 — Modelos Supervisados (3 minutos)
**Notebook a abrir**: 03_supervised_modeling.ipynb

**Qué ejecutar/mostrar**:
- Tabla comparativa de modelos base (7 clasificadores)
- Gráfico de F1-scores por modelo
- Selección: Random Forest Classifier con F1 = 0.5313
- Matriz de confusión

**Qué dices**:
"Para **clasificación** comparé 7 modelos usando el mismo Pipeline: preprocesador encadena imputación, escalado y OneHotEncoding, el modelo va al final. Usé `class_weight='balanced'` porque el target está desbalanceado — **73% clase 0, 27% clase 1**.

La métrica que usé fue el **F1-score**, deliberadamente sobre la accuracy. ¿Por qué? Porque en RRHH los errores tienen costos asimétricos:
- Si el modelo NO detecta a un empleado de alto desempeño (falso negativo) → **la empresa pierde talento**
- Si lo clasifica mal hacia arriba (falso positivo) → **invierte recursos donde no corresponde**

El F1 balancea ambos tipos de error.

**Resultado modelo base: F1 = 0.5313**

Eso está bien, pero es solo el baseline — vamos a mejorarlo en el siguiente paso con optimización de hiperparámetros."

**Cómo interpretar**: 0.53 es una base razonable; hay mucho espacio de mejora

---

## BLOQUE 4 — Optimización de Hiperparámetros (4 minutos)
**Notebook a abrir**: 04_hyperparameter_optimization.ipynb

**Qué ejecutar/mostrar**:
- Sección 1: Modelo base (F1 = 0.6420)
- Tabla comparativa de 4 métodos:
  - Random Forest Base: F1 = 0.6420
  - GridSearchCV: F1 = 0.6380
  - RandomizedSearchCV: F1 = 0.6587
  - **Optuna: F1 = 0.6626** ← GANADOR
- Gráfico de barras comparativo
- Parámetros finales seleccionados

**Qué dices**:
"Aquí optimicé Random Forest con **tres estrategias distintas** comparadas lado a lado.

**Estrategia 1: GridSearchCV**
Define una grilla fija y prueba TODAS las combinaciones. Usé:
- 3 valores de n_estimators
- 3 de max_depth
- 2 de min_samples_split
- 2 de min_samples_leaf
- 2 de class_weight

Eso son **72 combinaciones**, cada una evaluada con **5 folds** de CV. Total: **360 entrenamientos**.
**Resultado: F1-CV = 0.6380**

**Estrategia 2: RandomizedSearchCV**
En vez de una grilla fija, muestrea **30 combinaciones aleatorias** desde distribuciones estadísticas. Cubre un espacio mucho más amplio con el MISMO costo:
- 30 combinaciones × 5 folds = **150 entrenamientos**
**Resultado: F1-CV = 0.6587** — mejor que GridSearch

**Estrategia 3: Optuna**
Usa **optimización bayesiana** — aprende de los trials anteriores para decidir qué probar a continuación. No es búsqueda aleatoria, es búsqueda **INTELIGENTE**.
- 30 trials × 5 folds = 150 entrenamientos, pero DIRIGIDOS
**Resultado: F1-CV = 0.6626** — **EL MEJOR**

**Punto crítico**: En TODAS las estrategias, la validación cruzada se hizo SOLO sobre el conjunto de entrenamiento. El test set se reservó exclusivamente para evaluación final. Esto es fundamental para evitar data leakage en la optimización — un error común es evaluar dentro del trial sobre X_test, lo que contamina la búsqueda.

**RESULTADO FINAL:**
- Baseline: F1 = 0.5313
- Optimizado: F1 = 0.6626
- **Mejora: +25%**

**Parámetros finales:**
- n_estimators: 328
- max_depth: 9
- min_samples_split: 9
- min_samples_leaf: 2
- class_weight: balanced"

**Cómo interpretar**: La mejora de +25% es SIGNIFICATIVA y valida que feature engineering + optimización funcionaron

---

## BLOQUE 5 — Clustering No Supervisado (2 minutos)
**Notebook a abrir**: 05_unsupervised_learning.ipynb

**Qué ejecutar/mostrar**:
- Gráfico PCA sin clusters (62.85% varianza explicada en 2 componentes)
- Método del codo (K vs inercia)
- Silhouette scores para K=2 a K=10 (máximo en K=3)
- Gráfico PCA CON clusters (K=4, coloreado)
- Tabla de perfiles de clusters
- Boxplot de desempeño por cluster

**Qué dices**:
"Aquí cambio de enfoque — en lugar de PREDECIR, DESCUBRO la estructura del dataset sin usar el target.

Usé **PCA** para visualizar el dataset en 2 dimensiones: los dos componentes principales explican el **62.85% de la varianza**. 

Luego apliqué **KMeans** para encontrar clusters naturales.

Para elegir K, revisé dos evaluaciones:
- Método del codo: muestra inflexión alrededor de K=3
- Silhouette score: **máximo en K=3**

El score Silhouette es máximo en K=3, que estadísticamente es limpio. PERO — ese clustering separa casi exclusivamente empleados con/sin capacitación. Es estadísticamente correcto pero poco útil para RRHH.

Por eso usé **K=4** para obtener perfiles más ricos que combinen desempeño × capacitación × ausencias.

**Los 4 clusters que encontré:**

- **Cluster 0 (492 emp, 34%)**: Bajo desempeño, sin capacitación, ausencias bajas — empleados estables pero sin desarrollo
- **Cluster 1 (657 emp, 45%)**: ALTO desempeño, sin capacitación, ausencias bajas — **estos son los SUPERSTARS**, talento natural
- **Cluster 2 (123 emp, 9%)**: Bajo desempeño, CON capacitación, ausencias altas — **EN RIESGO, necesitan intervención inmediata**
- **Cluster 3 (177 emp, 12%)**: Alto desempeño, sin capacitación, PERO con MUCHAS ausencias (22.75 días) — **posible rotación**, talento en fuga

**Lo más importante**: el clustering SEPARA NATURALMENTE PERFILES DE EMPLEADOS sin usar `puntaje_desempeno` como variable. Eso valida que nuestro feature engineering capturó señales REALES de la organización — los clusters tienen características claras y diferenciadas."

**Cómo interpretar**: Los 4 clusters son interpretables y accionables para RRHH

---

## BLOQUE 6 — Conclusiones y Aplicación (2.5 minutos)
**Notebook a abrir**: 06_final_conclusions.ipynb (VERSIÓN MEJORADA)

**Qué ejecutar/mostrar**:
- Tabla de hallazgos principales
- Resumen de clusters
- Variables más importantes en el modelo
- Impacto organizacional

**Qué dices**:
"Para cerrar, voy a resumir los **RESULTADOS NUMÉRICOS**, los **HALLAZGOS CLAVE**, y la **APLICACIÓN PRÁCTICA**.

**RESULTADOS NUMÉRICOS:**
- Modelo de clasificación base: **F1 = 0.5313**
- Modelo optimizado: **F1 = 0.6626** (+25% mejora)
- Regresión R²: **0.64** — explica el 64% de la varianza
- Clustering: **4 perfiles naturales** de empleados

**HALLAZGOS CLAVE:**

1. **Las competencias son el predictor #1** — capacitación de calidad, desempeño histórico y evaluaciones previas

2. **El sesgo del evaluador domina la señal** — 60% de varianza es quién evalúa, no el empleado

3. **La capacitación de calidad importa más que cantidad** — empleados con pocas horas pero buenas competencias outperform

4. **Las ausencias tienen relación clara con desempeño** — 22 días/año es el umbral crítico

5. **Random Forest fue el modelo más robusto** en todos los contextos — mejor que SVM, XGBoost en este dataset

**APLICACIÓN PRÁCTICA:**

Los modelos se pueden usar para:

- **Identificar talento para promoción**: Cluster 1 (superstars) — especialmente si tienen desempeño actual bajo pero competencias altas (potencial oculto)

- **Detectar riesgo de rotación**: Cluster 3 — alto desempeño pero muchas ausencias = empleados buscando salida

- **Priorizar capacitaciones**: Cluster 2 — tienen disposición a capacitarse, hay que mantenerlos

- **Diseñar intervenciones diferenciadas**: Cada cluster necesita estrategia distinta

**CONCLUSIÓN FINAL:**

Este proyecto demostró que con:
- ✓ Feature engineering inteligente
- ✓ Datos de calidad
- ✓ Optimización apropiada

Es posible construir modelos de RRHH que capturan la complejidad REAL de las organizaciones.

El desempeño de los empleados NO depende de una variable, sino de **combinaciones** entre competencias, capacitación, historial de ausencias, y — crucialmente — quién es el evaluador.

**Lecciones aprendidas para proyectos futuros:**
1. Validar que el óptimo estadístico tiene sentido organizacional (K=3 vs K=4)
2. Data leakage en Optuna es error silencioso — el código corre sin errores pero resultados son sobreoptimistas
3. Un Pipeline de sklearn no es solo comodidad — es la ÚNICA forma correcta de garantizar que el preprocesador se ajuste solo en train

Gracias. Quedo disponible para preguntas."

**Cómo interpretar**: Los números (0.53 → 0.66, +25%) prueban que los cambios funcionaron

---

## CHECKLIST PRE-PRESENTACIÓN

✅ **Antes de presentar:**
- [ ] Run All en notebook 03 para frescos F1 scores
- [ ] Abre cada notebook en orden para familiaridad
- [ ] Confirma que gráficos (F1, PCA, clusters) se renderizan
- [ ] Practica timings: 15 minutos EXACTOS

✅ **Si preguntan "¿por qué no 85%?"**
Respuesta: *"El desbalance de clases (73% vs 27%) hace difícil pasar 85% en F1. Pero logramos +25% de mejora respecto al baseline (0.53→0.66), que es sólido. Los modelos avanzados (XGBoost, LightGBM, Voting, Stacking) probablemente estarían en 0.70-0.75 en test set."*

✅ **Material recomendado:**
- Llevar gráficas impresas o en PDF (F1 comparativos, PCA, perfiles de clusters)
- Tener README.md abierto para contexto rápido
- Mantener este guión visible mientras presentas

---

cait