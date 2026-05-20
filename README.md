# Informe Técnico — Sistema de Predicción de Desempeño de Empleados

**Asignatura:** SCY1101 — Programación para la Ciencia de Datos  
**Evaluación:** Parcial N°2  
**Integrantes:** Arturo  
**Fecha:** Mayo 2025

---

## Tabla de contenidos

1. [Resumen ejecutivo](#1-resumen-ejecutivo)
2. [Descripción del dataset](#2-descripción-del-dataset)
3. [Marco metodológico](#3-marco-metodológico)
4. [Preparación de datos y feature engineering](#4-preparación-de-datos-y-feature-engineering)
5. [Modelado supervisado — Clasificación](#5-modelado-supervisado--clasificación)
6. [Modelado supervisado — Regresión](#6-modelado-supervisado--regresión)
7. [Optimización de hiperparámetros](#7-optimización-de-hiperparámetros)
8. [Aprendizaje no supervisado](#8-aprendizaje-no-supervisado)
9. [Conclusiones y recomendaciones](#9-conclusiones-y-recomendaciones)
10. [Referencias](#10-referencias)
11. [Guía de uso y reproducibilidad](#11-guía-de-uso-y-reproducibilidad)

---

## 1. Resumen ejecutivo

Este proyecto desarrolla un sistema completo de análisis y predicción de desempeño de empleados utilizando técnicas de Machine Learning sobre datos históricos de Recursos Humanos. Se integraron cuatro fuentes de datos (empleados, ausencias, capacitaciones y evaluaciones) para construir un dataset de **1.449 evaluaciones** correspondientes a **~430 empleados únicos** distribuidos en 4 períodos semestrales (2021-S1 a 2023-S2).

Se abordó el problema desde tres enfoques complementarios:

- **Clasificación binaria:** identificar si un empleado tendrá alto desempeño (`desempeno_alto`, umbral P75 = 5.60 puntos). Mejor resultado: **SVM con F1 = 0.790** en comparación inicial; tras optimización, **Random Forest con Optuna alcanzó F1-CV = 0.789**.
- **Regresión continua:** estimar el puntaje numérico de desempeño. Mejor resultado: **Regresión Lineal con R² = 0.643** en comparación inicial; tras optimización con Optuna, **Random Forest redujo RMSE-CV a 1.250**.
- **Clustering no supervisado:** segmentar empleados en perfiles homogéneos. Se identificaron **3–4 clusters** con PCA explicando **51.0%** de la varianza en 2 componentes.

El hallazgo más relevante del análisis es de naturaleza metodológica: **el sesgo del evaluador domina la señal del target**, con una desviación estándar entre evaluadores de 1.82 versus 0.84 intra-evaluador. Esto implica que modelar el comportamiento del evaluador (variable `evaluador_media`) es tan importante como las variables del propio empleado.

---

## 2. Descripción del dataset

### 2.1 Fuentes de datos

El dataset se construyó integrando cuatro tablas de origen:

| Tabla | Registros | Variables principales |
|---|---|---|
| `empleados.csv` | ~430 | cargo, departamento, tipo_contrato, jornada |
| `ausencias.csv` | Variable | tipo_ausencia, dias, fecha |
| `capacitaciones.csv` | Variable | curso, horas, nota_final, estado |
| `evaluaciones.csv` | 1.449 | puntaje_desempeno, competencias, periodo, evaluador |

### 2.2 Dataset final: `dataset_rrhh_limpio.csv`

Después del proceso de integración, limpieza y transformación:

| Dimensión | Valor |
|---|---|
| Registros | 1.449 |
| Empleados únicos | ~430 |
| Variables originales | 37 |
| Variables tras feature engineering | 36 predictoras + 1 target |
| Períodos | 2021-S1, 2021-S2, 2022-S1, 2022-S2 |
| Valores nulos tras limpieza | < 2% (imputados por mediana/moda) |

### 2.3 Variable objetivo

| Target | Tipo | Descripción |
|---|---|---|
| `puntaje_desempeno` | Regresión continua | Puntaje 0–10 asignado por el evaluador |
| `desempeno_alto` | Clasificación binaria | 1 si puntaje ≥ P75 (5.60), 0 si no |

**Distribución de clases:**
- Clase 0 (desempeño normal): 1.058 registros (73.0%)
- Clase 1 (alto desempeño): 391 registros (27.0%)
- Desbalance moderado → se aplicó `class_weight='balanced'` en todos los clasificadores

### 2.4 Variables excluidas (data leakage)

Se excluyeron del entrenamiento:
- `desempeno_bajo`: derivada directamente del target
- `riesgo_rotacion`: combinación del target con ausencias
- `id_empleado`: identificador sin valor predictivo

---

## 3. Marco metodológico

### 3.1 Justificación del enfoque

Se eligió un enfoque de modelado en tres niveles por las siguientes razones:

**¿Por qué clasificación Y regresión?**
La clasificación (`desempeno_alto`) permite tomar decisiones binarias operacionales: ¿a quién retener, promover o intervenir? La regresión (`puntaje_desempeno`) permite ordenar y priorizar: ¿cuánto mejor es un empleado que otro? Ambos son necesarios en RRHH real.

**¿Por qué Random Forest como modelo base?**
Random Forest es robusto ante variables con diferentes escalas, tolera valores nulos (con imputer), maneja variables categóricas (con OHE), y no requiere normalización estricta. Es el modelo de ensamble más utilizado como baseline en problemas tabulares de RRHH (Fernández-Delgado et al., 2014).

**¿Por qué incluir XGBoost y LightGBM?**
Son los algoritmos de gradient boosting más competitivos en benchmarks de datos tabulares (Chen & Guestrin, 2016; Ke et al., 2017). Se añadieron como modelos avanzados para comparación y para potencialmente superar el RF base.

**¿Por qué PCA + KMeans?**
PCA reduce la dimensionalidad antes del clustering, eliminando ruido y correlaciones entre variables. KMeans es interpretable y escalable; sus centroides permiten describir perfiles de empleados de forma accionable para RRHH.

### 3.2 Pipeline de procesamiento

Todos los modelos usan el mismo preprocesador encapsulado en un `sklearn.Pipeline`:

```
ColumnTransformer
├── Variables numéricas → SimpleImputer(median) → StandardScaler
└── Variables categóricas → SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown='ignore')
```

**Ventajas del Pipeline:**
1. Previene data leakage: el scaler se ajusta solo en train
2. Reproducibilidad: una sola llamada `.fit()` / `.predict()`
3. Permite serialización completa con `joblib`

### 3.3 Estrategia de validación

| Técnica | Configuración | Uso |
|---|---|---|
| Train/Test split | 80/20, `random_state=42`, `stratify=y` | Evaluación final de todos los modelos |
| StratifiedKFold | `n_splits=5, shuffle=True, random_state=42` | CV en optimización de clasificación |
| KFold | `n_splits=5, shuffle=True, random_state=42` | CV en optimización de regresión |

**Tamaños de conjuntos:**
- Train clasificación: 1.159 registros
- Test clasificación: 290 registros
- Train regresión: 1.159 registros
- Test regresión: 290 registros

---

## 4. Preparación de datos y feature engineering

### 4.1 Limpieza de datos

Las operaciones de limpieza aplicadas en `notebooks/00_data_cleaning_feature_engineering.ipynb`:

- **Valores nulos:** imputación por mediana en variables numéricas, moda en categóricas
- **Outliers:** clipping al percentil 99 en `total_dias_ausencia` y `puntaje_desempeno`
- **Transformaciones log:** aplicadas a variables con distribución asimétrica severa (`total_dias_ausencia`, `num_capacitaciones`, `total_horas_capacitacion`, `riesgo_operacional_rrhh`)
- **Normalización categórica:** `.strip().title()` en columnas de texto para eliminar inconsistencias de capitalización

### 4.2 Variables derivadas

Se crearon 10 variables adicionales con alto valor predictivo:

#### Variables temporales y de sesgo evaluador

| Variable | Fórmula | Justificación |
|---|---|---|
| `prev_puntaje` | lag-1 por empleado | Correlación 0.47 con target — mayor predictor individual |
| `evaluador_media` | promedio histórico por evaluador | Captura sesgo sistemático: std_entre_eval = 1.82 |
| `dept_media` | promedio por departamento | Captura diferencias estructurales entre áreas |

**Hallazgo crítico sobre sesgo evaluador:**

```
std_entre_evaluadores = 1.82  (varianza ENTRE evaluadores)
std_intra_evaluador   = 0.84  (varianza del mismo evaluador en distintos períodos)
```

Esto significa que **~60% de la varianza del target es atribuible al evaluador**, no al desempeño real del empleado. La variable `evaluador_media` captura este sesgo y es la más importante en los modelos finales.

#### Variables de interacción

| Variable | Fórmula | Significado |
|---|---|---|
| `comp_asistencia` | `competencia_combinada × (1 − ausencia_alta)` | Competencia ajustada por asistencia |
| `nota_cap_ponderada` | `promedio_nota × log(1 + n_cap)` | Nota ponderada por volumen de capacitación |
| `ratio_tec_blanda` | `comp_tec / comp_blanda` | Balance entre competencias técnicas y blandas |
| `cap_score` | `max_nota × (1 − sin_capacitacion)` | Score de capacitación (0 si sin capacitación) |

### 4.3 Impacto del feature engineering

La inclusión de `prev_puntaje` + `evaluador_media` mejoró el F1 en validación cruzada de **0.46 → 0.65** respecto al conjunto base de variables. Esto valida que el historial del empleado y el criterio del evaluador son las señales más fuertes del dataset.

---

## 5. Modelado supervisado — Clasificación

### 5.1 Modelos evaluados

Se compararon **7 clasificadores** con configuración `class_weight='balanced'` para compensar el desbalance de clases, usando el mismo Pipeline de preprocesamiento:

| Modelo | Justificación de uso |
|---|---|
| Logistic Regression | Baseline lineal interpretable |
| Decision Tree | Modelo simple, reglas de decisión visibles |
| Random Forest | Ensamble robusto, principal candidato |
| KNN | Basado en similitud, no paramétrico |
| SVM (RBF) | Buen desempeño en espacios de alta dimensión |
| Gaussian NB | Baseline probabilístico |
| Gradient Boosting | Boosting secuencial, buen rendimiento tabular |

### 5.2 Resultados comparativos

Evaluación sobre test set (290 registros, split 80/20):

| Modelo | Accuracy | Balanced Acc. | Precision | Recall | **F1-score** |
|---|---|---|---|---|---|
| **SVM** | 0.766 | 0.763 | 0.727 | **0.865** | **0.790** |
| Gradient Boosting | 0.772 | 0.772 | 0.763 | 0.804 | 0.783 |
| Logistic Regression | 0.752 | 0.750 | 0.721 | 0.838 | 0.775 |
| Random Forest | 0.759 | 0.758 | 0.753 | 0.784 | 0.768 |
| Decision Tree | 0.745 | 0.744 | 0.728 | 0.797 | 0.761 |
| KNN | 0.738 | 0.737 | 0.731 | 0.770 | 0.750 |
| Gaussian NB | 0.721 | 0.720 | 0.719 | 0.743 | 0.731 |

**Mejor modelo inicial: SVM con F1 = 0.790**

### 5.3 Análisis de métricas

**¿Por qué F1 es la métrica principal?**

En el contexto de RRHH, los errores tienen costos asimétricos:
- **Falso negativo** (predecir bajo cuando es alto): se pierde talento clave — costo alto
- **Falso positivo** (predecir alto cuando es bajo): se invierte en un empleado que no lo merece — costo moderado

El F1-score balancea Precision y Recall, siendo más adecuado que Accuracy cuando el costo de los errores es diferente y las clases están desbalanceadas.

**Interpretación del SVM:**
- Recall = 0.865 → detecta el 86.5% de los empleados de alto desempeño reales
- Precision = 0.727 → el 72.7% de los predichos como alto desempeño realmente lo son
- Balanced Accuracy = 0.763 → rendimiento equilibrado entre ambas clases

### 5.4 Curva ROC y umbral óptimo

Se graficó la curva ROC para el mejor clasificador y se calculó el umbral óptimo mediante la estadística de Youden (maximiza Sensitivity + Specificity − 1). El umbral por defecto de 0.5 fue ajustado según el balance deseado entre precisión y cobertura en el contexto organizacional.

---

## 6. Modelado supervisado — Regresión

### 6.1 Modelos evaluados

Se compararon **5 regresores** con el mismo pipeline de preprocesamiento:

| Modelo | Justificación |
|---|---|
| Linear Regression | Baseline lineal, máxima interpretabilidad |
| Decision Tree Regressor | Captura no linealidades simples |
| Random Forest Regressor | Ensamble robusto, reduce varianza |
| KNN Regressor | No paramétrico, basado en similitud |
| SVR (RBF) | Robusto ante outliers con margen de tolerancia |

### 6.2 Resultados comparativos

Evaluación sobre test set (290 registros):

| Modelo | MAE | RMSE | **R²** |
|---|---|---|---|
| **Linear Regression** | 0.951 | 1.185 | **0.643** |
| Random Forest | 0.973 | 1.253 | 0.601 |
| SVR | 1.038 | 1.322 | 0.556 |
| Decision Tree | 1.072 | 1.379 | 0.516 |
| KNN | 1.166 | 1.486 | 0.438 |

**Mejor modelo inicial: Regresión Lineal con R² = 0.643**

### 6.3 Análisis de métricas

**Interpretación del RMSE:**
Un RMSE = 1.185 en una escala de puntaje 0–10 significa que el modelo se equivoca en promedio ±1.2 puntos. En contexto RRHH esto equivale a la diferencia entre un puntaje de 5.0 y 6.2 — margen aceptable para priorización, no para decisiones individuales absolutas.

**Interpretación del R²:**
R² = 0.643 indica que el modelo explica el 64.3% de la varianza del puntaje de desempeño. El 35.7% restante corresponde principalmente al ruido del evaluador (varianza no capturable sin datos perfectos de evaluación).

**¿Por qué Regresión Lineal supera a Random Forest?**
Con variables fuertemente correlacionadas con el target (`evaluador_media`, `prev_puntaje`), la relación es aproximadamente lineal. Random Forest captura mejor las interacciones no lineales, pero con 1.449 registros la ganancia es marginal y la varianza del ensamble suma error. Tras optimización, Random Forest recupera terreno.

---

## 7. Optimización de hiperparámetros

### 7.1 Modelo seleccionado para optimización

Se seleccionó **Random Forest** (clasificación y regresión) para la etapa de optimización por ser el modelo de ensamble más utilizado en benchmarks tabulares y por su capacidad de mejora sustancial mediante ajuste de hiperparámetros.

**Hiperparámetros optimizados:**

| Hiperparámetro | Rango | Impacto |
|---|---|---|
| `n_estimators` | 100–400 | Más árboles → menor varianza, mayor costo |
| `max_depth` | 4–15 | Profundidad máxima → controla sobreajuste |
| `min_samples_split` | 2–10 | Mínimo de muestras para dividir un nodo |
| `min_samples_leaf` | 1–5 | Mínimo de muestras en hoja → regularización |
| `class_weight` | balanced / None | Solo clasificación — manejo de desbalance |

### 7.2 Estrategias comparadas

#### GridSearchCV

Prueba **todas** las combinaciones de una grilla definida manualmente. Configuración utilizada:

```python
param_grid = {
    "n_estimators":      [100, 200, 300],
    "max_depth":         [6, 8, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf":  [1, 2],
    "class_weight":      ["balanced", None]   # solo clf
}
# Total combinaciones: 3×3×2×2×2 = 72 (clf) / 3×3×2×2 = 36 (reg)
# CV=5 → 72×5 = 360 entrenamientos
```

**Ventaja:** exhaustivo, reproducible, sin aleatoriedad.  
**Desventaja:** costo computacional crece exponencialmente con el espacio de búsqueda.

#### RandomizedSearchCV

Muestrea `n_iter=30` combinaciones **aleatorias** desde distribuciones estadísticas:

```python
param_distributions = {
    "n_estimators":      randint(100, 400),       # uniforme discreta
    "max_depth":         randint(4, 15),
    "min_samples_split": randint(2, 10),
    "min_samples_leaf":  randint(1, 5),
    "class_weight":      ["balanced", None]
}
# 30 combinaciones × CV=5 = 150 entrenamientos
# Cubre espacio más amplio que GridSearchCV con 30% del costo
```

**Ventaja:** eficiente con espacios grandes, cubre rangos continuos.  
**Desventaja:** resultados no deterministas (mitigado con `random_state=42`).

#### Optuna (Bayesian Optimization)

Usa el algoritmo TPE (*Tree-structured Parzen Estimator*) para seleccionar inteligentemente los próximos hiperparámetros basándose en los resultados anteriores:

```python
# 30 trials × CV=5 = 150 entrenamientos (mismo costo que RandomizedSearch)
# Sin data leakage: CV solo sobre X_train, test set reservado para evaluación final
```

**Ventaja:** converge más rápido que búsqueda aleatoria al usar el historial de trials.  
**Desventaja:** mayor complejidad de implementación, menos transparente.

### 7.3 Resultados — Clasificación

Mejor F1 en validación cruzada (CV=5, sobre train únicamente):

| Enfoque | Mejor F1-CV | Mejores parámetros |
|---|---|---|
| GridSearchCV | 0.7860 | max_depth=10, n_est=100, leaf=2, split=5 |
| RandomizedSearchCV | 0.7853 | max_depth=12, n_est=150, leaf=3, split=2 |
| **Optuna** | **0.7895** | max_depth=13, n_est=240, leaf=2, split=2 |

**Mejor estrategia de clasificación: Optuna (F1-CV = 0.7895)**

### 7.4 Resultados — Regresión

Mejor RMSE en validación cruzada (CV=5, sobre train únicamente):

| Enfoque | Mejor RMSE-CV | Mejores parámetros |
|---|---|---|
| GridSearchCV | 1.2934 | max_depth=10, n_est=200, leaf=1, split=2 |
| RandomizedSearchCV | 1.2918 | max_depth=12, n_est=300, leaf=1, split=3 |
| **Optuna** | **1.2502** | max_depth=12, n_est=132, leaf=2, split=5 |

**Mejor estrategia de regresión: Optuna (RMSE-CV = 1.2502)**

### 7.5 Análisis del impacto de la optimización

**Clasificación:**
- GridSearchCV vs RandomizedSearchCV: diferencia marginal (0.7860 vs 0.7853 en CV)
- Optuna supera a ambos en F1-CV (+0.0035 sobre GridSearch)
- La diferencia entre estrategias es pequeña porque el espacio de hiperparámetros de Random Forest es relativamente pequeño — el modelo es intrínsecamente robusto

**Regresión:**
- Optuna logra una mejora más sustancial: RMSE-CV = 1.2502 vs 1.2934 de GridSearch (−3.3%)
- RandomizedSearchCV y GridSearchCV son casi equivalentes en regresión

**¿Vale la pena la optimización?**
Para clasificación, la mejora es marginal (< 1% en F1-CV). Para regresión, Optuna ofrece una mejora real de ~3% en RMSE. En ambos casos, el impacto más grande vino del feature engineering (de F1=0.46 a F1=0.65) más que de la optimización de hiperparámetros.

### 7.6 Prevención de data leakage en Optuna

Un error común en implementaciones de Optuna es evaluar el objetivo sobre el test set dentro de cada trial:

```python
# ❌ INCORRECTO — data leakage
def objetivo(trial):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)   # test set contamina la búsqueda
    return f1_score(y_test, y_pred)

# ✅ CORRECTO — sin leakage
def objetivo(trial):
    return cross_val_score(
        pipeline, X_train, y_train,     # solo train, CV interno
        cv=cv_opt, scoring="f1"
    ).mean()
```

Esta corrección se aplicó en `notebooks/04_hyperparameter_optimization.ipynb`.

---

## 8. Aprendizaje no supervisado

### 8.1 Objetivo del análisis no supervisado

Mientras los modelos supervisados predicen el desempeño, el análisis no supervisado busca **describir** la población de empleados sin usar el target. Los clusters resultantes permiten diseñar intervenciones diferenciadas de RRHH para cada perfil.

### 8.2 Reducción de dimensionalidad con PCA

Se aplicó PCA sobre 13 variables de RRHH (ausencias, capacitaciones, competencias):

| Componente | Varianza explicada | Varianza acumulada | Eje principal |
|---|---|---|---|
| PC1 | 27.93% | 27.93% | Capacitación (nota, horas, frecuencia) |
| PC2 | 23.05% | 51.0% | Ausencias y competencias |
| PC3 | 11.89% | 62.9% | — |
| PC4 | 7.04% | 70.0% | — |
| PC5 | 4.34% | 74.3% | — |

**Interpretación:**
- **PC1 (27.9%):** representa el eje de capacitación. Empleados con alto PC1 tienen más horas de formación, mejores notas y mayor cantidad de cursos.
- **PC2 (23.0%):** representa el eje de ausencias y competencias. Alto PC2 corresponde a empleados con pocas ausencias y alta competencia combinada.
- Con 2 componentes se explica el **51.0% de la varianza total** — suficiente para visualización, pero el 49% restante no es capturable en el plano 2D.

### 8.3 Segmentación con KMeans

**Selección del K óptimo:**

| K | Silhouette Score | Inercia |
|---|---|---|
| 2 | 0.2206 | 31.854 |
| **3** | **0.2306** | 25.394 |
| 4 | 0.1964 | 23.363 |
| 5 | 0.1871 | 21.587 |
| 6 | 0.1811 | 20.238 |

El Silhouette Score máximo ocurre en **K=3**. Sin embargo, se aplicó un mínimo de K=4 por razones organizacionales:

> K=2 y K=3 producen separaciones dominadas por la variable binaria `sin_capacitacion`, creando clusters triviales (con/sin formación). K=4 obliga al algoritmo a segmentar dentro de cada grupo de capacitación según ausencias y desempeño, generando perfiles más accionables para intervenciones de RRHH.

**Perfiles de clusters identificados (K=4):**

| Cluster | Perfil | Características principales |
|---|---|---|
| A | Sin capacitación · riesgo alto | `sin_capacitacion=1`, ausencias altas, riesgo operacional alto |
| B | Sin capacitación · estable | `sin_capacitacion=1`, ausencias bajas, menor riesgo inmediato |
| C | Con capacitación · alto desempeño | Notas altas, competencias altas, pocas ausencias |
| D | Con capacitación · en desarrollo | Formados pero puntaje medio — candidatos a intervención focalizada |

**Validación externa del clustering:**
Los clusters fueron validados cruzando con `puntaje_desempeno` (variable no usada en el clustering). El Cluster C presenta el puntaje promedio más alto y el Cluster A el más bajo, confirmando que la segmentación captura señal real de desempeño.

---

## 9. Conclusiones y recomendaciones

### 9.1 Conclusiones del modelado

#### Clasificación

El mejor modelo final es **Random Forest optimizado con Optuna** (F1-CV = 0.789). En test set, los clasificadores alcanzaron hasta F1 = 0.790 (SVM sin optimizar). La diferencia entre modelos es pequeña, lo que sugiere que el techo del modelo está limitado por el ruido del evaluador, no por la elección del algoritmo.

**Resultados clave:**
- F1 ≥ 0.75 en todos los modelos evaluados → señal predictiva sólida
- Recall alto (SVM: 0.865) → el modelo detecta la mayoría del talento real
- Balanced Accuracy ≥ 0.72 → funciona bien en ambas clases a pesar del desbalance

#### Regresión

El mejor modelo tras optimización es **Random Forest con Optuna** (RMSE-CV = 1.250). La Regresión Lineal tiene el mejor R² en test (0.643), pero Random Forest Optuna ofrece mejor generalización en CV.

**Resultados clave:**
- R² ≈ 0.64 → el modelo explica el 64% de la varianza del puntaje
- RMSE ≈ 1.19 → error promedio de ±1.2 puntos en escala 0–10
- MAE ≈ 0.95 → la mitad de las predicciones tienen error < 1 punto

#### Clustering

La segmentación en K=4 grupos reveló perfiles organizacionalmente significativos y validados externamente con el target de desempeño. El análisis PCA confirma que capacitación y ausencias son los dos ejes principales de variación en la población.

### 9.2 Hallazgos principales

| # | Hallazgo | Implicancia |
|---|---|---|
| 1 | **Sesgo del evaluador domina la señal** (std_entre_eval=1.82 >> std_intra=0.84) | El 60% de la varianza del target es atribuible al evaluador, no al empleado |
| 2 | **`prev_puntaje` es el mejor predictor individual** (correlación 0.47) | El desempeño pasado predice el futuro mejor que cualquier variable de capacitación |
| 3 | **Feature engineering supera a la optimización** | Agregar `evaluador_media` + `prev_puntaje` mejoró F1-CV de 0.46 → 0.65 |
| 4 | **K=2 en clustering es trivial** | `sin_capacitacion` binaria domina KMeans; K=4 mínimo para perfiles útiles |
| 5 | **Optuna supera a GridSearch y RandomizedSearch** | Optimización bayesiana converge a mejores soluciones con el mismo número de evaluaciones |

### 9.3 Limitaciones del estudio

1. **Calidad del target:** el puntaje es subjetivo y varía por evaluador. Cualquier modelo entrenado sobre este target hereda ese ruido estructural.
2. **Tamaño del dataset:** 1.449 registros es suficiente para resultados válidos pero insuficiente para modelos profundos o validaciones muy robustas.
3. **Ventana temporal:** 4 semestres no capturan tendencias de largo plazo ni efectos de carrera de los empleados.
4. **Variables no disponibles:** factores como clima laboral, satisfacción, remuneración y carga de trabajo no están en el dataset y podrían tener alta correlación con el desempeño.

### 9.4 Recomendaciones para RRHH

1. **Estandarizar el proceso de evaluación:** la mayor mejora al modelo vendría de reducir el sesgo entre evaluadores mediante rúbricas calibradas, no de agregar más datos del mismo proceso actual.
2. **Priorizar capacitación en Cluster A:** los empleados sin capacitación y con alto riesgo operacional (Cluster A) son los candidatos con mayor potencial de mejora mediante intervención formativa.
3. **Focalizar retención en Cluster C:** los empleados con alta nota de capacitación y alto desempeño son los más valiosos y probablemente los más en riesgo de ser atraídos por la competencia.
4. **Usar evaluaciones 360°:** incorporar retroalimentación de pares y subordinados reduciría el sesgo de evaluador y mejoraría sustancialmente la calidad del target.
5. **Re-entrenar el modelo periódicamente:** con nuevos períodos de evaluación, el modelo debería actualizarse para capturar tendencias organizacionales recientes.

---

## 10. Referencias

1. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
2. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD '16. ACM.
3. Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS 2017.
4. Akiba, T., et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework*. KDD '19.
5. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825–2830.
6. Fernández-Delgado, M., et al. (2014). *Do we Need Hundreds of Classifiers to Solve Real World Classification Problems?* JMLR, 15, 3133–3181.
7. Bergstra, J., & Bengio, Y. (2012). *Random Search for Hyper-Parameter Optimization*. JMLR, 13, 281–305.
8. James, G., et al. (2021). *An Introduction to Statistical Learning* (2a ed.). Springer.

---

## 11. Guía de uso y reproducibilidad

### 11.1 Estructura del proyecto

```
prueba2/
├── notebooks/
│   ├── 00_data_cleaning_feature_engineering.ipynb
│   ├── 01_business_context_and_eda.ipynb
│   ├── 02_target_selection_and_problem_definition.ipynb
│   ├── 03_supervised_modeling.ipynb
│   ├── 04_hyperparameter_optimization.ipynb  ← GridSearchCV + RandomizedSearchCV + Optuna
│   ├── 05_unsupervised_learning.ipynb
│   ├── 06_final_conclusions.ipynb
│   └── 07_project_summary_for_presentation_PRO.ipynb
├── src/
│   ├── data_preprocessing.py    ← carga, FE, preprocesador
│   ├── model_training.py        ← definición y entrenamiento
│   ├── model_evaluation.py      ← métricas y comparación
│   └── hyperparameter_tuning.py ← GridSearch, RandomizedSearch, Optuna
├── data/
│   ├── 01_raw/                  ← CSV originales
│   └── 05_model_input/          ← dataset_rrhh_limpio.csv
├── models/trained_models/       ← modelos serializados .pkl
├── results/
│   ├── plots/                   ← gráficos generados
│   └── metrics/                 ← métricas en JSON
└── requirements.txt
```

### 11.2 Instalación

```bash
# Clonar repositorio
git clone https://github.com/MiguelTr7/Prueba2.0.git
cd Prueba2.0

# Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt
```

### 11.3 Dependencias principales

| Librería | Versión | Uso |
|---|---|---|
| scikit-learn | ≥ 1.4 | Modelos, pipelines, métricas |
| pandas | ≥ 2.0 | Manipulación de datos |
| numpy | ≥ 1.26 | Operaciones numéricas |
| matplotlib / seaborn | ≥ 3.8 | Visualizaciones |
| xgboost | ≥ 2.0 | XGBoost Classifier/Regressor |
| lightgbm | ≥ 4.0 | LightGBM Classifier/Regressor |
| optuna | ≥ 3.0 | Optimización bayesiana |
| scipy | ≥ 1.11 | Distribuciones para RandomizedSearchCV |
| joblib | ≥ 1.3 | Serialización de modelos |

### 11.4 Orden de ejecución

Ejecutar los notebooks en orden para reproducir todos los resultados:

```
1. 00_data_cleaning_feature_engineering.ipynb   (~2 min)
2. 01_business_context_and_eda.ipynb            (~3 min)
3. 02_target_selection_and_problem_definition   (~2 min)
4. 03_supervised_modeling.ipynb                 (~5 min)
5. 04_hyperparameter_optimization.ipynb         (~15 min) ← genera modelos .pkl
6. 05_unsupervised_learning.ipynb               (~3 min)
7. 06_final_conclusions.ipynb                   (~2 min)
8. 07_project_summary_for_presentation_PRO      (~5 min)
```

### 11.5 Reproducibilidad

Todas las semillas están fijadas:
- `random_state=42` en todos los modelos sklearn
- `random_state=42` en todos los splits train/test
- `random_state=42` en `RandomizedSearchCV`
- `optuna.create_study(direction=...)` sin semilla global — los trials son estocásticos pero los resultados finales son estables

Los modelos finales entrenados están disponibles en `models/trained_models/` para uso sin reentrenamiento:

```python
import joblib
modelo_clf = joblib.load('models/trained_models/modelo_clasificacion_final.pkl')
modelo_reg = joblib.load('models/trained_models/modelo_regresion_final.pkl')
```
