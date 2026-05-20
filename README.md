# Sistema de Análisis y Predicción de Desempeño de Empleados

**Asignatura:** SCY1101 — Programación para la Ciencia de Datos  
**Evaluación:** Parcial N°2  
**Integrantes:** Arturo  

---

## Descripción del proyecto

Proyecto de Machine Learning aplicado a Recursos Humanos. Se desarrolló un sistema completo de análisis y predicción de desempeño de empleados utilizando datos históricos de ausencias, capacitaciones, evaluaciones y competencias.

**Objetivos:**
- Predecir empleados de alto desempeño (clasificación binaria).
- Estimar el puntaje numérico de desempeño (regresión).
- Segmentar empleados en perfiles para intervenciones diferenciadas (clustering).

---

## Estructura del proyecto

```
prueba2/
│
├── notebooks/
│   ├── 00_data_cleaning_feature_engineering.ipynb  ← Limpieza y FE
│   ├── 01_business_context_and_eda.ipynb           ← EDA y contexto de negocio
│   ├── 02_target_selection_and_problem_definition.ipynb
│   ├── 03_supervised_modeling.ipynb                ← Modelos supervisados
│   ├── 04_hyperparameter_optimization.ipynb        ← GridSearchCV + RandomizedSearchCV + Optuna
│   ├── 05_unsupervised_learning.ipynb              ← PCA + KMeans
│   ├── 06_final_conclusions.ipynb
│   └── 07_project_summary_for_presentation_PRO.ipynb
│
├── src/
│   ├── data_preprocessing.py    ← Limpieza, FE, preprocesador sklearn
│   ├── model_training.py        ← Definición y entrenamiento de modelos
│   ├── model_evaluation.py      ← Métricas y comparación de modelos
│   └── hyperparameter_tuning.py ← GridSearchCV, RandomizedSearchCV, Optuna
│
├── data/
│   ├── 01_raw/                  ← Datos originales (empleados, ausencias, etc.)
│   ├── 05_model_input/          ← dataset_rrhh_limpio.csv
│   └── 06_models/               ← Modelos serializados (.pkl)
│
├── models/
│   └── trained_models/          ← Copia de modelos finales para entrega
│       ├── modelo_clasificacion_final.pkl
│       └── modelo_regresion_final.pkl
│
├── results/
│   ├── plots/                   ← Gráficos generados por los notebooks
│   ├── metrics/                 ← Métricas exportadas en JSON
│   └── reports/                 ← Informes técnicos
│
└── requirements.txt             ← Dependencias del proyecto
```

---

## Instalación y dependencias

**Requisito previo:** Python 3.12

### Opción 1 — entorno virtual con pip

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

### Opción 2 — uv (más rápido)

```bash
uv sync
```

### Dependencias principales

| Librería | Versión | Uso |
|---|---|---|
| scikit-learn | ≥1.4 | Modelos, pipelines, métricas |
| pandas | ≥2.0 | Manipulación de datos |
| numpy | ≥1.26 | Operaciones numéricas |
| matplotlib / seaborn | ≥3.8 | Visualizaciones |
| xgboost | ≥2.0 | XGBoost Classifier/Regressor |
| lightgbm | ≥4.0 | LightGBM Classifier/Regressor |
| optuna | ≥3.0 | Optimización bayesiana de hiperparámetros |
| scipy | ≥1.11 | Distribuciones para RandomizedSearchCV |
| joblib | ≥1.3 | Serialización de modelos |
| jupyter / jupyterlab | — | Ejecución de notebooks |

---

## Orden de ejecución de notebooks

Ejecutar en orden para reproducir todos los resultados:

```
1. 00_data_cleaning_feature_engineering.ipynb
2. 01_business_context_and_eda.ipynb
3. 02_target_selection_and_problem_definition.ipynb
4. 03_supervised_modeling.ipynb
5. 04_hyperparameter_optimization.ipynb   ← genera modelos finales .pkl
6. 05_unsupervised_learning.ipynb
7. 06_final_conclusions.ipynb
8. 07_project_summary_for_presentation_PRO.ipynb
```

> **Nota:** El notebook 04 es el más costoso computacionalmente (~15 min).  
> Genera `models/trained_models/modelo_clasificacion_final.pkl` y `modelo_regresion_final.pkl`.

---

## Uso de los módulos src/

Los módulos pueden importarse directamente desde Python:

```python
import sys
sys.path.insert(0, 'src')

from data_preprocessing import cargar_dataset, crear_features_derivadas, crear_preprocesador
from model_training import obtener_modelos_clasificacion, entrenar_todos
from model_evaluation import comparar_clasificadores, graficar_comparacion_clf
from hyperparameter_tuning import tune_grid_search_clf, tune_random_search_clf, tune_optuna_clf

# Cargar y preparar datos
df = cargar_dataset(r'C:\Users\Arturo\prueba2')
df = crear_features_derivadas(df)

# Entrenar y comparar modelos
from sklearn.model_selection import train_test_split
X = df[obtener_variables_modelo(df)]
y = df['desempeno_alto']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=42, stratify=y)
modelos = obtener_modelos_clasificacion()
pipelines = entrenar_todos(modelos, X_train, y_train)
tabla = comparar_clasificadores(pipelines, X_test, y_test)
print(tabla)
```

---

## Resultados principales

### Clasificación — `desempeno_alto` (P75 = 5.60)

| Modelo | F1-score | Balanced Acc. | Recall |
|---|---|---|---|
| RF GridSearchCV | **0.713** | 0.816 | 0.782 |
| RF RandomizedSearchCV | ~0.710 | ~0.810 | ~0.77 |
| RF Base | 0.675 | 0.785 | 0.731 |

### Regresión — `puntaje_desempeno`

| Modelo | RMSE | R² |
|---|---|---|
| RF Base | **1.141** | **0.669** |
| RF GridSearchCV | 1.143 | 0.668 |

### Clustering

- **PCA:** 55.65% varianza en 2 componentes (PC1: capacitación, PC2: ausencias).
- **KMeans K=4:** 4 perfiles de empleados (sin_cap+riesgo, sin_cap+estable, con_cap+alto, con_cap+desarrollo).

---

## Hallazgos clave

1. **Sesgo del evaluador domina ~60% de la varianza** — `evaluador_media` es la variable más importante.
2. **Lag feature `prev_puntaje`** (correlación 0.47) — mayor predictor individual.
3. **K=2 trivial en clustering** — Silhouette óptimo era K=2 por `sin_capacitacion` binaria; se usó K≥4.
4. **Data leakage corregido en Optuna** — CV solo sobre train, test set reservado para evaluación final.

---

## Reproducibilidad

- Todas las semillas fijadas con `random_state=42`.
- Dataset original en `data/01_raw/` (no versionado en git).
- Dataset limpio reproducible ejecutando notebook 00.
- Modelos finales en `models/trained_models/`.
