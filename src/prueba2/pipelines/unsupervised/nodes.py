"""Nodes for the unsupervised analysis pipeline."""

import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from ..model_training import crear_preprocesador, obtener_variables_modelo


def prepare_unsupervised_data(df_limpio):
    """Prepare cleaned data for unsupervised learning."""
    X = df_limpio[obtener_variables_modelo(df_limpio)]
    preprocessor = crear_preprocesador(X)
    X_transformed = preprocessor.fit_transform(X)
    return X_transformed


def run_unsupervised_analysis(X_unsupervised):
    """Run PCA and KMeans analysis for unsupervised insights."""
    resumen = {}
    n_components = min(5, X_unsupervised.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    resumen['pca_explained_variance_ratio'] = pca.fit(X_unsupervised).explained_variance_ratio_.tolist()
    resumen['pca_cumulative_variance'] = np.cumsum(resumen['pca_explained_variance_ratio']).tolist()

    opciones_k = [2, 3, 4, 5, 6]
    resultados_k = []
    for k in opciones_k:
        modelo_k = KMeans(n_clusters=k, random_state=42, n_init=10)
        etiquetas = modelo_k.fit_predict(X_unsupervised)
        puntaje = float(silhouette_score(X_unsupervised, etiquetas))
        resultados_k.append({
            'n_clusters': k,
            'silhouette_score': puntaje,
            'inertia': float(modelo_k.inertia_),
        })

    mejor = max(resultados_k, key=lambda item: item['silhouette_score'])
    resumen['kmeans_options'] = resultados_k
    resumen['best_k'] = int(mejor['n_clusters'])
    resumen['best_silhouette'] = mejor['silhouette_score']
    resumen['best_inertia'] = mejor['inertia']

    ruta_salida = Path('data') / '08_reporting' / 'unsupervised_summary.json'
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    with ruta_salida.open('w', encoding='utf-8') as archivo:
        json.dump(resumen, archivo, indent=2, ensure_ascii=False)

    print(f"Resumen unsupervised guardado en: {ruta_salida}")
    return resumen
