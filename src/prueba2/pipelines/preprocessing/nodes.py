"""Nodes for the preprocessing pipeline."""

import os

from ..data_preprocessing import limpiar_datos_basico


def run_preprocessing():
    project_root = os.getcwd()
    ruta_raw = os.path.join(project_root, "data", "01_raw")
    ruta_output = os.path.join(project_root, "data", "05_model_input")
    return limpiar_datos_basico(ruta_raw, ruta_output)
