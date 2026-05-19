"""
Project tests for the Kedro pipeline registry and discovered workflows.
"""
from pathlib import Path

import pytest
from kedro.framework.startup import bootstrap_project

from src.prueba2.pipeline_registry import register_pipelines


class TestKedroPipelines:
    def test_default_pipeline_is_registered(self):
        bootstrap_project(Path.cwd())
        pipelines = register_pipelines()

        assert 'data_science' in pipelines
        assert '__default__' in pipelines
        assert pipelines['__default__'] is pipelines['data_science']
        assert len(pipelines['__default__'].nodes) > 0

    def test_unsupervised_pipeline_is_available(self):
        bootstrap_project(Path.cwd())
        pipelines = register_pipelines()

        assert 'unsupervised' in pipelines
        assert len(pipelines['unsupervised'].nodes) >= 2
