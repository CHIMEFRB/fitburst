#!/usr/bin/env python
"""Generic tests for frb-datatrails."""
import fitburst


def test_project_import():
    """Simple check to test for package importability."""
    assert isinstance(fitburst.__file__, str)


def test_analysis_function():
    """Check if the seed function works."""
    flavor = "str"
    uuid = fitburst.analysis.seed.get_uuid(flavor=flavor)
    assert isinstance(uuid, str)
