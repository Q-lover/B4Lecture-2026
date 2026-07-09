import importlib

import nf_assignment


def test_package_version_is_available() -> None:
    assert nf_assignment.__version__


def test_scaffold_subpackages_import() -> None:
    modules = [
        "nf_assignment.flows",
        "nf_assignment.networks",
        "nf_assignment.toy",
        "nf_assignment.speech",
        "nf_assignment.speech.features",
        "nf_assignment.training",
        "nf_assignment.utils",
    ]
    for name in modules:
        importlib.import_module(name)
