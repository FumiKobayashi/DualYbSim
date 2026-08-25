"""Pytest configuration.

Slow tests (marked with ``@pytest.mark.slow``) are skipped by default so CI runs
quickly. Pass ``--run-slow`` to execute them locally, e.g.::

    pytest tests/ --run-slow -v
"""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run tests marked as slow (heavy numerical workloads).",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="skipped by default; pass --run-slow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
