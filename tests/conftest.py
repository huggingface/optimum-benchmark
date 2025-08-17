"""Pytest configuration for optimum-benchmark tests."""

import pytest


def pytest_sessionfinish(session, exitstatus):
    """
    Hook to modify the exit status when no tests are collected.

    This prevents pytest from returning a non-zero exit code when no tests
    are found, which is useful for CI/CD pipelines where some test runs
    might legitimately have no tests to run.
    """
    if exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED:
        session.exitstatus = pytest.ExitCode.OK
