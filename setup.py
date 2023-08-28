from setuptools import find_packages, setup

setup(
    name="optimum-benchmark",
    version="0.0.1",
    packages=find_packages(),
    extras_require={
        "test": ["pytest"],
    },
    entry_points={
        "console_scripts": [
            "optimum-benchmark=optimum_benchmark.experiment:run_experiment",
            "optimum-report=optimum_benchmark.report:generate_report",
        ]
    },
)
