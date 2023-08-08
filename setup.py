from setuptools import setup, find_packages


setup(
    name="optimum-benchmark",
    version="0.0.1",
    packages=find_packages(),
    # add pytest as for optimum-benchmark[test]
    extras_require={
        "test": ["pytest"],
    },
    entry_points={
        "console_scripts": [
            "optimum-benchmark=optimum_benchmark.main:run_experiment",
            "optimum-report=optimum_benchmark.report:generate_report",
        ]
    },
)
