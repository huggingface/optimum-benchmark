from setuptools import setup, find_packages


setup(
    name="optimum-benchmark",
    version="0.0.1",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "optimum-benchmark=optimum_benchmark.main:run_experiment",
            "optimum-report=optimum_benchmark.report:generate_report",
        ]
    },
)
