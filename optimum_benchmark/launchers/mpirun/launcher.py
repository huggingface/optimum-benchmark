import os
import traceback
from contextlib import ExitStack
from logging import Logger
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Any, Callable, List

import psutil

from ...benchmark.report import BenchmarkReport
from ...logging_utils import setup_logging
from ...process_utils import sync_with_child, sync_with_parent
from ..base import Launcher
from .config import MPIrunConfig


class MPIrunLauncher(Launcher[MPIrunConfig]):
    NAME = "mpirun"

    def __init__(self, config: MPIrunConfig):
        super().__init__(config)

    def launch(self, worker: Callable[..., BenchmarkReport], worker_args: List[Any]) -> BenchmarkReport:
        parent_connection, child_connection = Pipe()
        main_process_pid = os.getpid()
        isolated_process = Process(
            target=target,
            args=(worker, worker_args, child_connection, main_process_pid, self.config, self.logger),
            daemon=False,
        )

        with ExitStack() as stack:
            if self.config.numactl:
                stack.enter_context(self.numactl_executable())

            isolated_process.start()

            if isolated_process.is_alive():
                sync_with_child(parent_connection)
            else:
                raise RuntimeError("Could not synchronize with isolated process")

            if self.config.device_isolation:
                stack.enter_context(self.device_isolation(isolated_process.pid))

            if isolated_process.is_alive():
                sync_with_child(parent_connection)
            else:
                raise RuntimeError("Could not synchronize with isolated process")

            isolated_process.join()

        if isolated_process.exitcode != 0:
            raise RuntimeError(f"Isolated process exited with non-zero code {isolated_process.exitcode}")

        if parent_connection.poll():
            response = parent_connection.recv()
        else:
            raise RuntimeError("Isolated process did not send any response")

        reports = []

        for output in response:
            if "traceback" in output:
                if "rank" in output:
                    self.logger.error(f"\t+ Received traceback from rank process [{output['rank']}]")
                    raise ChildProcessError(output["traceback"])
                else:
                    self.logger.error("\t+ Received traceback from isolated process")
                    raise ChildProcessError(output["traceback"])

            elif "report" in output:
                self.logger.info(f"\t+ Received report from rank process [{output['rank']}]")
                reports.append(BenchmarkReport.from_dict(output["report"]))

            else:
                raise RuntimeError(f"Received an unexpected response from isolated process: {output}")

        self.logger.info("\t+ Aggregating reports from all rank processes")
        report = BenchmarkReport.aggregate(reports)
        report.log()

        return report


def target(
    worker: Callable[..., BenchmarkReport],
    worker_args: List[Any],
    child_connection: Connection,
    main_process_pid: int,
    config: MPIrunConfig,
    logger: Logger,
):
    main_process = psutil.Process(main_process_pid)

    if main_process.is_running():
        sync_with_parent(child_connection)
    else:
        raise RuntimeError("Could not synchronize with main process")

    log_level = os.environ.get("LOG_LEVEL", "INFO")
    log_to_file = os.environ.get("LOG_TO_FILE", "1") == "1"
    setup_logging(level=log_level, to_file=log_to_file, prefix="ISOLATED-PROCESS")

    if main_process.is_running():
        sync_with_parent(child_connection)
    else:
        raise RuntimeError("Could not synchronize with main process")

    try:
        from mpi4py.futures import MPIPoolExecutor, wait

        with MPIPoolExecutor(max_workers=config.num_processes) as executor:
            fs = [executor.submit(entrypoint, i, worker, worker_args, logger) for i in range(config.num_processes)]
            wait(fs)
    except Exception:
        logger.error("\t+ Sending traceback to main process")
        child_connection.send([{"traceback": traceback.format_exc()}])
    else:
        logger.info("\t+ Sending outputs to main process")
        child_connection.send([f.result() for f in fs])
    finally:
        logger.info("\t+ Exiting isolated process")
        child_connection.close()
        exit(0)


def entrypoint(rank: int, worker: Callable[..., BenchmarkReport], worker_args: List[Any], logger: Logger):
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    log_to_file = os.environ.get("LOG_TO_FILE", "1") == "1"
    log_all_ranks = os.environ.get("LOG_ALL_RANKS", "0") == "1"

    if log_all_ranks or rank == 0:
        setup_logging(level=log_level, to_file=log_to_file, prefix=f"RANK-PROCESS-{rank}")
    else:
        setup_logging(level="ERROR", to_file=log_to_file, prefix=f"RANK-PROCESS-{rank}")

    try:
        report = worker(*worker_args)
    except Exception:
        logger.error("\t+ Benchmark failed with an exception")
        output = {"rank": rank, "traceback": traceback.format_exc()}
    else:
        logger.info("\t+ Benchmark completed successfully")
        output = {"rank": rank, "report": report.to_dict()}
    finally:
        logger.info("\t+ Exiting rank process")
        return output
