import logging
import logging.config
import os
import threading
import time
from pathlib import Path

import psutil
from dask.callbacks import Callback


class ResourceMonitor:
    def __init__(self, interval: float = 5.0, level: int = logging.INFO):
        """Context manager that logs process/system resource usage in the background.

        Parameters
        ----------
        interval :
            Sleep interval between log outputs (seconds).
        level :
            Logging level for performance stats.
        """
        self.interval = interval
        self.level = level

        self._stop_event = threading.Event()
        self._monitor_thread = None
        self._process = psutil.Process(os.getpid())
        self.logger = logging.getLogger(self.__class__.__name__)
        self._should_run = self.logger.isEnabledFor(self.level)

    def __enter__(self):
        # If the level is below what's configured, do nothing
        if not self._should_run:
            return self

        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        if not self._should_run:
            return False

        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join()
        return False

    def _monitor_loop(self):
        try:
            last_disk = psutil.disk_io_counters()
        except (OSError, RuntimeError, NotImplementedError):
            last_disk = None

        last_time = time.time()

        time.sleep(0.1)

        while not self._stop_event.wait(timeout=self.interval):
            try:
                current_time = time.time()
                dt = current_time - last_time

                cpu_p = self._process.cpu_percent()

                mem_info = self._process.memory_info()
                rss_mb = mem_info.rss / (1024 * 1024)

                current_disk = psutil.disk_io_counters()

                read_speed_mb = 0.0
                write_speed_mb = 0.0

                if current_disk and last_disk and dt > 0:
                    read_diff = current_disk.read_bytes - last_disk.read_bytes
                    write_diff = current_disk.write_bytes - last_disk.write_bytes

                    read_speed_mb = (read_diff / (1024 * 1024)) / dt
                    write_speed_mb = (write_diff / (1024 * 1024)) / dt

                self.logger.log(
                    self.level,
                    f"CPU: {cpu_p:.1f}% | "
                    f"RSS: {rss_mb:.2f} MB | "
                    f"Disk Read: {read_speed_mb:.2f} MB/s | "
                    f"Disk Write: {write_speed_mb:.2f} MB/s",
                )

                last_disk = current_disk
                last_time = current_time

            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                self.logger.warning(
                    f"Monitoring thread hit a metric collection error: {e}"
                )
                break


class LogProgress(Callback):
    """A Dask callback that logs progress at specified intervals instead of animating a bar."""

    def __init__(self, log_interval: int = 5, level: int = logging.INFO):
        super().__init__()
        self.log_interval = log_interval  # Log every X percent, say 10%
        self.level = level
        self.last_logged = None
        self.logger = logging.getLogger("dask_progress")

    def _start(self, dsk):
        self.logger.log(self.level, f"Beginning dask calculation with {len(dsk)} tasks")

    def _pretask(self, key, dsk, state):
        # Calculate current percentage
        ndone = len(state["finished"])
        ntasks = len(state["dependencies"])

        if ntasks > 0:
            percent = round((ndone / ntasks) * 100)
            if percent % self.log_interval == 0 and percent != self.last_logged:
                self.logger.log(
                    self.level,
                    f"Progress: {percent}% completed ({ndone}/{ntasks} tasks)",
                )
                self.last_logged = percent

    def _finish(self, dsk, state, errored):
        if errored:
            self.logger.error("Dask computation failed.")
        else:
            self.logger.log(self.level, "Dask calculation completed.")


def configure_logging(level: str, log_path: Path | None) -> None:
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(threadName)s | %(name)s: %(message)s"
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": level,
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "": {  # root logger
                "handlers": ["console"],
                "level": level,
                "propagate": True,
            },
            "dask": {"level": level, "propagate": True},
            "hdf5plugin": {"level": level, "propagate": True},
        },
    }

    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logging_config["handlers"]["file"] = {  # ty: ignore[invalid-assignment]
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "standard",
            "level": level,
            "filename": str(log_path),
            "maxBytes": 10485760,
            "backupCount": 5,
            "encoding": "utf8",
        }
        logging_config["loggers"][""]["handlers"] = ["file"]

    logging.config.dictConfig(logging_config)
