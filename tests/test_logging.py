"""Tests for :mod:`nzcvm.logging`.

Covers the background :class:`ResourceMonitor`, the :class:`LogProgress`
dask callback and :func:`configure_logging`. The tests step the monitor loop
synchronously with stubbed clocks and disk counters so the reported rates
are deterministic.
"""

from __future__ import annotations

import itertools
import logging
import logging.handlers
from collections import namedtuple
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import dask
import psutil
import pytest

from nzcvm import logging as nzlogging
from nzcvm.logging import LogProgress, ResourceMonitor, configure_logging

MIB = 1024 * 1024
DiskCounters = namedtuple("DiskCounters", ["read_bytes", "write_bytes"])


@pytest.fixture
def restore_logging() -> Iterator[None]:
    """Snapshot and restore the logger state touched by ``configure_logging``."""
    names = ["", "dask", "hdf5plugin"]
    loggers = [logging.getLogger(name) for name in names]
    saved = [(lg.level, list(lg.handlers), lg.propagate) for lg in loggers]
    yield
    for lg, (level, handlers, propagate) in zip(loggers, saved, strict=True):
        for handler in lg.handlers:
            if handler not in handlers:
                handler.close()
        lg.handlers[:] = handlers
        lg.setLevel(level)
        lg.propagate = propagate


# ---------------------------------------------------------------------------
# ResourceMonitor
# ---------------------------------------------------------------------------


def _run_loop_once(monitor: ResourceMonitor) -> None:
    """Run ``_monitor_loop`` in this thread, stopping after the first record."""

    class StopAfterFirst(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            monitor._stop_event.set()

    handler = StopAfterFirst()
    monitor.logger.addHandler(handler)
    try:
        monitor._monitor_loop()
    finally:
        monitor.logger.removeHandler(handler)


def test_resource_monitor_reports_disk_rates(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Disk read/write speeds are byte deltas in MiB divided by elapsed time."""
    disk = iter([DiskCounters(0, 0), DiskCounters(6 * MIB, 3 * MIB)])
    clock = iter([10.0, 12.0])
    monkeypatch.setattr(nzlogging.psutil, "disk_io_counters", lambda: next(disk))
    monkeypatch.setattr(
        nzlogging,
        "time",
        SimpleNamespace(time=lambda: next(clock), sleep=lambda _: None),
    )

    monitor = ResourceMonitor(interval=0.0, level=logging.WARNING)
    monitor._process = SimpleNamespace(  # ty: ignore[invalid-assignment]
        cpu_percent=lambda: 12.345,
        memory_info=lambda: SimpleNamespace(rss=int(1.5 * MIB)),
    )
    with caplog.at_level(logging.WARNING, logger="ResourceMonitor"):
        _run_loop_once(monitor)

    records = [r for r in caplog.records if r.name == "ResourceMonitor"]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert records[0].getMessage() == (
        "CPU: 12.3% | RSS: 1.50 MB | Disk Read: 3.00 MB/s | Disk Write: 1.50 MB/s"
    )


@pytest.mark.parametrize(
    ("initial_disk", "times"),
    [
        (OSError("no disks"), [0.0, 1.0]),  # initial counters unavailable
        (DiskCounters(0, 0), [5.0, 5.0]),  # zero elapsed time
    ],
)
def test_resource_monitor_zero_rates_without_baseline(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    initial_disk: object,
    times: list[float],
) -> None:
    """Without a previous disk sample or with ``dt == 0`` the rates are zero."""
    calls = itertools.count()

    def disk_io_counters() -> DiskCounters:
        if next(calls) == 0:
            if isinstance(initial_disk, Exception):
                raise initial_disk
            assert isinstance(initial_disk, DiskCounters)
            return initial_disk
        return DiskCounters(8 * MIB, 8 * MIB)

    clock = iter(times)
    monkeypatch.setattr(nzlogging.psutil, "disk_io_counters", disk_io_counters)
    monkeypatch.setattr(
        nzlogging,
        "time",
        SimpleNamespace(time=lambda: next(clock), sleep=lambda _: None),
    )

    monitor = ResourceMonitor(interval=0.0)
    monitor._process = SimpleNamespace(  # ty: ignore[invalid-assignment]
        cpu_percent=lambda: 0.0,
        memory_info=lambda: SimpleNamespace(rss=0),
    )
    with caplog.at_level(logging.INFO, logger="ResourceMonitor"):
        _run_loop_once(monitor)

    (record,) = [r for r in caplog.records if r.name == "ResourceMonitor"]
    assert record.getMessage().endswith("Disk Read: 0.00 MB/s | Disk Write: 0.00 MB/s")


@pytest.mark.parametrize("error", [psutil.NoSuchProcess(1), psutil.AccessDenied(1)])
def test_resource_monitor_stops_on_metric_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error: Exception,
) -> None:
    """A psutil metric error logs a warning and ends the loop."""
    monkeypatch.setattr(
        nzlogging, "time", SimpleNamespace(time=lambda: 0.0, sleep=lambda _: None)
    )

    def fail() -> float:
        raise error

    monitor = ResourceMonitor(interval=0.0)
    monitor._process = SimpleNamespace(cpu_percent=fail)  # ty: ignore[invalid-assignment]
    with caplog.at_level(logging.INFO, logger="ResourceMonitor"):
        # Returns on its own: the stop event is never set.
        monitor._monitor_loop()

    records = [r for r in caplog.records if r.name == "ResourceMonitor"]
    assert [r.levelno for r in records] == [logging.WARNING]
    assert (
        records[0]
        .getMessage()
        .startswith("Monitoring thread hit a metric collection error:")
    )


def test_resource_monitor_context_runs_and_joins_thread(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """With the level enabled, the monitor logs from a thread joined on exit."""
    with caplog.at_level(logging.INFO, logger="ResourceMonitor"):
        monitor = ResourceMonitor(interval=0.01)
        with monitor as entered:
            assert entered is monitor
            thread = monitor._monitor_thread
            assert thread is not None
            assert thread.daemon
            assert thread.is_alive()
            thread.join(timeout=0.5)  # let it log a few samples
        assert not thread.is_alive()
        assert monitor._stop_event.is_set()

    messages = [r.getMessage() for r in caplog.records if r.name == "ResourceMonitor"]
    assert messages
    assert all(m.startswith("CPU: ") and "| RSS: " in m for m in messages)


def test_resource_monitor_disabled_level_starts_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If the logger ignores the level, entering the monitor is a no-op."""
    with caplog.at_level(logging.WARNING, logger="ResourceMonitor"):
        monitor = ResourceMonitor(interval=0.01, level=logging.INFO)
        with monitor as entered:
            assert entered is monitor
            assert monitor._monitor_thread is None
        assert not monitor._stop_event.is_set()
    assert not [r for r in caplog.records if r.name == "ResourceMonitor"]


def test_resource_monitor_does_not_suppress_exceptions(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="ResourceMonitor"):
        monitor = ResourceMonitor(interval=0.01)
        with pytest.raises(ValueError, match="boom"), monitor:
            raise ValueError("boom")
    assert monitor._monitor_thread is not None
    assert not monitor._monitor_thread.is_alive()


# ---------------------------------------------------------------------------
# LogProgress
# ---------------------------------------------------------------------------


def _progress_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "dask_progress"]


def test_log_progress_logs_each_interval_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Progress logs at multiples of the interval, skipping repeats."""
    progress = LogProgress(log_interval=25)
    dependencies = {i: set() for i in range(8)}
    with caplog.at_level(logging.INFO, logger="dask_progress"):
        for ndone in [0, 0, 1, 2, 2, 3, 5, 6, 8]:
            state = {"finished": set(range(ndone)), "dependencies": dependencies}
            progress._pretask(None, {}, state)

    assert _progress_messages(caplog) == [
        "Progress: 0% completed (0/8 tasks)",
        "Progress: 25% completed (2/8 tasks)",
        "Progress: 75% completed (6/8 tasks)",
        "Progress: 100% completed (8/8 tasks)",
    ]
    assert progress.last_logged == 100


def test_log_progress_rounds_percentage(caplog: pytest.LogCaptureFixture) -> None:
    """The check rounds the percentage to the nearest integer first."""
    progress = LogProgress(log_interval=33)
    with caplog.at_level(logging.INFO, logger="dask_progress"):
        progress._pretask(
            None, {}, {"finished": {0}, "dependencies": {i: set() for i in range(3)}}
        )
    assert _progress_messages(caplog) == ["Progress: 33% completed (1/3 tasks)"]


def test_log_progress_ignores_empty_graph(caplog: pytest.LogCaptureFixture) -> None:
    progress = LogProgress()
    with caplog.at_level(logging.INFO, logger="dask_progress"):
        progress._pretask(None, {}, {"finished": set(), "dependencies": {}})
    assert _progress_messages(caplog) == []
    assert progress.last_logged is None


def test_log_progress_during_dask_compute(caplog: pytest.LogCaptureFixture) -> None:
    """A real dask computation logs start, progress and completion at the level."""

    @dask.delayed
    def inc(x: int) -> int:
        return x + 1

    total = dask.delayed(sum)([inc(i) for i in range(3)])
    with (
        caplog.at_level(logging.DEBUG, logger="dask_progress"),
        LogProgress(log_interval=1, level=logging.DEBUG),
    ):
        assert total.compute(scheduler="sync") == 6

    records = [r for r in caplog.records if r.name == "dask_progress"]
    assert {r.levelno for r in records} == {logging.DEBUG}
    messages = [r.getMessage() for r in records]
    assert messages[0] == "Beginning dask calculation with 4 tasks"
    assert messages[-1] == "Dask calculation completed."
    assert all(m.startswith("Progress: ") for m in messages[1:-1])
    assert len(messages) > 2


def test_log_progress_reports_failure(caplog: pytest.LogCaptureFixture) -> None:
    @dask.delayed
    def fail() -> None:
        raise RuntimeError("task failed")

    with (
        caplog.at_level(logging.INFO, logger="dask_progress"),
        LogProgress(),
        pytest.raises(RuntimeError, match="task failed"),
    ):
        fail().compute(scheduler="sync")

    records = [r for r in caplog.records if r.name == "dask_progress"]
    assert records[-1].levelno == logging.ERROR
    assert records[-1].getMessage() == "Dask computation failed."
    assert "Dask calculation completed." not in [r.getMessage() for r in records]


# ---------------------------------------------------------------------------
# configure_logging
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("restore_logging")
def test_configure_logging_console(capsys: pytest.CaptureFixture[str]) -> None:
    """Without a path, the root logger writes formatted records to stdout."""
    configure_logging("WARNING", None)

    root = logging.getLogger()
    assert root.level == logging.WARNING
    assert logging.getLogger("dask").level == logging.WARNING
    assert logging.getLogger("hdf5plugin").level == logging.WARNING
    (handler,) = root.handlers
    assert type(handler) is logging.StreamHandler
    assert handler.level == logging.WARNING

    logging.getLogger("nzcvm.test").info("hidden")
    logging.getLogger("nzcvm.test").warning("shown")
    out = capsys.readouterr().out
    assert "hidden" not in out
    assert "[WARNING] MainThread | nzcvm.test: shown" in out


@pytest.mark.usefixtures("restore_logging")
def test_configure_logging_file(tmp_path: Path) -> None:
    """With a path, it creates parent directories and writes records to the file."""
    log_path = tmp_path / "nested" / "dir" / "run.log"
    configure_logging("DEBUG", log_path)

    root = logging.getLogger()
    assert root.level == logging.DEBUG
    (handler,) = root.handlers
    assert isinstance(handler, logging.handlers.RotatingFileHandler)
    assert handler.baseFilename == str(log_path)
    assert handler.maxBytes == 10 * MIB
    assert handler.backupCount == 5
    assert handler.level == logging.DEBUG

    logging.getLogger("nzcvm.test").debug("written to file")
    handler.flush()
    contents = log_path.read_text(encoding="utf8")
    assert "[DEBUG] MainThread | nzcvm.test: written to file" in contents


@pytest.mark.usefixtures("restore_logging")
def test_configure_logging_keeps_existing_loggers() -> None:
    existing = logging.getLogger("nzcvm.preexisting")
    existing.disabled = False
    configure_logging("INFO", None)
    assert not existing.disabled
