from __future__ import annotations
import json
import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional


def _supports_unicode(stream) -> bool:
    encoding = getattr(stream, "encoding", None) or ""
    try:
        "\u2713\u2717".encode(encoding)
        return True
    except (UnicodeEncodeError, LookupError, TypeError):
        return False


_UNICODE_OK = _supports_unicode(sys.stdout)
_MARK_OK = "\u2713" if _UNICODE_OK else "[PASS]"
_MARK_FAIL = "\u2717" if _UNICODE_OK else "[FAIL]"



def get_logger(name: str = "arxiv_pipeline") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


@dataclass
class SectorStatus:
    name: str
    pulled: int = 0
    retained: int = 0
    duration_seconds: float = 0.0
    ok: bool = True
    error: Optional[str] = None

    def as_status_line(self, name_width: int) -> str:
        mark = _MARK_OK if self.ok else _MARK_FAIL
        base = (
            f"{self.name:<{name_width}}{mark}  "
            f"pulled {self.pulled:>3}  retained {self.retained:>3}"
        )
        if self.ok:
            return base + "  0 errors"
        return base + f"  1 error: {self.error}"


@dataclass
class RunSummary:
    run_timestamp: str
    cycle_hours: int
    duration_seconds: float
    sectors: dict = field(default_factory=dict)   # name -> SectorStatus (as dict)
    totals: dict = field(default_factory=dict)     # {"pulled": int, "retained": int, "failed_sectors": int}


class RunTracker:
    def __init__(self, cycle_hours: int = 48):
        self.cycle_hours = cycle_hours
        self._run_start = time.monotonic()
        self._statuses: list[SectorStatus] = []
        self._logger = get_logger()

    @contextmanager
    def track_sector(self, sector_name: str) -> Iterator[SectorStatus]:
        status = SectorStatus(name=sector_name)
        start = time.monotonic()
        self._logger.debug(f"[{sector_name}] starting")
        try:
            yield status
        except Exception as exc: 
            status.ok = False
            status.error = f"{type(exc).__name__}: {exc}"
            self._logger.exception(f"[{sector_name}] failed: {status.error}")
        else:
            self._logger.debug(
                f"[{sector_name}] done — pulled {status.pulled}, retained {status.retained}"
            )
        finally:
            status.duration_seconds = round(time.monotonic() - start, 2)
            self._statuses.append(status)

    def print_status_table(self) -> None:
        print("\n--- Run status summary ---")
        width = max((len(s.name) for s in self._statuses), default=14) + 2
        for status in self._statuses:
            print(status.as_status_line(name_width=width))
        print("---------------------------\n")

    def finalize(self, run_log_path: str | Path = "data/run_log.json") -> RunSummary:
        duration = round(time.monotonic() - self._run_start, 2)
        pulled_total = sum(s.pulled for s in self._statuses)
        retained_total = sum(s.retained for s in self._statuses)
        failed = sum(1 for s in self._statuses if not s.ok)

        summary = RunSummary(
            run_timestamp=datetime.now(timezone.utc).isoformat(),
            cycle_hours=self.cycle_hours,
            duration_seconds=duration,
            sectors={s.name: asdict(s) for s in self._statuses},
            totals={
                "pulled": pulled_total,
                "retained": retained_total,
                "failed_sectors": failed,
            },
        )

        path = Path(run_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(summary), indent=2))

        self._logger.info(
            f"Run complete in {duration}s — "
            f"{retained_total} retained across {len(self._statuses)} sectors "
            f"({failed} failed)"
        )
        self.print_status_table()
        return summary