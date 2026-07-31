"""
Run provenance: every script that writes to disk records how it was invoked.

Usage
-----
    with RunRecorder("build_scores", config=cfg.to_dict()) as rec:
        ...
        rec.add_output(score_json)

On exit this writes results/runs/<UTC>_<script>/manifest.json containing the
argv, the resolved config, the git SHA plus a dirty flag, the interpreter and
torch/CUDA versions, the seed, wall-clock timings, and the sha256 + size of
every declared output. stdout is teed to stdout.log in the same directory.

processed/ and results/ are gitignored, so these manifests are the only durable
link between a number in the paper and the commit that produced it.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_DIR = REPO_ROOT / "results" / "runs"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=15,
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip()
    except Exception:
        return None


def git_state() -> Dict[str, Any]:
    sha = _git("rev-parse", "HEAD")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "sha": sha,
        "branch": branch,
        "dirty": bool(status),
        # Truncated: the working tree here routinely has dozens of changes and
        # the full list would swamp the manifest.
        "dirty_files": (status.splitlines()[:40] if status else []),
    }


def env_state() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "interpreter": sys.executable,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": os.getcwd(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover
        info["torch"] = f"unavailable: {exc}"
    return info


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    """Seed every RNG that could affect a reported number."""
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# stdout tee
# ---------------------------------------------------------------------------


class _Tee:
    """Mirror writes to the real stream and to a log file."""

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, data):
        self._stream.write(data)
        try:
            self._fh.write(data)
        except Exception:
            pass
        return len(data)

    def flush(self):
        self._stream.flush()
        try:
            self._fh.flush()
        except Exception:
            pass

    def isatty(self):
        return getattr(self._stream, "isatty", lambda: False)()

    def __getattr__(self, item):
        return getattr(self._stream, item)


# ---------------------------------------------------------------------------
# recorder
# ---------------------------------------------------------------------------


@dataclass
class _Step:
    name: str
    seconds: float


class RunRecorder:
    def __init__(
        self,
        script: str,
        config: Optional[Dict[str, Any]] = None,
        notes: str = "",
        runs_dir: Optional[Path] = None,
        tee: bool = True,
    ):
        self.script = script
        self.config = config or {}
        self.notes = notes
        self.started_at = datetime.now(timezone.utc)
        stamp = self.started_at.strftime("%Y%m%dT%H%M%SZ")

        self.dir = Path(runs_dir or DEFAULT_RUNS_DIR) / f"{stamp}_{script}"
        self.dir.mkdir(parents=True, exist_ok=True)

        self.outputs: List[Path] = []
        self.steps: List[_Step] = []
        self.extra: Dict[str, Any] = {}
        self.status = "running"

        self._t0 = time.time()
        self._tee = tee
        self._log_fh = None
        self._saved_stdout = None
        self._saved_stderr = None

    # -- context management ------------------------------------------------

    def __enter__(self) -> "RunRecorder":
        if self._tee:
            self._log_fh = open(self.dir / "stdout.log", "w", encoding="utf-8")
            self._saved_stdout, self._saved_stderr = sys.stdout, sys.stderr
            sys.stdout = _Tee(self._saved_stdout, self._log_fh)
            sys.stderr = _Tee(self._saved_stderr, self._log_fh)

        print(f"[run] {self.script}  ->  {self.dir}")
        git = git_state()
        print(f"[run] git {git['sha'][:8] if git['sha'] else '?'} "
              f"({git['branch']}){' DIRTY' if git['dirty'] else ''}")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.status = "ok" if exc_type is None else f"failed: {exc_type.__name__}"
        if exc_type is not None:
            self.extra["error"] = f"{exc_type.__name__}: {exc}"

        try:
            self.write_manifest()
            print(f"[run] status={self.status}  elapsed={self.elapsed:.1f}s")
            print(f"[run] manifest: {self.dir / 'manifest.json'}")
        finally:
            if self._saved_stdout is not None:
                sys.stdout, sys.stderr = self._saved_stdout, self._saved_stderr
            if self._log_fh is not None:
                self._log_fh.close()
        return False  # never swallow the exception

    # -- recording ---------------------------------------------------------

    @property
    def elapsed(self) -> float:
        return time.time() - self._t0

    def step(self, name: str):
        """Time a named phase:  with rec.step("D"): ..."""
        return _StepTimer(self, name)

    def add_output(self, path) -> None:
        p = Path(path)
        if p not in self.outputs:
            self.outputs.append(p)

    def add(self, **kwargs) -> None:
        """Attach arbitrary summary facts to the manifest."""
        self.extra.update(kwargs)

    def write_manifest(self) -> Path:
        outputs = []
        for p in self.outputs:
            entry: Dict[str, Any] = {"path": str(p)}
            if p.exists():
                entry["bytes"] = p.stat().st_size
                entry["sha256"] = sha256_file(p)
            else:
                entry["missing"] = True
            outputs.append(entry)

        manifest = {
            "script": self.script,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "elapsed_seconds": round(self.elapsed, 2),
            "argv": sys.argv,
            "config": self.config,
            "git": git_state(),
            "env": env_state(),
            "steps": [{"name": s.name, "seconds": round(s.seconds, 2)} for s in self.steps],
            "outputs": outputs,
            "notes": self.notes,
            **({"extra": self.extra} if self.extra else {}),
        }

        path = self.dir / "manifest.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        return path


class _StepTimer:
    def __init__(self, rec: RunRecorder, name: str):
        self.rec = rec
        self.name = name

    def __enter__(self):
        self._t = time.time()
        print(f"[step] {self.name} ...")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        dt = time.time() - self._t
        self.rec.steps.append(_Step(self.name, dt))
        flag = "ok" if exc_type is None else "FAILED"
        print(f"[step] {self.name} {flag} in {dt:.1f}s")
        return False
