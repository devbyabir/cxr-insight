"""Record what a run depended on: code revision, package versions, seed, data hashes."""
import importlib.metadata
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PACKAGES = ("torch", "torchvision", "numpy", "pillow", "transformers", "unsloth", "peft",
            "trl", "streamlit")


def git_revision(cwd=None):
    def run(*args):
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                              check=True).stdout.strip()
    try:
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def package_versions(names=PACKAGES):
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def run_metadata(seed, extra=None, cwd=None):
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git": git_revision(cwd or Path(__file__).resolve().parent),
        "packages": package_versions(),
        "seed": seed,
        **(extra or {}),
    }
