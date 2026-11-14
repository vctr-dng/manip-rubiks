"""Third-party robot assets.

The Sharpa Wave meshes are too large to vendor into the repository, so the
model is fetched from MuJoCo Menagerie at a pinned commit into a local cache
(``~/.cache/manip_rubiks`` or ``$MANIP_RUBIKS_CACHE``).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
MENAGERIE_COMMIT = "370c43965323c0944b433b202aac5058b5f907a9"


def cache_dir() -> Path:
    root = os.environ.get("MANIP_RUBIKS_CACHE")
    return Path(root) if root else Path.home() / ".cache" / "manip_rubiks"


def menagerie_dir() -> Path:
    return cache_dir() / f"mujoco_menagerie-{MENAGERIE_COMMIT[:12]}"


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def ensure_menagerie_model(name: str) -> Path:
    """Return the directory of a Menagerie model, fetching it on first use."""
    root = menagerie_dir()
    target = root / name
    if target.is_dir():
        return target
    root.mkdir(parents=True, exist_ok=True)
    if not (root / ".git").exists():
        _git("init", "-q", cwd=root)
        _git("remote", "add", "origin", MENAGERIE_URL, cwd=root)
        _git("config", "core.sparseCheckout", "true", cwd=root)
    sparse = root / ".git" / "info" / "sparse-checkout"
    entries = set(sparse.read_text().split()) if sparse.exists() else set()
    entries.add(f"/{name}/")
    sparse.write_text("\n".join(sorted(entries)) + "\n")
    _git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", MENAGERIE_COMMIT, cwd=root)
    _git("checkout", "-q", MENAGERIE_COMMIT, cwd=root)
    if not target.is_dir():
        raise FileNotFoundError(f"{name} not found in Menagerie at {MENAGERIE_COMMIT}")
    return target


def sharpa_wave_xml(side: str = "right") -> Path:
    if side not in ("left", "right"):
        raise ValueError(side)
    return ensure_menagerie_model("sharpa_wave") / f"{side}_hand.xml"
