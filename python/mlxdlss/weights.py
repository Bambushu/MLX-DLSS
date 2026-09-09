"""Resolve a weights file from an explicit path, a local directory, or the Hugging Face hub.

Lets the CLIs and nodes run without a hand-typed ``--weights`` path: a bare name (or nothing) is
resolved against ``MLXDLSS_WEIGHTS`` / the repo ``weights/`` dir, and only downloaded from the hub as
a last resort. Auto-download needs the weights to be published at ``MLXDLSS_HF_REPO``; until they are,
the error explains exactly what to do. An explicit path with directory components must exist as given
(no silent basename fallback), so a typo fails loudly instead of loading the wrong file or hitting the hub.
"""
from __future__ import annotations

import os
from pathlib import Path

# The default weights when the caller does not name one. A fine-tune, not the stock renderer:
# on soft/upscaled video the stock weights are inert (see docs/super-resolution.md).
DEFAULT_NAME = "dlssnr-ft-real-v2.safetensors"


def _search_dirs() -> list[Path]:
    dirs: list[Path] = []
    env = os.environ.get("MLXDLSS_WEIGHTS")
    if env:
        dirs.append(Path(env).expanduser())
    dirs.append(Path("~/mlx-dlss/weights").expanduser())
    dirs.append(Path(__file__).resolve().parents[2] / "weights")  # repo weights/ next to python/
    dirs.append(Path.cwd() / "weights")
    seen, out = set(), []
    for d in dirs:
        rd = d.resolve()
        if rd not in seen:
            seen.add(rd)
            out.append(d)
    return out


def resolve_weights(explicit: str | os.PathLike | None = None, name: str = DEFAULT_NAME) -> Path:
    """Return a path to the weights, searching local dirs then (optionally) the hub.

    ``explicit`` wins if given. An explicit path that names a directory (``foo/bar.safetensors`` or an
    absolute path) must exist exactly; only a bare filename falls through to the search dirs and hub.
    Raises ``FileNotFoundError`` with actionable guidance when nothing is found.
    """
    if explicit is not None:
        p = Path(str(explicit)).expanduser()
        if p.is_file():
            return p.resolve()
        if p.is_dir():
            raise FileNotFoundError(f"expected a weights file, not a directory: {p}")
        if p.is_absolute() or p.parent != Path("."):
            raise FileNotFoundError(f"weights path does not exist: {p}")
        name = p.name  # bare filename -> resolve against the search dirs / hub below

    for d in _search_dirs():
        hit = d / name
        if hit.is_file():
            return hit.resolve()

    hf_repo = os.environ.get("MLXDLSS_HF_REPO", "")  # read at call time, not import time
    if hf_repo:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise FileNotFoundError(
                f"weights '{name}' not found locally and huggingface_hub is not installed "
                f"(pip install huggingface_hub) to fetch from {hf_repo}"
            ) from exc
        try:
            revision = os.environ.get("MLXDLSS_HF_REVISION") or None
            return Path(hf_hub_download(repo_id=hf_repo, filename=name, revision=revision))
        except Exception as exc:  # EntryNotFound, HTTP, offline, auth — all become the actionable error
            raise FileNotFoundError(f"could not fetch '{name}' from {hf_repo}: {exc}") from exc

    searched = ", ".join(str(d) for d in _search_dirs())
    raise FileNotFoundError(
        f"weights '{name}' not found. Put the file in one of: {searched} "
        f"(or set MLXDLSS_WEIGHTS to its directory), or set MLXDLSS_HF_REPO to a hub repo that hosts it."
    )
