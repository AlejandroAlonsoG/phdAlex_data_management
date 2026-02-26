"""
Path Utilities for the data ordering tool.
==========================================

Centralises all path conversion logic so that every module stores
**relative** paths in registries and state files.  This makes the
entire project portable across drives and mount points.

Design
------
* ``original_path`` stays **absolute** — it records the historical location
  of the source file, which may be on a disconnected drive.
* ``current_path`` (and ``file_path`` in hashes.xlsx) is stored as a
  **relative** path with respect to the output/base directory.
* CLI arguments accept both absolute and relative paths; everything is
  resolved to absolute ``Path`` objects internally for file I/O, but
  converted back to relative strings before persisting.

Usage
-----
>>> from data_ordering.path_utils import to_relative, to_absolute, ensure_relative
>>> to_relative(Path("D:/output/Campaña_2009/img.jpg"), Path("D:/output"))
'Campaña_2009/img.jpg'
>>> to_absolute("Campaña_2009/img.jpg", Path("D:/output"))
Path('D:/output/Campaña_2009/img.jpg')
"""

from pathlib import Path, PurePosixPath
from typing import Optional, Union


def to_relative(path: Union[str, Path], base_dir: Union[str, Path]) -> str:
    """Convert *path* to a POSIX-style relative string w.r.t. *base_dir*.

    If *path* is already relative or cannot be made relative to *base_dir*
    (e.g. it lives on a different drive), the original string representation
    is returned unchanged.

    Always uses forward slashes (``/``) regardless of OS so that the stored
    value is cross-platform.

    Args:
        path: The path to convert.
        base_dir: The base directory for computing the relative path.

    Returns:
        A POSIX-style relative path string, or the original string if
        relativisation is not possible.
    """
    if not path or not base_dir:
        return str(path) if path else ''

    p = Path(path)
    b = Path(base_dir)

    # Already relative — just normalise separators
    if not p.is_absolute():
        return p.as_posix()

    try:
        rel = p.relative_to(b)
        return rel.as_posix()
    except ValueError:
        # On Windows, paths on different drives raise ValueError.
        # Return the absolute path as-is (best effort).
        return p.as_posix()


def to_absolute(
    path: Union[str, Path],
    base_dir: Union[str, Path],
) -> Path:
    """Resolve *path* to an absolute ``Path`` using *base_dir* as anchor.

    If *path* is already absolute it is returned as-is.

    Args:
        path: A relative or absolute path (string or Path).
        base_dir: The base directory used to resolve relative paths.

    Returns:
        An absolute ``Path`` object.
    """
    if not path:
        return Path(base_dir)

    p = Path(path)
    if p.is_absolute():
        return p
    return Path(base_dir) / p


def ensure_relative(
    path: Union[str, Path, None],
    base_dir: Union[str, Path],
) -> Optional[str]:
    """Like ``to_relative`` but returns ``None`` when *path* is falsy."""
    if not path:
        return None
    return to_relative(path, base_dir)


def ensure_absolute(
    path: Union[str, Path, None],
    base_dir: Union[str, Path],
) -> Optional[Path]:
    """Like ``to_absolute`` but returns ``None`` when *path* is falsy."""
    if not path:
        return None
    return to_absolute(path, base_dir)


def convert_semicolon_paths(
    paths_str: str,
    base_dir: Union[str, Path],
    *,
    to_rel: bool = True,
) -> str:
    """Convert a semicolon-separated list of paths (like ``original_path``).

    Each segment is individually converted to relative (if *to_rel*) or
    absolute, and then re-joined with ``'; '``.

    Args:
        paths_str: e.g. ``"D:/data/a.jpg; D:/data/b.jpg"``
        base_dir: The base directory.
        to_rel: If True convert to relative; if False convert to absolute.

    Returns:
        The converted semicolon-separated string.
    """
    if not paths_str:
        return paths_str

    parts = [p.strip() for p in paths_str.split(';') if p.strip()]
    if not parts:
        return paths_str

    converter = to_relative if to_rel else lambda p, b: str(to_absolute(p, b))
    converted = [converter(p, base_dir) for p in parts]
    return '; '.join(converted)
