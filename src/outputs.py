"""Guards for writing outputs.

Nothing may be written inside the saved Colab archive, and file writers create
new files only: they never replace an existing one.
"""

import errno
import os
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Outputs of the original Colab analysis, kept read-only as a record.
ARCHIVE = REPO_ROOT / "results" / "colab_legacy"
# Committed inside the archive, so any copy of the repo (another clone, a
# worktree, an installed package run from elsewhere) recognises its archive.
ARCHIVE_MARKER = ".read-only-archive"


def _inside_archive(path, archive) -> bool:
    resolved = Path(path).resolve()
    archive = Path(archive).resolve()
    for folder in (resolved, *resolved.parents):
        if folder == archive or (folder / ARCHIVE_MARKER).exists():
            return True
        # samefile also catches a differently cased path on macOS.
        if folder.exists() and archive.exists() and os.path.samefile(folder, archive):
            return True
    return False


def check_output_dir(path, archive=None) -> Path:
    """Return ``path`` if outputs may be written there (it may already exist).

    Raises:
        PermissionError: If ``path`` is inside the read-only Colab archive,
            recognised either as ``archive`` or by the marker file it holds.
    """
    archive = ARCHIVE if archive is None else archive
    path = Path(path)
    if _inside_archive(path, archive):
        raise PermissionError(
            f"{path} is inside the read-only Colab archive; "
            "write new outputs elsewhere under results/"
        )
    return path


def check_new_output(path, archive=None) -> Path:
    """Return ``path`` if a new file may be created there.

    Raises:
        PermissionError: If ``path`` is inside the read-only Colab archive.
        FileExistsError: If a file already exists at ``path``.
    """
    path = check_output_dir(path, archive)
    if path.exists():
        raise FileExistsError(f"{path} already exists; refusing to overwrite it")
    return path


def _publish(temporary: Path, path: Path) -> None:
    """Give ``temporary`` the name ``path`` without ever replacing a file there."""
    try:
        os.link(temporary, path)
    except FileExistsError:
        raise FileExistsError(f"{path} already exists; refusing to overwrite it") from None
    except OSError as error:
        # Some network and FAT file systems cannot hard-link.
        if error.errno not in (errno.EPERM, errno.ENOTSUP, errno.EXDEV, errno.EOPNOTSUPP):
            raise
        if path.exists():
            raise FileExistsError(f"{path} already exists; refusing to overwrite it") from None
        os.replace(temporary, path)


def write_new_file(path, write, archive=None) -> Path:
    """Create ``path`` by calling ``write(temporary_path)``, all or nothing.

    ``write`` fills a temporary file in the same folder (same suffix, so
    NumPy and h5py keep the name). Only a complete file is given the final
    name; if ``write`` fails, nothing is left behind.

    Raises:
        PermissionError: If ``path`` is inside the read-only Colab archive.
        FileExistsError: If ``path`` exists, before or after writing (for
            example, another process wrote the same cache file meanwhile).
    """
    path = check_new_output(path, archive)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.",
                                    suffix=path.suffix)
    os.close(handle)
    temporary = Path(name)
    try:
        write(temporary)
        _publish(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
