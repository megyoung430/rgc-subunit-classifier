"""Guards for writing outputs: never into the saved Colab archive, never over a file."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Outputs of the original Colab analysis, kept read-only as a record.
ARCHIVE = REPO_ROOT / "results" / "colab_legacy"


def _inside(path: Path, folder: Path) -> bool:
    path, folder = Path(path).resolve(), Path(folder).resolve()
    return path == folder or folder in path.parents


def check_output_dir(path, archive=None) -> Path:
    """Return ``path`` if outputs may be written there (it may already exist).

    Raises:
        PermissionError: If ``path`` is inside the read-only Colab archive.
    """
    archive = ARCHIVE if archive is None else archive
    path = Path(path)
    if _inside(path, archive):
        raise PermissionError(
            f"{path} is inside the read-only Colab archive {archive}; "
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
