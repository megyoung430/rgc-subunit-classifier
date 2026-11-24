import pytest

from src.outputs import (
    ARCHIVE_MARKER,
    check_new_output,
    check_output_dir,
    write_new_file,
)


def test_new_output_inside_archive_is_refused(tmp_path):
    archive = tmp_path / "results" / "colab_legacy"
    (archive / "sta").mkdir(parents=True)
    with pytest.raises(PermissionError):
        check_new_output(archive / "sta" / "new.h5", archive=archive)
    with pytest.raises(PermissionError):
        check_output_dir(archive / "sweeps", archive=archive)


def test_existing_file_is_never_overwritten(tmp_path):
    existing = tmp_path / "subunits.h5"
    existing.write_bytes(b"saved")
    with pytest.raises(FileExistsError):
        check_new_output(existing, archive=tmp_path / "archive")
    assert existing.read_bytes() == b"saved"


def test_paths_outside_archive_are_allowed(tmp_path):
    archive = tmp_path / "results" / "colab_legacy"
    target = tmp_path / "results" / "sta_cache" / "a.h5"
    assert check_new_output(target, archive=archive) == target
    assert check_output_dir(tmp_path / "results" / "sweeps", archive=archive)


def test_archive_lookalike_sibling_is_not_mistaken_for_archive(tmp_path):
    archive = tmp_path / "results" / "colab_legacy"
    sibling = tmp_path / "results" / "colab_legacy_v2" / "a.h5"
    assert check_new_output(sibling, archive=archive) == sibling


def test_archive_is_recognised_by_its_marker_wherever_the_code_lives(tmp_path):
    other_repo_archive = tmp_path / "clone" / "results" / "colab_legacy"
    other_repo_archive.mkdir(parents=True)
    (other_repo_archive / ARCHIVE_MARKER).write_text("")
    elsewhere = tmp_path / "unrelated_archive_location"
    with pytest.raises(PermissionError):
        check_new_output(other_repo_archive / "sta" / "new.h5", archive=elsewhere)
    with pytest.raises(PermissionError):
        check_output_dir(other_repo_archive, archive=elsewhere)


def test_write_new_file_is_all_or_nothing(tmp_path):
    target = tmp_path / "cache" / "a.npy"

    def fail(temporary):
        temporary.write_bytes(b"half")
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError):
        write_new_file(target, fail, archive=tmp_path / "archive")
    assert not target.parent.exists() or list(target.parent.iterdir()) == []

    write_new_file(target, lambda temporary: temporary.write_bytes(b"done"),
                   archive=tmp_path / "archive")
    assert target.read_bytes() == b"done"
    assert list(target.parent.iterdir()) == [target]
    with pytest.raises(FileExistsError):
        write_new_file(target, lambda temporary: temporary.write_bytes(b"new"),
                       archive=tmp_path / "archive")
    assert target.read_bytes() == b"done"


def test_write_new_file_temporary_keeps_the_suffix(tmp_path):
    seen = []
    write_new_file(tmp_path / "a.npz", lambda temporary: seen.append(temporary.suffix)
                   or temporary.write_bytes(b"x"), archive=tmp_path / "archive")
    assert seen == [".npz"]
