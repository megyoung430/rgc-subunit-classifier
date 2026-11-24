import pytest

from src.outputs import check_new_output, check_output_dir


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
