"""resolve_weights: explicit path wins, dir components must exist, local dirs searched, missing raises."""
import pytest

from mlxdlss.weights import resolve_weights


@pytest.fixture(autouse=True)
def _no_hub(monkeypatch):
    monkeypatch.delenv("MLXDLSS_HF_REPO", raising=False)


def test_explicit_absolute_path(tmp_path):
    f = tmp_path / "w.safetensors"
    f.write_bytes(b"x")
    assert resolve_weights(f) == f


def test_name_resolved_under_env_dir(tmp_path, monkeypatch):
    (tmp_path / "dlssnr-ft-real-v2.safetensors").write_bytes(b"x")
    monkeypatch.setenv("MLXDLSS_WEIGHTS", str(tmp_path))
    assert resolve_weights() == tmp_path / "dlssnr-ft-real-v2.safetensors"


def test_missing_raises_with_guidance(tmp_path, monkeypatch):
    monkeypatch.setenv("MLXDLSS_WEIGHTS", str(tmp_path))
    with pytest.raises(FileNotFoundError) as exc:
        resolve_weights(name="nope.safetensors")
    assert "MLXDLSS_WEIGHTS" in str(exc.value)


def test_explicit_relative_path_that_exists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sub").mkdir()
    f = tmp_path / "sub" / "w.safetensors"
    f.write_bytes(b"x")
    assert resolve_weights("sub/w.safetensors") == f.resolve()


def test_explicit_path_with_dirs_missing_raises_not_basename(tmp_path, monkeypatch):
    # a decoy with the same basename in the search dir must NOT be picked
    monkeypatch.setenv("MLXDLSS_WEIGHTS", str(tmp_path))
    (tmp_path / "w.safetensors").write_bytes(b"x")
    with pytest.raises(FileNotFoundError) as exc:
        resolve_weights("does/not/exist/w.safetensors")
    assert "does not exist" in str(exc.value)


def test_directory_is_not_accepted(tmp_path, monkeypatch):
    monkeypatch.setenv("MLXDLSS_WEIGHTS", str(tmp_path))
    (tmp_path / "zzz_unique_test.safetensors").mkdir()  # a dir with the weights name (unique so real dirs can't supply it)
    with pytest.raises(FileNotFoundError):
        resolve_weights(name="zzz_unique_test.safetensors")
