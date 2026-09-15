from pathlib import Path

from src.utils.config import load_config, resolve_path


def test_configuration_loading_and_path_resolution(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 42\npaths:\n  value: relative/path\n")
    config = load_config(config_path)
    assert config["seed"] == 42
    assert resolve_path(config, config["paths"]["value"]).is_absolute()

