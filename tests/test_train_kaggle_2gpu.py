from argparse import Namespace

import pytest

from train_kaggle_2gpu import accumulation_steps, apply_overrides


def test_accumulation_uses_global_ddp_batch_size():
    assert accumulation_steps(32, 4, 2) == 4


def test_accumulation_rejects_inexact_effective_batch():
    with pytest.raises(ValueError, match="must be divisible"):
        accumulation_steps(30, 4, 2)


def test_kaggle_cli_overrides_paths_precision_and_batch_settings():
    config = {
        "dataset": {"processed_dir": "old-data"},
        "training": {
            "output_dir": "old-output",
            "epochs": 20,
            "micro_batch_size": 4,
            "effective_batch_size": 32,
        },
        "runtime": {"workers": 4},
        "model": {"precision": "bf16"},
        "wandb": {"enabled": True},
    }
    args = Namespace(
        data_dir="/kaggle/input/data",
        output_dir="/kaggle/working/run",
        epochs=2,
        micro_batch_size=2,
        effective_batch_size=16,
        workers=3,
        precision="fp16",
        no_wandb=True,
    )
    result = apply_overrides(config, args)
    assert result["dataset"]["processed_dir"] == "/kaggle/input/data"
    assert result["training"] == {
        "output_dir": "/kaggle/working/run",
        "epochs": 2,
        "micro_batch_size": 2,
        "effective_batch_size": 16,
    }
    assert result["runtime"]["workers"] == 3
    assert result["model"]["precision"] == "fp16"
    assert result["wandb"]["enabled"] is False
