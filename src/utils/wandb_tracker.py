"""Optional Weights & Biases tracking for SurroundSearch-v4 training."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def _serializable(value: Any):
    if isinstance(value, dict):
        return {key: _serializable(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, list | tuple):
        return [_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


class WandbTracker:
    def __init__(self, module, run, settings: dict):
        self.module = module
        self.run = run
        self.settings = settings
        self.finished = False

    @classmethod
    def start(cls, config: dict, output_dir: Path, command: str, wandb_module=None):
        settings = config.get("wandb", {})
        enabled = bool(settings.get("enabled", False))
        if command == "smoke" and not settings.get("log_smoke", False):
            enabled = False
        if command not in ("train", "smoke") or not enabled:
            return None
        if wandb_module is None:
            try:
                import wandb as wandb_module
            except ImportError as error:
                raise RuntimeError(
                    "W&B is enabled but not installed. Run: pip install -r requirements-training.txt"
                ) from error
        kwargs = {
            "project": settings.get("project", "surroundsearch-v4"),
            "config": _serializable(config),
            "dir": str(output_dir),
            "job_type": command,
            "mode": settings.get("mode", "online"),
            "tags": list(settings.get("tags", [])),
            "save_code": bool(settings.get("save_code", True)),
        }
        for key in ("entity", "name", "group", "notes"):
            if settings.get(key):
                kwargs[key] = settings[key]
        try:
            run = wandb_module.init(**kwargs)
        except Exception as error:
            raise RuntimeError(
                "W&B initialization failed. For online mode run `wandb login`; "
                "or set `wandb.mode: offline` / `wandb.enabled: false`."
            ) from error
        run.define_metric("progress/optimizer_step")
        for pattern in ("train/*", "lr/*", "system/*", "model/*"):
            run.define_metric(pattern, step_metric="progress/optimizer_step")
        if getattr(run, "url", None):
            print(f"W&B run: {run.url}")
        return cls(wandb_module, run, settings)

    def log(self, metrics: dict):
        self.run.log(metrics)

    def complete(self, report: dict, checkpoint_path: Path):
        self.run.summary.update({
            "result/steps": report["steps"],
            "result/optimizer_steps": report["optimizer_steps"],
            "result/forward_seconds": report["forward_seconds"],
            "result/backward_seconds": report["backward_seconds"],
            "result/peak_allocated_gib": report["peak_allocated_gib"],
            "result/peak_reserved_gib": report["peak_reserved_gib"],
            "result/checkpoint_path": str(checkpoint_path),
        })
        if self.settings.get("log_checkpoint", False):
            artifact = self.module.Artifact(
                f"{self.run.id}-checkpoint", type="model",
                metadata={"architecture": "SurroundSearch-v4", "steps": report["steps"]})
            artifact.add_file(str(checkpoint_path))
            self.run.log_artifact(artifact)

    def finish(self, exit_code: int):
        if not self.finished:
            self.run.finish(exit_code=exit_code)
            self.finished = True


def start_wandb(config: dict, output_dir: Path, command: str, wandb_module=None):
    return WandbTracker.start(config, output_dir, command, wandb_module)
