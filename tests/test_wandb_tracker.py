from src.utils.wandb_tracker import start_wandb


class FakeArtifact:
    def __init__(self, name, type, metadata):
        self.name, self.type, self.metadata = name, type, metadata
        self.files = []

    def add_file(self, path):
        self.files.append(path)


class FakeRun:
    id = "run-123"
    url = "https://wandb.local/run-123"

    def __init__(self):
        self.definitions, self.history, self.summary, self.artifacts = [], [], {}, []
        self.exit_code = None

    def define_metric(self, *args, **kwargs):
        self.definitions.append((args, kwargs))

    def log(self, value):
        self.history.append(value)

    def log_artifact(self, artifact):
        self.artifacts.append(artifact)

    def finish(self, exit_code):
        self.exit_code = exit_code


class FakeWandb:
    Artifact = FakeArtifact

    def __init__(self):
        self.run, self.kwargs = FakeRun(), None

    def init(self, **kwargs):
        self.kwargs = kwargs
        return self.run


def test_wandb_tracker_logs_metrics_summary_and_optional_checkpoint(tmp_path):
    module = FakeWandb()
    config = {"seed": 42, "_project_root": tmp_path,
              "wandb": {"enabled": True, "project": "test", "mode": "offline",
                        "tags": ["unit"], "log_checkpoint": True}}
    tracker = start_wandb(config, tmp_path, "train", module)
    assert module.kwargs["project"] == "test"
    assert "_project_root" not in module.kwargs["config"]
    tracker.log({"progress/optimizer_step": 1, "train/loss_total": 2.5})
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    report = {"steps": 16, "optimizer_steps": 1, "forward_seconds": 2.0,
              "backward_seconds": 3.0, "peak_allocated_gib": 1.5, "peak_reserved_gib": 2.0}
    tracker.complete(report, checkpoint)
    tracker.finish(0)
    tracker.finish(1)
    assert module.run.history[-1]["train/loss_total"] == 2.5
    assert module.run.summary["result/optimizer_steps"] == 1
    assert module.run.artifacts[0].files == [str(checkpoint)]
    assert module.run.exit_code == 0


def test_wandb_is_disabled_for_smoke_by_default(tmp_path):
    module = FakeWandb()
    tracker = start_wandb({"wandb": {"enabled": True}}, tmp_path, "smoke", module)
    assert tracker is None
    assert module.kwargs is None
