# W&B tracking for SurroundSearch-v4

Install the training dependencies and authenticate once:

```powershell
conda activate scenesearch
pip install -r requirements-training.txt
wandb login
```

Training automatically creates a run in the `surroundsearch-v4` project:

```powershell
python surround_v4.py train --device cuda
```

The dashboard receives averaged view, scene, hard-negative, object, count,
spatial, distance, regularization and total losses after each optimizer update.
It also receives learning rates, epoch/microstep progress, throughput, forward
and backward time, logit scale, CUDA memory, and W&B's automatic system metrics.
The run URL is printed when training starts.

Edit the `wandb` section in `configs/surround_v4.yaml` to set an entity, run
name, group, or tags. Use `mode: offline` when training without Internet, then
sync the generated run later with `wandb sync <run-directory>`. Set
`enabled: false` to disable tracking. Smoke runs are not logged unless
`log_smoke: true` is set.

Checkpoints stay local by default to avoid adding upload time after training.
Set `log_checkpoint: true` to upload `best.pt` as a W&B model artifact.
