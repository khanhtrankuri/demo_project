# Training speed on RTX 4060 Laptop 8 GB

The default command is unchanged:

```powershell
conda activate scenesearch
python surround_v4.py train --device cuda
```

The optimized path keeps CLIP ViT-L/14, all six cameras, the global view plus
three crops, vision/text LoRA, geometry, auxiliary losses and the memory queue.
The user's 20 epochs, effective batch size 16 and four data workers are preserved.

Changes:

- Cache structured annotation facts once. Sample candidates in a lazy random
  permutation and stop after finding enough eligible negatives, instead of
  re-extracting annotations for the entire dataset on every step. Classification
  rules and uniform sampling without replacement are preserved; the exact
  negatives selected by a particular seed can differ from the old shuffle.
- Batch eight crops per vision encoder call (previously one).
- Separate fused scene/camera tokens from optional dense patch tokens. The
  reranker does not use dense patches, so its token request no longer causes a
  second pass through the vision backbone.
- Skip unused scene text tokens when the reranker loss cannot run.
- Construct one training DataLoader and keep its workers alive across epochs;
  use nonblocking image/mask transfers from pinned memory.

## Measured comparison

Real nuScenes images, first six train samples, six microsteps, micro batch 1,
effective batch 16, BF16, CUDA, zero data workers to compare the compute paths.
The baseline trainer/model/sampler were loaded read-only from commit
`bc0b05a0e78019abfacf6104108f5f808e8a1bec`, with the current training settings.
Neither run saved or replaced a model checkpoint.

| Measurement | Before | Optimized, crop batch 8 |
|---|---:|---:|
| Training-loop wall time, six steps | 69.20 s | 12.68 s |
| Forward total | 35.33 s | 3.32 s |
| Backward total | 28.56 s | 4.54 s |
| Peak allocated VRAM | 2.98 GiB | 1.79 GiB |
| Peak reserved VRAM | 3.03 GiB | 1.93 GiB |
| Mine four anchors against full train manifest | 10.115 s | 0.000133 s |

The new sampler's one-time fact preparation took 2.21 seconds on the full
manifest. Mining was timed separately; the six-step loop uses a small subset.
The measured loop speedup is about 5.46x. This short test includes startup and
an optimizer update, but not loading pretrained weights. It does not predict
full-epoch wall time or establish retrieval quality. GPU temperature, power and
data-worker behavior can change the result. See `reports/v4_speed_before.json`
and `reports/v4_speed_crop8.json` for the raw measurements.

Reproduce a small benchmark without writing checkpoints:

```powershell
python experiments/benchmark_v4_speed.py --steps 6 --output reports/v4_speed_current.json
python experiments/benchmark_v4_speed.py --baseline-ref bc0b05a0e78019abfacf6104108f5f808e8a1bec --steps 6 --output reports/v4_speed_baseline_repeat.json
```

Use `--workers 4` to include Windows worker startup and image prefetching, or
`--crop-batch 4` to compare a smaller crop batch.

Validation: all 89 repository tests passed and Ruff passed for the changed
Python files. A separate 16-step real-image CUDA run with four workers and
effective batch 16 completed an optimizer update without OOM (peak allocated
1.79 GiB). Its 36.03-second wall time includes Windows worker startup; see
`reports/v4_speed_workers4_validation.json`. No full training or retrieval
quality evaluation was run as part of this optimization.

## Existing training limitations

With `micro_batch_size: 1`, the existing reranker loss is skipped because it
needs at least two scenes in the same microbatch. Gradient accumulation does
not change that. These speed measurements concern dual-encoder training;
they do not establish that the reranker has been trained. Dense patch outputs
remain available explicitly via `return_tokens=True, return_patches=True`.

`use_fast=True` does not speed up image augmentation here: this trainer reads
only mean/std from the Hugging Face image processor and uses its own
`MultiScaleTransform` to generate input tensors.
