# SurroundSearch-v1

A new, randomly initialized image/text retrieval baseline for **all six camera
keyframes available locally** in `nuScense/extracted`. No OpenCLIP checkpoint is
loaded. This is an experimental domain model, not a pretrained foundation model;
training on these repeated driving scenes does not establish open-world or
multilingual language understanding. A UTF-8 tokenizer accepts arbitrary text,
but the current training captions are English.

## Data preparation

```powershell
python scripts/prepare_surround.py --config configs/surround.yaml
```

The local extraction contains 10,120 frames per camera, 60,720 camera images
in total. The pipeline streams large JSON tables, fully decodes images, and
exports every valid image to `data_processed/nuscenes_surround/images.jsonl`.
It groups synchronized camera keyframes by `sample_token` into train/val/test
JSONL manifests. Missing cameras are masked, rather than discarding an entire
sample. Rejected images and counts are reported. There is no 10K selection cap,
and images are referenced rather than copied. LiDAR, radar and map files are
not RGB camera inputs and are not consumed by this model. This extraction has
keyframes only; interpolated camera sweeps would need a separate label policy.

Scene groups never cross partitions. The existing 10K test scene IDs are kept
in test when that CSV exists; validation receives about 10% of all available
scenes, and the remaining scenes train the model. All images participate in
one partition; validation/test are not optimizer inputs. These are local scene
splits, not the official nuScenes train/val benchmark split.

Verified local preparation (seed 42, preserving 51 old test scenes):

| Partition | Scenes | Six-camera samples | Images |
| --- | ---: | ---: | ---: |
| Train | 178 | 7,068 | 42,408 |
| Validation | 26 | 1,028 | 6,168 |
| Test | 51 | 2,024 | 12,144 |
| Total | 255 | 10,120 | 60,720 |

All local camera images decoded successfully. All samples have six cameras.
No scene, sample or image ID crosses splits. `stats.json` records the inventory,
class vocabulary and split fingerprint. The dataset contains 834 distinct
per-view captions; 8,771 views have no boxes passing the geometric FOV rule.

Per-camera object labels are projected from 3D boxes through world -> ego ->
camera transforms. The visibility rule follows the nuScenes devkit ANY-corner
criterion: some corner projects inside the image and all corners are in front
of the near plane. This checks field of view, **not occlusion**. Source:
[nuScenes geometry utilities](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/geometry_utils.py).
Scene descriptions supervise only the combined six-view representation;
they describe a time interval, so this is weak supervision rather than exact
frame captions. Per-view captions contain only the projected object classes.
Location strings and camera names are not inserted into text targets.

## Model

| Component | Specification |
| --- | --- |
| Input | RGB, 160 x 288, six ordered cameras per sample |
| Shared image encoder | Residual CNN, GroupNorm, 32/64/128/192/256 channels |
| Image embedding | 256 dimensions, L2 normalized |
| Text encoder | UTF-8 byte tokens, length 192, width 256, 3 Transformer layers |
| Surround fusion | Camera embeddings + scene token, 2 Transformer layers, masked attention |
| Auxiliary task | Multi-label object prediction for each camera |
| Parameters | Approximately 6.28 million, all trained from scratch |

The per-image projection supports single-camera retrieval. Fusion adds a
separate scene embedding for retrieving complete surround samples. The
backbone is shared by all camera positions. Camera embeddings enter fusion
only, not single-view embeddings. Camera dropout regularizes fusion while
all available images still receive per-view supervision each epoch.

Loss = per-view contrastive loss + 0.25 x scene contrastive loss +
0.5 x object BCE. Equal captions are multiple positives; the target distributes
probability uniformly among them. Class weights are computed from train only.
Full-field resize and mild color jitter preserve the field of view used by
object labels; there is no horizontal flip or random crop.

## Train, evaluate, search

Activate the existing `scenesearch` environment first:

```powershell
conda activate scenesearch
python surround.py train
python surround.py eval
python surround.py index --split all
python surround.py query --query "A road view containing car, person." --level view --top-k 5
python surround.py query --query "Cars waiting at an intersection." --level scene --top-k 5
```

`configs/surround.yaml` uses batch size 16 samples (96 images), AMP, AdamW,
warmup/cosine learning rate and a maximum of 40 epochs. Adjust `--batch-size 8`
if another process uses GPU memory. Resume exactly the same run with
`python surround.py train --resume`. Configuration and manifest fingerprints
must match on resume. Existing runs require resume or a new `--output-dir`.

Every epoch evaluates **the full validation gallery**, reporting image-to-text
and text-to-image Recall@1/5/10 for views and surround samples. Selection score
is the mean of the four Recall@10 values (two directions x two levels).
`best.pt` records the highest score; early stopping uses six epochs without
an improvement greater than 0.001. The untrained model is evaluated as epoch
zero and retained if training never improves validation. Test is evaluated
only by the explicit `eval` command, not during training. Train loss is not a
checkpoint selection criterion.

Metrics use exact-caption matches, a weak-label proxy for relevance. A small
number of caption types can inflate recall; reports include unique-caption
counts. Old CLIP results with different captions/galleries are not directly
comparable. Neither lower train loss nor successful smoke tests guarantee
better held-out retrieval: compare full test reports after training.

Artifacts live in `artifacts/nuscenes_surround`: `best.pt`, resumable `last.pt`,
`history.json`, `summary.json`, test reports, and separate normalized view and
scene indexes. These weights are **not compatible with the old OpenCLIP
scripts/UI**; use `surround.py` for both indexing and querying. Each index stores
the checkpoint hash and querying rejects mismatched weights.

Quick end-to-end check, separate from the real training run:

```powershell
python surround.py train --epochs 2 --max-samples 12 --batch-size 2 --workers 0 --output-dir artifacts/surround_smoke
python surround.py eval --max-samples 12 --batch-size 2 --workers 0 --output-dir artifacts/surround_smoke
python surround.py index --max-samples 12 --batch-size 2 --workers 0 --output-dir artifacts/surround_smoke
python surround.py query --query "A road view containing car." --output-dir artifacts/surround_smoke
```

Smoke metrics are only functional checks. A tiny gallery is not a quality
benchmark. Use a new output directory when rerunning a smoke experiment.
