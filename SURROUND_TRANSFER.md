# SurroundTransfer-v2

This experiment targets the original four-way mean Recall@10 score. It keeps
the v1 scene partitions, captions, gallery contents, and `retrieval_metrics`
implementation unchanged. A score of 0.8 is a target, not a guaranteed result.

The v1 experiment selected epoch 10 with validation score 0.4015667; its full
test score is 0.2154024. V1 learns both language and vision from scratch on
178 training scenes. V2 instead uses the locally cached LAION OpenCLIP ViT-B/32
base checkpoint, with frozen image and text encoders. It does not load the
previous fine-tuned CAM_FRONT checkpoint.

Images are padded to preserve their entire field of view before CLIP
preprocessing. The backbone runs once per image and caches normalized features.
Train and validation are cached first; the test cache is built after checkpoint
selection. Caches validate both the manifest hash and backbone weights hash.

Trainable modules are small residual image and scene adapters, camera-aware
attention pooling, an auxiliary object classifier, and contrastive temperature.
Adapters start as identity maps. The training loss balances distinct caption
groups and deduplicates text candidates, so common captions do not dominate
rare combinations. Duplicate captions remain multiple positives. A cosine
anchor penalizes drift away from frozen pretrained embeddings. Text weights
and text embeddings are never optimized using evaluation captions.

Validation is evaluated over the entire gallery every epoch. The exact same
mean of view/scene image-to-text/text-to-image Recall@10 selects `best.pt`.
The frozen baseline is retained if adaptation never improves validation.
Training stops after six epochs without a significant validation improvement.
All outputs are separate from v1 at `artifacts/nuscenes_surround_transfer`.

## Measured result

The completed run selected epoch 6 using validation (0.467842), then stopped
at epoch 12. The selected checkpoint was subsequently evaluated on the full
held-out test gallery. No test results were used to select its epoch.

| Test metric | Scratch v1 | Transfer v2 |
| --- | ---: | ---: |
| View image-to-text Recall@10 | 0.328063 | 0.371871 |
| View text-to-image Recall@10 | 0.126374 | 0.181319 |
| Scene image-to-text Recall@10 | 0.289526 | 0.632411 |
| Scene text-to-image Recall@10 | 0.117647 | 0.392157 |
| **Selection score (four-way mean)** | **0.215402** | **0.394439** |

This is an absolute gain of 0.179037 (approximately 83.1% relative), **not
the requested 0.8**. Both runs use 12,144 view images / 364 distinct view
captions and 2,024 six-camera samples / 51 distinct scene captions. Both test
manifest hashes are
`8bffe87230178cd2f914c48defd5a7e43a1974129188b91c3f0be3256e18aba5`.

The trained adapter is already at `artifacts/nuscenes_surround_transfer/best.pt`;
its report is `artifacts/nuscenes_surround_transfer/evaluation_test.json`.
The base OpenCLIP weights must also remain in the local Hugging Face cache.
The test search index has been built at
`artifacts/nuscenes_surround_transfer/index_test`.
To use these existing artifacts, run only `eval` or `query` below. Do not
rerun a fresh `train` into this directory; it deliberately refuses to
overwrite an existing checkpoint. Use a new output directory for a new
experiment. `--resume` restores an interrupted run with identical settings;
an already early-stopped run remains stopped.

Training loss still decreased after validation peaked: early stopping limits
overtraining rather than proving that overfitting has disappeared. The next
experiments should be selected on validation, not repeatedly tuned against
this test result. Weak object captions and the gap between whole-scene
descriptions and individual instants remain limitations of this benchmark.

## Commands

```powershell
conda activate scenesearch
python surround_transfer.py cache
python surround_transfer.py train
# Resume an interrupted run with: python surround_transfer.py train --resume
python surround_transfer.py cache --split test
python surround_transfer.py eval
python surround_transfer.py index --split test
python surround_transfer.py query --split test --query "A road view containing car, person."
```

For indexes of train or validation, pass `--split train` or `--split val` to
both `index` and `query`. `--level scene` queries six-camera samples. V2 uses
its own CLI and checkpoint format. It does not replace v1's `surround.py`.

Configuration: `configs/surround_transfer.yaml`. Requires the base weights
already in the local Hugging Face cache; inference and training do not contact
an external API. The installed OpenCLIP implementation loads the local model
as described in [OpenCLIP documentation](https://github.com/mlfoundations/open_clip).

`baseline_validation.json`, `history.json`, `summary.json`, and
`evaluation_test.json` record the actual experiment. Full test quality is only
known after evaluation. Exact-caption relevance remains weak supervision:
camera FOV labels do not establish occlusion, and a scene description can refer
to events outside a single sample's instant. An 80% score may require better
visual grounding or supervision, not just changing optimizer settings.
