# nuScenes Surround Retrieval with CLIP LoRA

## SurroundSearch-v4 and Kaggle 2 x T4

The repository also includes the camera-aware `SurroundSearch-v4` architecture
from P-056. Prepare the rich v4 manifests first, then launch distributed
training on a Kaggle notebook with both T4 GPUs enabled:

```bash
!torchrun --standalone --nproc_per_node=2 train_kaggle_2gpu.py \
  --config configs/surround_v4.yaml \
  --data-dir /kaggle/input/<dataset>/nuscenes_surround \
  --output-dir /kaggle/working/surround_v4 \
  --precision fp16
```

T4 does not provide native BF16 acceleration, so the Kaggle trainer defaults
to FP16 autocast with dynamic loss scaling. It uses one process per GPU,
`DistributedSampler`, gradient synchronization only at optimizer boundaries,
rank-0 checkpointing, and resumable optimizer/scaler/scheduler state. The
effective batch size is global across both GPUs.

Run a short integration check before a full job:

```bash
!torchrun --standalone --nproc_per_node=2 train_kaggle_2gpu.py \
  --data-dir /kaggle/input/<dataset>/nuscenes_surround \
  --output-dir /kaggle/working/surround_v4_smoke \
  --max-steps 2 --no-wandb
```

The current architecture uses `openai/clip-vit-large-patch14` with LoRA on the
vision transformer's query and value projections. It trains on synchronized
six-camera nuScenes samples and is designed for an 8 GB local NVIDIA GPU.

Main components:

- CLIP ViT-L/14 at 224 x 224 with 768-dimensional embeddings.
- Rank-8 vision LoRA across 24 transformer layers.
- Frozen CLIP text encoder and frozen base vision weights.
- Camera-aware fusion for scene-level retrieval.
- BF16, gradient checkpointing, PyTorch SDPA and logical-batch gradient cache.
- Scene-separated train, validation and test manifests.
- Full-gallery view and scene retrieval evaluation.

Only about 1.20 million of 428.81 million parameters are trainable. The local
RTX 4060 smoke run used approximately 1.07 GiB peak allocated and 1.22 GiB peak
reserved memory as reported by PyTorch.

For installation, archive extraction, dataset splitting, training, evaluation,
indexing and query commands, see [SURROUND_LORA.md](SURROUND_LORA.md).

Quick start after preparing the dataset:

```powershell
conda activate scenesearch
python surround_lora.py download
python surround_lora.py smoke
python surround_lora.py train
python surround_lora.py eval
```

Configuration is stored in `configs/surround_lora.yaml`. Generated datasets,
pretrained weights, checkpoints and indexes are excluded from Git.

Dataset preparation reads every `nuScense/*.tgz` archive. The nuScenes
selection config explicitly lists all six cameras and uses `target_size: all`.
The LoRA manifests include every valid annotated keyframe, split by complete
scenes.
