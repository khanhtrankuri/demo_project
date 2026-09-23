# nuScenes Surround Retrieval with CLIP LoRA

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
