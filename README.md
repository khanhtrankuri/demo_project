# RAV-14 SceneSearch

Local semantic and metadata-aware search over 10,000 real BDD100K road-scene images. The MVP uses pretrained OpenCLIP ViT-B/32, normalized image/text embeddings, FAISS inner-product search, deterministic metadata parsing, hybrid reranking, and a Streamlit gallery. It does not call an external API.

## Dataset found in this workspace

The local dataset is a DatasetNinja/Supervisely-style BDD100K export. Images are stored under `train/img`, `val/img`, and `test/img`; annotations are per-image `*.jpg.json` sidecars under matching `ann` directories. Scene attributes appear in `tags`, and object classes appear in `objects[].classTitle`.

Dataset images, archives, model weights, generated metadata, embeddings, indexes, and feedback are excluded from Git. After cloning, place your dataset at `bdd100k/bdd100k_-images-100k-DatasetNinja` or set `paths.dataset_root` in `configs/default.yaml`, then run the preparation and index commands below. The first model load downloads the pretrained weights; subsequent inference runs locally.

The validated MVP pool uses meaningful metadata only. It samples 8,000 images from the native train split and 2,000 from native validation with seed 42. Image files are not copied; CSV files store absolute paths.

## Windows PowerShell setup

```powershell
conda create -n scenesearch python=3.11 -y
conda activate scenesearch
pip install -r requirements.txt
```

Inspect the dataset:

```powershell
python scripts/check_bdd100k.py bdd100k/bdd100k_-images-100k-DatasetNinja --output bdd100k_audit_report.json
```

Prepare the deterministic 10K subset:

```powershell
python scripts/prepare_bdd100k.py --config configs/default.yaml --dataset-root bdd100k/bdd100k_-images-100k-DatasetNinja
```

Build all 10,000 CLIP embeddings and the FAISS index:

```powershell
python scripts/build_index.py --config configs/default.yaml --split all
```

For lower-memory GPUs, append `--batch-size 32` or `--batch-size 16`. To force CPU, append `--device cpu`.

On NVIDIA Windows machines, install the matching CUDA wheels if the default PyPI Torch build is CPU-only. For example:

```powershell
pip install --force-reinstall torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
```

Run the end-to-end smoke test:

```powershell
python scripts/smoke_test.py --config configs/default.yaml --split all
```

Launch the demo:

```powershell
streamlit run app/app.py
```

Evaluate Semantic Only versus Hybrid:

```powershell
python scripts/evaluate.py --config configs/default.yaml --split all
```

Run unit tests:

```powershell
pytest -q
```

To build native splits separately, run `build_index.py` with `--split dev` or `--split test`. Only artifacts for the requested split are written.

## Retrieval behavior

Semantic Only encodes the full query with CLIP and returns FAISS nearest neighbors. Hybrid first retrieves 100 semantic candidates, extracts supported metadata keywords without an LLM, calculates exact-condition metadata scores, min-max normalizes candidate semantic scores to `[0,1]`, then combines semantic and metadata scores at 0.8/0.2. `nighttime` is matched to BDD100K's `night`; `motorcycle` is matched to its `motor` class.

Feedback buttons append JSON lines to `data_processed/feedback.jsonl`. No database or authentication is used.

## Evaluation scope

The benchmark contains only queries whose relevance can be computed from available weather, time-of-day, scene, and object labels. Precision@5, Precision@10, MRR, Recall@10, and latency are written to `reports/baseline_metrics.csv` and `.json`. Recall is relative to the complete indexed set's metadata-derived relevance set; no action labels or fake semantic ground truth are generated.

## Future phases (not implemented)

- Phase 2: generate captions, contrastively fine-tune CLIP on BDD100K, then rebuild FAISS.
- Phase 3: retrieve with the domain-adapted CLIP model, combine metadata, and rerank a small candidate set using a local Qwen2.5-VL model.

The encoder, semantic index wrapper, metadata scorer, and hybrid reranker are separate components so those later replacements do not require redesigning the UI or dataset layer.

The index wrapper uses genuine `faiss.IndexFlatIP` whenever the FAISS native module can load. On Windows machines where Application Control blocks `_swigfaiss.pyd`, it records that error and uses an exact NumPy inner-product fallback so the local demo remains operational; rebuild the index after policy approval to produce a genuine FAISS artifact.
