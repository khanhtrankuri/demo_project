# SurroundSearch-v4 design

## Scope and invariants

SurroundSearch-v4 is a new, opt-in retrieval path. It does not replace or
rename `SurroundLoRA`, its checkpoints, or its CLI. V4 writes only below
`artifacts/surround_v4/` and is selected through `configs/surround_v4.yaml`.
The target task is annotation-grounded natural-language retrieval over either
one nuScenes camera view or one six-camera sample.

## Diagnosis of the current implementation

The following findings refer to the current files and call sites, rather than
to generic CLIP limitations.

1. **The visual bottleneck is one global vector per camera.**
   `SurroundLoRA.encode_views()` in `src/models/surround_lora.py` reads only
   `vision_model(...).pooler_output` and immediately applies
   `visual_projection`. The OpenCLIP equivalent in
   `src/models/surround_openclip_lora.py` calls `encode_image()`. Neither path
   exposes patch tokens to the fusion module or to a later reranker. An object
   and its location can therefore only survive if already compressed into the
   pretrained global token.

2. **The six-camera fusion can express importance, but not relationships.**
   In `SurroundTransfer.forward()` (`src/models/surround_transfer.py`),
   `nn.Linear(width, 1) + camera_bias` produces one scalar per camera and the
   result is a weighted sum. The score for a camera is computed independently
   of all other cameras. It cannot model front/left agreement, distinguish an
   object spanning adjacent views, or represent two simultaneous facts in
   different cameras. The six-entry `camera_bias` encodes only a prior weight;
   it is not a camera token participating in pairwise attention.

3. **Text is fixed while the visual space moves.**
   Both `SurroundLoRA.encode_text()` and `OpenClipSurroundLoRA.encode_text()`
   are decorated with `torch.no_grad()`. `surround_lora.py:text_bank()` encodes
   every unique caption once before training, and `optimizer_for()` has no
   text-encoder group. Consequently domain phrases such as “front-right”,
   distance bins, counts, and ego-relative relations cannot adapt, while
   vision LoRA and the retrieval heads change the opposite side of the shared
   space.

4. **The supervision deletes most of the annotation semantics.**
   `src/data/surround.py:prepare_dataset()` reduces visible 3D boxes to a
   `Counter`, then emits `A road view containing <sorted unique classes>.`.
   Although `object_counts` is stored, counts are absent from the caption.
   Projected position, metric distance, camera direction, instance attributes,
   motion state, and object-object combinations are not stored. Scene text is
   only `scene.description`, which is weak metadata shared by many samples and
   is not generated from the sample's boxes.

5. **The effective negative set is small and mostly accidental.**
   `make_loss()` in `surround_lora.py` uses only the current logical batch.
   `balanced_contrastive_loss()` deduplicates byte-identical captions, which
   correctly avoids treating identical strings as negatives, but there is no
   sample identity or positive matrix. Two paraphrases of the same sample are
   not represented at all; if added naively, they would become negatives.
   There is no semantic/count/distance/temporal/cross-camera negative policy
   and no cross-batch queue.

6. **The reported score is an exact-string proxy.**
   `surround_lora.py:metrics_for()` calls `select_texts()` using caption strings;
   `surround.py:retrieval_metrics()` accepts only the resulting group id. Thus
   relevance means exact caption equality, as the code itself records in the
   `relevance` field. A scene satisfying “at least three cars” is counted wrong
   unless its stored caption is exactly the query group, while multiple scenes
   with the same generic caption are all treated as interchangeable positives.
   The metric has no annotation predicate, AP, nDCG, or per-query-category
   breakdown.

7. **Wide-image detail is sacrificed before the backbone.**
   `LoRASurroundDataset.__getitem__()` pads each image to a square with
   `ImageOps.pad()` and then `Resize((size, size))`. For a 1600x900 nuScenes
   image at size 224, the real image content is approximately 224x126 with
   about 49 pixels of padding above and below. Small pedestrians and distant
   vehicles receive far fewer source-to-token pixels than a three-crop path.
   The older `SurroundDataset` directly resizes to a configured rectangle,
   which preserves all pixels but distorts aspect ratio.

8. **Spatial information is lost at three explicit boundaries.**
   First, data preparation keeps only class counts after `boxes_in_camera()`.
   Second, the model discards patch tokens at `pooler_output`/`encode_image`.
   Third, `SurroundTransfer` collapses all cameras with a scalar weighted sum.
   The object head is applied to the already-global per-view embedding, so it
   cannot restore left/center/right or metric depth.

9. **Similar scenes are not distinguishable under the current labels/loss.**
   Any two views with the same sorted object-class set receive the same caption
   in `prepare_dataset()`. `balanced_contrastive_loss()` then deliberately maps
   those strings to the same class. Count, side, distance, camera, and state are
   absent from both the positive definition and negative sampling, so scenes
   differing only in those properties are mathematically encouraged to share
   the same retrieval target.

## Proposed architecture

```text
text -> shared VL text tower + LoRA(last N q/v) -> token sequence -> text CLS
                                                               |
camera image -> [global, left, center, right] crops             |
             -> shared VL vision tower + LoRA                  |
             -> crop global tokens + optional patch tokens     |
             -> intra-camera transformer -> camera token       |
             -> + camera-id embedding + geometry MLP           |
 six camera tokens + learned SCENE_CLS                         |
             -> masked 2-4 layer camera transformer            |
             -> scene CLS + contextualized camera tokens       |
             -> projection + normalization <-------------------+
             -> object/count/spatial/distance training heads

stage 1: persistent view and scene embeddings -> cosine top-50
stage 2: query tokens cross-attend to candidate visual/camera tokens -> score
         (query-conditioned fusion exists only here; it is never indexed)
```

The backbone contract is deliberately small:

```python
encode_image(pixel_values) -> normalized [N, D]
encode_text(token_batch) -> normalized [T, D]
get_patch_tokens(pixel_values) -> [N, P, D]
projection_dim: int
```

`CLIPBackbone` and `SigLIPBackbone` implement this contract. Base parameters
remain frozen. Vision and optional text LoRA are separate parameter groups.
The fusion, auxiliary heads, and reranker are FP32 by default; frozen backbone
forward passes use BF16 and PyTorch SDPA. No FlashAttention or bitsandbytes
extension is required.

## Tensor flow

Let `B` be scenes per logical batch, `C=6`, `S=1+num_spatial_crops=4`,
`D` the common projection size (default 768), `P` retained patch tokens per
crop, `K_v` view captions, and `K_s` scene captions.

| Boundary | Shape | Notes |
|---|---:|---|
| dataset images | `[B,C,S,3,H,W]` | missing cameras are zero with `[B,C]` mask |
| crop global features | `[B,C,S,D]` | only valid cameras are encoded |
| optional patch features | `[B,C,S,P,D]` | retained for top-K reranking, not the index |
| crop tokens + crop position | `[B*C,S,D]` | global/left/center/right identities |
| intra-camera output | `[B,C,D]` | learned camera CLS, not average pooling |
| geometry input | `[B,C,5]` | `sin(yaw), cos(yaw), tx, ty, tz` |
| fusion input | `[B,1+C,D]` | scene CLS followed by fixed-order camera tokens |
| contextual camera tokens | `[B,C,D]` | mask preserves missing-camera semantics |
| scene embedding | `[B,D]` | normalized projected scene CLS |
| view text embeddings | `[sum(B*C*K_v),D]` | ids define a many-to-many positive matrix |
| scene text embeddings | `[sum(B*K_s),D]` | all captions from one scene are positives |
| queue | `[Q,D]` | detached BF16/FP16 embeddings plus ids/signatures |
| reranker candidate tokens | `[Bq,topK,1+C(+patch),D]` | computed/read only for shortlisted items |

Patch tokens are optional during stage-1 training to fit 8 GB. Multi-scale
crop global tokens already prevent the single-global-view information collapse;
patch tokens can be recomputed for top-K candidates in stage 2.

## Rich supervision and data flow

Preparation retains one structured object record per projected box:

```text
class, count contribution, camera, horizontal bin, distance metres,
distance bin, motion/parked/standing attribute, sample/scene/instance token
```

View captions and scene captions are deterministic templates seeded by sample
id. Each record receives 3-5 variants: a full sentence, compact form,
count-focused form, spatial/distance form, and (where valid) an attribute or
multi-object form. Empty fields are omitted rather than invented. Query ground
truth is a serialized annotation predicate; it never depends on caption text.

Hard negatives are selected using structured signatures:

- random: disjoint or low-overlap class set;
- semantic/spatial: same class, conflicting side;
- count: same class, different count bin;
- distance: same class and optionally side, conflicting distance bin;
- temporal: same nuScenes scene, nearby timestamp, changed signature;
- cross-camera: same class/relation in a different camera.

## Losses

For normalized image embeddings `v_i`, text embeddings `t_j`, temperature
`tau`, and boolean positive matrix `P_ij`, the multi-positive image-to-text
term is:

```text
L_i2t = -mean_i log( sum_{j:P_ij} exp(v_i.t_j/tau)
                       / sum_j exp(v_i.t_j/tau) )
```

The text-to-image term is symmetric, excluding anchors with no positive. Queue
entries participate only in denominators (and may be positives when their
stable sample id matches); they are detached, so the queue never retains an
autograd graph. `L_hard` is a margin/log-sum-exp ranking loss over explicitly
typed hard negatives. Auxiliary targets are masked when an annotation is
unknown.

```text
L = lambda_v  L_view_mp
  + lambda_s  L_scene_mp
  + lambda_h  L_hard
  + lambda_o  BCE(object presence)
  + lambda_c  CE(count bins per class)
  + lambda_sp CE(left/center/right per class)
  + lambda_d  CE(distance bins per class)
  + lambda_r  L2/anchor regularization
```

Every component is returned under its own name before weighting. This avoids a
single opaque scalar in logs and makes each ablation auditable.

## Training strategy on an RTX 4060 Laptop 8 GB

The safe default is CLIP ViT-L/14 at 224, BF16, one physical scene per
micro-batch, one crop forward at a time, gradient checkpointing, and a logical
batch of 8 accumulated through gradient cache. SigLIP base/patch16-224 is the
recommended SigLIP-class alternative; 336 resolution is an opt-in profile and
should reduce retained patch tokens or logical batch size.

Approximate peak components for the default profile (implementation-dependent,
so smoke output is authoritative):

| Component | Estimate |
|---|---:|
| frozen ViT-L/14 BF16 weights | 0.8-0.9 GiB |
| frozen text weights + projections | 0.25-0.35 GiB |
| one 224 crop activation with checkpointing | 0.7-1.3 GiB |
| LoRA/fusion/head params, grads, Adam states | 0.2-0.5 GiB |
| four-crop CPU staging + one-crop GPU staging | 0.1-0.3 GiB |
| CUDA kernels/allocator/runtime reserve | 1.5-2.5 GiB |
| queue 4096 x 768 x two modalities, BF16 | about 12 MiB |
| expected peak allocated / reserved | 3.5-5.5 / 5-7 GiB |

The 24-layer ViT-H model currently selected by `surround_lora.yaml` is not the
v4 8-GB default. V4 exposes it only as an explicit high-memory override.
Smoke reports peak allocated/reserved memory, trainable/total parameters, and
separately timed forward/backward phases.

## Evaluation

`query_generator.py` produces annotation predicates for object presence,
count, spatial relation, distance, attribute, multi-object, camera-specific,
and scene-level conditions. A query can have any number of relevant samples.
`semantic_retrieval.py` evaluates ranked ids against those relevance sets and
reports Recall@1/5/10, MRR, mAP, and nDCG@10 globally and per category. Exact
caption equality remains available only as a labeled legacy metric.

## Ablation plan

`experiments/run_ablation.py` defines cumulative, config-only variants so the
same data split and seed are reused:

| ID | Change from previous row | Main question |
|---|---|---|
| A0 | current SurroundLoRA v3 | legacy reference |
| A1 | rich captions | does grounded language help? |
| A2 | multi-positive loss | do paraphrases stop acting as negatives? |
| A3 | typed hard negatives | do fine distinctions improve? |
| A4 | cross-batch queue | does negative diversity help small batches? |
| A5 | camera transformer | do cross-view relations improve? |
| A6 | global + three crops | does small-object/spatial recall improve? |
| A7 | text LoRA | does domain language alignment improve? |
| A8 | geometry MLP | does ego-relative/camera retrieval improve? |
| A9 | SigLIP/SigLIP2 | does the pretrained objective/backbone help? |
| A10 | top-50 reranker | does token-level matching improve top ranks? |

Rows append to `reports/ablation.csv` with config hash, checkpoint hash, seed,
all global metrics, category metrics, peak VRAM, and timing. A failed command
is recorded as failed and blocks later rows unless explicitly resumed.

## Implementation roadmap and gates

1. Phase 1: this diagnosis/design document. Gate: code references and tensor
   contracts reviewed against the repository.
2. Phase 2: structured caption/query generation and semantic metrics. Gate:
   deterministic caption tests, predicate/metric tests, legacy tests.
3. Phase 3: multi-scale encoder, camera transformer, geometry MLP. Gate: shape,
   mask, missing-camera, no-average-pooling, and gradient tests plus smoke.
4. Phase 4: multi-positive loss, typed hard-negative sampler, detached queue.
   Gate: false-negative, queue wraparound/detach, and negative-type tests.
5. Phase 5: CLIP/SigLIP abstraction and separate text LoRA group. Gate: common
   contract tests and trainable-parameter audit.
6. Phase 6: top-K cross-modal reranker and query-aware camera attention. Gate:
   index remains query-independent, reranker shape/gradient tests, end-to-end
   synthetic retrieval smoke.

## Expected bottlenecks and mitigations

- Four crops multiply backbone compute even though peak activation is bounded;
  encode crops sequentially and use cached-gradient replay.
- Token APIs differ between Hugging Face CLIP, SigLIP, and OpenCLIP; isolate
  those differences inside backbone adapters and test the contract with tiny
  local models.
- nuScenes projected boxes are FOV labels, not perfect visible/occlusion masks;
  carry an `annotation_quality` field and avoid claiming pixel visibility.
- Template captions can leak phrasing shortcuts; randomize deterministic
  templates, evaluate only with annotation predicates, and report by category.
- False negatives remain possible across visually equivalent scenes; mask
  known predicate-equivalent items in the loss and prefer typed contradictions.
- Reranking can dominate latency if tokens are stored densely; stage 1 stores
  only compact embeddings, while stage 2 recomputes or loads tokens for top-K.
- Count/spatial/distance labels are class-imbalanced; use masked class weights
  and report each category instead of hiding it in a single aggregate.
