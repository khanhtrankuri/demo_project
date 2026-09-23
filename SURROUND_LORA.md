# SurroundLoRA-v3: CLIP ViT-L/14 trên GPU local

Pipeline riêng cho CLIP lớn hơn, giữ nguyên split theo scene và metric của v1/v2.
Không ghi đè checkpoint, cache hoặc index cũ.

## Kiến trúc và bộ nhớ

- Backbone: `openai/clip-vit-large-patch14`, ảnh 224 x 224, embedding 768.
  Revision được cố định trong `configs/surround_lora.yaml`.
- LoRA rank 8, alpha 16, dropout 0.05 trên `q_proj` và `v_proj` của 24 vision
  transformer layers (48 projections). Base weights và text encoder đóng băng.
- Tổng 428.814.105 tham số; chỉ 1.197.592 tham số trainable (khoảng 0,28%),
  trong đó 786.432 là LoRA. Phần còn lại là head truy hồi / fusion / object.
- Base weights dùng BF16; LoRA, heads, gradients và optimizer states dùng FP32.
  Gradient checkpointing non-reentrant + PyTorch SDPA, không cần FlashAttention
  extension, bitsandbytes, PEFT hay quantization 4-bit trên Windows.
- Micro-batch = 1 mẫu sáu camera (tối đa 6 ảnh). Logical contrastive batch = 16
  mẫu (tối đa 96 ảnh), không phải 16 ảnh.
- Gradient cache hai lượt: encode từng micro-batch không giữ graph, tính loss
  trên toàn bộ logical batch, rồi replay từng micro-batch để backpropagate.
  RNG được replay để LoRA dropout trùng khớp. Cách này giữ negatives giữa các
  micro-batch; gradient accumulation thông thường không làm được điều đó.
- Chỉ text features được giữ trong RAM khi train. **Không dùng image feature
  cache của v2**: image features thay đổi theo LoRA ở mỗi optimizer step.

Ảnh được pad vuông để giữ full field of view, resize bicubic và normalize bằng
mean/std của CLIP-L processor. Train dùng color jitter nhẹ, không flip camera.
Loss contrastive cân bằng caption và nhận duplicate captions là multiple
positives; auxiliary object loss, head-anchor regularization, camera dropout,
weight decay, cosine LR/warmup và early stopping theo full validation.
Head-anchor chỉ hạn chế head lệch khỏi feature LoRA hiện tại; nó không phải
distillation từ một bản sao frozen CLIP-L.

## Sử dụng

Dùng Windows PowerShell tại thư mục gốc của repository. Pipeline đã được kiểm
tra với Python 3.11, torch 2.7.1+cu128 và transformers 4.57.6. Không cần cài
thêm PEFT.

Đặt các archive nuScenes trong `nuScense/`:

```text
nuScense/
  v1.0-trainval_meta.tgz
  v1.0-trainval01_keyframes.tgz
  v1.0-trainval02_keyframes.tgz
  ...
```

Chạy lần lượt từ cài thư viện đến train:

```powershell
# 1. Tạo environment và cài thư viện.
conda create -n scenesearch python=3.11 -y
conda activate scenesearch
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, 'CUDA available:', torch.cuda.is_available())"

# 2. Giải nén nuScenes vào nuScense/extracted.
# Bước này cũng tạo data_processed/nuscenes_10k/test.csv để khóa test scenes.
python scripts/prepare_nuscenes.py --config configs/nuscenes.yaml

# 3. Xử lý cả sáu camera và chia scene thành train/val/test.
python scripts/prepare_surround.py --config configs/surround.yaml

# 4. Tải CLIP ViT-L/14, khoảng 1,71 GB safetensors + tokenizer/config.
python surround_lora.py download

# 5. Chạy hai optimizer steps và encode một subset validation để kiểm tra VRAM.
python surround_lora.py smoke

# 6. Full training trên train split; checkpoint được chọn bằng validation.
python surround_lora.py train

# Resume từ checkpoint cuối epoch đã hoàn thành với cùng cấu hình.
python surround_lora.py train --resume

# 7. Chỉ đánh giá test sau khi best.pt đã được chọn bằng validation.
python surround_lora.py eval
python surround_lora.py index --split test
python surround_lora.py query --split test --query "A road view containing car, person."
python surround_lora.py query --split test --level scene --query "A busy intersection with pedestrians."
```

Nếu environment `scenesearch` đã tồn tại thì bỏ qua `conda create`. Nếu dữ liệu
đã được giải nén và `data_processed/nuscenes_10k/test.csv` đã tồn tại thì bỏ
qua bước 2. Bước 3 tạo `data_processed/nuscenes_surround/train.jsonl`,
`val.jsonl`, `test.jsonl` và `stats.json`; ảnh gốc chỉ được tham chiếu, không bị
sao chép. Split được thực hiện theo toàn bộ `scene_token`, không theo FPS, để
các frame gần nhau của cùng một scene không xuất hiện ở nhiều partition.

Pipeline hiện dùng annotated keyframes khoảng 2 Hz. Sáu camera tại cùng một
`sample_token` được ghép thành một mẫu. Các camera sweep trung gian không có
annotation tương đương keyframe và không được đưa vào supervised training.

Trong workspace này, model gốc đã được tải vào
`artifacts/pretrained/clip-vit-large-patch14`. Ngoài lệnh `download`, các lệnh
khác chỉ đọc local files, không gọi inference API và không tải weights ngầm.

Checkpoint LoRA nằm tại `artifacts/nuscenes_surround_lora`:

- `smoke_adapter.pt`, `smoke_report.json`: chỉ kiểm tra kỹ thuật; **không phải
  model đã train đầy đủ**, không tạo `best.pt` và không báo cáo test score.
- `best.pt`: chỉ LoRA + heads, chọn bằng full validation, kể cả frozen baseline
  epoch 0 nếu training không cải thiện. Cần giữ base weights để inference.
- `last.pt`: LoRA + heads + optimizer/RNG/history để resume tại ranh giới epoch.
- `baseline_validation.json`, `history.json`, `summary.json`,
  `evaluation_test.json`: kết quả thực của full training/evaluation.

Train sẽ từ chối ghi đè `best.pt` nếu không có `--resume`. Resume yêu cầu cùng
weights, manifests, config và runtime; không đổi `--workers` giữa hai lần.
Nếu dừng giữa epoch, phần đó sẽ chạy lại từ checkpoint cuối epoch trước.
Nếu chưa hoàn thành epoch 1 thì chưa có `last.pt`: dùng output_dir mới để bắt
đầu lại, giữ nguyên folder cũ. Early-stopped run sẽ không tự train tiếp.

## Kiểm tra và giới hạn

Smoke run trên RTX 4060 Laptop 8 GB đã chạy 2 optimizer steps, LoRA weights
thay đổi, frozen weights không có gradients. Cấu hình micro-batch 1 đã đo
được khoảng 1,07 GiB PyTorch peak allocated / 1,22 GiB peak reserved.
Đây **không phải tổng VRAM của máy**: CUDA context, Windows desktop và ứng dụng
khác không nằm trong số PyTorch này. Xem `smoke_report.json` cho lần đo mới nhất.

LoRA giảm gradients/optimizer memory, không loại bỏ bộ nhớ của base weights
hay activations. Gradient cache/checkpointing đổi thêm compute lấy bộ nhớ;
full training chậm hơn pipeline cache features v2. Không suy ra thời gian
cả epoch hay chất lượng từ hai bước smoke.

Nếu muốn tăng throughput, thử `micro_batch_size: 2` hoặc `4`, giữ nguyên
`effective_batch_size: 16`, chạy lại `smoke` để đo VRAM trước full training.
Nếu OOM khi validation, giảm `runtime.eval_batch_size` về 1. Dùng output_dir
mới khi thay cấu hình của một run đã train. Đóng ứng dụng GPU không cần thiết.

Metric vẫn là trung bình bốn Recall@10 trên FULL gallery view/scene theo hai
chiều. Tập test vẫn 12.144 ảnh / 2.024 mẫu sáu camera; test không tham gia
optimizer hay checkpoint selection. Model lớn hơn + LoRA **không đảm bảo
score 0,8**. Chưa có kết quả full training/test v3 để so sánh với v2 = 0,394439.

Kiểm thử: 25 tests surround/LoRA pass, gồm gradient-cache so với gradient của
cả logical batch, LoRA không cập nhật base, missing-camera masks, lưu/nạp
adapter và resume. Toàn bộ suite: 56 tests pass; bộ nạp FAISS cũ vẫn in lỗi
native DLL trên Windows rồi dùng fallback. Pipeline LoRA không dùng FAISS.

Tài liệu gốc: [CLIP-L checkpoint](https://huggingface.co/openai/clip-vit-large-patch14),
[LoRA paper](https://arxiv.org/abs/2106.09685),
[Transformers CLIP](https://huggingface.co/docs/transformers/model_doc/clip).
