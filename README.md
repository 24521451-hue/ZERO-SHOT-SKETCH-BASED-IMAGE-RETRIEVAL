[readme (1).md](https://github.com/user-attachments/files/34084528/readme.1.md)
# ZERO-SHOT-SKETCH-BASED-IMAGE-RETRIEVAL# Zero-Shot & Generalized Sketch-Based Image Retrieval (CS331)

Kiến trúc **Asymmetric Hybrid LoRA-CLIP kết hợp Semantic Consistency & Adaptive OOD Detection** cho bài toán Truy vấn ảnh từ nét vẽ tay (Sketch-to-Image Retrieval) trên bộ dữ liệu **Sketchy-Extended** (73,002 ảnh tự nhiên và 75,471 ảnh phác họa thuộc 125 danh mục).

## 1. Cấu trúc Repository Chuẩn (Modular Architecture)

Toàn bộ mã nguồn được tách biệt thành 5 mô-đun độc lập theo đúng chuẩn phát triển phần mềm AI:

| Tên File | Vai trò & Trách nhiệm Kỹ thuật | Phân công Phụ trách |
| :--- | :--- | :--- |
| `config.py` | Quản lý tập trung toàn bộ siêu tham số (`SBIRConfig`), danh sách chuẩn 21 lớp Unseen (`SKETCHY_SPLIT2_UNSEEN_21`), hằng số chuẩn hóa CLIP và hàm khóa seed tái lập kết quả (`set_reproducible_seed`). | Dùng chung (Shared Interface) |
| `dataset.py` | Cài đặt `SketchCVPreprocessor` (chuyển đổi RGBA an toàn, `Tight-Crop + Symmetric Padding 1:1`, `Morphological Dilation/Erosion`, `White Random Erasing`), trình quét 2 thư mục `SketchyExtendedSplitManager` và `SeCoLoRATripletDataset`. | Thành viên B (Data & CV Lead) |
| `models.py` | Định nghĩa `LowRankLinearAdapter` (LoRA cho tầng Self-Attention), `BottleneckResidualAdapter` (khởi tạo 0 bảo toàn không gian CLIP ở Epoch 0), kiến trúc `AsymmetricHybridLoRACLIP` và hàm sinh `build_clip_text_prototypes`. | Thành viên A (Model Lead) |
| `losses.py` | Cài đặt hàm mục tiêu đa thành phần `SeCoLoRAMultiObjectiveLoss` kết hợp 4 hàm mất mát: $\mathcal{L}_{\text{triplet}}$, $\mathcal{L}_{\text{NT-Xent}}$, $\mathcal{L}_{\text{class}}$ (Text Semantic Prototype) và $\mathcal{L}_{\text{consist}}$ (Asymmetric Perturbation InfoNCE). | Thành viên A (Optimization Lead) |
| `evaluate.py` | Bộ đo lường vector hóa tốc độ cao cho Category-level (`mAP@all`, `mAP@200`, `P@100`, `P@200`), Fine-Grained (`Acc@1`, `Acc@5`, `Acc@10`) và bộ lọc ngoại lai `AdaptiveOODDetector` (`AUROC`, `FPR@95TPR`). | Thành viên B (Eval & OOD Lead) |
| `train.py` | Điểm vào dòng lệnh (CLI Entrypoint) điều phối toàn bộ quy trình: kiểm thử tự động (`--self-test`), đánh giá Baseline 0 (`--eval-baseline-only`), huấn luyện Mixed Precision (`fp16`) và lưu checkpoint tốt nhất. | Tích hợp chung |

## 2. Hướng dẫn Cài đặt & Khởi chạy trên Kaggle / Google Colab

### Bước 1: Clone Repository & Cài đặt Thư viện

```bash
git clone <URL_GITHUB_CỦA_NHÓM>
cd <TÊN_REPO>
pip install -r requirements.txt
```

### Bước 2: Chạy Kiểm thử Tự động (Unit Verification)

```bash
python3 train.py --self-test
```

### Bước 3: Chạy Đánh giá Baseline 0 (Frozen CLIP ViT-B/32)

```bash
python3 train.py \
  --data-root /kaggle/input/sketchy-extended \
  --photo-subdir photo \
  --sketch-subdir sketch \
  --split split2 \
  --eval-baseline-only
```

### Bước 4: Huấn luyện Toàn diện Mô hình Đề xuất (Asymmetric LoRA-CLIP + 4 Losses)

```bash
python3 train.py \
  --data-root /kaggle/input/sketchy-extended \
  --photo-subdir photo \
  --sketch-subdir sketch \
  --split split2 \
  --epochs 15 \
  --batch-size 64 \
  --lr 2e-5 \
  --lora-rank 8 \
  --lambda-cls 1.0 \
  --lambda-consist 1.0
```

## 3. Tài liệu Tham khảo Học thuật Nền tảng

1. Sangkloy et al., *"The Sketchy Database: Learning to Retrieve Badly Drawn Bunnies"*, ACM SIGGRAPH 2016.
2. Yelamarthi et al., *"A Zero-Shot Framework for Sketch Based Image Retrieval"*, ECCV 2018 (Giao thức Split 2: 104 Seen / 21 Unseen).
3. Sain et al., *"CLIP for All Things Zero-Shot Sketch-Based Image Retrieval, Fine-Grained or Not"* (CLIP-AT), CVPR 2023.
4. Koley et al., *"SketchFusion: Learning Universal Sketch Features through Fusing Foundation Models"*, CVPR 2025.
5. Long et al., *"SeCo-SBIR: Semantically Consistent Prompt Learning for Zero-Shot Sketch-Based Image Retrieval"*, ACM MM 2026.
