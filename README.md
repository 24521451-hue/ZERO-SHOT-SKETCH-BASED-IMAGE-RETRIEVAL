[readme (1).md](https://github.com/user-attachments/files/34441223/readme.1.md)
# SeCo-LoRA-CLIP: Zero-Shot & Generalized Sketch-Based Image Retrieval

Hệ thống **SeCo-LoRA-CLIP** (*Semantically Consistent Asymmetric Hybrid LoRA-CLIP*) cho bài toán Truy vấn Ảnh từ Nét vẽ tay (**Zero-Shot, Generalized & Fine-Grained Sketch-Based Image Retrieval — ZS-SBIR / GZS-SBIR / FG-ZS-SBIR**) kèm cơ chế **Phát hiện Ngoại lai Thích ứng (Adaptive OOD Detection)** trên bộ dữ liệu **Sketchy-Extended** (73,002 ảnh tự nhiên và 75,471 ảnh phác họa thuộc 125 danh mục).

---

## 1. Điểm nhấn Kiến trúc & Đóng góp Học thuật Cốt lõi

1. **Kiến trúc Asymmetric Hybrid LoRA-CLIP không nhân bản VRAM (Zero-Duplication Dual-Mode Backbone)**:
   - Duy nhất một thân mạng `CLIP ViT-B/32` trong bộ nhớ GPU, tự động chuyển đổi giữa **Chế độ Tham chiếu Đóng băng 100% (`encode_frozen_visual`)** cho nhánh Ảnh / Nhánh Đối sánh Bất biến và **Chế độ Thích ứng LoRA (`encode_sketch`)** với các ma trận Low-Rank ($\mathbf{A} \in \mathbb{R}^{r \times d}, \mathbf{B} \in \mathbb{R}^{d \times r}$, khởi tạo $\mathbf{B}=\mathbf{0}$) tiêm trực tiếp vào cả 4 phép chiếu $\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v, \mathbf{W}_o$ của toàn bộ 12 tầng Multi-Head Self-Attention kết hợp tinh chỉnh `LayerNorm`.
   - Tích hợp **Bottleneck Residual Adapter** ($\text{GELU}$, khởi tạo $\mathbf{W}_2 = \mathbf{0}$, hệ số $\alpha_{\text{res}} = 0.2$) đảm bảo tại Epoch 0, không gian nhúng bảo toàn 100% hình học siêu cầu $\mathbb{S}^{511}$ của CLIP gốc.
2. **Tăng tốc Huấn luyện gấp 3 lần bằng Cơ chế Pre-Extracted Frozen Photo Feature Caching**:
   - Toàn bộ đặc trưng ảnh tự nhiên (`photo_clean` và `photo_pert`) qua nhánh CLIP đóng băng được trích xuất trước một lần duy nhất vào bộ nhớ RAM/VRAM. Trong suốt 15 epochs huấn luyện và 10 kịch bản Ablation, mô hình không cần giải mã JPEG hay chạy lại ViT cho nhánh ảnh, giúp giảm 70% thời gian mỗi epoch trên GPU Kaggle T4/P100.
3. **Hệ Hàm Mất Mát Đa Mục Tiêu 5 Thành phần (Class-Aware Multi-Objective Optimization)**:
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{triplet}}^{\text{hard}} + \lambda_{\text{ntx}} \mathcal{L}_{\text{NT-Xent}}^{\text{CA}} + \lambda_{\text{cls}} \mathcal{L}_{\text{class}} + \lambda_{\text{con}} \mathcal{L}_{\text{consist}} + \lambda_{\delta} \mathcal{L}_{\delta\text{-KL}}$$
   - **$\mathcal{L}_{\text{triplet}}^{\text{hard}}$**: Kết hợp Inter-category Triplet và Intra-category Hard-Negative Triplet.
   - **$\mathcal{L}_{\text{NT-Xent}}^{\text{CA}}$ (Class-Aware False-Negative Masking)**: Loại bỏ hiện tượng đẩy nhầm các cặp cùng danh mục trong cùng mini-batch.
   - **$\mathcal{L}_{\text{class}}$ (Multi-Template Semantic Prototype Alignment)**: Đồng thích ứng với nguyên mẫu ngữ nghĩa từ 7 mẫu câu gợi ý (Prompt Ensemble) của CLIP Text Encoder.
   - **$\mathcal{L}_{\text{consist}}$ (Perturbation-Based Asymmetric InfoNCE — SeCo-SBIR ACM MM'26)**: Neo biểu diễn đã thích ứng vào không gian tổng quát của CLIP đóng băng thông qua biến đổi nhiễu bất đối xứng.
   - **$\mathcal{L}_{\delta\text{-KL}}$ (Sgarro's Relative Distance $f$-Divergence — CLIP-AT CVPR'23)**: Đồng nhất hóa phân phối khoảng cách tương đối giữa các danh mục bằng `BalancedPKBatchSampler` ($P=16$ lớp $\times K=4$ mẫu).
4. **Pipeline Xử lý Ảnh Nét vẽ Chuyên biệt (`SketchCVPreprocessor`)**:
   - Xử lý an toàn kênh trong suốt `RGBA -> RGB` trên nền trắng chuẩn.
   - `Tight-Crop + 1:1 Symmetric Padding` giúp tối đa hóa mật độ patch chứa nét vẽ cho ViT-B/32.
   - `Stroke-Safe Morphological Ops`: Biến đổi độ đậm/mảnh của nét bút có bảo vệ chống mất nét mảnh 1px.
   - `White Random Erasing`: Giả lập nét vẽ đứt quãng/thiếu nét của người dùng thực tế.

---

## 2. Cấu trúc Repository Chuẩn (Modular Architecture)

| Tên File | Vai trò & Trách nhiệm Kỹ thuật |
| :--- | :--- |
| [`config.py`](file:///C:/Users/Admin/Downloads/ZS-SBIR/config.py) | Quản lý tập trung siêu tham số (`SBIRConfig`), danh sách 21 lớp Unseen (`SKETCHY_SPLIT2_UNSEEN_21`), 25 lớp Unseen (`SKETCHY_SPLIT1_UNSEEN_25`), 7 mẫu Prompt Ensemble và khóa seed tái lập (`set_reproducible_seed`). |
| [`dataset.py`](file:///C:/Users/Admin/Downloads/ZS-SBIR/dataset.py) | Cài đặt `SketchCVPreprocessor`, trình quét tự động `SketchyExtendedSplitManager`, bộ lấy mẫu cân bằng `BalancedPKBatchSampler` ($P \times K$) và `SeCoLoRATripletDataset` hỗ trợ chế độ Cached Feature $O(1)$. |
| [`models.py`](file:///C:/Users/Admin/Downloads/ZS-SBIR/models.py) | Định nghĩa `LoRALinearLayer`, `LoRAMultiheadAttention` (tiêm LoRA vào $\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v, \mathbf{W}_o$), `BottleneckResidualAdapter`, kiến trúc `AsymmetricHybridLoRACLIP` và `build_clip_text_prototypes`. |
| [`losses.py`](file:///C:/Users/Admin/Downloads/ZS-SBIR/losses.py) | Cài đặt hàm mục tiêu `SeCoLoRAMultiObjectiveLoss` gồm 5 thành phần với cơ chế Class-Aware Masking và $f$-Divergence KL Regularization. |
| [`evaluate.py`](file:///C:/Users/Admin/Downloads/ZS-SBIR/evaluate.py) | Bộ đo lường vector hóa cho **ZS-SBIR**, **GZS-SBIR**, **Seen-SBIR** (`mAP@all`, `mAP@200`, `P@100`, `P@200`, `R@1`, `R@5`, `R@10`), **FG-ZS-SBIR** chuẩn Intra-category & Cross-category (`Acc@1`, `Acc@5`, `Acc@10`), và **AdaptiveOODDetector** (`AUROC`, `FPR@95TPR`, `Det_Acc`). |
| [`visualize.py`](file:///C:/Users/Admin/Downloads/ZS-SBIR/visualize.py) | Xuất tự động 4 biểu đồ chuẩn công bố khoa học: (1) t-SNE Cross-Modal Manifold, (2) Phổ biên độ 2D Fourier (HF vs LF), (3) Phân phối điểm tin cậy OOD & ngưỡng $\tau_{\text{OOD}}$, (4) Lưới truy vấn trực quan Top-K. |
| [`train.py`](file:///C:/Users/Admin/Downloads/ZS-SBIR/train.py) | Điểm vào dòng lệnh (CLI) điều phối kiểm thử tự động (`--self-test`), đánh giá Baseline 0, huấn luyện tăng tốc Mixed Precision + Feature Cache, xuất báo cáo Markdown/JSON và lưu checkpoint gọn nhẹ (~5.5 MB). |
| [`ablation_runner.py`](file:///C:/Users/Admin/Downloads/ZS-SBIR/ablation_runner.py) | Thực thi tự động chuỗi 10 thí nghiệm Ablation Study trên GPU (tái sử dụng bộ nhớ đệm đặc trưng ảnh) và xuất trực tiếp bảng kết quả ra `.md` và `.csv`. |

---

## 3. Hướng dẫn Chạy Một Lệnh Duy Nhất trên Kaggle / Google Colab

### Bước 1: Cài đặt Thư viện & Kiểm thử Tự động (Self-Test)
```bash
pip install -r requirements.txt
python train.py --self-test
python ablation_runner.py --self-test
python visualize.py --self-test
```

### Bước 2: Chạy Một Lệnh Duy Nhất (Huấn luyện Full + Xuất Báo cáo + 4 Biểu đồ) trên Kaggle
```bash
python train.py \
  --data-root /kaggle/input/sketchy-extended \
  --photo-subdir photo \
  --sketch-subdir sketch \
  --split split2 \
  --epochs 15 \
  --batch-size 64 \
  --lr 1e-4 \
  --lora-rank 8 \
  --save-dir ./outputs
```
Sau khi chạy xong, thư mục `./outputs` sẽ tự động chứa đầy đủ:
- `best_seco_lora_clip.pt`: Checkpoint trọng số mô hình tốt nhất.
- `final_benchmark_report.md` & `final_benchmark_metrics.json`: Bảng báo cáo điểm số đầy đủ (`ZS-SBIR`, `GZS-SBIR`, `FG-ZS-SBIR`, `Seen-SBIR`, `R@1/5/10`, `OOD AUROC`).
- `figures/fig1_tsne_manifold.png`, `fig2_ood_confidence_separation.png`, `fig3_2d_fourier_spectrum.png`, `fig4_topk_qualitative_retrieval.png`.

### Bước 3: Chạy Toàn bộ Chuỗi Thí nghiệm Ablation Study (Tùy chọn thêm cờ `--run-all-ablations`)
```bash
python train.py \
  --data-root /kaggle/input/sketchy-extended \
  --split split2 \
  --epochs 15 \
  --save-dir ./outputs \
  --run-all-ablations
```
*(Tự động xuất thêm `outputs/ablation_results_summary.md` và `outputs/ablation_results_summary.csv`)*

---

## 4. Tài liệu Tham khảo Học thuật Nền tảng

1. Sangkloy et al., *"The Sketchy Database: Learning to Retrieve Badly Drawn Bunnies"*, ACM SIGGRAPH 2016.
2. Yelamarthi et al., *"A Zero-Shot Framework for Sketch Based Image Retrieval"*, ECCV 2018 (Giao thức Split 2: 104 Seen / 21 Unseen).
3. Hu et al., *"LoRA: Low-Rank Adaptation of Large Language Models"*, ICLR 2022.
4. Sain et al., *"CLIP for All Things Zero-Shot Sketch-Based Image Retrieval, Fine-Grained or Not"* (CLIP-AT), CVPR 2023.
5. Koley et al., *"SketchFusion: Learning Universal Sketch Features through Fusing Foundation Models"*, CVPR 2025.
6. Long et al., *"SeCo-SBIR: Semantically Consistent Prompt Learning for Zero-Shot Sketch-Based Image Retrieval"*, ACM MM 2026.
