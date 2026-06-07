# Báo Cáo Repository Template

**Ngày tạo**: 2026-06-07  
**Phiên bản**: 1.0.0  
**Mục đích**: Template tái sử dụng cho Kaggle, nghiên cứu ML, và huấn luyện LLM

---

## 1. Tổng Quan

Repository này là một template nghiên cứu hoàn chỉnh, được thiết kế để:

- Hỗ trợ các cuộc thi Kaggle từ đầu đến cuối
- Theo dõi thí nghiệm một cách có hệ thống
- Lưu trữ kiến thức lâu dài (không phụ thuộc vào lịch sử chat)
- Chạy được trên máy cục bộ (CPU-only) và trên Kaggle (2x T4 GPU)
- Tích hợp với Claude Code và các LLM agent (OpenRouter)

---

## 2. Cấu Trúc Thư Mục

```
repo/
├── competition/          # Thông tin cuộc thi
├── data/
│   ├── raw/              # Dữ liệu gốc (không chỉnh sửa)
│   ├── external/         # Dữ liệu bên ngoài
│   └── processed/        # Dữ liệu đã xử lý
├── notebooks/            # Notebook Jupyter
├── src/                  # Source code Python
│   ├── features/         # Feature engineering
│   ├── models/           # Model wrappers
│   └── utils/            # Tiện ích
├── research/             # Kế hoạch và phát hiện nghiên cứu
├── memory/               # Hệ thống ghi nhớ lâu dài
├── references/           # Tài liệu tham khảo
├── prompts/              # Prompt cho LLM agent
├── results/              # Kết quả thí nghiệm
├── skills/               # Claude Code custom skills
├── mcp/                  # Cấu hình MCP server
└── docs/                 # Tài liệu
```

**Tổng số file**: 62 (bao gồm thư mục)  
**File code và tài liệu**: 31

---

## 3. Các File Đã Tạo

### 3.1 competition/

| File | Mô tả |
|------|-------|
| `competition_info.md` | Thông tin cơ bản cuộc thi: tên, URL, loại bài toán, giải thưởng |
| `dataset_info.md` | Mô tả dataset: cột, kiểu dữ liệu, phân phối, missing values |
| `evaluation.md` | Metric đánh giá, baseline scores, cách tính metric cục bộ |

### 3.2 research/

| File | Mô tả |
|------|-------|
| `current_plan.md` | Sprint hiện tại, priority queue, trạng thái thí nghiệm |
| `experiment_queue.md` | Danh sách thí nghiệm với đầy đủ template (objective, hypothesis, risk...) |
| `findings.md` | Ghi chép toàn bộ phát hiện (append-only, không xóa) |
| `hypothesis.md` | Log giả thuyết: chưa kiểm tra, đã xác nhận, đã bác bỏ |
| `open_questions.md` | Câu hỏi đang mở, phân loại theo mức độ ưu tiên |

### 3.3 memory/

| File | Mô tả |
|------|-------|
| `previous_runs.md` | Lịch sử toàn bộ lần chạy: config, CV score, LB score |
| `failed_attempts.md` | Những gì đã thất bại và lý do — phải đọc trước khi bắt đầu |
| `leaderboard_progress.md` | Theo dõi thứ hạng và điểm theo thời gian |

### 3.4 references/

| File | Mô tả |
|------|-------|
| `useful_links.md` | Link cuộc thi, notebook công khai, thư viện |
| `kaggle_discussions.md` | Thread forum quan trọng, insights đã rút ra |
| `papers.md` | Bài báo khoa học liên quan, phương pháp áp dụng |
| `repos.md` | Repository tham khảo, code nguồn hữu ích |

### 3.5 prompts/

| File | Mô tả |
|------|-------|
| `planner_prompt.md` | Prompt lên kế hoạch thí nghiệm cho LLM |
| `executor_prompt.md` | Prompt thực thi thí nghiệm cụ thể |
| `reviewer_prompt.md` | Prompt phân tích kết quả và đề xuất bước tiếp theo |

### 3.6 notebooks/

| File | Runtime CPU | Mô tả |
|------|-------------|-------|
| `01_eda.ipynb` | < 2 phút | EDA: missing values, distributions, correlations, train/test so sánh |
| `02_training.ipynb` | < 5 phút | 5-fold CV, OOF predictions, lưu kết quả tự động |
| `03_inference.ipynb` | < 1 phút | Blend predictions từ nhiều thí nghiệm |
| `04_submission.ipynb` | < 1 phút | Tạo submission.csv có timestamp |

### 3.7 src/

| File | Mô tả |
|------|-------|
| `features/feature_engineering.py` | Ratio features, aggregation features, lag features, label encoding |
| `models/lgb_model.py` | LightGBM wrapper với 5-fold CV, OOF, test preds, config lưu tự động |
| `models/accelerate_trainer.py` | Template huấn luyện multi-GPU với HuggingFace Accelerate |
| `utils/metrics.py` | Registry metric: AUC, LogLoss, RMSE, MAE, F1, MAP |
| `utils/reproducibility.py` | Set seed (Python, NumPy, PyTorch), log môi trường |

---

## 4. Quy Trình Làm Việc

### 4.1 Workflow Nghiên Cứu (mỗi phiên)

```
Bắt đầu phiên
  ├── Đọc memory/failed_attempts.md    → tránh lặp lỗi cũ
  ├── Đọc research/current_plan.md     → tiếp tục từ chỗ dừng
  └── Chọn EXP từ experiment_queue.md

Chạy thí nghiệm (02_training.ipynb)
  ├── Ghi kết quả → research/experiment_queue.md
  ├── Ghi phát hiện → research/findings.md
  └── Cập nhật → memory/previous_runs.md

Nộp bài (nếu có)
  └── Cập nhật → memory/leaderboard_progress.md

Kết thúc phiên
  ├── Cập nhật → research/current_plan.md
  └── Ghi câu hỏi mới → research/open_questions.md
```

### 4.2 Workflow Planner / Executor

1. **Planner** — Dùng `prompts/planner_prompt.md` + nội dung file hiện tại → LLM đề xuất thí nghiệm tiếp theo
2. **Executor** — Dùng `prompts/executor_prompt.md` + spec thí nghiệm → LLM tạo code hoàn chỉnh
3. **Reviewer** — Dùng `prompts/reviewer_prompt.md` + kết quả → LLM phân tích, cập nhật hướng nghiên cứu

> **Quan trọng**: Luôn paste nội dung file trực tiếp vào prompt. Không dựa vào bộ nhớ chat của LLM.

### 4.3 Workflow Kaggle

| Bước | Môi trường | Config |
|------|-----------|--------|
| EDA cục bộ | CPU | SAMPLE_SIZE = 1000 |
| Validation cục bộ | CPU | SAMPLE_SIZE = 500, RF thay LGB |
| Training đầy đủ | Kaggle 2x T4 | SAMPLE_SIZE = -1, LGB hoặc Accelerate |
| Blend + Submit | CPU | Dùng file .npy đã lưu |

---

## 5. Quy Tắc Bắt Buộc

### Compute

- **Cục bộ**: CPU only. Tất cả notebook phải chạy < 30 phút (mục tiêu < 5 phút).
- **Kaggle**: 2x NVIDIA T4. Dùng Accelerate/DDP cho neural network. LGB/XGB không cần.
- **Không giả định GPU cục bộ**.

### Thí Nghiệm

Mỗi thí nghiệm trong `experiment_queue.md` phải có đủ 7 trường:
1. Objective
2. Hypothesis
3. Expected gain
4. Risk
5. Compute estimate
6. Status
7. Result + Notes (sau khi chạy)

### Bộ Nhớ

- Mọi phát hiện → ghi vào file ngay
- Không xóa lịch sử cũ
- Không dựa vào chat history qua phiên làm việc
- File `memory/failed_attempts.md` phải được đọc trước mỗi thí nghiệm mới

### Tái Tạo (Reproducibility)

- Gọi `set_seed()` ở đầu mỗi script training
- Lưu config JSON cùng với OOF/test predictions
- Ghi phiên bản thư viện trong log

---

## 6. Kết Quả Kiểm Tra

### Notebook CPU Test

| Notebook | Kết quả | Thời gian |
|----------|---------|-----------|
| 01_eda.ipynb | PASSED (valid nbformat4, 11 cells) | — |
| 02_training.ipynb | PASSED (valid nbformat4, 7 cells) | — |
| 03_inference.ipynb | PASSED (valid nbformat4, 5 cells) | — |
| 04_submission.ipynb | PASSED (valid nbformat4, 5 cells) | — |
| Core training logic (sklearn RF, 5-fold, n=500) | PASSED | **0.4 giây** |

### Cấu Trúc Thư Mục

Tất cả thư mục theo CLAUDE.md đã được tạo:

- [x] competition/
- [x] data/raw/, data/external/, data/processed/
- [x] notebooks/
- [x] src/features/, src/models/, src/utils/
- [x] research/
- [x] memory/
- [x] references/
- [x] prompts/
- [x] results/experiments/, results/submissions/, results/plots/
- [x] skills/
- [x] mcp/
- [x] docs/

### File Bắt Buộc

Tất cả file trong CLAUDE.md đã được tạo với nội dung có ý nghĩa:

- [x] competition/competition_info.md
- [x] competition/dataset_info.md
- [x] competition/evaluation.md
- [x] research/current_plan.md
- [x] research/experiment_queue.md
- [x] research/findings.md
- [x] research/hypothesis.md
- [x] research/open_questions.md
- [x] memory/previous_runs.md
- [x] memory/failed_attempts.md
- [x] memory/leaderboard_progress.md
- [x] references/useful_links.md
- [x] references/kaggle_discussions.md
- [x] references/papers.md
- [x] references/repos.md
- [x] prompts/planner_prompt.md
- [x] prompts/executor_prompt.md
- [x] prompts/reviewer_prompt.md
- [x] README.md
- [x] PLAN.md (có sẵn)
- [x] CLAUDE.md (có sẵn)

---

## 7. Cách Sử Dụng Template Này

### Bắt đầu cuộc thi mới

1. Clone/copy repo này
2. Xóa file `.gitkeep` trong `data/`
3. Điền thông tin vào `competition/competition_info.md`
4. Copy dữ liệu vào `data/raw/`
5. Đặt tên thí nghiệm đầu tiên trong `research/experiment_queue.md`
6. Chạy `notebooks/01_eda.ipynb`

### Tiếp tục sau thời gian dừng

1. Đọc `research/current_plan.md`
2. Đọc `memory/failed_attempts.md`
3. Đọc `research/findings.md`
4. Tiếp tục từ EXP tiếp theo trong `research/experiment_queue.md`

### Dùng với Claude Code

Mở repo bằng Claude Code. CLAUDE.md sẽ được đọc tự động. Paste nội dung file vào prompt khi cần — không dựa vào bộ nhớ chat.

---

## 8. Phụ Thuộc

### Tối thiểu (CPU)

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
matplotlib>=3.7
```

### Tabular ML

```
lightgbm>=4.0
xgboost>=2.0
optuna>=3.0
```

### Neural Networks (Kaggle)

```
torch>=2.0
accelerate>=0.25
transformers>=4.35
```

### Tiện ích

```
jupyter>=1.0
polars>=0.19   # thay thế pandas khi dataset lớn
```

---

*Báo cáo được tạo tự động bởi Claude Code — 2026-06-07*
