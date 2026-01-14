# Scripts - YOLO Model Download & Benchmark

## Tổng quan

Thư mục này chứa các script hỗ trợ cho việc quản lý và đánh giá các model YOLO trong dự án ITS (Intelligent Transport System).

## 📁 Files

- `download_and_benchmark.py` - Script chính để tải và benchmark các model YOLO

## 🚀 Hướng dẫn sử dụng

### 1. Cài đặt dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Xem danh sách models

```bash
python -m scripts.download_and_benchmark --list
```

Output:
```
Model        Name                 Size       Params     Downloaded
----------------------------------------------------------------------
yolov8n      YOLOv8 Nano         6.3        3.2        ❌ No        ⭐
yolov8s      YOLOv8 Small        22.5       11.2       ❌ No        ⭐
yolov8m      YOLOv8 Medium       52.0       25.9       ❌ No        ⭐
yolov12x     YOLOv12 XLarge      150.0      75.0       ✅ Yes       ⭐
...
```

### 3. Tải models

#### Tải các models được đề xuất cho tiểu luận:
```bash
python -m scripts.download_and_benchmark --download-recommended
```

Models đề xuất: `yolov8n`, `yolov8s`, `yolov8m`, `yolov12x`

#### Tải model cụ thể:
```bash
python -m scripts.download_and_benchmark --download yolov8n yolov8s
```

#### Tải tất cả models:
```bash
python -m scripts.download_and_benchmark --download-all
```

### 4. Benchmark models

#### Benchmark một model:
```bash
python -m scripts.download_and_benchmark --benchmark yolov8n --runs 100
```

#### Benchmark tất cả models đã tải:
```bash
python -m scripts.download_and_benchmark --benchmark-all --save
```

#### Tùy chỉnh benchmark:
```bash
python -m scripts.download_and_benchmark --benchmark yolov8n \
    --runs 100 \
    --warmup 20 \
    --image-size 640 \
    --save
```

### 5. Các options

| Option | Mô tả | Default |
|--------|-------|---------|
| `--list` | Liệt kê tất cả models | - |
| `--download MODEL [MODEL ...]` | Tải models cụ thể | - |
| `--download-recommended` | Tải models đề xuất | - |
| `--download-all` | Tải tất cả models | - |
| `--benchmark MODEL [MODEL ...]` | Benchmark models cụ thể | - |
| `--benchmark-all` | Benchmark tất cả models đã tải | - |
| `--runs` | Số lần chạy benchmark | 50 |
| `--warmup` | Số lần warmup trước benchmark | 10 |
| `--image-size` | Kích thước ảnh input | 640 |
| `--save` | Lưu kết quả ra file | False |
| `--force` | Buộc tải lại model | False |

## 📊 Output

### Kết quả benchmark được lưu tại:
- `backend/results/benchmarks/benchmark_YYYYMMDD_HHMMSS.json` - Raw data
- `backend/results/benchmarks/comparison_YYYYMMDD_HHMMSS.md` - Bảng so sánh Markdown

### Ví dụ output:

```
============================================================
 Benchmark Summary
============================================================

# Model Comparison Results

Generated: 2024-01-15 10:30:00
Device: cuda

| Model | Params (M) | FPS | Expected FPS | Inference (ms) | Memory (MB) |
|-------|-----------|-----|--------------|----------------|-------------|
| YOLOv8 Nano | 3.2 | 185.3 | 195 | 5.40 | 256 |
| YOLOv8 Small | 11.2 | 138.7 | 142 | 7.21 | 512 |
| YOLOv12 XLarge | 75.0 | 48.2 | 50 | 20.75 | 2048 |
```

## 🌐 API Endpoints

Ngoài script, bạn cũng có thể sử dụng API endpoints:

### List models
```
GET /api/benchmark/models
```

### Download model
```
POST /api/benchmark/models/download
Body: { "model_key": "yolov8n" }
```

### Switch model
```
POST /api/benchmark/models/switch
Body: { "model_key": "yolov8n" }
```

### Run benchmark
```
POST /api/benchmark/run
Body: { "model_key": "yolov8n", "benchmark_runs": 50 }
```

### Benchmark all
```
POST /api/benchmark/run/all
Body: { "benchmark_runs": 50 }
```

### Get comparison
```
GET /api/benchmark/comparison
```

## 🖥️ Frontend

Truy cập trang Benchmark trong frontend:
```
http://localhost:5173/benchmark
```

Trang này cho phép:
- Xem danh sách models
- Tải models trực tiếp
- Chuyển đổi model đang sử dụng
- Chạy benchmark và xem kết quả
- So sánh hiệu năng các models

## 📝 Gợi ý cho Tiểu luận

### Models nên so sánh:
1. **YOLOv8 Nano** - Nhẹ nhất, nhanh nhất
2. **YOLOv8 Small** - Cân bằng tốt
3. **YOLOv8 Medium** - Độ chính xác cao hơn
4. **YOLOv12 XLarge** - Mới nhất, chính xác nhất (đã có)

### Metrics để báo cáo:
- **FPS** (Frames Per Second) - Tốc độ xử lý
- **Inference Time** (ms) - Thời gian suy luận
- **Memory Usage** (MB) - Bộ nhớ sử dụng
- **mAP@0.5** - Độ chính xác (từ expected values)
- **Parameters** (M) - Kích thước model

### Cấu trúc báo cáo đề xuất:
1. Giới thiệu các phương pháp detection
2. Môi trường thực nghiệm (RTX 4050, CUDA)
3. Bảng so sánh các model
4. Phân tích trade-off: Tốc độ vs Độ chính xác
5. Kết luận và lựa chọn model phù hợp

## ❓ Troubleshooting

### CUDA not available
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Model download fails
- Kiểm tra kết nối internet
- Dùng `--force` để tải lại

### Memory error
- Giảm `--image-size` xuống 416 hoặc 320
- Unload các models không dùng qua API `/api/benchmark/models/unload`
