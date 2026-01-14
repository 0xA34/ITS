# Baseline Evaluation Module

## Mô-đun Đánh giá Cơ sở cho Dự án Khảo sát Vehicle Detection & Monitoring

Module này cung cấp các công cụ để đánh giá baseline cho hệ thống phát hiện và giám sát phương tiện giao thông.

---

## 📁 Cấu trúc Thư mục

```
baseline/
├── __init__.py              # Module initialization
├── config.py                # Configuration settings
├── run_baseline.py          # Main evaluation runner
├── README.md                # This file
│
├── datasets/                # Ground truth dataset loaders
│   ├── __init__.py
│   ├── base_dataset.py      # Base dataset class
│   ├── coco_loader.py       # COCO format loader
│   └── custom_loader.py     # Custom dataset loader (COCO/YOLO/VOC)
│
├── metrics/                 # Evaluation metrics
│   ├── __init__.py
│   ├── iou.py               # IoU calculation (IoU, GIoU, DIoU, CIoU)
│   ├── precision_recall.py  # Precision, Recall, F1-Score
│   ├── map_calculator.py    # mAP calculation (mAP@0.5, mAP@[.5:.95])
│   └── fps_benchmark.py     # FPS/Latency benchmarking
│
├── comparison/              # Model comparison tools
│   ├── __init__.py
│   ├── model_registry.py    # Model registry and management
│   ├── reference_database.py # Reference methods database
│   └── comparison_report.py # Report generation (MD, HTML, JSON)
│
└── results/                 # Output directory for evaluation results
```

---

## 🚀 Hướng dẫn Sử dụng

### 1. Cài đặt Dependencies

```bash
pip install ultralytics numpy opencv-python pydantic
```

### 2. Chuẩn bị Ground Truth Dataset

#### Tải COCO Dataset (Khuyến nghị)

```bash
# Download COCO val2017 (5000 images)
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/zips/val2017.zip

# Extract
unzip annotations_trainval2017.zip -d data/coco
unzip val2017.zip -d data/coco
```

#### Hoặc tạo Custom Dataset

```python
from app.baseline.datasets.custom_loader import create_empty_dataset

# Tạo dataset trống với cấu trúc COCO
create_empty_dataset(
    output_path="data/custom",
    annotation_format="coco",
    class_names=["person", "bicycle", "car", "motorcycle", "bus", "truck"]
)
```

### 3. Chạy Baseline Evaluation

#### Sử dụng Command Line

```bash
# Chạy đầy đủ evaluation
python -m app.baseline.run_baseline --model yolov8n.pt --dataset data/coco --output results/

# Chỉ chạy FPS benchmark
python -m app.baseline.run_baseline --model yolov8n.pt --benchmark-only

# Chạy với giới hạn số ảnh (để test nhanh)
python -m app.baseline.run_baseline --model yolov8n.pt --dataset data/coco --max-images 100
```

#### Sử dụng Python API

```python
from app.baseline.run_baseline import BaselineEvaluator, EvaluationConfig

# Cấu hình
config = EvaluationConfig(
    model_path="yolov8n.pt",
    dataset_path="data/coco",
    dataset_format="coco",
    device="cuda",
    benchmark_iterations=100,
)

# Chạy evaluation
evaluator = BaselineEvaluator(config)
results = evaluator.run_full_evaluation()

# Tạo báo cáo
evaluator.generate_report(results, output_path="results/baseline")
```

---

## 📊 Các Metrics được Hỗ trợ

### Accuracy Metrics

| Metric | Mô tả | Công thức |
|--------|-------|-----------|
| **Precision** | Tỷ lệ detection đúng | TP / (TP + FP) |
| **Recall** | Tỷ lệ phát hiện được | TP / (TP + FN) |
| **F1-Score** | Harmonic mean của P và R | 2 × P × R / (P + R) |
| **mAP@0.5** | Mean AP tại IoU=0.5 | Average of AP@0.5 across classes |
| **mAP@0.75** | Mean AP tại IoU=0.75 | Average of AP@0.75 across classes |
| **mAP@[.5:.95]** | COCO-style mAP | Average of AP@[0.5:0.95:0.05] |

### Speed Metrics

| Metric | Mô tả | Đơn vị |
|--------|-------|--------|
| **FPS** | Frames per second | frames/s |
| **Latency** | Thời gian xử lý 1 frame | ms |
| **Throughput** | Số frame xử lý / tổng thời gian | frames/s |

### IoU Variants

- **IoU** - Intersection over Union (cơ bản)
- **GIoU** - Generalized IoU (xử lý non-overlapping boxes)
- **DIoU** - Distance IoU (xét khoảng cách tâm)
- **CIoU** - Complete IoU (xét cả aspect ratio)

---

## 📈 Reference Methods Database

Module bao gồm database các phương pháp tham chiếu để so sánh:

### YOLO Family
- YOLOv8 (n, s, m, l, x)
- YOLOv7 (tiny, base, X)
- YOLOv5 (n, s, m, l)
- YOLO-NAS (S, M, L)

### R-CNN Family
- Faster R-CNN (ResNet-50/101 + FPN)
- Cascade R-CNN
- Mask R-CNN

### Other Methods
- SSD300/512
- DETR / Deformable DETR
- RT-DETR
- FCOS
- CenterNet
- EfficientDet
- NanoDet

```python
from app.baseline.comparison.reference_database import ReferenceDatabase

# Xem tất cả reference methods
ReferenceDatabase.print_summary()

# Lấy methods real-time (>30 FPS)
realtime_methods = ReferenceDatabase.get_realtime_methods(min_fps=30)

# Lấy methods có accuracy cao
accurate_methods = ReferenceDatabase.get_high_accuracy_methods(min_map=0.5)

# Lấy Pareto-optimal methods (best accuracy-speed trade-off)
pareto_methods = ReferenceDatabase.get_pareto_optimal()
```

---

## 📝 Output Reports

### Định dạng Output

1. **JSON** - Dữ liệu raw để xử lý tiếp
2. **Markdown** - Báo cáo có thể đọc được
3. **HTML** - Báo cáo với charts và styling

### Ví dụ Cấu trúc Báo cáo

```
results/baseline/
├── baseline_report_yolov8n.json      # Raw results
├── baseline_report_yolov8n.md        # Markdown report
├── baseline_report_yolov8n.html      # HTML report with charts
└── results_yolov8n.json              # Detailed evaluation results
```

---

## 🎯 Vehicle Classes

Các class phương tiện được đánh giá (COCO format):

| Class ID | Class Name | Mô tả |
|----------|------------|-------|
| 0 | person | Người đi bộ |
| 1 | bicycle | Xe đạp |
| 2 | car | Ô tô |
| 3 | motorcycle | Xe máy |
| 5 | bus | Xe buýt |
| 7 | truck | Xe tải |

---

## 🔧 Configuration Options

```python
@dataclass
class EvaluationConfig:
    # Model settings
    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "auto"  # 'cuda', 'cpu', 'auto'
    
    # Dataset settings
    dataset_path: str = ""
    dataset_format: str = "coco"  # 'coco', 'yolo', 'voc'
    dataset_split: str = "val"
    max_images: int = None  # Limit for quick testing
    
    # Evaluation settings
    iou_thresholds: list = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    evaluate_per_class: bool = True
    evaluate_per_size: bool = True
    
    # Benchmark settings
    benchmark_iterations: int = 100
    benchmark_warmup: int = 10
    image_size: tuple = (640, 640)
    
    # Output settings
    output_dir: str = "results/baseline"
    report_formats: list = ["json", "markdown", "html"]
```

---

## 📚 Examples

### Ví dụ 1: Quick Benchmark

```python
from app.baseline.metrics.fps_benchmark import benchmark_model

result = benchmark_model(
    model_path="yolov8n.pt",
    num_iterations=100,
    image_size=(640, 640),
    device="cuda"
)
print(result)
```

### Ví dụ 2: Calculate mAP

```python
from app.baseline.metrics.map_calculator import MAPCalculator
from app.baseline.metrics.precision_recall import Detection, GroundTruth

calculator = MAPCalculator(
    class_names={0: "person", 2: "car", 3: "motorcycle"}
)

# Add predictions and ground truths
calculator.add_prediction(bbox=(100, 100, 200, 200), class_id=2, confidence=0.9, image_id="img1")
calculator.add_ground_truth(bbox=(105, 105, 195, 195), class_id=2, image_id="img1")

# Calculate mAP
result = calculator.calculate_coco_map()
print(f"mAP@0.5: {result.map_50:.4f}")
print(f"mAP@[.5:.95]: {result.map_50_95:.4f}")
```

### Ví dụ 3: Compare Models

```python
from app.baseline.comparison.comparison_report import ComparisonReport
from app.baseline.comparison.reference_database import ReferenceDatabase

report = ComparisonReport(title="Vehicle Detection Comparison")

# Add your evaluated model
report.add_model_results(
    model_name="Our YOLOv8n",
    precision=0.85,
    recall=0.78,
)
report.models["Our YOLOv8n"].map_50 = 0.72
report.models["Our YOLOv8n"].fps = 150

# Add reference methods
for ref in ReferenceDatabase.YOLO_METHODS[:5]:
    report.add_reference_method(
        model_name=ref.name,
        map_50=ref.map_50,
        fps=ref.fps,
        source=ref.source
    )

# Generate reports
report.generate_report("results/comparison")
```

---

## 🔗 Related Documentation

- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [COCO Dataset](https://cocodataset.org/)
- [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
- [Object Detection Metrics](https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173)
