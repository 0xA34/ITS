# Baseline Evaluation Module

A comprehensive evaluation framework for assessing YOLO-based vehicle detection models used in the Intelligent Traffic System.

## Overview

The baseline evaluation module provides tools and utilities to evaluate object detection models against ground truth datasets. It supports multiple dataset formats, calculates standard detection metrics, performs FPS benchmarking, and generates detailed comparison reports.

## Features

- Ground truth dataset loading for COCO, YOLO, and VOC formats
- Comprehensive accuracy metrics: mAP@0.5, mAP@0.75, mAP@[.5:.95], Precision, Recall, F1-Score
- IoU variant calculations: IoU, GIoU, DIoU, CIoU
- FPS and latency benchmarking
- Model comparison and ranking
- Persistent benchmark history storage
- Multi-format report generation: JSON, Markdown, HTML
- Validation system with cross-validation and statistical analysis


## Installation

The baseline module dependencies are included in the main project requirements:

```bash
pip install -r requirements.txt
```

Additional dependencies:
```bash
pip install ultralytics numpy opencv-python pydantic shapely
```

## Usage

### Command Line Interface

Run full evaluation with a dataset:

```bash
python -m app.baseline.run_baseline --model yolov8n.pt --dataset data/coco --output results/
```

Run FPS benchmark only:

```bash
python -m app.baseline.run_baseline --model yolov8n.pt --benchmark-only
```

Run with limited images for quick testing:

```bash
python -m app.baseline.run_baseline --model yolov8n.pt --dataset data/coco --max-images 100
```

Specify device:

```bash
python -m app.baseline.run_baseline --model yolov8n.pt --device cuda
```

### Python API

```python
from app.baseline.run_baseline import BaselineEvaluator, EvaluationConfig

# Configure evaluation
config = EvaluationConfig(
    model_path="yolov8n.pt",
    dataset_path="data/coco",
    dataset_format="coco",
    device="cuda",
    benchmark_iterations=100,
    max_images=1000,
)

# Initialize evaluator
evaluator = BaselineEvaluator(config)

# Run full evaluation
results = evaluator.run_full_evaluation()

# Generate reports
evaluator.generate_report(results, output_path="results/baseline")
```

## Dataset Preparation

### COCO Dataset

Download and extract COCO val2017:

```bash
# Download annotations and images
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/zips/val2017.zip

# Extract
unzip annotations_trainval2017.zip -d data/coco
unzip val2017.zip -d data/coco
```

### Custom Dataset

Create a custom dataset in COCO format:

```python
from app.baseline.datasets.custom_loader import create_empty_dataset

create_empty_dataset(
    output_path="data/custom",
    annotation_format="coco",
    class_names=["person", "bicycle", "car", "motorcycle", "bus", "truck"]
)
```

## Supported Metrics

### Accuracy Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| Precision | Ratio of correct detections | TP / (TP + FP) |
| Recall | Ratio of detected ground truths | TP / (TP + FN) |
| F1-Score | Harmonic mean of precision and recall | 2 * P * R / (P + R) |
| mAP@0.5 | Mean Average Precision at IoU=0.5 | Average of AP@0.5 across classes |
| mAP@0.75 | Mean Average Precision at IoU=0.75 | Average of AP@0.75 across classes |
| mAP@[.5:.95] | COCO-style mAP | Average of AP@[0.5:0.95:0.05] |

### Speed Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| FPS | Frames per second | frames/s |
| Latency | Processing time per frame | ms |
| Throughput | Frames processed per total time | frames/s |

### IoU Variants

- **IoU** - Intersection over Union (standard)
- **GIoU** - Generalized IoU (handles non-overlapping boxes)
- **DIoU** - Distance IoU (considers center point distance)
- **CIoU** - Complete IoU (includes aspect ratio)

## Configuration Options

```python
@dataclass
class EvaluationConfig:
    # Model settings
    model_path: str = "yolov8n.pt"
    model_name: Optional[str] = None
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "auto"  # 'cuda', 'cpu', 'auto'
    
    # Dataset settings
    dataset_path: str = ""
    dataset_format: str = "coco"  # 'coco', 'yolo', 'voc'
    dataset_split: str = "val"
    max_images: Optional[int] = None
    
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
    save_predictions: bool = False
    save_visualizations: bool = False
```

## Benchmark History

The module stores benchmark results for comparison across multiple runs:

```python
from app.baseline.comparison.benchmark_history import BenchmarkHistory

# Initialize history
history = BenchmarkHistory()

# Save benchmark result
history.save_result(
    model_name="yolov8n",
    map_50=0.72,
    fps=150,
    latency_ms=6.7,
)

# Load all results
all_results = history.load_all()

# Get model names
model_names = history.get_model_names()

# Remove a result
history.remove_result("yolov8n")

# Clear all history
history.clear()
```

Benchmark results are automatically saved when running evaluations and displayed in comparison reports.

## Report Generation

The module generates three types of reports:

### JSON Report
Raw evaluation data for programmatic processing:
```
results/baseline/baseline_report_yolov8n.json
```

### Markdown Report
Human-readable report with tables and metrics:
```
results/baseline/baseline_report_yolov8n.md
```

### HTML Report
Interactive report with charts and styling:
```
results/baseline/baseline_report_yolov8n.html
```

## Vehicle Classes

The evaluation focuses on traffic-related classes from COCO:

| Class ID | Class Name | Description |
|----------|------------|-------------|
| 0 | person | Pedestrians |
| 1 | bicycle | Bicycles |
| 2 | car | Cars |
| 3 | motorcycle | Motorcycles |
| 5 | bus | Buses |
| 7 | truck | Trucks |

## Examples

### Quick FPS Benchmark

```python
from app.baseline.metrics.fps_benchmark import benchmark_model

result = benchmark_model(
    model_path="yolov8n.pt",
    num_iterations=100,
    image_size=(640, 640),
    device="cuda"
)
print(f"FPS: {result.fps:.2f}")
print(f"Latency: {result.avg_inference_time_ms:.2f}ms")
```

### Calculate mAP

```python
from app.baseline.metrics.map_calculator import MAPCalculator

calculator = MAPCalculator(
    class_names={0: "person", 2: "car", 3: "motorcycle"}
)

# Add predictions and ground truths
calculator.add_prediction(
    bbox=(100, 100, 200, 200),
    class_id=2,
    confidence=0.9,
    image_id="img1"
)
calculator.add_ground_truth(
    bbox=(105, 105, 195, 195),
    class_id=2,
    image_id="img1"
)

# Calculate mAP
result = calculator.calculate_coco_map()
print(f"mAP@0.5: {result.map_50:.4f}")
print(f"mAP@[.5:.95]: {result.map_50_95:.4f}")
```

### Compare Models

```python
from app.baseline.comparison.comparison_report import ComparisonReport

report = ComparisonReport(title="Vehicle Detection Comparison")

# Add evaluated model
report.add_model_results(
    model_name="Our YOLOv8n",
    precision=0.85,
    recall=0.78,
)
report.models["Our YOLOv8n"].map_50 = 0.72
report.models["Our YOLOv8n"].fps = 150

# Generate comparison report
report.generate_report("results/comparison")
```

### Custom Dataset Evaluation

```python
from app.baseline.datasets.custom_loader import CustomDatasetLoader

# Load custom dataset
dataset = CustomDatasetLoader(
    dataset_path="data/custom",
    annotation_format="yolo",
    class_mapping={0: "car", 1: "truck", 2: "bus"}
)

# Get annotations
annotations = dataset.get_all_annotations()
print(f"Loaded {len(annotations)} images")
```

## Output Structure

After running evaluation, results are organized as:

```
results/baseline/
├── baseline_report_yolov8n.json      # Raw results
├── baseline_report_yolov8n.md        # Markdown report
├── baseline_report_yolov8n.html      # HTML report with charts
└── results_yolov8n.json              # Detailed evaluation data
```

## Performance Expectations

### YOLOv8 Models on COCO

| Model | mAP@0.5 | mAP@[.5:.95] | FPS (GPU) | Latency (ms) |
|-------|---------|--------------|-----------|--------------|
| YOLOv8n | 37.3% | 27.0% | 100+ | ~10 |
| YOLOv8s | 44.9% | 32.8% | 80+ | ~15 |
| YOLOv8m | 50.2% | 37.4% | 60+ | ~25 |
| YOLOv8l | 52.9% | 39.8% | 45+ | ~35 |
| YOLOv8x | 53.9% | 41.0% | 30+ | ~50 |

Note: FPS values are approximate and depend on hardware configuration.

## Advanced Features

### Per-Class Evaluation

Enable per-class metrics to analyze model performance on individual vehicle types:

```python
config = EvaluationConfig(
    model_path="yolov8n.pt",
    dataset_path="data/coco",
    evaluate_per_class=True
)
```

### Per-Size Evaluation

Analyze performance on small, medium, and large objects:

```python
config = EvaluationConfig(
    model_path="yolov8n.pt",
    dataset_path="data/coco",
    evaluate_per_size=True
)
```

### Visualization

Save prediction visualizations with bounding boxes:

```python
config = EvaluationConfig(
    model_path="yolov8n.pt",
    dataset_path="data/coco",
    save_visualizations=True,
    output_dir="results/visualizations"
)
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or use a smaller model:
```python
config = EvaluationConfig(
    model_path="yolov8n.pt",  # Use nano instead of larger models
    device="cuda"
)
```

### Slow Evaluation

Limit the number of images:
```python
config = EvaluationConfig(
    max_images=100,  # Evaluate on subset
    benchmark_iterations=50  # Fewer benchmark iterations
)
```

### Dataset Loading Issues

Verify dataset format matches the specified type:
```python
# For COCO format
dataset_format="coco"  # Expects annotations/instances_val2017.json

# For YOLO format
dataset_format="yolo"  # Expects labels/ directory with .txt files

# For VOC format
dataset_format="voc"  # Expects Annotations/ directory with .xml files
```

## References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [COCO Dataset](https://cocodataset.org/)
- [PASCAL VOC Dataset](http://host.robots.ox.ac.uk/pascal/VOC/)
- [mAP Calculation Explained](https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173)
- [IoU Variants Paper](https://arxiv.org/abs/1911.08287)

## Contributing

When adding new features to the baseline module:

1. Follow the existing code structure
2. Add type hints and docstrings
3. Update this README with new functionality
4. Add tests for new metrics or loaders
5. Generate sample reports to verify output

## License

This module is part of the Intelligent Traffic System project and is for educational and research purposes.
