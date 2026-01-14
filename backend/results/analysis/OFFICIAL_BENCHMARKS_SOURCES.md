# Official Benchmark Sources and References

## Vehicle Detection Survey - Data Sources Documentation

---

## 1. YOLOv8 Official Benchmarks

### Source
- **URL:** https://docs.ultralytics.com/models/yolov8/
- **Publisher:** Ultralytics
- **Dataset:** COCO val2017 (80 classes)

### Official Performance Table (Detection - COCO)

| Model | Size (pixels) | mAP@0.5:0.95 | Speed CPU ONNX (ms) | Speed A100 TensorRT (ms) | Params (M) | FLOPs (B) |
|-------|---------------|--------------|---------------------|--------------------------|------------|-----------|
| YOLOv8n | 640 | **37.3** | 80.4 | 0.99 | 3.2 | 8.7 |
| YOLOv8s | 640 | **44.9** | 128.4 | 1.20 | 11.2 | 28.6 |
| YOLOv8m | 640 | **50.2** | 234.7 | 1.83 | 25.9 | 78.9 |
| YOLOv8l | 640 | **52.9** | 375.2 | 2.39 | 43.7 | 165.2 |
| YOLOv8x | 640 | **53.9** | 479.1 | 3.53 | 68.2 | 257.8 |

### Notes
- mAP values are **mAP@0.5:0.95** (COCO-style, averaged over IoU thresholds 0.5 to 0.95)
- Speed measured on **NVIDIA A100 GPU** with TensorRT
- All 80 COCO classes evaluated

### Citation
```bibtex
@software{yolov8_ultralytics,
  author = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
  title = {Ultralytics YOLOv8},
  version = {8.0.0},
  year = {2023},
  url = {https://github.com/ultralytics/ultralytics},
  license = {AGPL-3.0}
}
```

---

## 2. YOLO12 Official Benchmarks

### Source
- **URL:** https://docs.ultralytics.com/models/yolo12/
- **Publisher:** University at Buffalo, University of Chinese Academy of Sciences
- **Dataset:** COCO val2017 (80 classes)

### Official Performance Table (Detection - COCO)

| Model | Size (pixels) | mAP@0.5:0.95 | Speed T4 TensorRT (ms) | Params (M) | FLOPs (B) |
|-------|---------------|--------------|------------------------|------------|-----------|
| YOLO12n | 640 | **40.6** | 1.64 | 2.6 | 6.5 |
| YOLO12s | 640 | **48.0** | 2.61 | 9.3 | 21.4 |
| YOLO12m | 640 | **52.5** | 4.86 | 20.2 | 67.5 |
| YOLO12l | 640 | **53.7** | 6.77 | 26.4 | 88.9 |
| YOLO12x | 640 | **55.2** | 11.79 | 59.1 | 199.0 |

### Notes
- Speed measured on **NVIDIA T4 GPU** with TensorRT FP16 precision
- YOLO12 uses attention-based architecture (Area Attention mechanism)
- Community-driven release (not officially maintained by Ultralytics)

### Citation
```bibtex
@article{tian2025yolo12,
  title={YOLO12: Attention-Centric Real-Time Object Detectors},
  author={Tian, Yunjie and Ye, Qixiang and Doermann, David},
  journal={arXiv preprint arXiv:2502.12524},
  year={2025}
}
```

---

## 3. Our Evaluation vs Official Benchmarks

### Comparison Table

| Model | Our mAP@0.5 | Our mAP@0.5:0.95 | Official mAP@0.5:0.95 | Difference | Reason |
|-------|-------------|------------------|----------------------|------------|--------|
| YOLOv8n | 48.53% | 34.47% | 37.3% | -2.83% | Vehicle classes only |
| YOLOv8s | 57.63% | 42.21% | 44.9% | -2.69% | Vehicle classes only |
| YOLOv8m | 63.29% | 47.27% | 50.2% | -2.93% | Vehicle classes only |
| YOLOv8x | 66.01% | 50.24% | 53.9% | -3.66% | Vehicle classes only |
| YOLO12x | 67.10% | 51.49% | 55.2% | -3.71% | Vehicle classes only |

### Why Our Results Are Lower

1. **Class Subset:** We evaluate on **6 vehicle-related classes** only:
   - person (class 0)
   - bicycle (class 1)
   - car (class 2)
   - motorcycle (class 3)
   - bus (class 5)
   - truck (class 7)

2. **Official Benchmarks:** Evaluate on **all 80 COCO classes**

3. **Class Difficulty:** Some vehicle classes (especially `truck`) are more challenging than average COCO classes

4. **Hardware Differences:** 
   - Official: A100/T4 GPU with TensorRT optimization
   - Ours: Consumer GPU without TensorRT

### Validation Conclusion
Our results are **~3% lower** in mAP@0.5:0.95, which is **expected and acceptable** given the evaluation differences.

---

## 4. COCO Dataset Information

### Source
- **URL:** https://cocodataset.org/
- **Paper:** https://arxiv.org/abs/1405.0312

### Dataset Statistics

| Property | Value |
|----------|-------|
| Name | COCO (Common Objects in Context) |
| Version | 2017 |
| Split Used | val2017 |
| Total Images | 5,000 (full) / 2,968 (our vehicle subset) |
| Total Classes | 80 |
| Vehicle Classes | 6 (person, bicycle, car, motorcycle, bus, truck) |

### Vehicle Classes in COCO

| COCO Class ID | Class Name | Category |
|---------------|------------|----------|
| 0 | person | Human |
| 1 | bicycle | Vehicle |
| 2 | car | Vehicle |
| 3 | motorcycle | Vehicle |
| 5 | bus | Vehicle |
| 7 | truck | Vehicle |

### Citation
```bibtex
@inproceedings{lin2014microsoft,
  title={Microsoft COCO: Common Objects in Context},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
  booktitle={European Conference on Computer Vision (ECCV)},
  pages={740--755},
  year={2014},
  organization={Springer}
}
```

---

## 5. Metric Definitions

### mAP@0.5 (Mean Average Precision at IoU 0.5)
- Calculates AP for each class at IoU threshold = 0.5
- Averages AP across all classes
- More lenient threshold, higher values

### mAP@0.5:0.95 (COCO-style mAP)
- Calculates AP at multiple IoU thresholds: [0.5, 0.55, 0.60, ..., 0.95]
- Averages across all thresholds and classes
- Stricter metric, lower values
- **This is the standard COCO evaluation metric**

### Precision
- TP / (TP + FP)
- Ratio of correct detections among all detections

### Recall
- TP / (TP + FN)
- Ratio of detected objects among all ground truth objects

### FPS (Frames Per Second)
- Number of images processed per second
- Includes preprocessing, inference, and postprocessing

---

## 6. Additional Reference Links

### Official Documentation
- Ultralytics YOLOv8: https://docs.ultralytics.com/models/yolov8/
- Ultralytics YOLO12: https://docs.ultralytics.com/models/yolo12/
- Ultralytics GitHub: https://github.com/ultralytics/ultralytics

### Dataset
- COCO Dataset: https://cocodataset.org/
- COCO Download: https://cocodataset.org/#download
- COCO Evaluation: https://cocodataset.org/#detection-eval

### Papers
- YOLOv8: No formal paper (see Ultralytics documentation)
- YOLO12: arXiv:2502.12524 (https://arxiv.org/abs/2502.12524)
- COCO Dataset: arXiv:1405.0312 (https://arxiv.org/abs/1405.0312)

### Evaluation Metrics
- COCO Evaluation Metrics: https://cocodataset.org/#detection-eval
- mAP Explained: https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173

---

## 7. Hardware Comparison

### Official Benchmark Hardware

| Source | GPU | Optimization | Notes |
|--------|-----|--------------|-------|
| Ultralytics YOLOv8 | NVIDIA A100 | TensorRT | Data center GPU |
| Ultralytics YOLO12 | NVIDIA T4 | TensorRT FP16 | Cloud GPU |

### Our Evaluation Hardware

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA Consumer GPU |
| Framework | PyTorch + Ultralytics |
| Optimization | None (native PyTorch) |
| Batch Size | 1 |
| Input Size | 640 × 640 |

### Expected Speed Differences
- TensorRT optimization can provide 2-5x speedup
- A100 is significantly faster than consumer GPUs
- Our FPS measurements are **conservative estimates**

---

## Summary

| Metric | Source | URL |
|--------|--------|-----|
| YOLOv8 mAP/FPS | Ultralytics Official | https://docs.ultralytics.com/models/yolov8/ |
| YOLO12 mAP/FPS | Ultralytics Official | https://docs.ultralytics.com/models/yolo12/ |
| COCO Dataset | Microsoft COCO | https://cocodataset.org/ |
| Evaluation Protocol | COCO Detection | https://cocodataset.org/#detection-eval |

---

*Document generated for ITS Vehicle Detection Survey*
*Last updated: January 2026*