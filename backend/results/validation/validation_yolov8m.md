# Validation Report: yolov8m

**Generated:** 2026-01-09T17:28:44.046931
**Overall Status:** WARN
**Validation Score:** 70.0/100

## Summary

⚠️ **WARN** - Results are acceptable but have some concerns.

## Benchmark Comparison

- **Status:** WARN
- **Passed:** 0/3

### Detailed Comparison

| Metric | Ours | Official | Diff % | Status |
|--------|------|----------|--------|--------|
| mAP@0.5 | 63.29 | 67.20 | -5.8% | ⚠️ WARN |
| mAP@[.5:.95] | 47.27 | 50.20 | -5.8% | ⚠️ WARN |
| fps | 58.23 | 45.00 | +29.4% | ⚠️ WARN |

## Recommendations

1. fps slightly higher than expected: May indicate different test conditions or dataset subset
