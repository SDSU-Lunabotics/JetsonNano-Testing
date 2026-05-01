# Evaluation Results

**Run:** `pole_rock_train-4`
**Date:** 2026-04-27
**Model:** YOLOv8n
**Epochs:** 100 | **Image size:** 640 | **Batch:** 16 | **Device:** CUDA

---

## Per-Class Metrics

| Class             | Precision | Recall | F1    | mAP50 |
|-------------------|-----------|--------|-------|-------|
| rock              | 1.000     | 0.783  | 0.878 | 0.874 |
| pole              | 0.642     | 0.667  | 0.654 | 0.555 |
| crater            | 0.858     | 0.900  | 0.878 | 0.895 |
| unknown_obstacle  | 0.000     | 0.000  | 0.000 | 0.000 |

**Overall mAP50: 0.775**

---

## Notes

- `unknown_obstacle` scored 0 across all metrics — no test examples or too few training samples for this class
- `pole` has the lowest mAP50 (0.555) — needs more training images
- `rock` and `crater` are performing well
