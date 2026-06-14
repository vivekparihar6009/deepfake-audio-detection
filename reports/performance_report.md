# Deepfake Audio Detection — Performance Report

**Evaluation Split:** testing

---

## Metric Summary

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Overall Accuracy | **94.04%** | ≥ 80% | ✅ |
| Equal Error Rate (EER) | **5.95%** | ≤ 12% | ✅ |
| F1 Score | **94.17%** | ≥ 80% | ✅ |
| Genuine (Real) Accuracy | **94.04%** | ≥ 75% | ✅ |
| Deepfake Accuracy | **94.05%** | ≥ 75% | ✅ |

> **EER Threshold (decision boundary):** 0.0060

---

## Confusion Matrix

![Confusion Matrix](confusion_matrix_testing.png)

---

## ROC Curve

![ROC Curve](roc_curve_testing.png)

---

## DET Curve

![DET Curve](det_curve_testing.png)

---

## Detailed Classification Report

```
              precision    recall  f1-score   support

     Genuine       0.94      0.94      0.94      2264
    Deepfake       0.94      0.94      0.94      2370

    accuracy                           0.94      4634
   macro avg       0.94      0.94      0.94      4634
weighted avg       0.94      0.94      0.94      4634

```

---

## Threshold Verification

- ✅ Accuracy >= 80%
- ✅ EER <= 12%
- ✅ F1 >= 80%
- ✅ Genuine Acc >= 75%
- ✅ Deepfake Acc >= 75%

**All thresholds met: YES ✅**

---
*Generated automatically by evaluate.py*