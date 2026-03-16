# Baseline Results

_Auto-generated from grouped cross-validation on the training wells._

---

## Overall metrics

- Raw macro-F1: `0.5932`
- Structured macro-F1: `0.6322`
- Raw weighted-F1: `0.9803`
- Structured weighted-F1: `0.9823`

## Interpretation

- Point-wise random forest already recovers classes `1` and `2` reasonably well
- Per-well structured decoding improves macro-F1 and stabilizes class ordering
- Class `3` remains the hardest target due to very low support and higher ambiguity
