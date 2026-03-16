# SINOPEC-02 序列模型结果

_记录轻量 BiLSTM 序列标注 baseline 的实验结果与结论。_

---

## 模型设置

- 框架：纯 `PyTorch`
- 结构：`BiLSTM`
- 输入：现有特征工程后的逐点特征，按井组织为变长序列
- 评估：`GroupKFold` 按井交叉验证
- 输出：raw 逐点预测与每井结构化解码结果

## 结果

- raw macro-F1: `0.4292`
- structured macro-F1: `0.5669`

分折结果可见：

- `reports/sequence/bilstm_cv_metrics.json`
- `reports/sequence/bilstm_oof_predictions.csv`

## 结论

- 当前轻量 `BiLSTM` 并没有超过经典模型 + 特征工程 + 结构化解码
- 这说明在当前样本规模下，强表格特征和显式结构约束仍然更重要
- 深度序列模型不是不能做，但需要更细的调参、正则化和任务改造

## 建议

下一步更值得做的是：

1. 两阶段候选点检测
2. 关键点排序学习
3. 显式约束 `1 -> 2 -> 3` 的结构化训练

