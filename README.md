# Telstra Network Disruptions Prediction

Telstra 电信网络故障预测分析 — Kaggle 竞赛项目。

## 📋 项目简介

基于 Telstra 提供的网络设备日志数据，对网络中断故障进行预测分析。通过探索性数据分析(EDA)发现数据规律，构建时序特征工程，最终通过机器学习分类模型预测故障类型。

## 📊 数据集

- **训练数据**: 设备日志、故障记录、位置信息等多表关联数据
- **目标**: 预测网络中断的故障类型（多分类问题）

## 🔬 分析流程

1. **EDA 探索性分析**: 数据分布、缺失值分析、异常检测
2. **特征工程**: 滑动窗口统计、故障间隔特征、时间序列特征
3. **模型构建**: 对比多种分类算法，调参优化
4. **结果评估**: 多分类评估指标（Log Loss、Accuracy）

## 📁 项目结构

```
├── code/               # 分析代码
├── data set/           # 原始数据集
├── processed data/     # 处理后的数据
├── visualization/      # 可视化输出
└── record/             # 实验记录
```

## 🛠️ 技术栈

- Python (Pandas, NumPy, Scikit-learn)
- Matplotlib / Seaborn
- Jupyter Notebook

## 📝 License

MIT
