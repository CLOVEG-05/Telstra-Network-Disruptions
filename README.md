# Telstra Network Fault Prediction

基于机器学习的网络故障严重程度预测系统，使用4模型集成（LightGBM + XGBoost + CatBoost + Random Forest）预测网络故障的三个严重等级。

## 项目背景

在实际网络运维中，传统故障处理存在以下痛点：

- 故障发生后才被动响应，缺乏预测能力
- 故障严重程度判断依赖人工经验，效率低下
- 多源故障数据分散，难以统一分析
- 故障特征复杂，单一模型效果有限

本项目通过机器学习方法，实现故障严重程度的自动预测，提升运维效率。

## 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                     4 Model Ensemble                        │
│        LightGBM + XGBoost + CatBoost + Random Forest        │
└──────────────────────────┬──────────────────────────────────┘
                           │
     ┌─────────────────────┼─────────────────────┐
     │                     │                     │
┌────▼────┐          ┌─────▼─────┐         ┌─────▼─────┐
│LightGBM │          │  XGBoost  │         │  CatBoost │
└─────────┘          └───────────┘         └───────────┘
     │
┌────▼────────────────────────────────────────────────────────┐
│                    Feature Engineering                       │
│  - Feature Cross (481 features)                              │
│  - Target Encoding (location_id)                            │
│  - Statistical Features (volume_mean, volume_std, etc.)     │
└──────────────────────────┬──────────────────────────────────┘
                           │
     ┌─────────────────────┼─────────────────────┐
     │                     │                     │
┌────▼────┐          ┌─────▼─────┐         ┌─────▼─────┐
│  Train  │          │   Test    │         │  Optuna   │
│ 7,381   │          │  11,171   │         │  Tuning   │
└─────────┘          └───────────┘         └───────────┘
```

## 技术栈

| 组件 | 技术 | 作用 |
|---|---|---|
| 数据处理 | Pandas, NumPy | 数据清洗与预处理 |
| 特征工程 | Feature Engineering | 特征交叉与统计特征 |
| 目标编码 | Target Encoding | 类别特征编码 |
| 梯度提升 | LightGBM, XGBoost, CatBoost | 核心预测模型 |
| 集成学习 | Voting Classifier | 模型集成 |
| 超参数优化 | Optuna | 自动调参 |
| 评估指标 | Log Loss, Accuracy | 模型评估 |

## 核心功能

### 1. 数据处理

- 多源数据合并（事件类型、日志特征、资源类型、严重程度）
- 自动化数据清洗与缺失值处理
- 训练/测试数据一致性保证

### 2. 特征工程

- 交互特征：event_log_ratio、event_log_product
- 统计特征：volume_mean、volume_std、volume_total
- 目标编码：location_id 编码
- 特征交叉：多样性×强度、体积相关特征
- **最终特征数：481**

### 3. 模型训练

- 4模型集成：LightGBM + XGBoost + CatBoost + Random Forest
- 软投票集成，权重优化分配
- 5折交叉验证
- Optuna超参数自动调优

### 4. 智能预测

- 故障严重程度三分类（0-低、1-中、2-高）
- 概率输出，支持置信度评估
- 自动化模型选择与部署

## 快速开始

### 环境要求

- Python 3.11+
- CUDA 11.8+ (可选，用于GPU加速)
- 内存 >= 8GB

### 安装依赖

```bash
conda create -n telstra python=3.11
conda activate telstra
pip install pandas numpy scikit-learn lightgbm xgboost catboost optuna
```

### 训练模型

```bash
# 1. 数据准备
python code/data_preparation.py

# 2. 特征工程
python code/feature_engineering_v2.py
python code/target_encoding.py
python code/advanced_features.py

# 3. 模型训练
python code/final_ensemble.py

# 4. 超参数优化（可选）
python code/optuna_tuning.py
```

### 生成预测

提交文件将保存到 `processed data/submission_final_v2.csv`

## 项目结构

```
telstra-recruiting-network/
├── code/                              # 核心代码
│   ├── data_preparation.py           # 数据准备和合并
│   ├── feature_engineering_v2.py      # 深度特征工程
│   ├── target_encoding.py             # 目标编码
│   ├── advanced_features.py           # 高级特征交叉
│   ├── final_ensemble.py              # 4模型集成
│   └── optuna_tuning.py               # 超参数调优
├── data set/                           # 原始数据集
│   ├── train.csv                      # 训练数据
│   ├── test.csv                       # 测试数据
│   └── sample_submission.csv          # 提交样例
├── processed data/                      # 处理后数据
│   ├── train_advanced.csv             # 最终训练数据
│   ├── test_advanced.csv              # 最终测试数据
│   ├── best_params.json               # Optuna最佳参数
│   └── submission_final_v2.csv        # 最终提交文件
├── record/                             # 项目记录
│   ├── project_record.md              # 详细项目记录
│   ├── project_flow.md                # 流程记录
│   └── project_record.txt             # 文本格式
├── visualization/                      # 可视化结果
└── README.md
```

## 模型性能

| 指标 | 结果 |
|------|------|
| CV Log Loss | 0.5214 |
| Optuna最佳 | 0.5179 |
| 5折平均准确率 | 76.58% |
| 预期Kaggle分数 | ~0.52 |

### 与排行榜对比

| 排名 | Log Loss |
|------|----------|
| 第1名 | 0.395 |
| 前10名 | ~0.40 |
| **本项目** | **~0.52** |

### 分类性能

| 类别 | 精确率 | 召回率 | F1分数 |
|------|--------|--------|--------|
| 0 (低) | 0.80 | 0.88 | 0.84 |
| 1 (中) | 0.61 | 0.47 | 0.53 |
| 2 (高) | 0.57 | 0.57 | 0.57 |

## 核心模型

### 4模型集成

```python
# 权重分配
LightGBM:  0.30
XGBoost:   0.30
CatBoost:  0.25
RF:        0.15
```

### Optuna优化参数

```json
{
  "learning_rate": 0.016,
  "max_depth": 10,
  "num_leaves": 28,
  "feature_fraction": 0.861,
  "bagging_fraction": 0.716,
  "min_child_samples": 7,
  "reg_alpha": 0.044,
  "reg_lambda": 1.305
}
```

## 数据处理流程

1. **数据准备** (`data_preparation.py`): 合并多源数据 → 462特征
2. **特征工程** (`feature_engineering_v2.py`): 交互特征 + 统计特征 → 471特征
3. **目标编码** (`target_encoding.py`): location_id编码 → 472特征
4. **高级特征** (`advanced_features.py`): 特征交叉 → 481特征
5. **模型训练** (`final_ensemble.py`): 4模型集成 → CV 0.5214
6. **参数调优** (`optuna_tuning.py`): Optuna优化 → Best 0.5179

## 项目收获

通过本项目，深入理解了：

- **特征工程**：从基础特征到高级特征交叉的全流程
- **集成学习**：多模型集成的策略与权重优化
- **超参数调优**：Optuna自动化调参的最佳实践
- **表格数据处理**：LightGBM/XGBoost/CatBoost的使用技巧
- **Kaggle竞赛**：从数据探索到模型提交的完整流程
- **模型评估**：Log Loss优化的技巧与注意事项

## 后续优化方向

- [ ] 增加Optuna训练轮数（50→200轮），进一步降低Log Loss
- [ ] 概率校准（Calibration）改善概率估计
- [ ] 接入更多基模型（ExtraTrees、AdaBoost）增加多样性
- [ ] 基于特征重要性进行高级特征选择
- [ ] Focal Loss针对类别不平衡问题
- [ ] 伪标签技术扩充训练集

## 联系方式

- GitHub: `https://github.com/yourname`
- Email: your@email.com