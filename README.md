# Gene-Disease Association Prediction

基于深度学习的基因-疾病关联预测模型

## 项目简介

本项目使用深度神经网络预测基因与疾病之间的关联强度。模型结合了基因嵌入、疾病嵌入以及多种证据特征，通过训练学习基因-疾病对的关联模式。

## 数据结构

### 输入数据
- `data/by_data_type/`: 按证据类型分类的基因-疾病关联数据
- `data/overall/`: 基因-疾病关联的整体评分数据

### 证据特征 (8列)
- `affected_pathway`: 受影响通路
- `animal_model`: 动物模型
- `genetic_association`: 遗传关联
- `genetic_literature`: 遗传文献
- `known_drug`: 已知药物
- `literature`: 文献证据
- `rna_expression`: RNA表达
- `somatic_mutation`: 体细胞突变

## 模型架构

```
输入层:
├── Gene ID → Embedding (32-dim)
├── Disease ID → Embedding (32-dim)
└── Explicit Features (8-dim, MinMax Scaled)
    ↓
拼接层 (32 + 32 + 8 = 72-dim)
    ↓
全连接层 1: 72 → 128
    ↓
BatchNorm + ReLU + Dropout(0.1)
    ↓
全连接层 2: 128 → 64
    ↓
ReLU + Dropout(0.1)
    ↓
全连接层 3: 64 → 1
    ↓
Sigmoid (输出 0-1 之间的关联分数)
```

## 环境要求

```bash
pip install torch pandas scikit-learn numpy
```

## 使用方法

### 训练模型

首次运行会自动训练模型并保存：

```bash
python train_and_predict.py
```

训练过程会：
1. 加载并预处理数据
2. 训练模型（默认25个epoch）
3. 保存最佳模型（基于验证集最低RMSE）
4. 在验证集上评估模型性能
5. 对示例疾病进行预测

### 加载已训练模型

如果已存在训练好的模型，再次运行会直接加载并跳过训练：

```bash
python train_and_predict.py
```

输出示例：
```
检测到缓存文件 training_data_merged.parquet，正在加载...
...
检测到已训练模型 best_gene_disease_model.pth，正在加载...
模型加载完成！
检测到编码器和缩放器 encoders_scaler.pkl，正在加载...
编码器和缩放器加载完成！
...
正在为 EFO_0000305 预测 Top 基因...
              Gene_ID  Predicted_Score
839   ENSG00000091831         0.740090
...
```

## 模型文件

训练后会生成以下文件（已添加到 `.gitignore`）：

- `best_gene_disease_model.pth`: 最佳模型权重
- `encoders_scaler.pkl`: 编码器和缩放器（用于推理）
- `training_data_merged.parquet`: 预处理后的数据缓存

## 训练配置

在 `train_and_predict.py` 中可调整以下参数：

```python
# 训练参数
epochs = 25                    # 训练轮数
learning_rate = 0.00008        # 学习率
batch_size = 2048              # 批次大小
test_size = 0.2                # 验证集比例

# 模型参数
embedding_dim = 32             # 嵌入维度
hidden_dim_1 = 128             # 第一隐藏层维度
hidden_dim_2 = 64              # 第二隐藏层维度
dropout_rate = 0.1             # Dropout率
```

## 预测函数

使用训练好的模型预测指定疾病的Top基因：

```python
def predict_genes_for_disease(disease_id_str, top_k=20):
    """
    为指定疾病预测Top关联基因
    
    Args:
        disease_id_str: 疾病ID (如 'EFO_0000305')
        top_k: 返回的基因数量
    
    Returns:
        DataFrame包含Gene_ID和Predicted_Score
    """
```

## 性能指标

模型使用以下指标评估：
- **RMSE (Root Mean Squared Error)**: 均方根误差
- **Baseline**: 随机猜测的RMSE作为对比基准

示例输出：
```
Baseline 随机猜测 RMSE: 0.5248
最优模型验证集 RMSE: 0.0087
模型优于随机猜测！
```

## 数据统计

- 唯一基因数: ~31,214
- 唯一疾病数: ~26,329
- 基因-疾病对数: ~4,492,971

## 项目结构

```
.
├── train_and_predict.py    # 主训练和预测脚本
├── README.md                # 项目说明文档
├── .gitignore              # Git忽略文件
├── data/                   # 原始数据目录
│   ├── by_data_type/       # 按证据类型分类的数据
│   └── overall/            # 整体评分数据
├── best_gene_disease_model.pth  # 训练好的模型
├── encoders_scaler.pkl     # 编码器和缩放器
└── training_data_merged.parquet  # 数据缓存
```

## 注意事项

1. 首次运行需要较长时间进行数据预处理和模型训练
2. 模型文件已添加到 `.gitignore`，不会被提交到版本控制
3. 预测的疾病ID必须存在于训练数据中
4. 使用CUDA加速训练（如果可用）

## 许可证

本项目用于学术研究目的。
