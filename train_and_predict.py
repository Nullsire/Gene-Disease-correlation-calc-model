import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
import pickle

# ==========================================
# 配置
# ==========================================
CACHE_PATH = 'training_data_merged.parquet'
DATA_TYPE_PATH = './data/by_data_type'
OVERALL_PATH = './data/overall'
MODEL_PATH = 'best_gene_disease_model.pth'
ENCODERS_SCALER_PATH = 'encoders_scaler.pkl'

# ==========================================
# 1. 数据加载与缓存
# ==========================================

if os.path.exists(CACHE_PATH):
    print(f"检测到缓存文件 {CACHE_PATH}，正在加载...")
    final_df = pd.read_parquet(CACHE_PATH)
else:  
    print("未检测到缓存，正在读取原始数据...")
    # 读取 by_data_type
    df_type = pd.read_parquet(DATA_TYPE_PATH)
    # 透视
    df_features = df_type.pivot_table(
        index=['targetId', 'diseaseId'], 
        columns='datatypeId', 
        values='score', 
        fill_value=0
    ).reset_index()
    df_features.columns.name = None
    # 读取 overall
    df_overall = pd.read_parquet(OVERALL_PATH)
    df_overall = df_overall.rename(columns={'score': 'overall_score'})
    df_overall = df_overall[['targetId', 'diseaseId', 'overall_score']]
    # 合并
    final_df = pd.merge(df_features, df_overall, on=['targetId', 'diseaseId'], how='inner')
    print(f"保存缓存到 {CACHE_PATH}")
    final_df.to_parquet(CACHE_PATH)

print(f"最终整合数据集形状: {final_df.shape}")

# ==========================================
# 2. 数据预处理
# ==========================================
print("正在预处理数据...")

gene_encoder = LabelEncoder()
disease_encoder = LabelEncoder()

final_df['gene_idx'] = gene_encoder.fit_transform(final_df['targetId'])
final_df['disease_idx'] = disease_encoder.fit_transform(final_df['diseaseId'])

num_genes = len(gene_encoder.classes_)
num_diseases = len(disease_encoder.classes_)

print(f"唯一基因数: {num_genes}, 唯一疾病数: {num_diseases}")

feature_cols = [c for c in final_df.columns if c not in ['targetId', 'diseaseId', 'overall_score', 'gene_idx', 'disease_idx']]
print(f"使用的证据特征 ({len(feature_cols)}列): {feature_cols}")

scaler = MinMaxScaler()
X_features = scaler.fit_transform(final_df[feature_cols].values)
X_features = torch.tensor(X_features, dtype=torch.float32)

gene_ids = torch.tensor(final_df['gene_idx'].values, dtype=torch.long)
disease_ids = torch.tensor(final_df['disease_idx'].values, dtype=torch.long)
y_scores = torch.tensor(final_df['overall_score'].values, dtype=torch.float32).view(-1, 1)

train_idx, val_idx = train_test_split(range(len(final_df)), test_size=0.2, random_state=42)

# ==========================================
# 3. Dataset 类
# ==========================================
class OTDataset(Dataset):
    def __init__(self, gene_ids, disease_ids, features, targets):
        self.gene_ids = gene_ids
        self.disease_ids = disease_ids
        self.features = features
        self.targets = targets
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, idx):
        return (self.gene_ids[idx], self.disease_ids[idx], self.features[idx]), self.targets[idx]

train_dataset = OTDataset(gene_ids[train_idx], disease_ids[train_idx], X_features[train_idx], y_scores[train_idx])
val_dataset = OTDataset(gene_ids[val_idx], disease_ids[val_idx], X_features[val_idx], y_scores[val_idx])

train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)

# ==========================================
# 4. 模型定义
# ==========================================
class GeneDiseasePredictor(nn.Module):
    def __init__(self, num_genes, num_diseases, num_features, embedding_dim=32):
        super(GeneDiseasePredictor, self).__init__()
        self.gene_embedding = nn.Embedding(num_genes, embedding_dim)
        self.disease_embedding = nn.Embedding(num_diseases, embedding_dim)
        input_dim = embedding_dim * 2 + num_features
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, gene_id, disease_id, explicit_features):
        g_emb = self.gene_embedding(gene_id)
        d_emb = self.disease_embedding(disease_id)
        combined = torch.cat([g_emb, d_emb, explicit_features], dim=1)
        output = self.layers(combined)
        return output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

model = GeneDiseasePredictor(
    num_genes=num_genes, 
    num_diseases=num_diseases, 
    num_features=len(feature_cols)
).to(device)

# ==========================================
# 5. 训练
# ==========================================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00008)
epochs = 25

# 检查是否存在已训练的模型
if os.path.exists(MODEL_PATH):
    print(f"检测到已训练模型 {MODEL_PATH}，正在加载...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("模型加载完成！")
    
    # 加载 encoders 和 scaler
    if os.path.exists(ENCODERS_SCALER_PATH):
        print(f"检测到编码器和缩放器 {ENCODERS_SCALER_PATH}，正在加载...")
        with open(ENCODERS_SCALER_PATH, 'rb') as f:
            saved_data = pickle.load(f)
            gene_encoder = saved_data['gene_encoder']
            disease_encoder = saved_data['disease_encoder']
            scaler = saved_data['scaler']
            feature_cols = saved_data['feature_cols']
            num_genes = saved_data['num_genes']
            num_diseases = saved_data['num_diseases']
        print("编码器和缩放器加载完成！")
    
    skip_training = True
else:
    skip_training = False
    print("\n开始训练...")
    best_val_rmse = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for (g_id, d_id, feats), targets in train_loader:
            g_id, d_id, feats, targets = g_id.to(device), d_id.to(device), feats.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(g_id, d_id, feats)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_rmse = np.sqrt(running_loss / len(train_loader))
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for (g_id, d_id, feats), targets in val_loader:
                g_id, d_id, feats, targets = g_id.to(device), d_id.to(device), feats.to(device), targets.to(device)
                outputs = model(g_id, d_id, feats)
                val_preds.append(outputs.cpu().numpy())
                val_targets.append(targets.cpu().numpy())
        val_preds = np.concatenate(val_preds, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        val_rmse = np.sqrt(np.mean((val_preds - val_targets) ** 2))
        
        # 保存最佳模型
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_state = model.state_dict().copy()
        
        print(f"Epoch [{epoch+1}/{epochs}], Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}")
    
    print("训练完成！")
    
    # 加载最佳模型
    model.load_state_dict(best_model_state)
    model.eval()
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for (g_id, d_id, feats), targets in val_loader:
            g_id, d_id, feats, targets = g_id.to(device), d_id.to(device), feats.to(device), targets.to(device)
            outputs = model(g_id, d_id, feats)
            val_preds.append(outputs.cpu().numpy())
            val_targets.append(targets.cpu().numpy())
    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)
    val_rmse = np.sqrt(np.mean((val_preds - val_targets) ** 2))
    print(f"最优模型验证集 RMSE: {val_rmse:.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"模型已保存至 {MODEL_PATH}")
    
    # 保存 encoders 和 scaler
    with open(ENCODERS_SCALER_PATH, 'wb') as f:
        pickle.dump({
            'gene_encoder': gene_encoder,
            'disease_encoder': disease_encoder,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'num_genes': num_genes,
            'num_diseases': num_diseases
        }, f)
    print(f"编码器和缩放器已保存至 {ENCODERS_SCALER_PATH}")

# ==========================================
# Baseline: 随机猜测
# ==========================================
np.random.seed(42)
baseline_preds = np.random.rand(len(val_targets), 1)
baseline_rmse = np.sqrt(np.mean((baseline_preds - val_targets) ** 2))
print(f"Baseline 随机猜测 RMSE: {baseline_rmse:.4f}")
print(f"模型 RMSE: {val_rmse:.4f}")
if val_rmse < baseline_rmse:
    print("模型优于随机猜测！")
else:
    print("模型未优于随机猜测，请检查数据或模型。")

# ==========================================
# 6. 推理函数
# ==========================================
def predict_genes_for_disease(disease_id_str, top_k=20):
    model.eval()
    if disease_id_str not in disease_encoder.classes_:
        return "错误：该疾病 ID 不在训练数据中。"
    disease_int = disease_encoder.transform([disease_id_str])[0]
    mask = (disease_ids == disease_int)
    candidate_gene_ids = gene_ids[mask]
    candidate_features = X_features[mask]
    batch_disease_ids = torch.full_like(candidate_gene_ids, disease_int)
    with torch.no_grad():
        candidate_gene_ids = candidate_gene_ids.to(device)
        batch_disease_ids = batch_disease_ids.to(device)
        candidate_features = candidate_features.to(device)
        predictions = model(candidate_gene_ids, batch_disease_ids, candidate_features)
        predictions = predictions.cpu().numpy().flatten()
    original_gene_ids = gene_encoder.inverse_transform(candidate_gene_ids.cpu().numpy())
    result_df = pd.DataFrame({
        'Gene_ID': original_gene_ids,
        'Predicted_Score': predictions
    })
    result_df = result_df.sort_values(by='Predicted_Score', ascending=False).head(top_k)
    return result_df

# ==========================================
# 7. 测试演示
# ==========================================
if __name__ == "__main__":
    test_disease = 'EFO_0000305'
    print(f"\n正在为 {test_disease} 预测 Top 基因...")
    top_genes = predict_genes_for_disease(test_disease)
    print(top_genes)
