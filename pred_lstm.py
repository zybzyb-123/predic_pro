import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import torch
from torch.utils.data import Dataset, DataLoader
from eval import evaluate_model_comprehensive, recall_at_k, precision_at_k
from run import  load_trace, simulate_cache_only,simulate_cache
import pickle
import os
from pathlib import Path


# ---------- 数据缓存功能 ----------
def get_cache_filename(log_path, window_minutes=5, min_access_count=2, top_k=None, 
                      seq_length=12, pred_length=6, data_type='processed'):
    """生成缓存文件名"""
    base_name = Path(log_path).stem
    params = f"win{window_minutes}_minacc{min_access_count}_seq{seq_length}_pred{pred_length}"
    if top_k:
        params += f"_topk{top_k}"
    return f"cache/{base_name}_{params}_{data_type}.pkl"

def save_processed_data(data_dict, cache_file):
    """保存处理后的数据到缓存"""
    os.makedirs('cache', exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"数据已保存到缓存: {cache_file}")

def load_processed_data(cache_file):
    """从缓存加载处理后的数据"""
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        print(f"从缓存加载数据: {cache_file}")
        return data
    else:
        print(f"缓存文件不存在: {cache_file}")
        return None


# ---------- 修改后的数据处理函数（支持缓存） ----------
def get_processed_data(log_path, window_minutes=5, min_access_count=2, top_k=None,
                      seq_length=12, pred_length=6, use_cache=True):
    """
    获取处理后的数据，支持缓存
    返回: pivot, page_to_idx, idx_to_page, num_pages, page_counts, sequences, labels
    """
    cache_file = get_cache_filename(log_path, window_minutes, min_access_count, 
                                   top_k, seq_length, pred_length)
    
    if use_cache:
        cached_data = load_processed_data(cache_file)
        if cached_data is not None:
            return cached_data
    
    # 如果没有缓存或禁用缓存，则重新处理数据
    pivot = collect_page_access_data_pivot(log_path, window_minutes)
    page_to_idx, idx_to_page, num_pages, page_counts = create_page_mapping_from_pivot(
        pivot, min_access_count, top_k
    )

    # 获取按索引排序的页面列表
    pages_in_model = sorted(page_to_idx.keys(), key=lambda x: page_to_idx[x])
    
    # 使用 create_sequences_binary_label
    pivot_array = pivot[pages_in_model].values
    
    sequences, labels = create_sequences_binary_label(
        pivot_array,
        seq_len=seq_length,
        pred_len=pred_length,
        stride=1,
        log_transform=True
    )
    
    result = {
        'pivot': pivot,
        'page_to_idx': page_to_idx,
        'idx_to_page': idx_to_page,
        'num_pages': num_pages,
        'page_counts': page_counts,
        'sequences': sequences,
        'labels': labels
    }
    
    if use_cache:
        save_processed_data(result, cache_file)
    
    return result


# ---------- 更稳健的数据预处理 ----------
def collect_page_access_data_pivot(log_path, window_minutes=5, time_col='timestamp', page_cols=None):
    """
    返回：pivot_df (index=time_window, columns=page_id, values=access_count)
    自动识别 page 列名 'page_id' / 'pageId' 等；将 timestamp 单位视作秒。
    """
    df = pd.read_csv(log_path)
    # 自动识别 page 列
    if page_cols is None:
        if 'page_id' in df.columns:
            page_col = 'page_id'
        elif 'pageId' in df.columns:
            page_col = 'pageId'
        elif 'page' in df.columns:
            page_col = 'page'
        else:
            raise KeyError(f"No page id column found. Available columns: {df.columns.tolist()}")
    else:
        page_col = page_cols

    if time_col not in df.columns:
        raise KeyError(f"No time column '{time_col}' in CSV. Columns: {df.columns.tolist()}")

    # 标准化列名
    df = df.rename(columns={page_col: 'page_id', time_col: 'timestamp'})

    # 时间转换（浮点 UNIX 秒）
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
    if df['timestamp'].isna().any():
        print("警告: timestamp 转换出 NaT 行（可能输入格式不是 unix seconds）")

    df['time_window'] = df['timestamp'].dt.floor(f'{window_minutes}min')

    grouped = df.groupby(['time_window', 'page_id']).size().reset_index(name='access_count')

    # 完整 time_window range
    all_windows = pd.date_range(start=grouped['time_window'].min(), end=grouped['time_window'].max(), freq=f'{window_minutes}min')

    # pivot：每行是一个 time_window，每列为 page_id
    pivot = grouped.pivot(index='time_window', columns='page_id', values='access_count')
    pivot = pivot.reindex(all_windows).fillna(0).astype(float)  # 确保连续窗口并用 0 填充

    pivot.index.name = 'time_window'
    return pivot

# ---------- page mapping，支持 top_k 筛选 ----------
def create_page_mapping_from_pivot(pivot_df, min_access_count=2, top_k=None):
    # pivot_df: DataFrame，列为 page_id
    page_counts = pivot_df.sum(axis=0)  # 每列总访问次数
    selected = page_counts[page_counts >= min_access_count].sort_values(ascending=False)
    if top_k is not None:
        selected = selected.iloc[:top_k]
    frequent_pages = selected.index.tolist()
    page_to_idx = {page_id: idx for idx, page_id in enumerate(frequent_pages)}
    idx_to_page = {idx: page_id for page_id, idx in page_to_idx.items()}
    return page_to_idx, idx_to_page, len(frequent_pages), selected  # 返回 selected 方便观察分布

# ---------- 构建序列（更简单/稳健） ----------
def create_sequences_from_pivot(pivot_df, page_to_idx, seq_length=12, pred_length=6, transform_fn=None):
    """
    pivot_df: rows=time_window, cols=page_id
    返回 sequences shape (N, seq_length, num_pages) 与 labels (N, num_pages)
    transform_fn: 对 counts 做转换（如 np.log1p）, 接受 numpy 矩阵返回矩阵
    """
    # 只保留映射中的列并按固定列顺序
    columns = [p for p in pivot_df.columns if p in page_to_idx]
    columns = sorted(columns, key=lambda x: page_to_idx[x])  # 保证列顺序与 mapping 一致
    data = pivot_df[columns].values  # shape: (T, P)
    T, P = data.shape

    if transform_fn is not None:
        data = transform_fn(data)

    sequences = []
    labels = []
    # inclusive range
    max_i = T - seq_length - pred_length + 1
    for i in range(max_i):
        hist = data[i:i+seq_length]               # (seq_length, P)
        pred_window = data[i+seq_length:i+seq_length+pred_length]  # (pred_length, P)
        label = (pred_window.sum(axis=0) > 0).astype(float)  # multi-hot: 在预测窗口内出现即为正
        sequences.append(hist)
        labels.append(label)

    return np.array(sequences), np.array(labels)

def create_sequences_binary_label(pivot_array: np.ndarray,
                                  seq_len: int,
                                  pred_len: int,
                                  stride: int = 1,
                                  log_transform: bool = True):
    """
    pivot_array: ndarray (T, P)，每行是5分钟窗口每页访问次数
    seq_len: 历史窗口数量 m
    pred_len: 预测窗口数量 n
    stride: 滑窗步长，默认1
    log_transform: 是否对输入访问次数取 log1p
    返回:
        sequences: (N, seq_len, P)  —— 历史访问次数特征
        labels   : (N, P)          —— 未来是否访问 (0/1)
    """
    T, P = pivot_array.shape
    seqs, labels = [], []
    for start in range(0, T - seq_len - pred_len + 1, stride):
        end_hist = start + seq_len
        end_pred = end_hist + pred_len

        # 历史特征：可以对次数做 log1p 压缩
        hist = pivot_array[start:end_hist].copy()  # 使用copy避免修改原数据
        if log_transform:
            hist = np.log1p(hist)

        # 标签：未来 pred_len 窗任意访问即置 1
        future_counts = pivot_array[end_hist:end_pred].sum(axis=0)
        lbl = (future_counts > 0).astype(np.float32)

        seqs.append(hist)
        labels.append(lbl)

    return np.stack(seqs), np.stack(labels)



class PageAccessDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]), 
            torch.FloatTensor(self.labels[idx])
        )

import torch.nn as nn
import torch.nn.functional as F

class SequentialMultiLabelPredictor(nn.Module):
    def __init__(self, input_size, total_pages, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.total_pages = total_pages
        
        # 时序编码器
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False  # 使用双向LSTM捕捉前后依赖
        )
        
        # 注意力机制
        # self.attention = nn.MultiheadAttention(
        #     embed_dim=hidden_size * 2,  # 双向LSTM输出维度加倍
        #     num_heads=8,
        #     dropout=dropout,
        #     batch_first=True
        # )
        
        # 输出层
        self.output_layer = nn.Linear(hidden_size, total_pages)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        # 使用Xavier初始化
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.1)
        
        # 输出层特殊初始化
        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.1)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, -3.0)
    
    def forward(self, x):
        # LSTM编码
        encoded, (hidden, cell) = self.encoder(x)
        
        context_vector = hidden[-1]
        context_vector = self.dropout(context_vector)
        logits = self.output_layer(context_vector)
        return logits

class WeightedBCELoss(nn.Module):
    """带权重的二元交叉熵损失，处理类别不平衡"""
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight
    
    def forward(self, inputs, targets):
        if self.pos_weight is not None:
            self.pos_weight = self.pos_weight.to(inputs.device)
        
        return F.binary_cross_entropy_with_logits(
            inputs, targets, 
            pos_weight=self.pos_weight,
            reduction='mean'
        )


# 考虑使用Focal Loss处理类别不平衡
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

def train_complete_model(log_path, model_name, num_epochs, use_cache=True):
    """
    完整的模型训练流程
    """
    # 1. 收集数据
    processed_data = get_processed_data(
        log_path, 
        window_minutes=5, 
        min_access_count=2, 
        top_k=None,
        seq_length=6, 
        pred_length=6,
        use_cache=use_cache
    )
    
    page_to_idx = processed_data['page_to_idx']
    idx_to_page = processed_data['idx_to_page']
    num_pages = processed_data['num_pages']
    page_counts = processed_data['page_counts']
    sequences = processed_data['sequences']
    labels = processed_data['labels']
    
    print("页面数 (被选中):", num_pages)
    print("页面访问量 top5:\n", page_counts.head())
    print("sequence shape:", sequences.shape, "labels shape:", labels.shape)
    print("labels positive rate (per class) min/max/mean:", 
          labels.sum(axis=0).min(), labels.sum(axis=0).max(), labels.sum(axis=0).mean())
    
    # 4. 划分训练集和验证集
    split_idx = int(0.8 * len(sequences))
    split_idx_test = int(0.9 * len(sequences))
    train_sequences, val_sequences, test_sequences = sequences[:split_idx], sequences[split_idx:split_idx_test], sequences[split_idx_test:]
    train_labels, val_labels, test_labels = labels[:split_idx], labels[split_idx:split_idx_test], labels[split_idx_test:]
    
    # 5. 创建数据加载器
    train_dataset = PageAccessDataset(train_sequences, train_labels)
    val_dataset = PageAccessDataset(val_sequences, val_labels)
    test_dataset = PageAccessDataset(test_sequences, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 6. 计算类别权重（处理不平衡）
    # 计算 pos_weight（更稳健）
    pos_count = np.sum(train_labels, axis=0)      # 每类正样本数
    neg_count = train_labels.shape[0] - pos_count
    # 避免除以0并对极端权重做裁剪
    pos_weight_np = (neg_count / (pos_count + 1e-6)).astype(np.float32)
    # 对从未出现的 class 设置为 1（或者直接移除该类）
    pos_weight_np[pos_count == 0] = 1.0
    # 裁剪上限，防止数值不稳定
    pos_weight_np = np.clip(pos_weight_np, 1.0, 100.0)  # 上限可调
    pos_weight = torch.FloatTensor(pos_weight_np)
    
    # 7. 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SequentialMultiLabelPredictor(
        input_size=num_pages,  # 输入维度等于页面数量
        total_pages=num_pages
    ).to(device)
    
    # 8. 定义损失函数和优化器
    criterion = WeightedBCELoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', patience=5, factor=0.5
    # )
    
    # 9. 训练循环
    best_val_loss = float('inf')
    best_recall = 0.0
    patience_counter = 0
    patience = 5
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_seq, batch_labels in train_loader:
            batch_seq, batch_labels = batch_seq.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_seq)
            loss = criterion(logits, batch_labels)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_seq, batch_labels in val_loader:
                batch_seq, batch_labels = batch_seq.to(device), batch_labels.to(device)
                logits = model(batch_seq)
                val_loss += criterion(logits, batch_labels).item()
        
        train_loss /= len(train_loader)
        val_metrics = calculate_metrics_at_k(model, val_loader, device, k=1024)
        val_loss /= len(val_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Recall@1024 = {val_metrics["Recall"]:.4f}, Val Precision@1024 = {val_metrics["Precision"]:.4f}')

        
        # 学习率调整
        # scheduler.step()
        
        # 早停检查
        if val_metrics["Precision"] > best_recall:
            best_recall = val_metrics["Precision"]
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'page_mapping': page_to_idx,
                'idx_mapping': idx_to_page
            }, model_name)
            print(f"保存最佳模型，验证损失: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("早停触发")
                break
    # torch.save({
    #             'epoch': epoch,
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'loss': val_loss,
    #             'page_mapping': page_to_idx,
    #             'idx_mapping': idx_to_page
    #         }, model_name)
    # print(f"保存最后模型，验证损失: {val_loss:.4f}")
    
    return model, page_to_idx, idx_to_page

def calculate_metrics_at_k(model, data_loader, device, k=1024):
    """计算Recall@K指标"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_seq, batch_labels in data_loader:
            batch_seq, batch_labels = batch_seq.to(device), batch_labels.to(device)
            logits = model(batch_seq)
            all_preds.append(logits.cpu().numpy())
            all_labels.append(batch_labels.cpu().numpy())
    
    pred_scores = np.concatenate(all_preds, axis=0)
    true_labels = np.concatenate(all_labels, axis=0)

    metrics = {}
    metrics[f'Precision'] = precision_at_k(pred_scores, true_labels, k)
    metrics[f'Recall'] = recall_at_k(pred_scores, true_labels, k)
    
    return metrics

def loadModel(train_file, model_name, use_cache=True):
    """加载模型和测试数据（使用缓存）"""
    processed_data = get_processed_data(
        train_file,
        window_minutes=5,
        min_access_count=2,
        top_k=None,
        seq_length=12, 
        pred_length=6,
        use_cache=use_cache
    )
    
    sequences = processed_data['sequences']
    labels = processed_data['labels']

    #得到测试集
    split_idx_test = int(0.9 * len(sequences))
    test_sequences, test_labels = sequences[split_idx_test:], labels[split_idx_test:]
    test_dataset = PageAccessDataset(test_sequences, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 加载模型
    model = SequentialMultiLabelPredictor(
        input_size=len(processed_data['page_to_idx']),
        total_pages=len(processed_data['page_to_idx'])
    ).to("cpu")
    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, test_loader



def getPredList(train_file, warm_file, model_name, seq_length=6, window_minutes=5, threshold=0.3, top_k=None, use_cache=True):

    """获取预测列表（使用缓存）"""
    processed_data = get_processed_data(
        train_file,
        window_minutes=5,
        min_access_count=2,
        top_k=None,
        seq_length=6,
        pred_length=6,
        use_cache=use_cache
    )
    
    page_to_idx = processed_data['page_to_idx']
    # 加载模型
    model = SequentialMultiLabelPredictor(
        input_size=len(page_to_idx),
        total_pages=len(page_to_idx)
    ).to("cpu")
    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint['model_state_dict'])

    # model.load_state_dict(torch.load(model_name, map_location="cpu"))

    result = predict_future_pages(
        model,
        log_path=warm_file,
        page_to_idx=page_to_idx,
        seq_length=seq_length,
        window_minutes=window_minutes,
        threshold=threshold,
        top_k=top_k,  # 添加top_k参数   
        device="cpu"
    )

    # 结果
    pred_list = result["pred_pages"]
    if top_k is not None:
        print(f"获取前{top_k}个概率最高的页面: {len(pred_list)}")
    else:
        print(f"预测概率≥{threshold}的页面数: {len(pred_list)}")
    print("示例页面:", pred_list[:20])  # 取前20个看看

    return pred_list


@torch.no_grad()
def predict_future_pages(
    model,
    log_path: str,
    page_to_idx: dict,
    seq_length: int = 6,
    window_minutes: int = 5,
    threshold: float = 0.5,
    top_k: int = None,
    device: str = "cpu",
):
    """
    使用训练好的模型对一份全新的日志数据进行预测。

    参数
    ----
    model : torch.nn.Module
        训练好的模型(已加载权重)。
    log_path : str
        新日志文件路径，包含 'timestamp' 和 'page_id' 两列。
    page_to_idx : dict
        训练时的 page_id -> index 映射。
    seq_length : int
        预测时使用的历史窗口长度（要与训练一致）。
    window_minutes : int
        将 timestamp 聚合到的时间粒度（要与训练一致）。
    threshold : float
        预测概率阈值。默认 0.5，可改成 0.3。
    device : str
        运行设备。

    返回
    ----
    dict : {
        "pred_scores": {page_id: prob, ...},
        "pred_pages": [page_id1, page_id2, ...]  # 概率 >= threshold
    }
    """
    model.eval()
    model.to(device)

    # ---------- 1. 读取并按时间窗口聚合 ----------
    df = pd.read_csv(log_path)
    # 统一列名
    if 'page_id' not in df.columns:
        if 'pageId' in df.columns:
            df = df.rename(columns={'pageId': 'page_id'})
        else:
            raise KeyError("日志文件中必须包含 page_id 列")

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
    df['time_window'] = df['timestamp'].dt.floor(f'{window_minutes}min')

    grouped = (
        df.groupby(['time_window', 'page_id'])
        .size()
        .reset_index(name='access_count')
    )

    # 按训练映射的页面顺序构建 pivot
    all_windows = pd.date_range(
        grouped['time_window'].min(),
        grouped['time_window'].max(),
        freq=f'{window_minutes}min'
    )
    pivot = grouped.pivot(index='time_window', columns='page_id', values='access_count')
    pivot = pivot.reindex(all_windows).fillna(0.0)

    # 只保留训练映射中的列，并按映射顺序排列
    pages_in_model = sorted(page_to_idx.keys(), key=lambda x: page_to_idx[x])
    pivot = pivot.reindex(columns=pages_in_model).fillna(0.0)

    data = np.log1p(pivot.values)  # 与训练时一致的 log1p 变换

    # ---------- 2. 取最后 seq_length 个时间窗口作为输入 ----------
    if len(data) < seq_length:
        raise ValueError(
            f"历史数据不足 {seq_length} 个窗口（当前 {len(data)}），无法预测"
        )
    input_seq = data[-seq_length:]  # shape: (seq_length, num_pages)

    x = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
    # ---------- 3. 模型推理 ----------
    logits = model(x)
    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # shape: (num_pages,)

    # ---------- 4. 生成输出 ----------
    idx_to_page = {v: k for k, v in page_to_idx.items()}
    pred_scores = {idx_to_page[i]: float(probs[i]) for i in range(len(probs))}

    # 根据阈值或top_k选择预测页面
    if top_k is not None:
        # 获取前k个概率最高的页面
        sorted_pages = sorted(pred_scores.items(), key=lambda x: x[1], reverse=True)
        pred_pages = [page for page, score in sorted_pages[:top_k]]
        print(f"获取前{top_k}个概率最高的页面")
    else:
        # 使用阈值筛选
        pred_pages = [pid for pid, p in pred_scores.items() if p >= threshold]
        print(f"使用阈值{threshold}筛选页面")

    return {
        "pred_scores": pred_scores,
        "pred_pages": pred_pages,
    }


def train_model(train_file, model_name, num_epochs):
    model, page_to_idx, idx_to_page = train_complete_model(train_file, model_name, num_epochs=num_epochs)

def eval_model(train_file, model_name):
    # 评估
    device = "cpu"
    model, test_loader = loadModel(train_file, model_name)
    comprehensive_results = evaluate_model_comprehensive(model, test_loader, device, topk_list=[256,512,768,1024,2048])
    print("综合指标:", comprehensive_results)
    return comprehensive_results

def getRit(train_file, warm_file, model_name, test_file, top_k):

    # 取预测数据
    # 记录开始时间
    import time
    start_time = time.time()
    pred_list = getPredList(train_file, warm_file, model_name, seq_length=6, window_minutes=5, threshold=0.3, top_k=top_k)
    # 记录结束时间
    end_time = time.time()
    # 计算运行时间
    run_time = end_time - start_time
    print(f"推理时间: {run_time:.4f} 秒")
    
    test_trace = load_trace(test_file)
    # lstm_curve, lstm_hit = simulate_cache_only(test_trace, pred_list)
    lstm_curve, lstm_hit = simulate_cache(test_trace, pred_list, 1024)
    print("命中率比较：")
    print(f"LSTM: {lstm_hit:.4f}")
    return lstm_hit


# 模型指标对比实验
def exp_assess():
    train_file = 'trace/trace0-11h.csv'
    model_name = "model/lstm_model_seq6.pt"

    precision256_list = []
    precision512_list = []
    precision768_list = []
    precision1024_list = []
    precision2048_list = []
    mse_list = []
    msa_list = []

    #模型对比
    for seed in range(42, 47):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # 执行训练
        train_model(train_file, model_name, num_epochs=80)

        # 评估
        result = eval_model(train_file, model_name)
        precision256_list.append(result['Precision@256'])
        precision512_list.append(result['Precision@512'])
        precision768_list.append(result['Precision@768'])
        precision1024_list.append(result['Precision@1024'])
        precision2048_list.append(result['Precision@2048'])
        mse_list.append(result['MSE'])
        msa_list.append(result['MAE'])

    print(f"Precision@256: {precision256_list}")
    print(f"Precision@256_mean: {np.mean(precision256_list):.4f}")
    
    print(f"Precision@512: {precision512_list}")
    print(f"Precision@512_mean: {np.mean(precision512_list):.4f}")
    
    print(f"Precision@768: {precision768_list}")
    print(f"Precision@768_mean: {np.mean(precision768_list):.4f}")
    
    print(f"Precision@1024: {precision1024_list}")
    print(f"Precision@1024_mean: {np.mean(precision1024_list):.4f}")

    print(f"MSE: {mse_list}")
    print(f"MSE_mean: {np.mean(mse_list):.4f}")

    print(f"MSA: {msa_list}")
    print(f"MSA_mean: {np.mean(msa_list):.4f}")

# 命中率实验
def exp_hit_rate():

    # 数据集1
    # train_file = 'trace/trace0-11h.csv'
    # warm_file = 'trace/trace13h.csv'
    # test_file = 'trace/trace14h.csv'

    # 数据集2
    # train_file = 'trace/trace0-11h.csv'
    # warm_file = 'trace/trace12h.csv'
    # test_file = 'trace/trace13h.csv'

    # 数据集2
    # train_file = 'trace/trace0-15h.csv'
    # warm_file = 'trace/trace16h.csv'
    # test_file = 'trace/trace17h.csv'

    # 数据集3
    # train_file = 'trace/trace0-6h.csv'
    # warm_file = 'trace/trace6h.csv'
    # test_file = 'trace/trace7h.csv'

    # 数据集4
    train_file = 'trace/trace11-22h.csv'
    warm_file = 'trace/trace22h.csv'
    test_file = 'trace/trace23h.csv'

    model_name = "model/lstm_model_seq6.pt"
    top_k_list = [768, 1024,2048]
    lstm768_hits = []
    lstm1024_hits = []
    lstm2048_hits = []

    for seed in range(42, 47):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # 执行训练
        train_model(train_file, model_name, num_epochs=80)
        
        for top_k in top_k_list:
            res = getRit(train_file, warm_file, model_name, test_file, top_k)
            if top_k == 768:
                lstm768_hits.append(res)
            elif top_k == 1024:
                lstm1024_hits.append(res)
            elif top_k == 2048:
                lstm2048_hits.append(res)
    
    print(f"LSTM@768: {lstm768_hits}")
    print(f"LSTM@768_mean: {np.mean(lstm768_hits):.4f}")
    
    print(f"LSTM@1024: {lstm1024_hits}")
    print(f"LSTM@1024_mean: {np.mean(lstm1024_hits):.4f}")
    
    print(f"LSTM@2048: {lstm2048_hits}")
    print(f"LSTM@2048_mean: {np.mean(lstm2048_hits):.4f}")


# top命中率实验
def exp_topk_hit_rate():

    #数据集11
    train_file = "trace/similar/train7.txt"
    warm_file = "trace/similar/warm7.txt"
    test_file = "trace/similar/test7.txt"

    model_name = "model/lstm_model_seq6.pt"
    top_k_list = [256,512, 768, 1024]
    lstm256_hits = []
    lstm512_hits = []
    lstm768_hits = []
    lstm1024_hits = []

    for seed in range(42, 43):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # 执行训练
        train_model(train_file, model_name, num_epochs=80)
        
        for top_k in top_k_list:
            res = getRit(train_file, warm_file, model_name, test_file, top_k)
            if top_k == 256:
                lstm256_hits.append(res)
            elif top_k == 512:
                lstm512_hits.append(res)
            elif top_k == 768:
                lstm768_hits.append(res)
            elif top_k == 1024:
                lstm1024_hits.append(res)
    
    print(f"LSTM@256: {lstm256_hits}")
    print(f"LSTM@256_mean: {np.mean(lstm256_hits):.4f}")
    
    print(f"LSTM@512: {lstm512_hits}")
    print(f"LSTM@512_mean: {np.mean(lstm512_hits):.4f}")
    
    print(f"LSTM@768: {lstm768_hits}")
    print(f"LSTM@768_mean: {np.mean(lstm768_hits):.4f}")
    
    print(f"LSTM@1024: {lstm1024_hits}")
    print(f"LSTM@1024_mean: {np.mean(lstm1024_hits):.4f}")

# 执行训练
if __name__ == "__main__":
    
    
    # 模型对比
    # exp_assess()

    # 命中率实验
    # exp_hit_rate()
    exp_topk_hit_rate()


    
