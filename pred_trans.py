import pandas as pd
import numpy as np
import itertools
from datetime import datetime, timedelta
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import math
from pred_lstm import get_processed_data, PageAccessDataset, WeightedBCELoss
from eval import evaluate_model_comprehensive, recall_at_k, precision_at_k
from run import  load_trace, simulate_cache_only, simulate_cache

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerMultiLabelPredictor(nn.Module):
    """基于Transformer的多标签预测模型"""
    def __init__(self, input_size, total_pages, d_model=128, nhead=4, 
                 num_layers=2, dim_feedforward=256, dropout=0.2):
        super(TransformerMultiLabelPredictor, self).__init__()
        
        self.total_pages = total_pages
        self.d_model = d_model
        
        # 输入投影层
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.output_layer = nn.Linear(d_model, total_pages)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        # 初始化输入投影层
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.constant_(self.input_projection.bias, 0)
        
        # 初始化输出层
        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.1)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, -3.0)

    def forward(self, x):
        """
        Args:
            x: 输入序列 [batch_size, seq_len, input_size]
        Returns:
            logits: 每个页面的预测分数 [batch_size, total_pages]
        """
        # 输入投影
        x = self.input_projection(x) * math.sqrt(self.d_model)  # [batch_size, seq_len, d_model]
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        x = self.transformer_encoder(x)  # [batch_size, seq_len, d_model]
        
        # 使用最后一层的所有时间步的平均值作为序列表示
        context_vector = torch.mean(x, dim=1)  # [batch_size, d_model]
        
        # 应用Dropout
        context_vector = self.dropout(context_vector)
        
        # 通过输出层
        logits = self.output_layer(context_vector)  # [batch_size, total_pages]
        
        return logits


def train_complete_model(log_path, model_name, num_epochs=50, window_minutes=5, seq_length=12, pred_length=6, min_access_count=2, use_cache=True):
    """
    完整的模型训练流程，固定训练轮数并只保存最后一次模型
    """
    # 1. 收集数据
    processed_data = get_processed_data(
        log_path, 
        window_minutes=window_minutes, 
        min_access_count=min_access_count, 
        top_k=None,
        seq_length=seq_length, 
        pred_length=pred_length,
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

    # 4. 划分训练集和验证集和测试集
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
    model = TransformerMultiLabelPredictor(
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
    
    # 10. 保存最后一次模型
    # torch.save({
    #     'epoch': num_epochs,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'loss': val_loss,
    #     'page_mapping': page_to_idx,
    #     'idx_mapping': idx_to_page,
    #     'input_size': num_pages,
    #     'total_pages': num_pages
    # }, model_name)
    # print(f"保存最终模型，验证损失: {val_loss:.4f}")
    
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
    
    # 添加调试信息
    print(f"预测概率统计: max={np.max(probs):.4f}, min={np.min(probs):.4f}, mean={np.mean(probs):.4f}")
    print(f"大于阈值{threshold}的概率数量: {np.sum(probs >= threshold)}")

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

# 修改getPredList函数以支持top_k参数
def getPredList(train_file, warm_file, model_name, seq_length=6, pred_length=6, window_minutes=5, threshold=0.3, top_k=None, min_access_count=2, use_cache=True):

    """获取预测列表（使用缓存）"""
    processed_data = get_processed_data(
        train_file,
        window_minutes=window_minutes,
        min_access_count=min_access_count,
        top_k=None,
        seq_length=seq_length,  # 注意：这里改为12以匹配训练
        pred_length=pred_length,
        use_cache=use_cache
    )
    
    page_to_idx = processed_data['page_to_idx']
    # 加载模型
    model = TransformerMultiLabelPredictor(
        input_size=len(page_to_idx),
        total_pages=len(page_to_idx)
    ).to("cpu")
    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint['model_state_dict'])

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
    print("示例页面:", pred_list[:20])  

    return pred_list

def loadModel(train_file, model_name, window_minutes=5, seq_length=12, pred_length=6, use_cache=True):

    """加载模型和测试数据（使用缓存）"""
    processed_data = get_processed_data(
        train_file,
        window_minutes=window_minutes,
        min_access_count=2,
        top_k=None,
        seq_length=seq_length, 
        pred_length=pred_length,
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
    model = TransformerMultiLabelPredictor(
        input_size=len(processed_data['page_to_idx']),
        total_pages=len(processed_data['page_to_idx'])
    ).to("cpu")
    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, test_loader


def train_model(train_file, model_name, num_epochs, window_minutes=5, seq_length=6, pred_length=6, min_access_count=2, use_cache=True):
    model, page_to_idx, idx_to_page = train_complete_model(train_file, model_name, num_epochs=num_epochs, window_minutes=window_minutes, seq_length=seq_length, pred_length=pred_length, min_access_count=min_access_count, use_cache=use_cache)

def eval_model(train_file, model_name, window_minutes=5, seq_length=6, pred_length=6, min_access_count=2, use_cache=True):
    # 评估
    device = "cpu"
    model, test_loader = loadModel(train_file, model_name, window_minutes=window_minutes, seq_length=seq_length, pred_length=pred_length)
    comprehensive_results = evaluate_model_comprehensive(model, test_loader, device, topk_list=[256, 512, 768, 1024,2048])
    print("综合指标:", comprehensive_results)
    return comprehensive_results

def getRit(train_file, warm_file, model_name, test_file, top_k, window_minutes=5, seq_length=6, pred_length=6, min_access_count=2, use_cache=True):
    #取预测数据
    # 记录开始时间
    import time
    start_time = time.time()
    pred_list = getPredList(train_file, warm_file, model_name, seq_length=seq_length,pred_length=pred_length, window_minutes=window_minutes, threshold=0.2, top_k=top_k, min_access_count=min_access_count, use_cache=use_cache)
    # 记录结束时间
    end_time = time.time()
    # 计算运行时间
    run_time = end_time - start_time
    print(f"推理时间: {run_time:.4f} 秒")
    
    test_trace = load_trace(test_file)
    # transformer_curve, transformer_hit_rate = simulate_cache_only(test_trace, pred_list)
    transformer_curve, transformer_hit_rate = simulate_cache(test_trace, pred_list, 1024)
    print("命中率比较：")
    print(f"Transformer: {transformer_hit_rate:.4f}")
    return transformer_hit_rate


# 消融实验
def exp_ablation():
    train_file = 'trace/trace0-11h.csv'
    warm_file = 'trace/trace13h.csv'
    test_file = 'trace/trace14h.csv'
    model_name = "model/transformer_model_min2.pt"
    top_k = (int)(1024*(1))

    window_minutes = 5
    seq_length = 6
    pred_length = 6
    min_access_count = 2


    res = []

    for seed in range(42, 47):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # 执行训练
        train_model(train_file, model_name, num_epochs=80, window_minutes=window_minutes, seq_length=seq_length, pred_length=pred_length, min_access_count=min_access_count)

        # 评估
        # eval_model(train_file, model_name, window_minutes=window_minutes, seq_length=seq_length, pred_length=pred_length, min_access_count=min_access_count)

        # 测命中率
        res.append(getRit(train_file, warm_file, model_name, test_file, top_k, window_minutes=window_minutes, seq_length=seq_length, pred_length=pred_length, min_access_count=min_access_count))
        
    print(res)
    print(f"Transformer: {np.mean(res):.4f}")

# 模型指标对比实验
def exp_assess():
    train_file = 'trace/trace0-11h.csv'
    model_name = "model/transformer_model_seq6.pt"

    window_minutes = 5
    seq_length = 6
    pred_length = 6

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
        train_model(train_file, model_name, num_epochs=80, window_minutes=window_minutes, seq_length=seq_length, pred_length=pred_length)

        # 评估
        result = eval_model(train_file, model_name, window_minutes=window_minutes, seq_length=seq_length, pred_length=pred_length)
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
    # train_file = 'trace/trace0-15h.csv'
    # warm_file = 'trace/trace16h.csv'
    # test_file = 'trace/trace17h.csv'

    # 数据集2
    # train_file = 'trace/trace0-11h.csv'
    # warm_file = 'trace/trace12h.csv'
    # test_file = 'trace/trace13h.csv'

    

    # 数据集3
    # train_file = 'trace/trace0-6h.csv'
    # warm_file = 'trace/trace6h.csv'
    # test_file = 'trace/trace7h.csv'

    # 数据集4
    # train_file = 'trace/trace11-22h.csv'
    # warm_file = 'trace/trace22h.csv'
    # test_file = 'trace/trace23h.csv'

    #数据集11
    train_file = "trace/similar/train7.txt"
    warm_file = "trace/similar/warm7.txt"
    test_file = "trace/similar/test7.txt"

    model_name = "model/transformer_model_seq6.pt"
    top_k_list = [768, 1024,2048]
    transformer768_hits = []
    transformer1024_hits = []
    transformer2048_hits = []

    for seed in range(42, 47):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # 执行训练
        train_model(train_file, model_name, num_epochs=80, window_minutes=5, seq_length=6, pred_length=6, min_access_count=2, use_cache=True)
        
        for top_k in top_k_list:
            res = getRit(train_file, warm_file, model_name, test_file, top_k, window_minutes=5, seq_length=6, pred_length=6, min_access_count=2, use_cache=True)
            if top_k == 768:
                transformer768_hits.append(res)
            elif top_k == 1024:
                transformer1024_hits.append(res)
            elif top_k == 2048:
                transformer2048_hits.append(res)
    
    print(f"Transformer@768: {transformer768_hits}")
    print(f"Transformer@768_mean: {np.mean(transformer768_hits):.4f}")
    
    print(f"Transformer@1024: {transformer1024_hits}")
    print(f"Transformer@1024_mean: {np.mean(transformer1024_hits):.4f}")
    
    print(f"Transformer@2048: {transformer2048_hits}")
    print(f"Transformer@2048_mean: {np.mean(transformer2048_hits):.4f}")
    

# top命中率实验
def exp_topk_hit_rate():

    #数据集11
    train_file = "trace/similar/train7.txt"
    warm_file = "trace/similar/warm7.txt"
    test_file = "trace/similar/test7.txt"

    model_name = "model/transformer_model_seq6.pt"
    top_k_list = [256,512, 768, 1024]
    transformer256_hits = []
    transformer512_hits = []
    transformer768_hits = []
    transformer1024_hits = []

    for seed in range(42, 43):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # 执行训练
        train_model(train_file, model_name, num_epochs=80, window_minutes=5, seq_length=6, pred_length=6, min_access_count=2, use_cache=True)
        
        for top_k in top_k_list:
            res = getRit(train_file, warm_file, model_name, test_file, top_k, window_minutes=5, seq_length=6, pred_length=6, min_access_count=2, use_cache=True)
            if top_k == 256:
                transformer256_hits.append(res)
            elif top_k == 512:
                transformer512_hits.append(res)
            elif top_k == 768:
                transformer768_hits.append(res)
            elif top_k == 1024:
                transformer1024_hits.append(res)
    
    print(f"Transformer@256: {transformer256_hits}")
    print(f"Transformer@256_mean: {np.mean(transformer256_hits):.4f}")
    
    print(f"Transformer@512: {transformer512_hits}")
    print(f"Transformer@512_mean: {np.mean(transformer512_hits):.4f}")
    
    print(f"Transformer@768: {transformer768_hits}")
    print(f"Transformer@768_mean: {np.mean(transformer768_hits):.4f}")
    
    print(f"Transformer@1024: {transformer1024_hits}")
    print(f"Transformer@1024_mean: {np.mean(transformer1024_hits):.4f}")


# 执行训练
if __name__ == "__main__":

    # 设置随机种子42-46
    seed = 42

    #模型对比
    # exp_assess()

    #运行消融实验
    # exp_ablation()

    #命中率实验
    # exp_hit_rate()
    exp_topk_hit_rate()

    
    