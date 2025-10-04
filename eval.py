import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Tuple
import torch
import torch.nn.functional as F

def precision_at_k(pred_scores: np.ndarray, true_labels: np.ndarray, k: int) -> float:
    """Precision@k for multi-label prediction - 改进版本"""
    if k > pred_scores.shape[1]:
        k = pred_scores.shape[1]
    
    # 获取每个样本的前k个预测
    topk_idx = np.argpartition(-pred_scores, k-1, axis=1)[:, :k]
    
    precisions = []
    for i in range(pred_scores.shape[0]):
        hits = true_labels[i, topk_idx[i]].sum()
        precisions.append(hits / k)
    
    return float(np.mean(precisions)) if precisions else 0.0

def recall_at_k(pred_scores: np.ndarray, true_labels: np.ndarray, k: int) -> float:
    """Recall@k for multi-label prediction - 改进版本"""
    if k > pred_scores.shape[1]:
        k = pred_scores.shape[1]
    
    topk_idx = np.argpartition(-pred_scores, k-1, axis=1)[:, :k]
    recalls = []
    
    for i in range(pred_scores.shape[0]):
        actual_pos = true_labels[i].sum()
        if actual_pos == 0:
            continue  # 跳过没有正样本的窗口
        hits = true_labels[i, topk_idx[i]].sum()
        recalls.append(hits / actual_pos)
    
    return float(np.mean(recalls)) if recalls else 0.0

def map_at_k(pred_scores: np.ndarray, true_labels: np.ndarray, k: int) -> float:
    """Mean Average Precision at k - 改进版本"""
    N, P = pred_scores.shape
    if k > P:
        k = P
        
    APs = []
    for i in range(N):
        # 获取排序后的索引（从高到低）
        sorted_idx = np.argsort(-pred_scores[i])
        # 只考虑前k个
        relevant_idx = sorted_idx[:k]
        
        # 计算平均精度
        num_hits = 0
        precision_sum = 0.0
        
        for rank, idx in enumerate(relevant_idx, start=1):
            if true_labels[i, idx] == 1:
                num_hits += 1
                precision_sum += num_hits / rank
        
        if num_hits > 0:
            APs.append(precision_sum / num_hits)
        # 如果没有命中，AP为0，不加入计算
    
    return float(np.mean(APs)) if APs else 0.0

def f1_score_at_k(pred_scores: np.ndarray, true_labels: np.ndarray, k: int) -> float:
    """F1-score@k for multi-label prediction"""
    precision = precision_at_k(pred_scores, true_labels, k)
    recall = recall_at_k(pred_scores, true_labels, k)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)

def mse_mae(pred_scores: np.ndarray, true_labels: np.ndarray) -> Tuple[float, float]:
    """改进的MSE和MAE计算，适用于多标签分类"""
    # 将预测分数通过sigmoid转换为概率
    pred_probs = 1 / (1 + np.exp(-pred_scores))  # 手动sigmoid
    
    # 限制概率范围，避免极端值
    pred_probs = np.clip(pred_probs, 1e-7, 1-1e-7)
    
    y_true = true_labels.ravel()
    y_pred = pred_probs.ravel()
    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    return float(mse), float(mae)

def binary_cross_entropy_loss(pred_scores: np.ndarray, true_labels: np.ndarray) -> float:
    """计算二元交叉熵损失"""
    # 将预测分数通过sigmoid转换为概率
    pred_probs = 1 / (1 + np.exp(-pred_scores))
    pred_probs = np.clip(pred_probs, 1e-7, 1-1e-7)
    
    # 计算BCE
    bce = -np.mean(true_labels * np.log(pred_probs) + (1 - true_labels) * np.log(1 - pred_probs))
    return float(bce)

def evaluate_model_comprehensive(model, test_loader, device, topk_list=None, threshold=0.5):
    """
    综合评估模型性能
    """
    if topk_list is None:
        topk_list = [256, 512, 1024]
    
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            all_preds.append(logits.cpu().numpy())
            all_labels.append(y.numpy())
    
    pred_scores = np.concatenate(all_preds, axis=0)
    true_labels = np.concatenate(all_labels, axis=0)
    
    metrics = {}
    
    # 排序指标
    for k in topk_list:
        metrics[f'Precision@{k}'] = precision_at_k(pred_scores, true_labels, k)
        metrics[f'Recall@{k}'] = recall_at_k(pred_scores, true_labels, k)
        metrics[f'MAP@{k}'] = map_at_k(pred_scores, true_labels, k)
        metrics[f'F1@{k}'] = f1_score_at_k(pred_scores, true_labels, k)
    
    # 回归指标
    metrics['MSE'], metrics['MAE'] = mse_mae(pred_scores, true_labels)
    metrics['BCE'] = binary_cross_entropy_loss(pred_scores, true_labels)
    
    # 基于阈值的分类指标
    pred_probs = 1 / (1 + np.exp(-pred_scores))
    pred_binary = (pred_probs > threshold).astype(float)
    
    # 准确率
    accuracy = np.mean((pred_binary == true_labels).astype(float))
    metrics['Accuracy'] = float(accuracy)
    
    # 正样本的统计
    positive_rate = np.mean(true_labels)
    predicted_positive_rate = np.mean(pred_binary)
    metrics['True_Positive_Rate'] = float(positive_rate)
    metrics['Predicted_Positive_Rate'] = float(predicted_positive_rate)
    
    return metrics

def evaluate_model(model, test_loader, device, topk_list=None):
    """保持向后兼容的评估函数"""
    comprehensive_metrics = evaluate_model_comprehensive(
        model, test_loader, device, topk_list
    )
    
    # 只返回原始函数定义的指标
    basic_metrics = {}
    for k in (topk_list or [256, 512, 1024]):
        basic_metrics[f'Precision@{k}'] = comprehensive_metrics[f'Precision@{k}']
        basic_metrics[f'Recall@{k}'] = comprehensive_metrics[f'Recall@{k}']
        basic_metrics[f'MAP@{k}'] = comprehensive_metrics[f'MAP@{k}']
    
    basic_metrics['MSE'] = comprehensive_metrics['MSE']
    basic_metrics['MAE'] = comprehensive_metrics['MAE']
    
    return basic_metrics