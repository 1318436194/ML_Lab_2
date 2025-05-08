#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(y_true, y_pred):
    """
    评估模型性能
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
    
    返回:
        metrics: 包含各种评价指标的字典
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics

def plot_training_history(history, title=None):
    """
    绘制训练历史记录
    
    参数:
        history: 包含损失值和准确率的字典
        title: 图表标题
    """
    plt.figure(figsize=(12, 5))
    
    # 绘制损失值
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    if 'test_loss' in history:
        plt.plot(history['test_loss'], label='Test Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over iterations')
    
    # 绘制准确率
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accuracy'], label='Training Accuracy')
    if 'test_accuracy' in history:
        plt.plot(history['test_accuracy'], label='Test Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over iterations')
    
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    
    return plt 