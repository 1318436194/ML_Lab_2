#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import load_svmlight_file

def load_data(train_path, test_path, n_features=123):
    """
    加载训练集和测试集
    
    参数:
        train_path: 训练集路径
        test_path: 测试集路径
        n_features: 特征维度，默认为123
    
    返回:
        X_train: 训练集特征矩阵
        y_train: 训练集标签
        X_test: 测试集特征矩阵
        y_test: 测试集标签
    """
    # 加载训练集
    X_train, y_train = load_svmlight_file(train_path, n_features=n_features)
    # 加载测试集
    X_test, y_test = load_svmlight_file(test_path, n_features=n_features)
    
    # 将稀疏矩阵转换为密集矩阵
    X_train = X_train.toarray()
    X_test = X_test.toarray()
    
    # 添加偏置项
    X_train = np.c_[np.ones(X_train.shape[0]), X_train]
    X_test = np.c_[np.ones(X_test.shape[0]), X_test]
    
    # 将标签转换为+1和-1
    y_train[y_train == 0] = -1
    y_test[y_test == 0] = -1
    
    return X_train, y_train, X_test, y_test

def batch_iterator(X, y, batch_size):
    """
    生成批量数据的迭代器
    
    参数:
        X: 特征矩阵
        y: 标签
        batch_size: 批量大小
    
    返回:
        X_batch: 批量特征矩阵
        y_batch: 批量标签
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    for start_idx in range(0, n_samples, batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        yield X[batch_indices], y[batch_indices] 