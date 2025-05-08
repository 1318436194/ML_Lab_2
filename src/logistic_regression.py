#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm

from data_utils import batch_iterator
from optimizers import SGD, Adam

class LogisticRegression:
    """
    逻辑回归模型
    """
    def __init__(self, input_dim, init_method='zeros', optimizer='sgd', learning_rate=0.01):
        """
        初始化逻辑回归模型
        
        参数:
            input_dim: 输入特征维度
            init_method: 参数初始化方法，可选 'zeros', 'random', 'normal'
            optimizer: 优化器类型，可选 'sgd', 'adam'
            learning_rate: 学习率
        """
        # 初始化参数
        if init_method == 'zeros':
            self.w = np.zeros(input_dim)
        elif init_method == 'random':
            self.w = np.random.random(input_dim) * 0.1
        elif init_method == 'normal':
            self.w = np.random.normal(0, 0.01, input_dim)
        else:
            raise ValueError("不支持的初始化方法")
        
        # 初始化优化器
        if optimizer == 'sgd':
            self.optimizer = SGD(learning_rate=learning_rate)
        elif optimizer == 'adam':
            self.optimizer = Adam(learning_rate=learning_rate)
        else:
            raise ValueError("不支持的优化器类型")
    
    def sigmoid(self, z):
        """
        sigmoid激活函数
        
        参数:
            z: 输入值
        
        返回:
            sigmoid(z)
        """
        # 使用截断避免溢出
        z = np.clip(z, -30, 30)
        return 1 / (1 + np.exp(-z))
    
    def forward(self, X):
        """
        前向传播
        
        参数:
            X: 输入特征
        
        返回:
            y_pred: 预测概率
        """
        z = np.dot(X, self.w)
        return self.sigmoid(z)
    
    def loss(self, y_true, y_pred):
        """
        计算损失函数值 (交叉熵损失)
        
        参数:
            y_true: 真实标签
            y_pred: 预测概率
        
        返回:
            loss: 损失函数值
        """
        # 避免对数中的0值
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        
        # 将y_true转换为二分类标签(0或1)
        y_binary = (y_true + 1) / 2
        
        # 计算交叉熵损失
        loss = -np.mean(y_binary * np.log(y_pred) + (1 - y_binary) * np.log(1 - y_pred))
        return loss
    
    def gradient(self, X, y_true, y_pred):
        """
        计算损失函数的梯度
        
        参数:
            X: 输入特征
            y_true: 真实标签
            y_pred: 预测概率
        
        返回:
            grad: 梯度
        """
        # 将y_true转换为二分类标签(0或1)
        y_binary = (y_true + 1) / 2
        
        # 计算梯度
        m = X.shape[0]
        grad = (1/m) * np.dot(X.T, (y_pred - y_binary))
        return grad
    
    def predict_proba(self, X):
        """
        预测概率
        
        参数:
            X: 输入特征
        
        返回:
            y_pred: 预测概率
        """
        return self.forward(X)
    
    def predict(self, X, threshold=0.5):
        """
        预测标签
        
        参数:
            X: 输入特征
            threshold: 阈值
        
        返回:
            y_pred: 预测标签 (-1或1)
        """
        y_proba = self.predict_proba(X)
        y_pred = np.where(y_proba >= threshold, 1, -1)
        return y_pred
    
    def train(self, X_train, y_train, X_test=None, y_test=None, 
              n_iterations=1000, batch_size=32, eval_interval=100):
        """
        训练模型
        
        参数:
            X_train: 训练集特征
            y_train: 训练集标签
            X_test: 测试集特征
            y_test: 测试集标签
            n_iterations: 迭代次数
            batch_size: 批量大小
            eval_interval: 评估间隔
            
        返回:
            history: 训练历史记录
        """
        history = {
            'train_loss': [],
            'train_accuracy': []
        }
        
        if X_test is not None and y_test is not None:
            history['test_loss'] = []
            history['test_accuracy'] = []
        
        for i in tqdm(range(n_iterations)):
            # 获取批量数据
            for X_batch, y_batch in batch_iterator(X_train, y_train, batch_size):
                # 前向传播
                y_pred = self.forward(X_batch)
                
                # 计算梯度
                grad = self.gradient(X_batch, y_batch, y_pred)
                
                # 更新参数
                self.w = self.optimizer.update(self.w, grad)
                break  # 每次迭代只使用一个批次
            
            # 定期评估模型
            if i % eval_interval == 0 or i == n_iterations - 1:
                # 评估训练集
                train_pred = self.predict(X_train)
                train_loss = self.loss(y_train, self.predict_proba(X_train))
                train_acc = np.mean(train_pred == y_train)
                
                history['train_loss'].append(train_loss)
                history['train_accuracy'].append(train_acc)
                
                # 评估测试集
                if X_test is not None and y_test is not None:
                    test_pred = self.predict(X_test)
                    test_loss = self.loss(y_test, self.predict_proba(X_test))
                    test_acc = np.mean(test_pred == y_test)
                    
                    history['test_loss'].append(test_loss)
                    history['test_accuracy'].append(test_acc)
        
        return history 