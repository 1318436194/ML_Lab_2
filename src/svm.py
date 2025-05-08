#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm

from data_utils import batch_iterator
from optimizers import SGD, Adam

class SVM:
    """
    支持向量机模型
    """
    def __init__(self, input_dim, init_method='zeros', optimizer='sgd', learning_rate=0.01, C=1.0, loss_type='hinge'):
        """
        初始化SVM模型
        
        参数:
            input_dim: 输入特征维度
            init_method: 参数初始化方法，可选 'zeros', 'random', 'normal'
            optimizer: 优化器类型，可选 'sgd', 'adam'
            learning_rate: 学习率
            C: 正则化参数
            loss_type: 损失函数类型，可选 'hinge', 'squared_hinge'
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
        
        self.C = C  # 正则化参数
        self.loss_type = loss_type
    
    def compute_margin(self, X):
        """
        计算间隔
        
        参数:
            X: 输入特征
        
        返回:
            margin: 间隔
        """
        return np.dot(X, self.w)
    
    def loss(self, y_true, margin):
        """
        计算损失函数值
        
        参数:
            y_true: 真实标签
            margin: 间隔
        
        返回:
            loss: 损失函数值
        """
        if self.loss_type == 'hinge':
            # Hinge Loss: max(0, 1 - y*f(x))
            hinge_loss = np.maximum(0, 1 - y_true * margin)
            loss = np.mean(hinge_loss) + 0.5 * (1/self.C) * np.sum(self.w[1:]**2)
        
        elif self.loss_type == 'squared_hinge':
            # Squared Hinge Loss: max(0, 1 - y*f(x))^2
            hinge_loss = np.maximum(0, 1 - y_true * margin)
            loss = np.mean(hinge_loss**2) + 0.5 * (1/self.C) * np.sum(self.w[1:]**2)
        
        else:
            raise ValueError("不支持的损失函数类型")
        
        return loss
    
    def gradient(self, X, y_true, margin):
        """
        计算损失函数的梯度
        
        参数:
            X: 输入特征
            y_true: 真实标签
            margin: 间隔
        
        返回:
            grad: 梯度
        """
        m = X.shape[0]
        
        # 计算hinge loss的梯度
        if self.loss_type == 'hinge':
            # Hinge Loss梯度: -y_i * x_i if y_i * f(x_i) < 1 else 0
            mask = (y_true * margin) < 1
            d_hinge = np.zeros_like(margin)
            d_hinge[mask] = -y_true[mask]
            grad_hinge = (1/m) * np.dot(X.T, d_hinge)
        
        elif self.loss_type == 'squared_hinge':
            # Squared Hinge Loss梯度: -2 * (1 - y_i * f(x_i)) * y_i * x_i if y_i * f(x_i) < 1 else 0
            hinge = 1 - y_true * margin
            mask = hinge > 0
            d_hinge = np.zeros_like(margin)
            d_hinge[mask] = -2 * hinge[mask] * y_true[mask]
            grad_hinge = (1/m) * np.dot(X.T, d_hinge)
        
        else:
            raise ValueError("不支持的损失函数类型")
        
        # 添加L2正则化的梯度
        grad_reg = np.zeros_like(self.w)
        grad_reg[1:] = (1/self.C) * self.w[1:]
        
        return grad_hinge + grad_reg
    
    def predict(self, X, threshold=0.0):
        """
        预测标签
        
        参数:
            X: 输入特征
            threshold: 阈值
        
        返回:
            y_pred: 预测标签 (-1或1)
        """
        scores = self.compute_margin(X)
        y_pred = np.where(scores >= threshold, 1, -1)
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
                # 计算间隔
                margin = self.compute_margin(X_batch)
                
                # 计算梯度
                grad = self.gradient(X_batch, y_batch, margin)
                
                # 更新参数
                self.w = self.optimizer.update(self.w, grad)
                break  # 每次迭代只使用一个批次
            
            # 定期评估模型
            if i % eval_interval == 0 or i == n_iterations - 1:
                # 评估训练集
                train_margin = self.compute_margin(X_train)
                train_loss = self.loss(y_train, train_margin)
                train_pred = self.predict(X_train)
                train_acc = np.mean(train_pred == y_train)
                
                history['train_loss'].append(train_loss)
                history['train_accuracy'].append(train_acc)
                
                # 评估测试集
                if X_test is not None and y_test is not None:
                    test_margin = self.compute_margin(X_test)
                    test_loss = self.loss(y_test, test_margin)
                    test_pred = self.predict(X_test)
                    test_acc = np.mean(test_pred == y_test)
                    
                    history['test_loss'].append(test_loss)
                    history['test_accuracy'].append(test_acc)
        
        return history 