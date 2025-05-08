#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class SGD:
    """
    随机梯度下降优化器
    """
    def __init__(self, learning_rate=0.01):
        """
        初始化SGD优化器
        
        参数:
            learning_rate: 学习率
        """
        self.learning_rate = learning_rate
    
    def update(self, w, grad):
        """
        更新参数
        
        参数:
            w: 当前参数
            grad: 梯度
        
        返回:
            w_updated: 更新后的参数
        """
        return w - self.learning_rate * grad


class Adam:
    """
    Adam优化器
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        初始化Adam优化器
        
        参数:
            learning_rate: 学习率
            beta1: 一阶矩估计的指数衰减率
            beta2: 二阶矩估计的指数衰减率
            epsilon: 小值，防止除零错误
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # 一阶矩估计
        self.v = None  # 二阶矩估计
        self.t = 0     # 时间步
    
    def update(self, w, grad):
        """
        更新参数
        
        参数:
            w: 当前参数
            grad: 梯度
        
        返回:
            w_updated: 更新后的参数
        """
        self.t += 1
        
        # 初始化动量项
        if self.m is None:
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)
        
        # 更新偏置校正的一阶和二阶矩估计
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(grad)
        
        # 计算偏置校正
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # 更新参数
        w_updated = w - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return w_updated 