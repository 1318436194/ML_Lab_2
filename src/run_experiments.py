#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
实验运行脚本
用于运行逻辑回归和SVM实验
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from main import run_logistic_regression_experiment, run_svm_experiment, compare_logistic_regression_and_svm

def run_all_experiments():
    """
    运行所有实验
    """
    start_time = time.time()
    
    # 创建报告目录
    os.makedirs('../report/my_report', exist_ok=True)
    
    print("="*50)
    print("开始逻辑回归实验...")
    print("="*50)
    run_logistic_regression_experiment()
    
    print("\n" + "="*50)
    print("开始SVM实验...")
    print("="*50)
    run_svm_experiment()
    
    print("\n" + "="*50)
    print("比较逻辑回归和SVM...")
    print("="*50)
    compare_logistic_regression_and_svm()
    
    end_time = time.time()
    print(f"\n所有实验完成！总用时: {(end_time - start_time) / 60:.2f} 分钟")

if __name__ == "__main__":
    run_all_experiments()