#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt

from data_utils import load_data
from logistic_regression import LogisticRegression
from svm import SVM
from evaluation import evaluate_model, plot_training_history

# 数据路径
TRAIN_PATH = '../datasets/a9a.txt'
TEST_PATH = '../datasets/a9a.t.txt'

# 实验参数
INIT_METHODS = ['zeros', 'random', 'normal']
OPTIMIZERS = ['sgd', 'adam']
LEARNING_RATES = [0.01, 0.001]
BATCH_SIZES = [32, 64, 128]
N_ITERATIONS = 1000
EVAL_INTERVAL = 100

def run_logistic_regression_experiment():
    """
    运行逻辑回归实验
    """
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_data(TRAIN_PATH, TEST_PATH)
    
    print("Running Logistic Regression experiments...")
    
    # 实验1：不同初始化方法
    init_histories = {}
    for init_method in INIT_METHODS:
        print(f"  Init method: {init_method}")
        lr = LogisticRegression(
            input_dim=X_train.shape[1],
            init_method=init_method,
            optimizer='sgd',
            learning_rate=0.01
        )
        history = lr.train(
            X_train, y_train, X_test, y_test,
            n_iterations=N_ITERATIONS,
            batch_size=32,
            eval_interval=EVAL_INTERVAL
        )
        init_histories[init_method] = history
    
    # 绘制不同初始化方法的结果
    plt.figure(figsize=(12, 10))
    
    # 训练损失
    plt.subplot(2, 2, 1)
    for init_method in INIT_METHODS:
        plt.plot(init_histories[init_method]['train_loss'], label=f'Train Loss ({init_method})')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Loss')
    plt.title('Training Loss for Different Initialization Methods')
    plt.legend()
    
    # 测试损失
    plt.subplot(2, 2, 2)
    for init_method in INIT_METHODS:
        plt.plot(init_histories[init_method]['test_loss'], label=f'Test Loss ({init_method})')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Loss')
    plt.title('Test Loss for Different Initialization Methods')
    plt.legend()
    
    # 训练准确率
    plt.subplot(2, 2, 3)
    for init_method in INIT_METHODS:
        plt.plot(init_histories[init_method]['train_accuracy'], label=f'Train Accuracy ({init_method})')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy for Different Initialization Methods')
    plt.legend()
    
    # 测试准确率
    plt.subplot(2, 2, 4)
    for init_method in INIT_METHODS:
        plt.plot(init_histories[init_method]['test_accuracy'], label=f'Test Accuracy ({init_method})')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy for Different Initialization Methods')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('../report/my_report/logistic_regression_init_methods.png')
    
    # 实验2：不同优化器
    opt_histories = {}
    for optimizer in OPTIMIZERS:
        print(f"  Optimizer: {optimizer}")
        lr = LogisticRegression(
            input_dim=X_train.shape[1],
            init_method='normal',
            optimizer=optimizer,
            learning_rate=0.01 if optimizer == 'sgd' else 0.001
        )
        history = lr.train(
            X_train, y_train, X_test, y_test,
            n_iterations=N_ITERATIONS,
            batch_size=32,
            eval_interval=EVAL_INTERVAL
        )
        opt_histories[optimizer] = history
    
    # 绘制不同优化器的结果
    plt.figure(figsize=(12, 10))
    
    # 训练损失
    plt.subplot(2, 2, 1)
    for optimizer in OPTIMIZERS:
        plt.plot(opt_histories[optimizer]['train_loss'], label=f'Train Loss ({optimizer})')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Loss')
    plt.title('Training Loss for Different Optimizers')
    plt.legend()
    
    # 测试损失
    plt.subplot(2, 2, 2)
    for optimizer in OPTIMIZERS:
        plt.plot(opt_histories[optimizer]['test_loss'], label=f'Test Loss ({optimizer})')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Loss')
    plt.title('Test Loss for Different Optimizers')
    plt.legend()
    
    # 训练准确率
    plt.subplot(2, 2, 3)
    for optimizer in OPTIMIZERS:
        plt.plot(opt_histories[optimizer]['train_accuracy'], label=f'Train Accuracy ({optimizer})')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy for Different Optimizers')
    plt.legend()
    
    # 测试准确率
    plt.subplot(2, 2, 4)
    for optimizer in OPTIMIZERS:
        plt.plot(opt_histories[optimizer]['test_accuracy'], label=f'Test Accuracy ({optimizer})')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy for Different Optimizers')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('../report/my_report/logistic_regression_optimizers.png')
    
    # 实验3：不同批量大小
    batch_histories = {}
    for batch_size in BATCH_SIZES:
        print(f"  Batch size: {batch_size}")
        lr = LogisticRegression(
            input_dim=X_train.shape[1],
            init_method='normal',
            optimizer='adam',
            learning_rate=0.001
        )
        history = lr.train(
            X_train, y_train, X_test, y_test,
            n_iterations=N_ITERATIONS,
            batch_size=batch_size,
            eval_interval=EVAL_INTERVAL
        )
        batch_histories[batch_size] = history
    
    # 绘制不同批量大小的结果
    plt.figure(figsize=(12, 10))
    
    # 训练损失
    plt.subplot(2, 2, 1)
    for batch_size in BATCH_SIZES:
        plt.plot(batch_histories[batch_size]['train_loss'], label=f'Train Loss (batch={batch_size})')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Loss')
    plt.title('Training Loss for Different Batch Sizes')
    plt.legend()
    
    # 测试损失
    plt.subplot(2, 2, 2)
    for batch_size in BATCH_SIZES:
        plt.plot(batch_histories[batch_size]['test_loss'], label=f'Test Loss (batch={batch_size})')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Loss')
    plt.title('Test Loss for Different Batch Sizes')
    plt.legend()
    
    # 训练准确率
    plt.subplot(2, 2, 3)
    for batch_size in BATCH_SIZES:
        plt.plot(batch_histories[batch_size]['train_accuracy'], label=f'Train Accuracy (batch={batch_size})')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy for Different Batch Sizes')
    plt.legend()
    
    # 测试准确率
    plt.subplot(2, 2, 4)
    for batch_size in BATCH_SIZES:
        plt.plot(batch_histories[batch_size]['test_accuracy'], label=f'Test Accuracy (batch={batch_size})')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy for Different Batch Sizes')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('../report/my_report/logistic_regression_batch_sizes.png')

def run_svm_experiment():
    """
    运行SVM实验
    """
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_data(TRAIN_PATH, TEST_PATH)
    
    print("Running SVM experiments...")
    
    # 实验1：不同初始化方法
    init_histories = {}
    for init_method in INIT_METHODS:
        print(f"  Init method: {init_method}")
        svm = SVM(
            input_dim=X_train.shape[1],
            init_method=init_method,
            optimizer='sgd',
            learning_rate=0.01,
            C=1.0,
            loss_type='hinge'
        )
        history = svm.train(
            X_train, y_train, X_test, y_test,
            n_iterations=N_ITERATIONS,
            batch_size=32,
            eval_interval=EVAL_INTERVAL
        )
        init_histories[init_method] = history
    
    # 绘制不同初始化方法的结果
    plt.figure(figsize=(12, 10))
    
    # 训练损失
    plt.subplot(2, 2, 1)
    for init_method in INIT_METHODS:
        plt.plot(init_histories[init_method]['train_loss'], label=f'Train Loss ({init_method})')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Loss')
    plt.title('Training Loss for Different Initialization Methods')
    plt.legend()
    
    # 测试损失
    plt.subplot(2, 2, 2)
    for init_method in INIT_METHODS:
        plt.plot(init_histories[init_method]['test_loss'], label=f'Test Loss ({init_method})')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Loss')
    plt.title('Test Loss for Different Initialization Methods')
    plt.legend()
    
    # 训练准确率
    plt.subplot(2, 2, 3)
    for init_method in INIT_METHODS:
        plt.plot(init_histories[init_method]['train_accuracy'], label=f'Train Accuracy ({init_method})')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy for Different Initialization Methods')
    plt.legend()
    
    # 测试准确率
    plt.subplot(2, 2, 4)
    for init_method in INIT_METHODS:
        plt.plot(init_histories[init_method]['test_accuracy'], label=f'Test Accuracy ({init_method})')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy for Different Initialization Methods')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('../report/my_report/svm_init_methods.png')
    
    # 实验2：不同优化器
    opt_histories = {}
    for optimizer in OPTIMIZERS:
        print(f"  Optimizer: {optimizer}")
        svm = SVM(
            input_dim=X_train.shape[1],
            init_method='normal',
            optimizer=optimizer,
            learning_rate=0.01 if optimizer == 'sgd' else 0.001,
            C=1.0,
            loss_type='hinge'
        )
        history = svm.train(
            X_train, y_train, X_test, y_test,
            n_iterations=N_ITERATIONS,
            batch_size=32,
            eval_interval=EVAL_INTERVAL
        )
        opt_histories[optimizer] = history
    
    # 绘制不同优化器的结果
    plt.figure(figsize=(12, 10))
    
    # 训练损失
    plt.subplot(2, 2, 1)
    for optimizer in OPTIMIZERS:
        plt.plot(opt_histories[optimizer]['train_loss'], label=f'Train Loss ({optimizer})')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Loss')
    plt.title('Training Loss for Different Optimizers')
    plt.legend()
    
    # 测试损失
    plt.subplot(2, 2, 2)
    for optimizer in OPTIMIZERS:
        plt.plot(opt_histories[optimizer]['test_loss'], label=f'Test Loss ({optimizer})')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Loss')
    plt.title('Test Loss for Different Optimizers')
    plt.legend()
    
    # 训练准确率
    plt.subplot(2, 2, 3)
    for optimizer in OPTIMIZERS:
        plt.plot(opt_histories[optimizer]['train_accuracy'], label=f'Train Accuracy ({optimizer})')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy for Different Optimizers')
    plt.legend()
    
    # 测试准确率
    plt.subplot(2, 2, 4)
    for optimizer in OPTIMIZERS:
        plt.plot(opt_histories[optimizer]['test_accuracy'], label=f'Test Accuracy ({optimizer})')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy for Different Optimizers')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('../report/my_report/svm_optimizers.png')
    
    # 实验3：不同损失函数
    loss_types = ['hinge', 'squared_hinge']
    loss_histories = {}
    for loss_type in loss_types:
        print(f"  Loss type: {loss_type}")
        svm = SVM(
            input_dim=X_train.shape[1],
            init_method='normal',
            optimizer='adam',
            learning_rate=0.001,
            C=1.0,
            loss_type=loss_type
        )
        history = svm.train(
            X_train, y_train, X_test, y_test,
            n_iterations=N_ITERATIONS,
            batch_size=32,
            eval_interval=EVAL_INTERVAL
        )
        loss_histories[loss_type] = history
    
    # 绘制不同损失函数的结果
    plt.figure(figsize=(12, 10))
    
    # 训练损失
    plt.subplot(2, 2, 1)
    for loss_type in loss_types:
        plt.plot(loss_histories[loss_type]['train_loss'], label=f'Train Loss ({loss_type})')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Loss')
    plt.title('Training Loss for Different Loss Functions')
    plt.legend()
    
    # 测试损失
    plt.subplot(2, 2, 2)
    for loss_type in loss_types:
        plt.plot(loss_histories[loss_type]['test_loss'], label=f'Test Loss ({loss_type})')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Loss')
    plt.title('Test Loss for Different Loss Functions')
    plt.legend()
    
    # 训练准确率
    plt.subplot(2, 2, 3)
    for loss_type in loss_types:
        plt.plot(loss_histories[loss_type]['train_accuracy'], label=f'Train Accuracy ({loss_type})')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy for Different Loss Functions')
    plt.legend()
    
    # 测试准确率
    plt.subplot(2, 2, 4)
    for loss_type in loss_types:
        plt.plot(loss_histories[loss_type]['test_accuracy'], label=f'Test Accuracy ({loss_type})')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy for Different Loss Functions')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('../report/my_report/svm_loss_functions.png')

def compare_logistic_regression_and_svm():
    """
    比较逻辑回归和SVM
    """
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_data(TRAIN_PATH, TEST_PATH)
    
    print("Running comparison between Logistic Regression and SVM...")
    
    # 逻辑回归
    print("  Training Logistic Regression...")
    lr = LogisticRegression(
        input_dim=X_train.shape[1],
        init_method='normal',
        optimizer='adam',
        learning_rate=0.001
    )
    lr_history = lr.train(
        X_train, y_train, X_test, y_test,
        n_iterations=N_ITERATIONS,
        batch_size=32,
        eval_interval=EVAL_INTERVAL
    )
    
    # SVM - 使用更合理的参数配置
    print("  Training SVM...")
    svm = SVM(
        input_dim=X_train.shape[1],
        init_method='normal',
        optimizer='adam',
        learning_rate=0.001,
        C=10.0,  # 增加C值以减少正则化强度
        loss_type='hinge'
    )
    svm_history = svm.train(
        X_train, y_train, X_test, y_test,
        n_iterations=N_ITERATIONS,
        batch_size=32,
        eval_interval=EVAL_INTERVAL
    )
    
    # 绘制比较结果
    plt.figure(figsize=(12, 10))
    
    # 训练损失
    plt.subplot(2, 2, 1)
    plt.plot(lr_history['train_loss'], label='Logistic Regression')
    plt.plot(svm_history['train_loss'], label='SVM')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    
    # 测试损失
    plt.subplot(2, 2, 2)
    plt.plot(lr_history['test_loss'], label='Logistic Regression')
    plt.plot(svm_history['test_loss'], label='SVM')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Loss')
    plt.title('Test Loss Comparison')
    plt.legend()
    
    # 训练准确率
    plt.subplot(2, 2, 3)
    plt.plot(lr_history['train_accuracy'], label='Logistic Regression')
    plt.plot(svm_history['train_accuracy'], label='SVM')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Comparison')
    plt.legend()
    
    # 测试准确率
    plt.subplot(2, 2, 4)
    plt.plot(lr_history['test_accuracy'], label='Logistic Regression')
    plt.plot(svm_history['test_accuracy'], label='SVM')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy Comparison')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('../report/my_report/lr_vs_svm_comparison.png')
    
    # 评估最终模型性能
    print("Final Logistic Regression performance on test set:")
    lr_pred = lr.predict(X_test)
    lr_metrics = evaluate_model(y_test, lr_pred)
    print(f"  Accuracy: {lr_metrics['accuracy']:.4f}")
    print(f"  Precision: {lr_metrics['precision']:.4f}")
    print(f"  Recall: {lr_metrics['recall']:.4f}")
    print(f"  F1 Score: {lr_metrics['f1']:.4f}")
    
    # 为SVM找到最佳阈值
    print("Finding best threshold for SVM...")
    # 使用训练集的一部分作为验证集来找最佳阈值
    # 这里使用20%的训练数据
    val_size = int(0.2 * len(X_train))
    indices = np.random.permutation(len(X_train))
    X_val, y_val = X_train[indices[:val_size]], y_train[indices[:val_size]]
    
    best_threshold = svm.find_best_threshold(X_val, y_val)
    print(f"  Best threshold: {best_threshold:.4f}")
    
    print("Final SVM performance on test set:")
    svm_pred = svm.predict(X_test, threshold=best_threshold)
    svm_metrics = evaluate_model(y_test, svm_pred)
    print(f"  Accuracy: {svm_metrics['accuracy']:.4f}")
    print(f"  Precision: {svm_metrics['precision']:.4f}")
    print(f"  Recall: {svm_metrics['recall']:.4f}")
    print(f"  F1 Score: {svm_metrics['f1']:.4f}")

if __name__ == "__main__":
    # 创建报告目录（如果不存在）
    os.makedirs('../report/my_report', exist_ok=True)
    
    # 运行逻辑回归实验
    run_logistic_regression_experiment()
    
    # 运行SVM实验
    run_svm_experiment()
    
    # 比较逻辑回归和SVM
    compare_logistic_regression_and_svm() 