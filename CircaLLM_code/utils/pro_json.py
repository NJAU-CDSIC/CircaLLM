import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    precision_recall_curve,
    auc
)
from sklearn.preprocessing import label_binarize
from typing import Dict, Tuple
import json


def calculate_binary_metrics(json_path):
    """
    计算二分类任务的六个核心评估指标
    
    参数：
    test_res -- 包含以下键的字典：
        - 'targets': 真实标签列表（0/1）
        - 'preds': 预测标签列表（0/1）
        - 'scores': 预测概率分数列表（正类概率，0-1）
    
    返回：
    包含以下指标的元组（顺序固定）：
    (accuracy, precision, recall, f1, auroc, aupr)
    """
    with open(json_path, 'r') as file:
        # 读取JSON数据
        test_res = json.load(file)
    y_true = np.array(test_res['targets'])
    y_pred = np.array(test_res['preds'])
    y_score = np.array(test_res['scores'])

    # 验证二分类格式
    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("检测到非二分类标签，请使用calculate_multiclass_metrics函数")

    # 计算基础指标
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # 计算AUROC和AUPR
    auroc = roc_auc_score(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)

    return (acc, pre, rec, f1, auroc, aupr)


def calculate_multiclass_metrics(json_path, verbose=True):
    """
    计算并格式化输出多标签分类指标
    
    参数：
    test_res -- 包含target/preds/scores的字典结构
    verbose -- 是否直接打印格式化报告（默认为True）
    
    返回：
    metrics_dict -- 包含所有计算指标的字典（格式已优化）
    """
    with open(json_path, 'r') as file:
        # 读取JSON数据
        test_res = json.load(file)
    metrics_dict = {
        'overall': {},
        'class_wise': {}
    }
    class_names = list(test_res['target'].keys())
    
    # 数据合并函数
    def aggregate_data(key):
        return np.concatenate([test_res[key][cn] for cn in class_names])
    
    # 计算整体指标
    targetss = aggregate_data('target')
    predss = aggregate_data('preds')
    scoress = aggregate_data('scores')
 
    # 整体分类报告
    metrics_dict['overall']['report'] = classification_report(
        targetss, predss, 
        target_names=['0', '1'],
        output_dict=True,
        zero_division=0
    )
    
    precision, recall, _ = precision_recall_curve(targetss, scoress)
    metrics_dict['overall'].update({
        'roc_auc': roc_auc_score(targetss, scoress),
        'aupr': auc(recall, precision)  # 正确参数顺序
    })
 
    # 计算每个类别的指标
    for class_name in class_names:
        target = np.array(test_res['target'][class_name])
        pred = np.array(test_res['preds'][class_name])
        score = np.array(test_res['scores'][class_name])
 
        # 分类报告
        metrics_dict['class_wise'][class_name] = {
            'report': classification_report(
                target, pred,
                target_names=['0', '1'],
                output_dict=True,
                zero_division=0
            ),
        }
        #计算PR曲线和AUPR
        precision, recall, _ = precision_recall_curve(target, score)
        metrics_dict['class_wise'][class_name]['aupr'] = auc(recall, precision)
        metrics_dict['class_wise'][class_name]['roc_auc'] = roc_auc_score(target, score)
 
    if verbose:
        _pretty_print(metrics_dict)
    
    return metrics_dict
 
def _pretty_print(metrics_dict):
    """格式化打印函数（内部使用）"""
    # 定义列格式模板
    COL_TEMPLATE = "{:<10}{:>12}{:>12}{:>12}{:>12}"
    
    # 打印整体指标
    print("=== 整体评估指标 ===")
    print(COL_TEMPLATE.format(
        '', 'precision', 'recall', 'f1-score', 'support'
    ))
    for label in ['0', '1']:
        print(COL_TEMPLATE.format(
            label,
            f"{metrics_dict['overall']['report'][label]['precision']:.4f}",
            f"{metrics_dict['overall']['report'][label]['recall']:.4f}",
            f"{metrics_dict['overall']['report'][label]['f1-score']:.4f}",
            f"{metrics_dict['overall']['report'][label]['support']}"
        ))
    
    # 打印关键指标
    print(f"\naccuracy     {metrics_dict['overall']['report']['accuracy']:.4f}")
    print(f"macro avg    {metrics_dict['overall']['report']['macro avg']['f1-score']:.4f}")
    print(f"weighted avg {metrics_dict['overall']['report']['weighted avg']['f1-score']:.4f}")
    print(f"\nAUROC: {metrics_dict['overall']['roc_auc']:.4f}")
    print(f"AUPR: {metrics_dict['overall']['aupr']:.4f}\n")
 
    # 打印分类指标
    print("=== 分类指标 ===")
    for cls, data in metrics_dict['class_wise'].items():
        print(f"\n{cls.capitalize()} 分类指标:")
        print(COL_TEMPLATE.format(
            '', 'precision', 'recall', 'f1-score', 'support'
        ))
        for label in ['0', '1']:
            print(COL_TEMPLATE.format(
                label,
                f"{data['report'][label]['precision']:.4f}",
                f"{data['report'][label]['recall']:.4f}",
                f"{data['report'][label]['f1-score']:.4f}",
                f"{data['report'][label]['support']}"
            ))
        
        print(f"\naccuracy     {data['report']['accuracy']:.4f}")
        print(f"macro avg    {data['report']['macro avg']['f1-score']:.4f}")
        print(f"weighted avg {data['report']['weighted avg']['f1-score']:.4f}")
        print(f"\nAUROC: {data['roc_auc']:.4f}")
        print(f"AUPR: {data['aupr']:.4f}")

def process_json_to_csv(json_path, output_csv):
    # 读取JSON文件
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 创建DataFrame（假设JSON结构包含ID/preds/scores字段）
    df = pd.DataFrame({
        'Symbol': data['ID'],
        'preds': data['preds'],
        'scores': data['scores']
    })
    df['preds'] = df['preds'].apply(lambda x: x[0])
    df['scores'] = df['scores'].apply(lambda x: x[0])
    
    # 计算p值（1 - scores）
    df['p'] = 1 - df['scores']
    
    # 按Symbol排序
    df = df.sort_values('Symbol').reset_index(drop=True)
    
    # 筛选条件：preds==1 且 p<0.05
    filtered_df = df[(df['preds'] == 1) & (df['p'] < 0.05)]
    
    # 选择输出列顺序
    result_df = filtered_df[['Symbol', 'preds', 'p']]
    
    # 保存为CSV
    result_df.to_csv(output_csv, index=False)
    print(f"处理完成！结果已保存至：{output_csv}")


def process_multiclass_json_to_csv(json_path, output_fold):
    # 读取JSON文件
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    output_period = output_fold + 'period.csv'
    output_amp = output_fold + 'amp.csv'
    output_phase = output_fold + 'phase.csv'
    output_mesor = output_fold + 'mesor.csv'
    id = data['ID']
    pre = data['preds']
    score = data['scores']

    df_period = pd.DataFrame({
        'Symbol': id,
        'preds': pre['period'],
        'scores': score['period']
    })
    df_period['preds'] = df_period['preds']
    df_period['scores'] = df_period['scores']
    df_period['p'] = 1 - df_period['scores']
    df_period = df_period.sort_values('Symbol').reset_index(drop=True)
    filtered_df = df_period[(df_period['preds'] == 1) & (df_period['p'] < 0.05)]
    result_df = filtered_df[['Symbol', 'preds', 'p']]
    result_df.to_csv(output_period, index=False)
    print(f"处理完成！结果已保存至：{output_period}")

    df_amp = pd.DataFrame({
        'Symbol': id,
        'preds': pre['amp'],
        'scores': score['amp']
    })
    df_amp['preds'] = df_amp['preds']
    df_amp['scores'] = df_amp['scores']
    df_amp['p'] = 1 - df_amp['scores']
    df_amp = df_amp.sort_values('Symbol').reset_index(drop=True)
    filtered_df = df_amp[(df_amp['preds'] == 1) & (df_amp['p'] < 0.05)]
    result_df = filtered_df[['Symbol', 'preds', 'p']]
    result_df.to_csv(output_amp, index=False)
    print(f"处理完成！结果已保存至：{output_amp}")

    df_phase = pd.DataFrame({
        'Symbol': id,
        'preds': pre['phase'],
        'scores': score['phase']
    })
    df_phase['preds'] = df_phase['preds']
    df_phase['scores'] = df_phase['scores']
    df_phase['p'] = 1 - df_phase['scores']
    df_phase = df_phase.sort_values('Symbol').reset_index(drop=True)
    filtered_df = df_phase[(df_phase['preds'] == 1) & (df_phase['p'] < 0.05)]
    result_df = filtered_df[['Symbol', 'preds', 'p']]
    result_df.to_csv(output_phase, index=False)
    print(f"处理完成！结果已保存至：{output_phase}")

    df_mesor = pd.DataFrame({
        'Symbol': id,
        'preds': pre['mesor'],
        'scores': score['mesor']
    })
    df_mesor['preds'] = df_mesor['preds']
    df_mesor['scores'] = df_mesor['scores']
    df_mesor['p'] = 1 - df_mesor['scores']
    df_mesor = df_mesor.sort_values('Symbol').reset_index(drop=True)
    filtered_df = df_mesor[(df_mesor['preds'] == 1) & (df_mesor['p'] < 0.05)]
    result_df = filtered_df[['Symbol', 'preds', 'p']]
    result_df.to_csv(output_mesor, index=False)
    print(f"处理完成！结果已保存至：{output_mesor}")