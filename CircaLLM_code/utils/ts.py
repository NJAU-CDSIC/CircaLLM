import pandas as pd
import numpy as np
from collections import defaultdict


def singal_convert_to_ts(input_csv, output_ts, problem_name, label_col='label'):
    # 读取基因表达矩阵（假设基因名为索引，最后一列为标签）
    df = pd.read_csv(input_csv, index_col=0)
    
    # 分离标签列
    if label_col in df.columns:
        labels = df.pop(label_col)
    else:
        raise ValueError(f"标签列 '{label_col}' 不存在于数据中")

    # 解析列名并构建重复组
    time_point_order = []     # 记录时间点首次出现顺序
    rep_groups = defaultdict(list)  # key=重复编号, value=[(时间点, 列名), ...]

    for col in df.columns:
        parts = col.split('_')
        if len(parts) == 2:
            # 时间点_重复编号格式（如："0_1"）
            time_point, rep_num = map(int, parts)
        else:
            raise ValueError(f"列名格式错误，预期格式为 '时间点' 或 '时间点_重复编号'，实际列名: {col}")
        
        # 记录时间点
        if time_point not in time_point_order:
            time_point_order.append(time_point)
        
        # 添加到对应重复组
        rep_groups[rep_num].append((time_point, col))

    # 验证数据完整性
    if rep_groups:
        num_time_points = len(time_point_order)
        num_reps = len(rep_groups[next(iter(rep_groups))])  # 任意取一个重复组获取重复次数
        
        # 验证所有重复组的时间点数一致
        for rep_num, cols in rep_groups.items():
            if len(cols) != num_time_points:
                raise ValueError(f"重复组 {rep_num} 的时间点数与其他组不一致")
            
    # 检查时间点间隔是否均匀
    equal_length = True
    if len(time_point_order) > 1:
        step = time_point_order[1] - time_point_order[0]
        for i in range(2, len(time_point_order)):
            if time_point_order[i] - time_point_order[i-1] != step:
                equal_length = False
                break

    # 构建元数据
    meta_info = [
        f"#{problem_name} provenance not determined yet",
        f"@problemName {problem_name}",
        # 使用观察到的时间点顺序
        f"@timeStamps {','.join(map(str, time_point_order))}",
        "@missing false",
        "@univariate true",
        f"@dup {num_reps}" if rep_groups else "@dup 1",
        f"@equalLength {str(equal_length).lower()}",
        f"@seriesLength {num_time_points * num_reps}" if rep_groups else f"@seriesLength {len(df.columns)}",
        "@classLabel true 0 1",
        "@data"
    ]

    # 构建转换后的数据行
    ts_lines = meta_info.copy()
    for gene_id in df.index:
        group_values = []
        # 按重复编号顺序处理（0,1,2...）
        for rep_num in sorted(rep_groups.keys()):
            # 对当前重复组按时间点顺序排序
            current_group = rep_groups[rep_num]
            current_group.sort(key=lambda x: time_point_order.index(x[0]))
            
            # 提取当前重复组的数值（按时间点顺序）
            current_values = [
                str(df.loc[gene_id, col]) 
                for (_, col) in current_group
            ]
            group_values.append(','.join(current_values))
        
        # 用冒号连接不同重复组
        values_str = ':'.join(group_values)
        final_line = f"{values_str}:{gene_id},{int(labels.loc[gene_id])}"
        ts_lines.append(final_line)

    # 写入文件
    with open(output_ts, 'w') as f:
        f.write('\n'.join(ts_lines))
    print(f"TS文件已生成：{output_ts}")
 
def multi_convert_to_ts(input_csv, output_ts, problem_name, label_cols):
    """
    将包含两个时间序列和多标签的CSV文件转换为TS格式
    
    参数：
    input_csv: 输入CSV文件路径
    output_ts: 输出TS文件路径
    problem_name: 问题名称
    label_cols: 标签列名列表（例如 ['label1', 'label2']）
    ts_prefixes: 时间序列列名前缀列表（默认 ['ts1', 'ts2']）
    """
    
    # 读取基因表达矩阵（假设基因名为索引）
    df = pd.read_csv(input_csv, index_col=0)
    
    # 分离标签列
    if all(col in df.columns for col in label_cols):
        labels = df[label_cols].copy()
        df = df.drop(columns=label_cols)
    else:
        missing = [col for col in label_cols if col not in df.columns]
        raise ValueError(f"标签列 {missing} 不存在于数据中")
 
    # 解析列名并构建时间序列数据结构
    time_point_order_1 = []  # 记录各时间序列的时间点顺序
    time_point_order_2 = []
    rep_groups_1 = defaultdict(list)  # key=重复编号, value=[(时间点, 列名), ...]
    rep_groups_2 = defaultdict(list)  # key=重复编号, value=[(时间点, 列名), ...]
 
    for col in df.columns:
        # 解析列名格式：time_rep_no（例如 1_0_1）
        parts = col.split('_')
        if len(parts) != 3:
            raise ValueError(f"列名格式错误，预期格式为 '时间点_重复编号_序列编号'，实际列名: {col}")
        
        time_point = int(parts[0])
        rep_num = int(parts[1])
        ts_idx = int(parts[2])

        if ts_idx == 0:
            # 记录时间点首次出现顺序
            if time_point not in time_point_order_1:
                time_point_order_1.append(time_point)
            # 添加到对应重复组
            rep_groups_1[rep_num].append((time_point, col))
        else:
            # 记录时间点首次出现顺序
            if time_point not in time_point_order_2:
                time_point_order_2.append(time_point)
            # 添加到对应重复组
            rep_groups_2[rep_num].append((time_point, col))
 
    # 验证数据完整性
    if rep_groups_1:
        num_time_points_1 = len(time_point_order_1)
        num_reps_1 = len(rep_groups_1[next(iter(rep_groups_1))])  # 任意取一个重复组获取重复次数
        # 验证所有重复组的时间点数一致
        for rep_num, cols in rep_groups_1.items():
            if len(cols) != num_time_points_1:
                raise ValueError(f"重复组 {rep_num} 的时间点数与其他组不一致")
    if rep_groups_2:
        num_time_points_2 = len(time_point_order_2)
        num_reps_2 = len(rep_groups_2[next(iter(rep_groups_2))])  # 任意取一个重复组获取重复次数
        # 验证所有重复组的时间点数一致
        for rep_num, cols in rep_groups_2.items():
            if len(cols) != num_time_points_2:
                raise ValueError(f"重复组 {rep_num} 的时间点数与其他组不一致")
 
    # 检查时间点间隔是否均匀
    equal_length_1 = True
    if len(time_point_order_1) > 1:
        step = time_point_order_1[1] - time_point_order_1[0]
        for i in range(2, len(time_point_order_1)):
            if time_point_order_1[i] - time_point_order_1[i-1] != step:
                equal_length_1 = False
                break

    # 检查时间点间隔是否均匀
    equal_length_2 = True
    if len(time_point_order_2) > 1:
        step = time_point_order_2[1] - time_point_order_2[0]
        for i in range(2, len(time_point_order_2)):
            if time_point_order_2[i] - time_point_order_2[i-1] != step:
                equal_length_2 = False
                break
 
    meta_info = [
        f"#{problem_name} provenance not determined yet",
        f"@problemName {problem_name}",
        f"@timeStamps {','.join(map(str, time_point_order_1))};{','.join(map(str, time_point_order_2))}",
        "@missing false",
        "@univariate false",  # 标记为多变量时间序列
        f"@dup {num_reps_1};{num_reps_2}",
        f"@equalLength {equal_length_1}",
        f"@seriesLength {len(time_point_order_1)};{len(time_point_order_2)}",
        # 多标签处理（假设为二元标签，逗号分隔）
        f"@classLabel true -1 0 1",
        "@data"
    ]
 
    # 构建数据行
    ts_lines = meta_info.copy()
    for gene_id in df.index:
        line_values = []
        group_values = []
        # 按重复编号顺序处理（0,1,2...）
        for rep_num in sorted(rep_groups_1.keys()):
            # 对当前重复组按时间点顺序排序
            current_group = rep_groups_1[rep_num]
            current_group.sort(key=lambda x: time_point_order_1.index(x[0]))
            
            # 提取当前重复组的数值（按时间点顺序）
            current_values = [
                str(df.loc[gene_id, col]) 
                for (_, col) in current_group
            ]
            group_values.append(','.join(current_values))
        # 用冒号连接不同重复组
        values_str = ':'.join(group_values)
        line_values.append(values_str)
        group_values = []
        for rep_num in sorted(rep_groups_2.keys()):

            # 对当前重复组按时间点顺序排序
            current_group = rep_groups_2[rep_num]
            current_group.sort(key=lambda x: time_point_order_2.index(x[0]))
            
            # 提取当前重复组的数值（按时间点顺序）
            current_values = [
                str(df.loc[gene_id, col]) 
                for (_, col) in current_group
            ]
            group_values.append(','.join(current_values))
        # 用冒号连接不同重复组
        values_str = ':'.join(group_values)
        line_values.append(values_str)

        values_str = ';'.join(line_values)

        final_line = f"{values_str};{gene_id},{','.join(labels.loc[gene_id].astype(int).astype(str))}"

        ts_lines.append(final_line)
    # 写入文件
    with open(output_ts, 'w') as f:
        f.write('\n'.join(ts_lines))
    print(f"多时间序列TS文件已生成：{output_ts}")