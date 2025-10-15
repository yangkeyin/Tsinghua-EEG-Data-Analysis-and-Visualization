#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RHD事件提取器

该脚本用于读取指定文件夹下的所有RHD文件，提取数字输入通道数据和对应的时间戳，并识别触发事件点。

实现步骤：
1. 遍历指定文件夹下所有扩展名为.rhd的文件
2. 对每个RHD文件，使用importrhdutilities.load_file()函数读取数据
3. 提取每个文件中的board_dig_in_data[1]通道数据和t_dig时间戳
4. 将所有文件的数据按顺序连接起来
5. 分析数字输入通道的状态变化，识别从False变为True的转换点（上升沿）
6. 返回这些触发事件对应的时间戳列表

使用方法：
    from utils.rhd_event_extractor import extract_events_from_rhd_folder
    events = extract_events_from_rhd_folder('path/to/rhd/files')
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入RHD文件读取工具
from utils.importrhdutilities import load_file


def load_single_rhd_file(file_path):
    """
    读取单个RHD文件并提取所需数据
    
    参数:
        file_path (str): RHD文件的路径
        
    返回:
        tuple: (digital_data, time_data)，其中
              digital_data是board_dig_in_data[1]通道的数据
              time_data是对应的t_dig时间戳数据
    """
    try:
        # 读取RHD文件
        result = load_file(file_path)
        # 提取第一个元素
        result0 = result[0]
        
        # 确保必要的数据存在
        if 'board_dig_in_data' not in result0 or 't_dig' not in result0:
            raise ValueError(f"文件 {file_path} 中缺少必要的数据字段")
            
        # 确保数字输入通道1存在
        if len(result0['board_dig_in_data']) < 2:
            raise ValueError(f"文件 {file_path} 中缺少数字输入通道1")
            
        # 提取数字输入通道1的数据和对应的时间戳
        digital_data = result0['board_dig_in_data'][1]
        time_data = result0['t_dig']
        
        return digital_data, time_data
        
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {str(e)}")
        return None, None


def concatenate_rhd_data(folder_path):
    """
    读取文件夹下所有RHD文件并连接数据
    
    参数:
        folder_path (str): 包含RHD文件的文件夹路径
        
    返回:
        tuple: (concatenated_digital, concatenated_time)，其中
              concatenated_digital是所有文件board_dig_in_data[1]通道数据的连接结果
              concatenated_time是所有文件t_dig时间戳数据的连接结果
    """
    # 获取文件夹下所有RHD文件
    rhd_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                if f.lower().endswith('.rhd')]
    
    # 按文件名排序（假设文件名包含时间信息）
    rhd_files.sort()
    
    if not rhd_files:
        raise ValueError(f"文件夹 {folder_path} 中没有找到RHD文件")
        
    # 初始化存储所有数据的列表
    all_digital_data = []
    all_time_data = []
    
    # 读取并连接所有文件
    for i, file_path in enumerate(rhd_files):
        print(f"正在读取文件 {i+1}/{len(rhd_files)}: {os.path.basename(file_path)}")
        
        digital_data, time_data = load_single_rhd_file(file_path)
        
        if digital_data is None or time_data is None:
            print(f"跳过文件: {file_path}")
            continue
            
        all_digital_data.append(digital_data)
        all_time_data.append(time_data)
    
    # 如果没有成功读取任何文件
    if not all_digital_data:
        raise ValueError("没有成功读取任何RHD文件的数据")
        
    # 连接所有数据
    concatenated_digital = np.concatenate(all_digital_data)
    concatenated_time = np.concatenate(all_time_data)
    
    return concatenated_digital, concatenated_time


def detect_rising_edges(digital_data, time_data):
    """
    检测数字信号的上升沿（从False变为True的转换点）
    
    参数:
        digital_data (ndarray): 数字信号数据
        time_data (ndarray): 对应的时间戳数据
        
    返回:
        list: 包含所有上升沿事件时间戳的列表
    """
    # 确保输入数据是布尔型
    if digital_data.dtype != bool:
        digital_data = digital_data.astype(bool)
        
    # 计算差分，找到上升沿（从False变为True的位置）
    # 注意：我们需要将digital_data转换为int来计算差分
    diff = np.diff(digital_data.astype(int))
    
    # 找到差分等于1的位置（上升沿）
    rising_edge_indices = np.where(diff == 1)[0]
    
    # 获取对应的时间戳
    # 注意：差分后的索引比原始数据少1，所以我们需要+1来获取原始数据中的索引
    event_times = [time_data[i+1] for i in rising_edge_indices]
    
    return event_times


def extract_events_from_rhd_folder(folder_path):
    """
    从指定文件夹下的所有RHD文件中提取触发事件时间戳
    
    参数:
        folder_path (str): 包含RHD文件的文件夹路径
        
    返回:
        list: 包含所有触发事件时间戳的列表
    """
    print(f"开始从文件夹 {folder_path} 提取事件...")
    
    # 读取并连接所有RHD文件的数据
    concatenated_digital, concatenated_time = concatenate_rhd_data(folder_path)
    plt.plot(concatenated_time, concatenated_digital)
    plt.show()
    
    # 检测上升沿触发事件
    event_times = detect_rising_edges(concatenated_digital, concatenated_time)
    
    print(f"找到 {len(event_times)} 个触发事件")
    
    return event_times


def save_events_to_file(event_times, output_file):
    """
    将事件时间戳保存到文件
    
    参数:
        event_times (list): 事件时间戳列表
        output_file (str): 输出文件路径
    """
    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 保存事件时间戳到文件
    with open(output_file, 'w') as f:
        f.write("event_time\n")
        for time in event_times:
            f.write(f"{time}\n")
            
    print(f"事件时间戳已保存到: {output_file}")


if __name__ == '__main__':
    # 定义默认的文件夹路径和输出文件路径
    FOLDER_PATH = r'd:\yangkeyin\datasets\251010清华数据分析\acute\20250612\visual\visual_250613_162743'
    OUTPUT_FILE = 'rhd_events.txt'
    
    try:
        # 提取事件
        events = extract_events_from_rhd_folder(FOLDER_PATH)
        
        # 保存事件到文件
        save_events_to_file(events, OUTPUT_FILE)
        
        # 打印前几个事件
        print("\n前5个事件时间戳:")
        for i, event_time in enumerate(events[:5]):
            print(f"{i+1}: {event_time:.6f} 秒")
            
    except Exception as e:
        print(f"错误: {str(e)}")
        sys.exit(1)