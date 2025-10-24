'''
此文件主要用于处理RHD格式的脑电数据，将其转换为MNE可处理的FIF格式文件。具体功能如下：
1. 从RHD文件中提取事件信息并转换为MNE事件数组
2. 在创建FIF数据前对数据进行预处理，包括陷波滤波、带通滤波、共模去噪
3. 将处理后的数据保存为FIF文件，同时保存对应的事件文件

此外，程序会从文件夹名称中提取被试ID、范式类型和试次编号，支持多个RHD文件的合并操作，
并能根据配置生成不同类型的事件文件。
'''

import os
import sys
from mne.io import Raw
import numpy as np
import mne
import shutil
from datetime import datetime
import glob

# 添加utils目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/utils')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/config')

# 导入RHD文件读取工具
from importrhdutilities import load_file
from rhd_event_extractor import extract_events_from_rhd_folder, save_events_to_file
from preprocess import preprocess

# 导入配置
import config as config



def get_subject_id_from_folder(folder_name):
    """从文件夹名称中提取被试ID"""
    # 从folder_name中提取日期部分
    date_part = folder_name.split('\\')[-3]
    # 从配置中获取被试映射
    subject_id = config.SUBJECT_MAPPING.get(date_part)
    if not subject_id:
        raise ValueError(f"未找到文件夹 {folder_name} 的被试映射")
    return subject_id


def get_paradigm_type_from_folder(folder_name):
    """从文件夹名称中提取范式类型"""
    if 'visual' in folder_name.lower():
        return 'visual'
    elif 'beard_fast' in folder_name.lower():
        return 'beard_fast'
    elif 'beard_slow' in folder_name.lower():
        return 'beard_slow'
    return 'unknown'


def get_trial_number_from_folder(folder_name):
    """从文件夹名称中提取试次编号"""
    # 拆分路径为文件名和上一级目录
    dirname, filename = os.path.split(folder_name)
    
    # 获取上一级目录下的所有文件夹名称
    parent_dir_contents = [d for d in os.listdir(dirname) if os.path.isdir(os.path.join(dirname, d))]
    
    # 对文件夹名称进行排序
    parent_dir_contents.sort()
    
    try:
        # 根据文件名定位其在排序后列表中的索引，加1作为试次编号
        trial_number = parent_dir_contents.index(filename) + 1
        return trial_number
    except ValueError:
        # 如果未找到文件名，默认返回1
        return 1


def merge_rhd_files(rhd_files):
    """
    合并多个RHD文件的数据
    args:
        rhd_files: 多个RHD文件的路径列表
    return:
        合并后的RHD数据字典
    """
    if not rhd_files:
        return None
        
    # 读取第一个文件，获取头部信息
    result = load_file(rhd_files[0])[0]
    
    # 如果只有一个文件，直接返回
    if len(rhd_files) == 1:
        return result
        
    # 合并多个文件的数据
    for file_path in rhd_files[1:]:
        next_result = load_file(file_path)[0]
        
        # 合并放大器数据
        if 'amplifier_data' in result and 'amplifier_data' in next_result:
            result['amplifier_data'] = np.hstack((
                result['amplifier_data'], 
                next_result['amplifier_data']
            ))
            
        # 更新时间戳
        if 't_amplifier' in result and 't_amplifier' in next_result:
            # 计算时间偏移
            time_offset = result['t_amplifier'][-1] - next_result['t_amplifier'][0] + 1/config.SAMPLING_RATE
            # 调整下一个文件的时间戳
            next_result['t_amplifier'] += time_offset
            # 合并时间戳
            result['t_amplifier'] = np.hstack((
                result['t_amplifier'], 
                next_result['t_amplifier'][1:]  # 避免重复第一个时间点
            ))
            
    return result


def create_mne_raw_from_rhd_data(rhd_data):
    """从RHD数据创建MNE Raw对象"""
    # 获取放大器数据
    data = rhd_data['amplifier_data']
    
    # 获取通道信息
    ch_names = [chan['custom_channel_name'] or chan['native_channel_name'] 
                for chan in rhd_data['amplifier_channels']]
    
    # 设置通道类型
    ch_types = [config.DEFAULT_CHANNEL_TYPE] * len(ch_names)
    
    # 创建info对象
    info = mne.create_info(ch_names=ch_names, sfreq=config.SAMPLING_RATE, ch_types=ch_types)
    
    # 注意：MNE需要数据是形状为(n_channels, n_samples)的数组
    # 但我们已经有了这种格式的数据，所以可以直接使用
    
    # 创建Raw对象
    raw = mne.io.RawArray(data, info)   
    
    return raw


def generate_events(root, subject_id, paradigm_type, trial_number):
    """生成事件文件"""
    events = []
    
    # 根据paradigm_type选择对应的triggers
    if paradigm_type in config.EVENTS_CONFIG:
        # 判断是否是defined_in_config类型
        if config.EVENTS_CONFIG[paradigm_type]['type'] == 'defined_in_config':
            for trigger_time in config.EVENTS_CONFIG[paradigm_type][subject_id]:
                # 转换时间为样本索引
                sample_idx = int(trigger_time * config.SAMPLING_RATE)
                # 添加事件
                events.append([sample_idx, 0, config.EVENT_IDS[paradigm_type]])
        # 其他类型的事件处理
        elif config.EVENTS_CONFIG[paradigm_type]['type'] == 'extract_from_rhd':
            # 利用rhd_event_extractor提取事件
            trigger_times = extract_events_from_rhd_folder(root, config.EVENTS_CONFIG[paradigm_type]['digital_channel_index'], if_plot=False)
            for trigger_time in trigger_times:
                # 转换时间为样本索引
                sample_idx = int(trigger_time * config.SAMPLING_RATE)
                # 添加事件
                events.append([sample_idx, 0, config.EVENT_IDS[paradigm_type]])
                
    # 转换为MNE事件数组
    events_array = np.array(events, dtype=int)
    
    return events_array


def main():
    """主函数"""
    # 遍历原始数据目录
    for root, dirs, files in os.walk(config.RAW_DATA_DIR):
        # 查找RHD文件
        rhd_files = [os.path.join(root, f) for f in files if f.endswith(config.RHD_FILE_EXTENSION)]
        
        # 如果当前目录包含RHD文件
        if rhd_files:
            # 获取被试ID
            subject_id = get_subject_id_from_folder(root)
            
            # 获取范式类型
            paradigm_type = get_paradigm_type_from_folder(root)
            
            # 获取试次编号
            trial_number = get_trial_number_from_folder(root)
            
            
            # 合并RHD文件
            print(f"处理被试 {subject_id}, 范式 {paradigm_type}, 试次 {trial_number}")
            print(f"合并 {len(rhd_files)} 个RHD文件...")
            rhd_data = merge_rhd_files(rhd_files)
            
            if rhd_data is None:
                print(f"警告: 无法读取RHD文件在 {root}")
                continue
            
            # 创建MNE Raw对象
            print("创建MNE Raw对象...")
            raw = create_mne_raw_from_rhd_data(rhd_data)

            # 重命名通道
            print("重命名通道...")
            raw.rename_channels(lambda x: x.replace('A-', 'C-'))
            
            # 预处理
            print("预处理数据...")
            raw_processed = preprocess(raw)
            
            # 生成事件
            print("生成事件文件...")
            events = generate_events(root, subject_id, paradigm_type, trial_number)
            
            # 创建输出目录
            output_dir = os.path.join(config.PROCESSED_DATA_DIR, subject_id)
            os.makedirs(output_dir, exist_ok=True)

            # 创建visual_events目录
            visual_events_dir = os.path.join(output_dir, 'visual_events')
            os.makedirs(visual_events_dir, exist_ok=True)  # 确保目录存在
            
            visual_events_filename = f"events_{subject_id}_trial{trial_number}.txt"
            visual_events_path = os.path.join(visual_events_dir, visual_events_filename)
            
            # 检查events是否为空
            if len(events) > 0:
                save_events_to_file(events, visual_events_path)
            else:
                print(f"警告: 未提取到任何事件，跳过保存visual_events文件")
            save_events_to_file(events, visual_events_path)
            
            # 生成FIF文件名
            fif_filename = f"{subject_id}_{paradigm_type}_trial{trial_number}{config.FIF_FILE_SUFFIX}"
            fif_path = os.path.join(output_dir, fif_filename)
            
            # 生成事件文件名
            events_filename = f"{subject_id}_{paradigm_type}_trial{trial_number}{config.EVENTS_FILE_SUFFIX}"
            events_path = os.path.join(output_dir, events_filename)
            
            # 检查文件是否存在
            if os.path.exists(fif_path) and not config.OVERWRITE_EXISTING:
                print(f"文件已存在，跳过: {fif_path}")
                continue
            
            # 保存Raw对象
            print(f"保存FIF文件到: {fif_path}")
            raw_processed.save(fif_path, overwrite=config.OVERWRITE_EXISTING)
            
            # 保存事件文件
            print(f"保存事件文件到: {events_path}")
            mne.write_events(events_path, events, overwrite=config.OVERWRITE_EXISTING)
            
            print(f"处理完成: {fif_path}")
            print("=" * 50)


if __name__ == '__main__':
    # 确保输出目录存在
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    
    # 运行主函数
    main()