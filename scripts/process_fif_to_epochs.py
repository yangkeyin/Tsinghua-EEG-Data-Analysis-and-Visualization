from pickle import TRUE
from pydoc import visiblename
import mne
import glob
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as config
import numpy as np

PARADIGMS = ['rest', 'visual', 'beard_slow', 'beard_fast']

def combine_rest_events(events_combined):
    """
    合并所有rest事件为一个连续的epoch
    """
    # --- 3. 创建 'Rest' 事件 ---
    rest_events_list = []
    print(f"\n--- 正在创建 'Rest' 事件 ---")
    # 检查配置中是否定义了用于参考的事件和Rest的ID
    ref_event_name = getattr(config, 'REFERENCE_EVENT_FOR_REST', None)
    rest_event_id_val = config.EVENT_IDS.get('rest')
    
    if ref_event_name and rest_event_id_val is not None:
        ref_event_id_val = config.EVENT_IDS.get(ref_event_name)
        if ref_event_id_val is not None:
                # 找到所有参考事件的样本点
                ref_event_samples = events_combined[events_combined[:, 2] == ref_event_id_val, 0]
                
                # 计算每个Rest事件的起始样本点
                # 确保 REST_T_BEFORE 在 config 中定义了
                rest_start_offset_samples = int(config.REST_T_BEFORE * config.SAMPLING_RATE)
                rest_start_samples = ref_event_samples + rest_start_offset_samples
                
                # 创建Rest事件数组: [sample, 0, rest_id]
                for sample in rest_start_samples:
                    rest_events_list.append([sample, 0, rest_event_id_val])
                print(f"  基于 '{ref_event_name}' 事件创建了 {len(rest_events_list)} 个 'Rest' 事件。")
        else:
                print(f"  错误: 未在 config.EVENT_IDS 中找到参考事件 '{ref_event_name}' 的 ID。")
            
    # --- 4. 合并所有事件 ---
    if rest_events_list:
        rest_events_array = np.array(rest_events_list, dtype=int)
        # 合并非Rest事件和Rest事件
        events_combined_all = np.concatenate((events_combined, rest_events_array), axis=0)
        # 按时间排序事件
        events_combined_all = events_combined_all[np.argsort(events_combined_all[:, 0])]
        print(f"合并 Rest 事件后总事件数: {len(events_combined_all)}")
    else:
        events_combined_all = events_combined # 如果没有Rest事件

    return events_combined_all

def load_and_epoch_data(paradigms):
    all_raws = []
    all_events_adjusted = [] # 存储调整过偏移量的事件
    last_sample_offset = 0 # 记录上一个文件结束时的样本点

    # --- 1. 按顺序加载 Raw 文件并调整 Events ---
    # 我们需要确保文件加载顺序一致，可以按文件名排序
    all_fif_files = []
    for paradigm in paradigms:
        if paradigm == 'rest': continue # Rest 稍后处理
        paradigm_actual = paradigm
        data_dir = config.PROCESSED_DATA_DIR
        # 查找所有被试的所有试次
        pattern = os.path.join(data_dir, '*', f'*_{paradigm_actual}_trial*_raw.fif')
        found_raws = glob.glob(pattern)
        all_fif_files.extend(found_raws)
        
    all_fif_files.sort() # 按文件名排序，确保拼接顺序稳定

    if not all_fif_files:
        print("错误: 未找到任何 .fif 文件。")
        return {}

    print(f"--- 按顺序加载 {len(all_fif_files)} 个 Raw 文件并调整事件 ---")
    for raw_file in all_fif_files:
        event_file = raw_file.replace(config.FIF_FILE_SUFFIX, config.EVENTS_FILE_SUFFIX)
        
        if not os.path.exists(event_file):
            print(f"  警告: 缺少对应的事件文件: {event_file}，跳过 {raw_file}")
            continue
            
        try:
            print(f"  加载: {os.path.basename(raw_file)}")
            raw = mne.io.read_raw_fif(raw_file, preload=False, verbose=False)
            events = mne.read_events(event_file, verbose=False)

            # --- 关键步骤：调整事件样本编号 ---
            if last_sample_offset > 0:
                print(f"    -> 事件样本编号增加偏移量: {last_sample_offset}")
                events[:, 0] += last_sample_offset
                
            # 降采样
            all_raws.append(raw)
            all_events_adjusted.append(events)
            
            # 更新下一个文件的偏移量
            last_sample_offset += raw.n_times # 使用 n_times 获取样本数

        except Exception as e:
            print(f"  处理文件 {raw_file} 或 {event_file} 时出错: {e}")
    # --- 2. 合并 Raw 和 Events ---
    print(f"\n--- 正在合并 {len(all_raws)} 个 Raw 对象 ---")
    raw_combined = mne.concatenate_raws(all_raws, verbose=False)
    
    if not all_events_adjusted:
         print("警告: 未找到任何有效的事件数据。")
         events_combined = np.array([], dtype=int).reshape(0, 3) # 创建空的事件数组
    else:
        print(f"--- 正在合并 {len(all_events_adjusted)} 个调整后的事件列表 ---")
        # 使用 numpy concatenate 合并调整后的事件
        events_combined = np.concatenate(all_events_adjusted, axis=0)
        print(f"合并后总事件数: {len(events_combined)}")
    
    # 对于events_combined为rest创建事件
    events_combined = combine_rest_events(events_combined)

    # 为每个paradigm创建Epochs
    epochs_dict = {}
    for condition in paradigms:
            event_id = {condition: config.EVENT_IDS[condition]}
            print(f"  创建 Epochs for condition: {condition}")
            if condition == 'rest': 
                try:
                    epochs = mne.Epochs(raw_combined, events_combined, event_id=event_id,
                                        tmin=0, tmax=2,
                                        baseline=None, # 不进行基线校正
                                        preload=True, verbose=False)
                    epochs.resample(config.RESAMPLE_FREQ, verbose=True)
                    epochs_dict[condition] = epochs
                except Exception as e:
                    print(f"  创建 Epochs for {condition} 失败: {e}")

            else:
                try:
                    epochs = mne.Epochs(raw_combined, events_combined, event_id=event_id,
                                        tmin=config.EPOCH_T_MIN, tmax=config.EPOCH_T_MAX,
                                        baseline=(-0.2,0), 
                                        preload=True, verbose=False)
                    epochs.resample(config.RESAMPLE_FREQ, verbose=True)
                    epochs_dict[condition] = epochs
                except Exception as e:
                    print(f"  创建 Epochs for {condition} 失败: {e}")

    return epochs_dict


def main():
    # 提取Epochs
    epochs_dict = load_and_epoch_data(PARADIGMS)
    os.makedirs(config.EPOCHS_DIR, exist_ok=True)
    for condition, epochs in epochs_dict.items():
        epochs.save(f'{config.EPOCHS_DIR}/{condition}_epo.fif', overwrite=True)

if __name__ == '__main__':
    main()
