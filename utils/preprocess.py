'''
# 此文件实现了对MNE Raw对象的预处理功能，借助配置文件中的参数，依次进行陷波滤波、带通滤波、设置共模平均参考（CAR）和降采样操作。
# 文件中还包含了一个示例用法，展示如何加载数据、执行预处理并保存处理后的数据。
'''

import mne
import sys
import os
# 添加utils目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/config')
import config as config


def preprocess(raw):
    """对MNE Raw对象进行预处理"""

    # 1. 陷波滤波
    notch_freqs = config.NOTCH_FREQS
    if notch_freqs:
        print(f"应用陷波滤波器: {notch_freqs} Hz...")
        raw.notch_filter(freqs=notch_freqs, verbose=config.VERBOSE)

    # 2. 带通滤波
    l_freq, h_freq = config.BANDPASS_L_FREQ, config.BANDPASS_H_FREQ
    if l_freq is not None or h_freq is not None:
        print(f"应用带通滤波器: {l_freq}-{h_freq} Hz...")
        raw.filter(l_freq=l_freq, h_freq=h_freq , verbose=config.VERBOSE)

    # 3. 设置共模平均参考 (CAR) 
    print("应用共模平均参考 (CAR)...")
    raw.set_eeg_reference('average', verbose=config.VERBOSE)

    # 4. 降采样
    # resample_freq = config.RESAMPLE_FREQ
    # if resample_freq and raw.info['sfreq'] > resample_freq:
    #     print(f"降采样至 {resample_freq} Hz...")
    #     # 在降采样之前加载数据到内存是推荐做法
    #     raw.load_data(verbose=config.VERBOSE)
    #     raw.resample(sfreq=resample_freq, verbose=config.VERBOSE)
    
    print("="*15, "预处理完成", "="*15)
    return raw

if __name__ == '__main__':
    # 加载数据
    raw = mne.io.read_raw_fif('data/fif_data/mouse1/mouse1_beard_fast_trial1_raw.fif', preload=True)
    
    # 执行预处理
    preprocess(raw)
    
    # 保存预处理后的数据
    raw.save('raw_preprocessed.fif', overwrite=True)