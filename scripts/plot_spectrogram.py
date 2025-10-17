"""
批量绘制多通道时频谱图脚本

功能:
1. 加载指定被试、范式和试次的单个预处理后的_raw.fif文件。
2. 截取用户感兴趣的时间窗口。
3. 对所有通道的数据执行短时傅里叶变换 (STFT) 来计算时频功率。
4. 将所有通道的时频谱图（Spectrogram）绘制在一个网格（Grid）中。
5. 保存生成的图像到plots文件夹。

"""

import os
import sys
import mne
import numpy as np
import matplotlib.pyplot as plt

# 导入配置
# (优化路径，确保无论从哪里运行都能找到config)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as config


# ===================================================================
# 在这里精确指定你要绘制的单个目标
# ===================================================================
TARGET_SUBJECT = 'mouse1'
TARGET_PARADIGM = 'visual'
TARGET_TRIAL = 1

# 设置要显示的时间窗口 (单位: 秒)
PLOT_T_MIN = 20.0
PLOT_T_MAX = 25.0
# ===================================================================


def load_and_prepare_raw(subject, paradigm, trial, tmin, tmax):
    """
    加载并截取指定trial的Raw数据。

    Args:
        subject (str): 被试ID, e.g., 'mouse1'.
        paradigm (str): 范式名称, e.g., 'visual'.
        trial (int): 试次编号, e.g., 1.
        tmin (float): 截取的开始时间 (秒).
        tmax (float): 截取的结束时间 (秒).

    Returns:
        mne.io.Raw | None: 加载并截取后的MNE Raw对象，如果文件不存在则返回None.
    """
    print(f"--- 正在加载数据: {subject}, {paradigm}, Trial {trial} ---")
    
    # 构建文件路径
    paradigm_dir = os.path.join(config.PROCESSED_DATA_DIR, TARGET_SUBJECT)
    fif_filename = f"{TARGET_SUBJECT}_{TARGET_PARADIGM}_trial{TARGET_TRIAL}{config.FIF_FILE_SUFFIX}"
    fif_path = os.path.join(paradigm_dir, fif_filename)

    if not os.path.exists(fif_path):
        print(f"错误: 文件未找到 -> {fif_path}")
        return None

    # 加载Raw数据
    print(f"加载文件: {fif_path}")
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)

    # 截取感兴趣的时间窗口
    if tmin is not None and tmax is not None:
        print(f"截取时间窗口: {tmin}s - {tmax}s")
        try:
            raw.crop(tmin=tmin, tmax=tmax)
        except ValueError as e:
            print(f"警告: 无法截取时间窗口。原始数据时长: {raw.times[-1]:.2f}s。错误: {e}")

    return raw


def calculate_all_channels_stft(raw):
    """
    对Raw对象中的所有通道数据进行STFT计算。

    Args:
        raw (mne.io.Raw): MNE Raw对象。

    Returns:
        tuple: 包含以下元素的元组:
            - db_data (np.ndarray): 功率谱密度矩阵 (dB), 形状 (n_channels, n_freqs, n_times).
            - freqs (np.ndarray): 频率轴向量.
            - times (np.ndarray): 时间轴向量.
    """
    print("--- 正在计算所有通道的短时傅里叶变换 (STFT) ---")
    
    # 从config获取STFT参数
    sfreq = raw.info['sfreq']
    wsize = 128
    tstep = wsize // 2

    # 一次性提取所有EEG通道的数据
    data, times = raw.get_data(picks='eeg', return_times=True)

    # 对所有通道同时进行STFT计算
    stft_data = mne.time_frequency.stft(data, wsize=wsize, tstep=tstep)
    
    # MNE STFT返回的是 (n_channels, n_freqs, n_times)
    #   取频率轴
    freqs = np.fft.rfftfreq(wsize, d=1./sfreq)

    # 计算功率谱密度 (PSD)
    psd_data = np.abs(stft_data)**2

    # 筛选感兴趣的频率范围
    freq_mask = (freqs >= config.TFR_FMIN) & (freqs <= config.TFR_FMAX)
    freqs = freqs[freq_mask]
    psd_data = psd_data[:, freq_mask, :]

    # 转换为分贝 (dB)
    db_data = 10 * np.log10(psd_data + 1e-20) # 增加一个极小数避免log(0)

    # STFT的时间轴需要重新计算
    stft_times = times[0] + np.arange(db_data.shape[2]) * (tstep / sfreq)

    return db_data, freqs, stft_times


def plot_spectrogram_by_region(db_data, freqs, times, ch_names_ordered, title):
    """
    将所有通道的时频谱图按照脑区布局绘制到4列网格中。

    Args:
        db_data (np.ndarray): 功率谱密度矩阵 (dB), 形状 (n_channels, n_freqs, n_times).
        freqs (np.ndarray): 频率轴向量.
        times (np.ndarray): 时间轴向量.
        ch_names_ordered (list[str]): 原始数据中的通道名称列表 (MNE Raw对象的ch_names).
        title (str): 图像的总标题.

    Returns:
        matplotlib.figure.Figure: 生成的Matplotlib Figure对象.
    """
    print("--- 正在按脑区布局绘制时频谱图 ---")
    
    # 创建从通道名到数据索引的映射
    ch_name_to_idx = {name: i for i, name in enumerate(ch_names_ordered)}
    
    # 根据config文件生成左右脑的有序通道列表
    left_chans_by_region = {}
    for region, channels in config.REGION_MAP['Left'].items():
        region_chans = []
        for ch_num in channels:
            ch_name = f"{config.CHANNEL_PREFIX}{ch_num:03d}"
            if ch_name in ch_name_to_idx:
                region_chans.append(ch_name)
        left_chans_by_region[region] = region_chans

    right_chans_by_region = {}
    for region, channels in config.REGION_MAP['Right'].items():
        region_chans = []
        for ch_num in channels:
            ch_name = f"{config.CHANNEL_PREFIX}{ch_num:03d}"
            if ch_name in ch_name_to_idx:
                region_chans.append(ch_name)
        right_chans_by_region[region] = region_chans

    # 将字典平铺成列表，以保持后续绘图逻辑
    left_chans = [chan for chans in left_chans_by_region.values() for chan in chans]
    right_chans = [chan for chans in right_chans_by_region.values() for chan in chans]

    # 计算布局
    n_cols = 4
    n_rows = 8
    if n_rows == 0:
        print("错误：根据脑区映射，没有找到任何可供绘制的通道。")
        return plt.figure() # 返回一个空图

    # 绘制 4 * 8 子图网络
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 2.5), 
                             sharex=True, sharey=True)
    fig.suptitle(title, fontsize=20, y=0.98)

    img = None
    
    #  绘制左半球 (前两列)
    for i, ch_name in enumerate(left_chans):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        data_idx = ch_name_to_idx[ch_name]
        
        img = ax.pcolormesh(times, freqs, db_data[data_idx, :, :], 
                            cmap='coolwarm', shading='auto')
        ax.set_title(ch_name, fontsize=8)

    # 绘制右半球 (后两列)
    for i, ch_name in enumerate(right_chans):
        row = i // 2
        col = (i % 2) + 2  # 关键：列索引+2
        ax = axes[row, col]
        data_idx = ch_name_to_idx[ch_name]

        img = ax.pcolormesh(times, freqs, db_data[data_idx, :, :], 
                            cmap='coolwarm', shading='auto')
        ax.set_title(ch_name, fontsize=8)

    # 在左侧添加标签
    ax_loc = 0 # 初始化添加标签的ax下标
    for i, region in enumerate(left_chans_by_region.keys()):
        # 处理四个通道的脑区
        if len(left_chans_by_region[region]) == 4:
            ax = axes[ax_loc, 0]
            ax.text(-0.3, 0.5, region, transform=ax.transAxes, 
                    fontsize=12, va='center', ha='right')
            ax_loc += 2 # 四个通道的脑区，ax下标增加2
        # 处理两个通道的脑区
        elif len(left_chans_by_region[region]) == 2:
            ax = axes[ax_loc, 0]
            ax.text(-0.3, 0.5, region, transform=ax.transAxes, 
                    fontsize=12, va='center', ha='right')
            ax_loc += 1 # 两个通道的脑区，ax下标增加1

    # for i, ax in enumerate(axes.flat):
    #     row, col = i // n_cols, i % n_cols
    #     # 判断当前ax是否被使用
    #     is_used = False
    #     if col < 2 and (row * 2 + col) < len(left_chans):
    #         is_used = True
    #     elif col >= 2 and (row * 2 + (col - 2)) < len(right_chans):
    #         is_used = True
        
    #     if not is_used:
    #         ax.axis('off')
    #     else:
    #         # 只在最外侧的子图标示坐标轴
    #         if ax.get_subplotspec().is_first_col() or col == 2:
    #             ax.set_ylabel("Freq (Hz)")
    #         if ax.get_subplotspec().is_last_row():
    #             ax.set_xlabel("Time (s)")


    # 添加半球标题
    fig.text(0.25, 0.95, "Left Hemisphere", ha='center', va='center', fontsize=16)
    fig.text(0.75, 0.95, "Right Hemisphere", ha='center', va='center', fontsize=16)

    # 添加颜色条
    fig.subplots_adjust(right=0.88, top=0.90, wspace=0.3, hspace=0.5)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.75])
    if img:
        cbar = fig.colorbar(img, cax=cbar_ax)
        cbar.set_label('Power (dB)', fontsize=12)

    return fig


def main():
    """主函数，执行整个流程"""
    raw = load_and_prepare_raw(TARGET_SUBJECT, TARGET_PARADIGM, TARGET_TRIAL, PLOT_T_MIN, PLOT_T_MAX)
    
    if raw is None:
        return

    db_data, freqs, times = calculate_all_channels_stft(raw)
    
    # 创建图像总标题
    title = f"Spectrograms: {TARGET_SUBJECT} - {TARGET_PARADIGM} - Trial {TARGET_TRIAL}"
    
    fig = plot_spectrogram_by_region(db_data, freqs, times, raw.ch_names, title)

    # 保存图像
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    output_filename = f"{TARGET_SUBJECT}_{TARGET_PARADIGM}_trial{TARGET_TRIAL}_spectrogram_grid.png"
    output_path = os.path.join(config.PLOTS_DIR, output_filename)
    print(f"保存图像到: {output_path}")
    fig.savefig(output_path, dpi=300)
    plt.show()

    print("--- 绘图完成 ---")


if __name__ == '__main__':
    main()