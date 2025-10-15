"""
# 此文件用于绘制原始脑电数据轨迹，按脑区和半球进行可视化展示。
# 程序会根据配置文件中的设置，加载指定被试、范式和试验的EEG数据，
# 并在指定时间窗口内绘制左右半球各脑区的原始信号轨迹，最后保存图像。
"""
import os
import mne
import numpy as np
import matplotlib.pyplot as plt

# 导入配置
# (优化路径，确保无论从哪里运行都能找到config)
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as config


# ===================================================================
# 在这里精确指定你要绘制的单个目标
# ===================================================================
TARGET_SUBJECT = 'mouse1'
TARGET_PARADIGM = 'visual'
TARGET_TRIAL = 1

# 设置要显示的时间窗口 (单位: 秒)。如果想看全部，可以设为 None。
PLOT_T_MIN = 10.0
PLOT_T_MAX = 15.0  # 例如，显示从第10秒到第15秒，共5秒的数据
# ===================================================================


def plot_raw_traces_by_region(ax, raw, region_map, hemisphere_name):
    """
    在给定的轴上按脑区堆叠绘制Raw数据轨迹。

    参数:
        ax (matplotlib.axes.Axes): 要绘制的子图对象.
        raw (mne.io.Raw): 包含连续数据的Raw对象.
        region_map (dict): 单个半球的脑区通道映射.
        hemisphere_name (str): 半球名称 ('Left' or 'Right').
    """
    ax.set_title(hemisphere_name, fontsize=14)
    
    y_ticks = []
    y_labels = []
    offset_step = 0 # 动态计算偏移量
    
    # 按照config中定义的脑区顺序从上到下绘制
    channel_counter = 0
    all_ch_names = []
    ordered_regions = list(region_map.keys())

    # 预先收集所有需要绘制的通道
    for region in ordered_regions:
        channels = region_map[region]
        # 将映射的通道编号转换为通道名称
        ch_names = [f"{config.CHANNEL_PREFIX}{c:03d}" for c in channels]
        ch_names_in_data = [name for name in ch_names if name in raw.ch_names]
        all_ch_names.extend(ch_names_in_data)

    if not all_ch_names:
        print(f"警告: 在 {hemisphere_name} 半球没有找到任何可供绘制的通道。")
        return

    # 提取所有相关通道的数据来计算合适的偏移量
    all_data, times = raw.get_data(picks=all_ch_names, return_times=True)
    all_data *= 1e6 # 转换为 µV
    
    # 基于数据的峰峰值范围来确定一个合理的偏移量
    offset_step = np.percentile(all_data.max(axis=1) - all_data.min(axis=1), 95) * 1.5
    if offset_step == 0: offset_step = 100 # 如果数据是平的，给一个默认值

    offset = 0

    # 再次遍历并绘制
    for region in ordered_regions:
        color = config.REGION_COLORS.get(region, 'black')
        channels = region_map[region]
        ch_names = [f"{config.CHANNEL_PREFIX}{c:03d}" for c in channels]

        y_ticks.append(channel_counter * -offset_step)
        y_labels.append(f"{region}")
        
        for ch_name in ch_names:
            if ch_name in raw.ch_names:
                # 获取单个通道的数据
                channel_data, times = raw.get_data(picks=[ch_name], return_times=True)
                channel_data *= 1e6 # 转换为 µV
                
                # 计算偏移量并绘制
                offset = channel_counter * -offset_step
                ax.plot(times, channel_data[0] + offset, color=color, linewidth=1.0)
                
                channel_counter += 1

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlim(times[0], times[-1])
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6, axis='x')
    
    # 在第一个子图上设置Y轴标签
    if 'Left' in hemisphere_name:
        ax.set_ylabel("Channels", fontsize=12)


def main():
    """主函数"""
    print(f"--- 绘制 Raw Trace: 被试 {TARGET_SUBJECT}, 范式 {TARGET_PARADIGM}, Trial {TARGET_TRIAL} ---")

    # 1. 精确构建文件路径
    paradigm_dir = os.path.join(config.PROCESSED_DATA_DIR, TARGET_SUBJECT)
    fif_filename = f"{TARGET_SUBJECT}_{TARGET_PARADIGM}_trial{TARGET_TRIAL}{config.FIF_FILE_SUFFIX}"
    fif_path = os.path.join(paradigm_dir, fif_filename)

    if not os.path.exists(fif_path):
        print(f"错误: 文件未找到 -> {fif_path}")
        return

    # 2. 加载Raw数据
    print(f"加载文件: {fif_path}")
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)

    # 3. （可选）截取时间段
    if PLOT_T_MIN is not None and PLOT_T_MAX is not None:
        print(f"截取时间窗口: {PLOT_T_MIN}s - {PLOT_T_MAX}s")
        try:
            raw.crop(tmin=PLOT_T_MIN, tmax=PLOT_T_MAX)
        except ValueError as e:
            print(f"警告: 无法截取时间窗口。可能设置的时间超出了数据范围 ({raw.times[-1]:.2f}s)。将显示全部数据。错误: {e}")


    # 4. 创建绘图画布
    fig, axes = plt.subplots(1, 2, figsize=(22, 12))
    fig.suptitle(f"Raw Trace: {TARGET_SUBJECT} - {TARGET_PARADIGM} - Trial {TARGET_TRIAL}", fontsize=18, y=0.98)

    # 5. 绘制左右脑
    print("正在绘制左半球...")
    plot_raw_traces_by_region(axes[0], raw, config.REGION_MAP['Left'], 'Left Hemisphere')
    
    print("正在绘制右半球...")
    plot_raw_traces_by_region(axes[1], raw, config.REGION_MAP['Right'], 'Right Hemisphere')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 6. 保存图像
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    output_filename = f"{TARGET_SUBJECT}_{TARGET_PARADIGM}_trial{TARGET_TRIAL}_raw_trace.png"
    output_path = os.path.join(config.PLOTS_DIR, output_filename)
    print(f"保存图像到: {output_path}")
    fig.savefig(output_path, dpi=300)
    plt.show()

    print("--- 绘图完成 ---")


if __name__ == '__main__':
    main()