"""
# 此文件用于“批量”绘制原始脑电数据轨迹
# 根据配置文件指定的被试、范式和试次，加载EEG数据后，
# 在设定“时间窗”内绘制左右半球各脑区原始波形，并导出图像。
"""
import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines

# 导入配置
# (优化路径，确保无论从哪里运行都能找到config)
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as config



def plot_raw_traces_by_region(ax, raw, region_map, hemisphere_name, fontsize=12, channel_prefix='C-'):
    """
    在给定的轴上按脑区堆叠绘制Raw数据轨迹。

    参数:
        ax (matplotlib.axes.Axes): 要绘制的子图对象.
        raw (mne.io.Raw): 包含连续数据的Raw对象.
        region_map (dict): 单个半球的脑区通道映射.
        hemisphere_name (str): 半球名称 ('Left' or 'Right').
    """
    ax.set_title(hemisphere_name, fontsize=fontsize)
    
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
        ch_names = [f"{channel_prefix}{c:03d}" for c in channels]
        ch_names_in_data = [name for name in ch_names if name in raw.ch_names]
        all_ch_names.extend(ch_names_in_data)

    if not all_ch_names:
        print(f"警告: 在 {hemisphere_name} 半球没有找到任何可供绘制的通道。")
        return

    # 提取所有相关通道的数据来计算合适的偏移量
    all_data, times = raw.get_data(picks=all_ch_names, return_times=True)
    
    # 基于数据的峰峰值范围来确定一个合理的偏移量
    offset_step = np.percentile(all_data.max(axis=1) - all_data.min(axis=1), 95) * 1.5
    if offset_step == 0: offset_step = 100 # 如果数据是平的，给一个默认值

    offset = 0

    # 再次遍历并绘制
    for region in ordered_regions:
        color = config.REGION_COLORS.get(region, 'black')
        channels = region_map[region]
        ch_names = [f"{channel_prefix}{c:03d}" for c in channels]

        y_ticks.append(channel_counter * -offset_step)
        y_labels.append(f"{region}")
        for ch_name in ch_names:
            if ch_name in raw.ch_names:
                # 获取单个通道的数据
                channel_data, times = raw.get_data(picks=[ch_name], return_times=True)
                
                channel_data *= config.SCALE_AMPLITUDE # 进行转换
                # 计算偏移量并添加向上的垂直偏移
                offset = channel_counter * -offset_step
                ax.plot(times, channel_data[0] + offset, color=color, linewidth=2.0)
                
                channel_counter += 1

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    # ax.set_xlim(times[0], times[-1])
    # ax.set_xlabel("Time (s)", fontsize=12)
    # ax.grid(True, linestyle=':', alpha=0.6, axis='x')

    # --- 隐藏边框和Y轴刻度线 ---
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # 隐藏X轴刻度标签和刻度
    ax.set_xticklabels([]) # 移除X轴刻度标签
    ax.set_xticks([]) # 移除X轴刻度
    
    # # 在第一个子图上设置Y轴标签
    # if 'Left' in hemisphere_name:
    #     ax.set_ylabel("Channels", fontsize=12)
        
        


def main():
    """主函数"""
    print(f"--- 绘制 Raw Trace: 被试 {config.TARGET_SUBJECT}, 范式 {config.TARGET_PARADIGM}, Trial {config.TARGET_TRIAL} ---")

    # 1. 精确构建文件路径
    paradigm_dir = os.path.join(config.PROCESSED_DATA_DIR, config.TARGET_SUBJECT)
    fif_filename = f"{config.TARGET_SUBJECT}_{config.TARGET_PARADIGM}_trial{config.TARGET_TRIAL}{config.FIF_FILE_SUFFIX}"
    fif_path = os.path.join(paradigm_dir, fif_filename)

    if not os.path.exists(fif_path):
        print(f"错误: 文件未找到 -> {fif_path}")
        return

    # 2. 加载Raw数据
    print(f"加载文件: {fif_path}")
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)

    # 降采样
    resample_freq = config.RESAMPLE_FREQ
    if resample_freq and raw.info['sfreq'] > resample_freq:
        print(f"降采样至 {resample_freq} Hz...")
        # 在降采样之前加载数据到内存是推荐做法
        raw.load_data(verbose=config.VERBOSE)
        raw.resample(sfreq=resample_freq, verbose=config.VERBOSE)
    
    # 提取通道前缀
    channel_prefix = raw.ch_names[0][:2]
    
    # ---批量绘制原始轨迹图---
    # 为每个被试每个范式每个试次创建一个文件夹
    output_dir = os.path.join(config.PLOTS_DIR, f'{config.TARGET_SUBJECT}_{config.TARGET_PARADIGM}_trial{config.TARGET_TRIAL}')
    os.makedirs(output_dir, exist_ok=True)
    # 计算需要绘制的时间窗口数量
    num_windows = int(np.ceil((config.PLOT_T_MAX - config.PLOT_T_MIN) / config.PLOT_WINDOW_SIZE))
    for window_idx in range(num_windows):
        t_min = config.PLOT_T_MIN + window_idx * config.PLOT_WINDOW_SIZE
        t_max = min(t_min + config.PLOT_WINDOW_SIZE, config.PLOT_T_MAX)
    
        # 3. （可选）截取时间段
        if t_min is not None and t_max is not None:
            print(f"截取时间窗口: {t_min}s - {t_max}s")
            try:
                raw_cropped = raw.copy().crop(tmin=t_min, tmax=t_max)
            except ValueError as e:
                print(f"错误: 无法截取时间窗口。可能设置的时间超出了数据范围 ({raw.times[-1]:.2f}s)。程序终止。错误: {e}")
                # 终止程序
                sys.exit(1)

        # 4. 创建绘图画布
        fig, axes = plt.subplots(1, 2, figsize=(22, 12))
        fig.suptitle(f"Raw Trace: {config.TARGET_SUBJECT} - {config.TARGET_PARADIGM} - Trial {config.TARGET_TRIAL} ({t_min:.2f}s - {t_max:.2f}s)", fontsize=18, y=0.98)
        fig.patch

        # 5. 绘制左右脑
        print("正在绘制左半球...")
        plot_raw_traces_by_region(axes[0], raw_cropped, config.REGION_MAP['Left'], 'Left', fontsize=20, channel_prefix=channel_prefix)
        
        print("正在绘制右半球...")
        plot_raw_traces_by_region(axes[1], raw_cropped, config.REGION_MAP['Right'], 'Right', fontsize=20, channel_prefix=channel_prefix)

        # 修改main函数中的图形布局部分
        # 在保存图像前添加以下代码，确保两个子图大小一致
        fig.subplots_adjust(wspace=0.3)  # 调整子图间距
        # 在右下角绘制时间和振幅尺度
        # 绘制垂直虚线
        fig.add_artist(lines.Line2D([0.98, 0.98], [0.05, 0.1], 
                color='black', linewidth=2, transform=fig.transFigure))
        # 绘制水平虚线
        fig.add_artist(lines.Line2D([0.98, 0.94], [0.05, 0.05], 
                color='black', linewidth=2, transform=fig.transFigure))
        fig.text(0.92, 0.05, f'{int(500 * (5 / (t_max - t_min)))}ms', transform=fig.transFigure, fontsize=15)
        fig.text(0.96, 0.03, f'{int(5 / config.SCALE_AMPLITUDE)}mv', transform=fig.transFigure, fontsize=15)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # 6. 保存图像
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"{t_min:.2f}s-{t_max:.2f}s_raw_trace.png"
        output_path = os.path.join(output_dir, output_filename)
        print(f"保存图像到: {output_path}")
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        plt.show()

        print(f"对于时间窗口 {t_min:.2f}s - {t_max:.2f}s 已保存至 {output_path}")

    print(f"--- 时间窗口 {config.PLOT_T_MIN:.2f}s - {config.PLOT_T_MAX:.2f}s 绘图完成，已保存至 {output_dir} --- ")


if __name__ == '__main__':
    main()