
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import mne
# 导入配置
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as config
import cv2
from matplotlib.path import Path
from scipy.interpolate import RBFInterpolator


# 定义要分析的状态列表 (必须有对应的 _epo.fif 文件)
TARGET_CONDITIONS = ['rest', 'visual', 'beard_slow', 'beard_fast'] 

def load_epochs(conditions):
    """
    加载指定被试和条件对应的 Epochs 文件。

    Args:
        subject (str): 被试ID.
        conditions (list[str]): 条件名称列表.

    Returns:
        dict: 字典，键是条件名，值是 MNE Epochs 对象。
    """
    epochs_dict = {}
    
    # 检查 Epochs 目录是否存在
    if not hasattr(config, 'EPOCHS_DIR') or not os.path.isdir(config.EPOCHS_DIR):
         print(f"错误: Config 文件中未定义 EPOCHS_DIR 或目录不存在: {getattr(config, 'EPOCHS_DIR', '未定义')}")
         return {}
         
    for condition in conditions:
        epo_filename = f"{condition}_epo.fif" 
        # 假设 Epochs 文件直接保存在 EPOCHS_DIR 下，且没有被试子目录
        # 如果 process_fif_to_epochs.py 的保存路径包含被试，需要相应修改
        epo_path = os.path.join(config.EPOCHS_DIR, epo_filename)
        
        if os.path.exists(epo_path):
            print(f"  加载: {epo_filename}")
            try:
                # MNE < 1.0 使用 preload=True/False, MNE >= 1.0 默认 preload=True
                # 为了兼容性，不显式设置 preload
                epochs_dict[condition] = mne.read_epochs(epo_path, verbose=False) 
            except Exception as e:
                print(f"  加载 Epochs 文件 {epo_path} 失败: {e}")
        else:
            print(f"  警告: 未找到 Epochs 文件: {epo_path}")
            
    return epochs_dict

def create_custom_outlines(image_path):
    """
    加载图像并提取轮廓，保持纵横比。
    """
    mouse_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if mouse_img is None:
        print(f"错误: 无法加载图片，请检查路径 {image_path}")
        return None, None
    
    # 二值化
    ret, thresh = cv2.threshold(mouse_img, 240, 255, cv2.THRESH_BINARY_INV) # 把小于240的像素设为0，大于240的像素设为255
    
    # 查找轮廓
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 返回所有外部轮廓，每个轮廓都是一个点集
    
    if not contours:
        print("错误: 未找到任何轮廓。")
        return None, None
    
    # 选最大的轮廓
    mouse_outline_contour = max(contours, key=cv2.contourArea)
    
    # --- 修正：保持纵横比的归一化 ---
    # (N, 1, 2) -> (N, 2)
    coords = mouse_outline_contour.squeeze()
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    
    # 将轮廓坐标归一化到-1到1的范围内
    x_coords = mouse_outline_contour[:, 0, 0]
    y_coords = mouse_outline_contour[:, 0, 1]
    x_norm = (x_coords - np.mean(x_coords)) / np.ptp(x_coords) * 2
    y_norm = (y_coords - np.mean(y_coords)) / np.ptp(y_coords) * 2

    # 3. 反转Y轴 (图像坐标系 -> 笛卡尔坐标系)
    y_norm *= -1
    
    return x_norm, y_norm

def plot_topomap(data, pos, ax=None, outline=None):
    """
    绘制topomap图。

    Args:
        data (array): 数据数组，形状为 (n_channels,)
        pos (dict): 通道位置信息，键是通道名，值是 (x, y) 坐标
        axes (plt.Axes, optional): 子图对象. 如果为 None，则创建新图.
    """
    # 获取轮廓
    x_outline, y_outline = outline
    # --- 网格范围必须匹配轮廓范围 ---
    grid_x_min, grid_x_max = np.min(x_outline) - 0.1, np.max(x_outline) + 0.1
    grid_y_min, grid_y_max = np.min(y_outline) - 0.1, np.max(y_outline) + 0.1

    # 1. 缩放电极位置
    # (这个 4,6.5 是 "magic number"，您需要确保这个缩放比例是正确的，
    #  它决定了电极在大脑轮廓内的相对大小)
    normalized_pos = np.stack((pos[:,0] * (grid_x_max / 4.5), 
                               pos[:,1] * (grid_y_max / 6)), axis=1)
    # normalized_pos[:,-1] = normalized_pos[:,1]+0.3
    
    # (注意: 1000x1000 网格 (100万点) 可能非常慢， 
    #  对于绘图来说 300x300 或 500x500 通常足够了)
    grid_res = 500 
    x_grid, y_grid = np.meshgrid(
        np.linspace(grid_x_min, grid_x_max, grid_res), 
        np.linspace(grid_y_min, grid_y_max, grid_res)
    )
    points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

    # 5. 创建遮罩 (Mask)
    path = Path(np.array([x_outline, y_outline]).T) # 创建路径对象，定义了多边形轮廓
    mask = path.contains_points(points).reshape(x_grid.shape) # 检查每个点是否在路径内

    # 6. --- 修正：使用 RBF 进行插值和外插 ---
    # kernel='thin_plate_spline' 是一种平滑且常用的RBF核
    rbfi = RBFInterpolator(normalized_pos, data, kernel='thin_plate_spline') #创建RBF插值器对象
    interpolated_data = rbfi(points).reshape(x_grid.shape) # 绘制插值网络

    # 7. 应用遮罩
    interpolated_data[~mask] = np.nan

    # 8. --- 绘图 ---

    # --- 修正：imshow 的 'extent' 必须匹配网格范围 ---
    extent = [grid_x_min, grid_x_max, grid_y_min, grid_y_max]
    
    im = ax.imshow(
        interpolated_data, 
        extent=extent, 
        origin='lower', 
        cmap='jet',
        interpolation='bicubic' # 'bicubic' 或 'quadric' 提供平滑效果

    )

    # 绘制轮廓线
    ax.plot(x_outline, y_outline, color='black', linewidth=1.5)
    
    # 绘制电极点，以检查对齐情况
    ax.scatter(normalized_pos[:, 0], normalized_pos[:, 1], c='black', s=10, zorder=10, label='电极位置')

    # 绘制电极索引
    # for i, (x, y) in enumerate(normalized_pos):
    #     ax.text(
    #         x, y+0.05, str(i), 
    #         color='black',       # 黑色字体
    #         fontsize=7,          # 字体大小 (可以调整)
    #         fontweight='bold',   # 粗体
    #         ha='center',         # 水平居中
    #         va='center',         # 垂直居中
    #         zorder=11            # 确保在白点之上
    #     )

    # ax.axis('off')
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=False)
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_aspect('equal') # 确保X和Y轴等比例

    return im

def normalize(tfr, times, baseline=(0,0.1)):
    """
    对TFR数据进行归一化处理。

    Args:
        tfr (array): TFR数据数组，形状为 (n_channels, n_freqs, n_times)

    Returns:
        array: 归一化后的TFR数据，形状与输入相同
    """
    # 提取baseline时间点
    baseline_idx = (times >= baseline[0]) & (times <= baseline[1])
    baseline_data = tfr[:, :, baseline_idx] 
    # 结果形状为 (n_channels, freqs, 1)
    mean_baseline = np.mean(baseline_data, axis=2, keepdims=True)
    std_baseline = np.std(baseline_data, axis=2, keepdims=True)

    # 避免除以0
    std_baseline[std_baseline == 0] = 1
    
    # 归一化公式：(x - mean_baseline) / std_baseline
    normalized_tfr = 10 * np.log10(tfr / mean_baseline)

    # 取所有频率段的平均值
    normalized_tfr = np.mean(normalized_tfr, axis=1, keepdims=False)

    return normalized_tfr
    

def main():
    # 假设 load_epochs 函数已正确实现并导入
    epochs_all_conditions = load_epochs(TARGET_CONDITIONS) 

    if not epochs_all_conditions:
        print("未能加载任何Epochs数据，程序退出。")
        return

    # 对每个epochs计算tfr
    tfr_all_conditions = {}
    for condition, epochs in epochs_all_conditions.items():
        tfr_all_conditions[condition] = {}
        print(f"  计算 {condition} 的 TFR...")
        tfr = epochs.compute_tfr(method='morlet', freqs=np.arange(config.TMAP_FREQ_RANGE[0], config.TMAP_FREQ_RANGE[1]), n_cycles=config.TMAP_N_CYCLES, average=True, verbose=False)
        if condition == 'rest':
            tfr_normalized = normalize(tfr.data, tfr.times, baseline=(0,2))    
        else:
            tfr_normalized = normalize(tfr.data, tfr.times, baseline=(-0.5,0))    
        tfr_all_conditions[condition]['times'] = tfr.times      
        tfr_all_conditions[condition]['tfr'] = tfr_normalized
    
    
    # 在一张图上绘制所有状态的tfr绘制在多个topomap子图中
    fig, axes = plt.subplots(4, 7, figsize=(20, 12))
    fig.suptitle(' Event-related response', fontsize=25)
    # 获取轮廓
    image_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'xiaoshunao_v4.png') 
    os.path.exists(image_file)
    x_outline, y_outline = create_custom_outlines(image_file)
    
    last_im = None
    for ax_i_index, (condition, value) in zip(range(4), tfr_all_conditions.items()):
        tfr = value['tfr']
        times = value['times']
        # 根据时间间隔绘制topomap
        for ax_j_index, t_index in enumerate(np.arange(-3, 4)):
            ax = axes[ax_i_index, ax_j_index]
            time = 0 # 初始化时间点
            if condition == 'rest':
                time = float((t_index + 3) * config.TMAP_TIME_GAP)     # 休息状态时间点从0开始
                times_points = np.where(np.isclose(times, time))[0][0]
            else:
                time = float(t_index * config.TMAP_TIME_GAP)     # 非休息状态时间点从3开始
                times_points = np.where(np.isclose(times, time))[0][0]
            # 绘制topomap
            print(f'绘制 {condition} 状态，时间点 {time:.2f}s')
            im = plot_topomap(tfr[:, times_points], pos = config.POSITION, ax=ax, outline=(x_outline, y_outline))
            last_im = im
            ax.set_xlabel(f'{time:.2f}s', loc='center', fontsize=12)
            ax.xaxis.set_label_position('top')
            if ax_j_index == 0:
                ax.set_title(condition, fontsize=18, fontweight='bold')
                
    
    # 保存图片
    if not hasattr(config, 'PLOTS_DIR') or not os.path.isdir(config.PLOTS_DIR):
        print(f"错误: Config 文件中未定义 PLOTS_DIR 或目录不存在: {getattr(config, 'PLOTS_DIR', '未定义')}")
        return
    output_dir = os.path.join(config.PLOTS_DIR, 'topomap')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    fig.subplots_adjust(left=0.05, right=0.88, top=0.9, bottom=0.1, hspace=0.4, wspace=0.3)
    cbar = fig.colorbar(last_im, cax=fig.add_axes([0.92, 0.15, 0.02, 0.7]))
    cbar.set_label('Power', fontsize=15)
    cbar.ax.tick_params(labelsize=12)
    fig.show()
    fig.savefig(os.path.join(output_dir, f'test_DB_normalize_0.2gap_{config.TMAP_FREQ_RANGE[0]}_{config.TMAP_FREQ_RANGE[1]}.png'))

if __name__ == '__main__':
    main()