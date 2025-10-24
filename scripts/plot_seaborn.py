# scripts/plot_connectivity.py

"""
基于Epochs文件计算并绘制脑区连接图脚本

功能:
1. 加载指定被试和条件的预处理后的_epo.fif文件。
2. 使用 MNE 计算指定频段内的通道间连接性 (例如 Coherence)。
3. 使用 MNE 的圆形布局可视化连接性，节点按脑区排序和着色。
4. 为每种条件和每个频段保存一张图。

"""

import os
import sys
import mne
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # 使用 seaborn 可以让热力图更美观
import mne_connectivity
from mne.time_frequency import csd_array_multitaper

# 导入配置
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as config


# ===================================================================
# 在这里设置目标
# ===================================================================
# 定义要分析的状态列表 (必须有对应的 _epo.fif 文件)
TARGET_CONDITIONS = ['rest', 'visual', 'beard_slow', 'beard_fast'] 

# 连接性计算方法: 'coh' (Coherence), 'imcoh' (Imaginary Coherence),
# 'pli' (Phase Lag Index), 'wpli' (Weighted PLI), 'plv' (Phase Locking Value) 等
CONNECTIVITY_METHOD = 'coh'  

# 可视化阈值：只显示绝对值大于此阈值的连接
CONN_THRESHOLD = 0.5 
# ===================================================================
def get_ordered_channels_and_indices(ch_names, region_map):
    ordered_ch_names = []
    original_indices = []
    ch_name_to_original_idx = {name: i for i, name in enumerate(ch_names)}
    region_boundaries = [0] # 记录每个区域结束的索引
    region_names_ordered = [] # 记录区域顺序

    for hemisphere in ['Left', 'Right']:
        for region, channels in region_map[hemisphere].items():
            region_has_channels = False
            for ch_num in channels:
                ch_name = f"C-{ch_num:03d}"
                if ch_name in ch_name_to_original_idx:
                    ordered_ch_names.append(ch_name)
                    original_indices.append(ch_name_to_original_idx[ch_name])
                    region_has_channels = True
            if region_has_channels:
                 region_boundaries.append(len(ordered_ch_names))
                 region_names_ordered.append(f"{hemisphere[0]}-{region}") # 例如 L-M2

    reorder_indices = np.array(original_indices)
    return ordered_ch_names, reorder_indices, region_boundaries, region_names_ordered

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

def calculate_connectivity_manual(epochs, fmin, fmax, method='coh'):
    """
    手动计算给定Epochs对象在指定频段内的频谱相干性 (Coherence)。

    Args:
        epochs (mne.Epochs): MNE Epochs 对象.
        fmin (float): 频段的最低频率.
        fmax (float): 频段的最高频率.
        method (str): 目前只支持 'coh'.

    Returns:
        np.ndarray | None: (n_channels, n_channels) 的相干性矩阵 (在频段内平均)。
                           如果计算失败或方法不支持则返回 None。
    """
    if method != 'coh':
        print(f" 错误: 手动计算目前只支持 'coh' 方法，不支持 '{method}'。")
        return None

    print(f" 手动计算 {fmin}-{fmax} Hz 的 {method}...")

    sfreq = epochs.info['sfreq']
    # 从 Epochs 对象获取数据，形状为 (n_epochs, n_channels, n_times)
    data = epochs.get_data()
    n_channels = data.shape[1]

    try:
        # --- 1. 计算平均互功率谱密度 (CSD) ---
        # csd_array_multitaper 返回 CrossSpectralDensity 对象，包含 CSD 矩阵和频率信息
        csd_obj = csd_array_multitaper(
            data,
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            adaptive=True,
            n_jobs=1,
            verbose=False
        )

        # 获取在频段内平均后的 CSD 矩阵 (n_channels, n_channels)
        # MNE 的 CrossSpectralDensity 对象有 .mean() 方法来获取平均后的数据
        csd = csd_obj.mean().get_data()

        # --- 2. 计算相干性 ---
        coh_matrix = np.zeros((n_channels, n_channels))

        # 提取自功率谱 (对角线元素)
        psd = np.diag(csd).real

        for i in range(n_channels):
            for j in range(n_channels):
                # 提取 Pxx(f), Pyy(f), Pxy(f) 在频段内的平均值
                psd_i = psd[i]  # Pii(f)
                psd_j = psd[j]  # Pjj(f)
                csd_ij = csd[i, j]  # Pij(f)

                # 计算 |Pxy(f)|^2
                abs_csd_ij_squared = np.abs(csd_ij) ** 2
                
                # 计算分母 Pxx(f) * Pyy(f)
                denominator = psd_i * psd_j

                # 避免除以零
                if denominator > 1e-10:
                    coh_matrix[i, j] = abs_csd_ij_squared / denominator
                else:
                    coh_matrix[i, j] = 0

        # --- 3. 检查对角线 ---
        diag_values = np.diag(coh_matrix)
        if not np.allclose(diag_values, 1.0):
            print(" 警告: 手动计算的相干性矩阵对角线不完全为1。可能的原因：数值精度、信号质量。")
            print(f" 对角线值示例: {diag_values[:5]}...")

        return coh_matrix

    except Exception as e:
        print(f" 手动计算连接性失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_connectivity(epochs, fmin, fmax, method='coh'):
    """
    计算给定Epochs对象在指定频段内的频谱连接性。

    Args:
        epochs (mne.Epochs): MNE Epochs 对象.
        fmin (float): 频段的最低频率.
        fmax (float): 频段的最高频率.
        method (str): 连接性计算方法.

    Returns:
        np.ndarray | None: (n_channels, n_channels) 的连接性矩阵 (频段内平均)。
    """
    print(f"  计算 {fmin}-{fmax} Hz 的 {method}...")
    try:
        # 使用 mne-connectivity 计算频谱连接性
        # epochs: 输入的 epochs 数据
        # method: 连接性度量方法（如 'coh' 表示相干）
        # mode='multitaper': 使用多窗法进行频谱估计，适合短数据段
        # sfreq: 采样频率，从 epochs.info 中读取
        # fmin/fmax: 感兴趣的频段范围
        # faverage=True: 在指定频段内对连接性结果做平均，返回单一值
        # mt_adaptive=True: 启用自适应多窗法，提高频谱估计稳定性
        # n_jobs=1: 单核运行，避免多线程带来的额外开销
        # verbose=False: 关闭详细日志输出
        con = mne_connectivity.spectral_connectivity_epochs(
            epochs, method=method, mode='multitaper', 
            sfreq=epochs.info['sfreq'], fmin=fmin, fmax=fmax, 
            faverage=True,  # 在频率范围内平均
            mt_adaptive=True, n_jobs=1, verbose=False)
        
        # 将连接性对象转换为稠密矩阵形式
        # output='dense' 返回完整的 (n_channels, n_channels) 矩阵
        return con.get_data(output='dense')
    except Exception as e:
        print(f"  计算连接性失败: {e}")
        return None


def plot_connectivity_matrix_heatmap(ax, con_matrix_reordered, ordered_labels, 
                                     region_boundaries, region_names, xlabel, 
                                     vmin=0, vmax=1, cmap='coolwarm', 
                                     show_cbar=False, cbar_label=''):
    """
    在指定的 Axes 上绘制排序后的连接性矩阵热力图。

    Args:
        ax (matplotlib.axes.Axes): 要绘制到的子图对象。
        con_matrix_reordered (np.ndarray): 排序后的连接矩阵。
        ordered_labels (list[str]): 排序后的通道标签。
        region_boundaries (list[int]): 脑区分隔线的索引列表。
        region_names (list[str]): 排序后的脑区名称列表。
        title (str): 子图标题。
        vmin (float): 颜色条最小值。
        vmax (float): 颜色条最大值。
        cmap (str): colormap 名称。
        show_cbar (bool): 是否在此子图旁边显示颜色条。
        cbar_label (str): 颜色条的标签。

    Returns:
        matplotlib.image.AxesImage: 返回绘制的热力图图像对象 (用于后续添加颜色条)。
    """
    n_nodes = len(ordered_labels)
    im = ax.imshow(con_matrix_reordered, cmap=cmap, vmin=vmin, vmax=vmax,
                   origin='lower', interpolation='nearest', 
                   extent=[0, n_nodes, 0, n_nodes])
    region_boundaries_middle = [(region_boundaries[i] + region_boundaries[i+1]) / 2 for i in range(len(region_boundaries) - 1)]
    # --- 简化标签和刻度 ---
    ax.invert_yaxis()  # 这将反转Y轴，使0在顶部
    ax.set_xticks(region_boundaries_middle)
    ax.set_yticks(region_boundaries_middle)
    ax.set_xticklabels(region_names, fontsize=8, rotation=45, ha='center')
    ax.set_yticklabels(region_names, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.xaxis.set_label_position('top')
    # 将y轴刻度线去掉
    ax.tick_params(axis='y', which='both', left=False)
    # 将x轴刻度线放在顶部
    ax.tick_params(axis='x', which='both', bottom=False)

    # --- 添加脑区分隔线 ---
    for boundary in region_boundaries[1:-1]: # 跳过首尾
        ax.axhline(boundary, color='white', lw=1.0, linestyle='--')
        ax.axvline(boundary, color='white', lw=1.0, linestyle='--')
    
    # 添加颜色条 (如果需要)
    if show_cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label(cbar_label, fontsize=10)
        cbar.ax.tick_params(labelsize=8)

    return im # 返回图像对


# --- 修改后的 main 函数 ---
def main():
    """主函数"""
    # 假设 load_epochs 函数已正确实现并导入
    epochs_all_conditions = load_epochs(TARGET_CONDITIONS) 

    if not epochs_all_conditions:
        print("未能加载任何Epochs数据，程序退出。")
        return

    first_epochs = next(iter(epochs_all_conditions.values()))
    ch_names = first_epochs.ch_names
    
    # 假设 get_ordered_channels_and_indices 已正确实现并导入
    ordered_labels, reorder_idx, region_boundaries, region_names = \
        get_ordered_channels_and_indices(ch_names, config.REGION_MAP)

    n_conditions = len(TARGET_CONDITIONS)
    n_bands = len(config.FREQUENCY_BANDS)
    band_names = list(config.FREQUENCY_BANDS.keys())

    # 创建大图网格
    fig, axes = plt.subplots(n_conditions, n_bands, 
                             figsize=(n_bands * 5, n_conditions * 4.5), # 调整尺寸
                             squeeze=False) # 确保 axes 总是 2D
    # plt.show()
    fig.suptitle(f'Connectivity Between Brain Regions', fontsize=20, y=0.98)
    # 用于颜色条的设置
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [左, 下, 宽, 高]
    # 存储所有子图的图像对象，用于统一颜色条
    last_im = None


    # 遍历计算和绘图
    for i, condition in enumerate(TARGET_CONDITIONS):
        print(f"\n--- Processing Condition: {condition} ---")
        
        # 确定这一行的颜色范围
        # all_matrices_in_row = []
        # for j, (band_name, (fmin, fmax)) in enumerate(config.FREQUENCY_BANDS.items()):
        #     if condition in epochs_all_conditions:
        #          epochs = epochs_all_conditions[condition]
        #          con_matrix = calculate_connectivity_manual(epochs, fmin, fmax, method=CONNECTIVITY_METHOD)
        #          if con_matrix is not None:
        #              all_matrices_in_row.append(con_matrix)
        
        # if not all_matrices_in_row: continue # 如果这一行都没有数据，跳过

        # # 计算这一行统一的 vmin, vmax (基于5%和95%百分位)
        # all_valid_con_values = np.concatenate([m[~np.isnan(m)].flatten() for m in all_matrices_in_row])
        # if all_valid_con_values.size == 0: continue # 如果没有有效值
        # vmin = np.percentile(all_valid_con_values, 5) if len(all_valid_con_values) > 0 else 0
        # vmax = np.percentile(all_valid_con_values, 95) if len(all_valid_con_values) > 0 else 1
        # # 确保 vmin 和 vmax 合理，例如 coherence 在 0-1 之间
        # vmin = max(0, vmin)
        # vmax = min(1, vmax) if CONNECTIVITY_METHOD in ['coh', 'plv'] else max(vmin + 1e-6, vmax) # 避免vmin=vmax
        # if vmax <= vmin: vmax = vmin + 0.1 # 如果范围太小，给个默认范围


        for j, (band_name, (fmin, fmax)) in enumerate(config.FREQUENCY_BANDS.items()):
            ax = axes[i, j]
            
            # 查找之前计算好的矩阵 (或重新计算一次，取决于内存)
            con_matrix = None
            if condition in epochs_all_conditions:
                epochs = epochs_all_conditions[condition]
                # 重新计算或从缓存读取 (这里简单起见重新计算)
                con_matrix = calculate_connectivity_manual(epochs, fmin, fmax, method=CONNECTIVITY_METHOD)

            if con_matrix is not None:
                con_matrix_reordered = con_matrix[reorder_idx][:, reorder_idx]
                
                # 调用修改后的绘图函数，不显示单独的颜色条
                img = plot_connectivity_matrix_heatmap(
                          ax, con_matrix_reordered, 
                          ordered_labels, 
                          region_boundaries, 
                          region_names, 
                          xlabel=band_name, # 子图标题留空或只写条件名
                          vmin=0, vmax=1, # 使用行统一的范围
                          show_cbar=False # 不单独显示颜色条
                      )
                last_im = img # 记录最后一个子图的图像对象
            else:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                ax.axis('off')

            # 添加行/列标签
            if ax.get_subplotspec().is_first_col():
                ax.set_title(condition, fontsize=18, fontweight='bold')
            


    # --- 添加共享颜色条 ---
    # --- 添加共享颜色条 ---
    # 使用之前创建的 cax 和保存的 last_im
    if last_im is not None:
        cbar = fig.colorbar(last_im, cax=cax)
        cbar.set_label(f'Coherence', fontsize=10) # 这里的band_name是最后一个
        cbar.ax.tick_params(labelsize=8)

    # 调整整体布局
    fig.subplots_adjust(left=0.05, right=0.9, top=0.9, bottom=0.05, wspace=0.5, hspace=0.4)
    # 保存图
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    output_filename = f"all_conditions_bands_{CONNECTIVITY_METHOD}_matrix_manual.png"
    output_path = os.path.join(config.PLOTS_DIR, output_filename)
    print(f"\n保存图像到: {output_path}")
    fig.savefig(output_path, dpi=300)
    plt.show()

    print("\n--- 连接性分析完成 ---")

if __name__ == '__main__':
    # 确保 load_epochs, calculate_connectivity, get_ordered_channels_and_indices 函数可用
    # (如果它们在其他文件，需要正确导入)
    main()