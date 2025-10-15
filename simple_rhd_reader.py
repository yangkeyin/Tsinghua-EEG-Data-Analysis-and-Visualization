import numpy as np
import matplotlib.pyplot as plt

# 这是读取RHD文件最常用的函数
# 来自intan-rhd-loader库
def read_rhd_file(file_path):
    """
    读取RHD文件的简单函数
    
    Args:
        file_path: RHD文件的路径
        
    Returns:
        data: 包含所有通道数据的字典
    """
    try:
        # 导入intan-rhd-loader库的读取函数
        from intan.load_intan_rhd_format import load_intan_rhd_format
        
        # 调用官方库函数读取文件
        data = load_intan_rhd_format(file_path)
        
        print(f"成功读取文件: {file_path}")
        print(f"通道数量: {len(data['amplifier_data'])}")
        print(f"采样率: {data['frequency_parameters']['amplifier_sample_rate']} Hz")
        
        return data
        
    except ImportError:
        print("错误: 未安装intan-rhd-loader库")
        print("请运行: pip install intan-rhd-loader")
        return None
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
        return None

# 一个简单的示例，展示如何使用这个函数
if __name__ == "__main__":
    # 替换为您的RHD文件路径
    # 例如: file_path = "d:\yangkeyin\datasets\251010清华数据分析\acute\20250612visual\visual_250613_162743\visual_250613_162743.rhd"
    file_path = "d:\yangkeyin\datasets\251010清华数据分析\acute\20250612visual\visual_250613_162743\visual_250613_162743.rhd"
    
    if file_path:
        # 读取文件
        data = read_rhd_file(file_path)
        
        # 如果读取成功，可以进行简单的可视化
        if data is not None:
            # 绘制第一个通道的前1000个采样点
            plt.figure(figsize=(10, 4))
            plt.plot(data['t'][:1000], data['amplifier_data'][0][:1000])
            plt.title('通道 0 数据示例')
            plt.xlabel('时间 (秒)')
            plt.ylabel('幅度 (微伏)')
            plt.grid(True)
            plt.tight_layout()
            plt.show()