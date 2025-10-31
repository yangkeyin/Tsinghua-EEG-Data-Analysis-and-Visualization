import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 加载图像 ---
# 替换为您的图片文件名
image_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'xiaoshunao_v4.png')
# 以灰度模式加载-
mouse_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if mouse_img is None:
    print(f"错误: 无法加载图片，请检查路径 {image_path}")
else:
    # --- 2. 图像预处理 (关键修改) ---
    # 我们的目标：将背景（白色）变为0，将大脑（彩色/黑色）变为255
    #
    # cv2.THRESH_BINARY_INV (反向二值化):
    # - 像素值 > 阈值 (240) 时, 设为 0 (黑色) -> 这会处理掉白色背景
    # - 像素值 <= 阈值 (240) 时, 设为 maxval (255) (白色) -> 这会保留大脑区域
    #
    # 我们选择一个高的阈值(如240)，以确保只有纯白色背景被去除。
    ret, thresh = cv2.threshold(mouse_img, 240, 255, cv2.THRESH_BINARY_INV) 

    # --- 3. 查找轮廓 ---
    # cv2.findContours 期望在黑色背景上找到白色物体。
    # 我们上一步得到的 'thresh' 图像（白色大脑剪影）正好符合要求。
    # cv2.RETR_EXTERNAL: 只查找最外层的轮廓。
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制所有轮廓
    cv2.drawContours(mouse_img, contours, -1, (0, 255, 0), 3)

    # 显示图像
    cv2.imshow('Contours', mouse_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if not contours:
        print("错误: 未找到任何轮廓。请尝试调整 cv2.threshold 的阈值 (例如 230 或 250)。")
    else:
        # --- 4. 选择最大的轮廓 ---
        # 假设最大的轮廓就是我们想要的脑部轮廓
        # (按面积大小排序)
        mouse_outline_contour = max(contours, key=cv2.contourArea)

                # 将轮廓坐标归一化到-1到1的范围内，以适应MNE的默认坐标系
        x_coords = mouse_outline_contour[:, 0, 0]
        y_coords = mouse_outline_contour[:, 0, 1]
        x_norm = (x_coords - np.mean(x_coords)) / np.ptp(x_coords) * 2
        y_norm = (y_coords - np.mean(y_coords)) / np.ptp(y_coords) * 2
        # 注意：y轴可能需要反转，取决于您的图像。
        y_norm *= -1

        # 创建一个包含轮廓坐标的字典，以备后用
        custom_outlines = {
            'head': (x_norm, y_norm)
        }   

        # --- 5. (可选) 可视化检查 ---
        # 创建一个彩色副本来绘制轮廓，以便检查是否正确
        vis_img = cv2.cvtColor(mouse_img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(vis_img, [mouse_outline_contour], -1, (0, 255, 0), 2) # 用绿色(0,255,0)画出轮廓

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("二值化图像 (大脑剪影)")
        plt.imshow(thresh, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title("提取的轮廓 (绿色)")
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.show()

        # --- 6. 坐标归一化 (保持纵横比) ---
        # Squeeze 掉多余的维度 (N, 1, 2) -> (N, 2)
        coords = mouse_outline_contour.squeeze()
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]
        
        # 计算中心点
        x_mean = np.mean(x_coords)
        y_mean = np.mean(y_coords)
        
        # 计算X和Y的范围
        x_range = np.ptp(x_coords) # ptp = Peak-to-Peak (max - min)
        y_range = np.ptp(y_coords)
        
        # 重点：使用X和Y中 *最大* 的范围作为归一化分母，以保持纵横比
        max_range = max(x_range, y_range)

        # (x - x_mean) / (max_range / 2)  等效于 (x - x_mean) * 2 / max_range
        x_norm = (x_coords - x_mean) * 2 / max_range
        y_norm = (y_coords - y_mean) * 2 / max_range
        
        # OpenCV的y轴（图像坐标）是向下的，
        # 而MNE的y轴（笛卡尔坐标）是向上的，所以需要反转。
        y_norm *= -1

        # --- 7. 创建字典 ---
        custom_outlines = {
            'head': (x_norm, y_norm)
        }

        print("轮廓坐标提取并归一化成功！")

        # (可选) 绘制归一化后的轮廓，检查其形状
        plt.figure(figsize=(5, 5))
        plt.title("归一化后的轮廓 (-1 to 1)")
        plt.plot(x_norm, y_norm, 'b-')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)
        plt.xlabel("归一化 X 坐标")
        plt.ylabel("归一化 Y 坐标")
        plt.show()