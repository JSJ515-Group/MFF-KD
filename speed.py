import matplotlib.pyplot as plt

# # 模拟数据
# models = ['RetinaNet','FSAF','Fovea','SABL','PETNet','GCFNet']
# speeds = [17.5,17.5,18.2,16.4,26.2,59.97]
#
# # 绘制柱状图
# plt.figure(figsize=(10, 8))
# bars = plt.bar(models, speeds, color='#CCA6BF')
#
# # 设置标签大小
# plt.xlabel('Model', fontsize=24)  # 设置横坐标标签字体大小为16
# plt.ylabel('FPS', fontsize=24)  # 设置纵坐标标签字体大小为16
# # plt.title('Model Speed Comparison', fontsize=24)  # 设置图表标题字体大小为24
#
# # 在每个柱状图上标注数据
# for bar in bars:
#     yval = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 1), ha='center', va='bottom', fontsize=16)  # 调整数据标签的字体大小为12
#
# plt.show()


#  检测单张图片
import netron
import torch
from PIL import Image
import onnx
import sys
import os
import numpy as np
from pathlib import Path
from typing import Union
import cv2
from ultralytics import YOLO
def img_test_01():
	# 训练好的模型权重路径
    model = YOLO("E:/gold_yolo/train/xiaorong/yolov8s_150/weights/best.pt")
    # 测试图片的路径
    img = cv2.imread("E:/xinVisDrone/images/val/0000242_01475_d_0000007.jpg")
    res = model(img)
    ann = res[0].plot()
    while True:
        cv2.imshow("yolo", ann)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # 设置保存图片的路径
    cur_path = sys.path[0]
    print(cur_path, sys.path)

    if os.path.exists(cur_path):
        cv2.imwrite(cur_path + "out.jpg", ann)
    else:
        os.mkdir(cur_path)
        cv2.imwrite(cur_path + "out.jpg", ann)


img_test_01()