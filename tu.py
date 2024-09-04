# -*- coding: utf-8 -*-
# import os
# import numpy as np
# import pandas as pd
# from sklearn.impute import SimpleImputer
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# from PIL import Image
#
# # 替换 'your_image_folder' 为包含图像的文件夹路径
# image_folder = 'E:/t-SNE/'
#
# # 读取图像文件列表
# image_files = ['n0153282900000046.jpg','n0153282900000413.jpg','n0170432300000726.jpg','n0170432300001154.jpg','n0174993900000164.jpg','n0174993900000717.jpg','n0177008100000313.jpg',
#                'n0177008100000503.jpg','n0184338300000153.jpg','n0184338300001238.jpg','n0185567200000147.jpg','n0185567200000952.jpg',
#                'n0191074700000152.jpg','n0191074700000728.jpg','n0193011200000097.jpg','n0193011200000399.jpg','n0198127600000351.jpg',
#                'n0198127600001128.jpg','n0207436700000273.jpg','n0207436700000623.jpg','n0208986700000337.jpg','n0209183100001045.jpg',
#                'n0211454800000790.jpg','n0211454800000889.jpg','n0211673800000109.jpg','n0211673800000387.jpg','n0212007900000929.jpg',
#                'n0212007900001203.jpg','n0212916500000315.jpg','n0212916500000896.jpg','n0213844100000148.jpg','n0213844100000584.jpg',
#                'n0216545600000131.jpg','n0216545600000546.jpg','n0217400100000077.jpg','n0217400100000306.jpg','n0221948600000207.jpg',
#                'n0221948600000419.jpg','n0245740800000524.jpg','n0245740800000976.jpg']  # 替换为实际的图像文件名
#
# # 为每个图像分配类别
# # 这里假设每个图像对应的类别标签存储在一个列表中
# labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] * 2  # 假设有20个类别，每个类别有两张图像
#
# # 读取图像数据并转换为numpy数组
# image_data = []
# for img_file in image_files:
#     img_path = os.path.join(image_folder, img_file)
#     img = Image.open(img_path).convert('L')  # 转为灰度图像
#     img_array = np.array(img).flatten()  # 将图像转为一维数组
#     image_data.append(img_array)
#
# # 创建一个DataFrame
# data = pd.DataFrame(image_data)
#
# # 使用 SimpleImputer 对象填充缺失值
# imputer = SimpleImputer(strategy='mean')
# data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
#
# # 初始化t-SNE模型
# tsne = TSNE(n_components=2, random_state=42)
#
# # 对图像数据进行降维
# image_tsne = tsne.fit_transform(data)
#
# # 创建一个散点图，根据类别用不同颜色绘制
# plt.scatter(image_tsne[:, 0], image_tsne[:, 1], c=labels, cmap='viridis')
# plt.title('t-SNE Visualization')
# plt.show()
import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from PIL import Image

# 替换 'your_image_folder' 为包含图像的文件夹路径
image_folder = 'E:/t-SNE/'

# 获取文件夹中的所有图像文件
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

# 为每个图像分配类别
# 这里假设每个图像对应的类别标签存储在一个列表中
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] * (len(image_files) // 20)  # 假设有20个类别，每个类别有若干张图像

# 读取图像数据并转换为numpy数组
image_data = []
for img_file in image_files:
    img_path = os.path.join(image_folder, img_file)
    img = Image.open(img_path).convert('L')  # 转为灰度图像
    img_array = np.array(img).flatten()  # 将图像转为一维数组
    image_data.append(img_array)

# 创建一个DataFrame
data = pd.DataFrame(image_data)

# 使用 SimpleImputer 对象填充缺失值
imputer = SimpleImputer(strategy='mean')
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# 特征选择/降维（这里使用PCA作为示例）
pca = PCA(n_components=None)  # 选择所有主成分
data_pca = pca.fit_transform(data)

# 初始化 t-SNE 模型
# 初始化 t-SNE 模型
tsne = TSNE(n_components=2, random_state=42, perplexity=100, learning_rate=200, n_iter=5000)

# 对数据进行 t-SNE 降维
data_tsne = tsne.fit_transform(data_pca)

# 创建一个散点图，根据类别用不同颜色绘制
plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels, cmap='viridis')
plt.title('t-SNE Visualization with All Samples')
plt.show()