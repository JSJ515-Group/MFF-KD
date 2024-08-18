# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import datasets
# from sklearn.manifold import TSNE
#
#
# # 加载数据
# def get_data():
#     """
#     :return: 数据集、标签、样本数量、特征数量
#     """
#     digits = datasets.load_digits(n_class=10)
#     data = digits.data
#     label = digits.target
#     n_samples, n_features = data.shape
#     return data, label, n_samples, n_features
#
#
# # 对样本进行预处理并画图
# def plot_embedding(data, label, title, selected_classes):
#     """
#     :param data: 数据集
#     :param label: 样本标签
#     :param title: 图像标题
#     :param selected_classes: 选择的类别
#     :return: 图像
#     """
#     x_min, x_max = np.min(data, 0), np.max(data, 0)
#     data = (data - x_min) / (x_max - x_min)
#
#     fig = plt.figure()
#     ax = plt.subplot(111)
#
#     for i in range(data.shape[0]):
#         if label[i] in selected_classes:
#             plt.scatter(data[i, 0], data[i, 1], color=plt.cm.Set1(label[i] / 10), s=10)
#
#     plt.xticks([])
#     plt.yticks([])
#     plt.title(title, fontsize=14)
#     return fig
#
#
# # 主函数，执行t-SNE降维
# def main():
#     data, label, n_samples, n_features = get_data()
#
#     # 选择要显示的类别（这里选择0到4）
#     selected_classes = [0, 1, 2, 3, 4]
#
#     print('Starting compute t-SNE Embedding...')
#     ts = TSNE(n_components=2, init='pca', random_state=0)
#     result = ts.fit_transform(data)
#
#     fig = plot_embedding(result, label, 't-SNE Embedding of digits (Selected Classes)', selected_classes)
#     plt.show()
#
#
# # 主函数
# if __name__ == '__main__':
#     main()
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.manifold import TSNE

def get_data():
    digits = datasets.load_digits(n_class=10)
    data = digits.data
    label = digits.target
    n_samples, n_features = data.shape
    return data, label, n_samples, n_features

def plot_embedding(data, label, title, display_classes=[0, 1, 2, 3, 4], custom_colors=None, dispersion=0.02):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    ax = plt.subplot(111)

    if custom_colors is None:
        custom_colors = plt.cm.Set1(np.arange(10) / 10)

    custom_cmap = ListedColormap(custom_colors)

    for i in range(data.shape[0]):
        if label[i] in display_classes:
            # 添加小的随机偏移
            x_disp = np.random.uniform(-dispersion, dispersion)
            y_disp = np.random.uniform(-dispersion, dispersion)
            plt.scatter(data[i, 0] + x_disp, data[i, 1] + y_disp, color=custom_cmap(label[i]),
                        s=10)

    plt.xticks()
    plt.yticks()
    plt.title(title, fontsize=14)
    return fig

def main():
    data, label, n_samples, n_features = get_data()
    print('Starting compute t-SNE Embedding...')
    # ts = TSNE(n_components=2, init='pca', random_state=0, perplexity=10, learning_rate=500, n_iter=1000)
    ts = TSNE(n_components=2, init='pca', random_state=0)
    result = ts.fit_transform(data)

    # 自定义颜色列表，使用类似 #008cff 的颜色设置
    custom_colors = ['#d3d55f', '#9ac5fa', '#f3b073', '#32731e', '#ce6f91']

    # 显示所有五个类别的图像，添加点的分散度
    fig = plot_embedding(result, label, 't-SNE Visualization with All Samples', display_classes=[0, 1, 2, 3, 4], custom_colors=custom_colors, dispersion=0.02)

    # 保存 t-SNE 可视化图形为图片文件
    plt.savefig('t-SNE_visualization.png')
    plt.show()

if __name__ == '__main__':
    main()
