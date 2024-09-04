# import pandas as pd
# import matplotlib.pyplot as plt
# import chardet  # 导入chardet库
# import matplotlib.font_manager
# # 设置字体为Times New Roman
# plt.rcParams['font.family'] = 'Times New Roman'
#
# # 在这之后继续你的绘图代码
#
#
# with open('E:/gold_yolo/train/model.xls', 'rb') as f:
#     result = chardet.detect(f.read())  # 检测文件编码格式
#
# # 读取保存ap值的Excel表格数据，无需指定编码格式
# df = pd.read_excel('E:/gold_yolo/train/model.xls')
#
# # 提取每个模型的ap50值列
# model1_ap = df['yolov5s'].tolist()
# model2_ap = df['yolov6s'].tolist()
# model3_ap = df['yolov8s'].tolist()
# model4_ap = df['GCFNet'].tolist()
#
# epochs = list(range(1, len(model1_ap) + 1))  # 横坐标为epoch数，假设从1开始
#
# # 绘制折线图
# plt.plot(epochs, model1_ap, label='yolov5s')
# plt.plot(epochs, model2_ap, label='yolov6s')
# plt.plot(epochs, model3_ap, label='yolov8s')
# plt.plot(epochs, model4_ap, label='GCFNet')
#
# plt.xlabel('Epoch')  # 横坐标标签
# plt.ylabel('AP$_{50}$')  # 纵坐标标签，使用下标形式展示50
# plt.title('AP$_{50}$ Values for Different Models')  # 图表标题，使用下标形式展示50
# plt.legend()  # 显示图例
# plt.show()


# 子图，可以考虑
# import pandas as pd
# import matplotlib.pyplot as plt
# import chardet  # 导入chardet库
# import matplotlib.font_manager
#
# # 设置字体为Times New Roman
# plt.rcParams['font.family'] = 'Times New Roman'
#
# # 检测文件编码格式
# with open('E:/SCAM_yolo/distiil/model.xls', 'rb') as f:
#     result = chardet.detect(f.read())
#
# # 读取保存ap值的Excel表格数据
# df = pd.read_excel('E:/SCAM_yolo/distiil/model.xls')
#
# # 提取每个β值的ap50值列，并将AP值乘以100
# beta_07_ap = (df['beta_0.7'] * 100).tolist()
# beta_06_ap = (df['beta_0.6'] * 100).tolist()
# beta_05_ap = (df['beta_0.5'] * 100).tolist()
# beta_04_ap = (df['beta_0.4'] * 100).tolist()
# beta_03_ap = (df['beta_0.3'] * 100).tolist()
#
# epochs = list(range(1, len(beta_07_ap) + 1))  # 横坐标为epoch数，假设从1开始
#
# # 创建子图
# fig, axs = plt.subplots(5, 1, figsize=(12, 24), sharex=True)
# fig.suptitle('AP$_{50}$ Values for Different β', fontsize=16)
#
# # 绘制每个 β 值的子图
# axs[0].plot(epochs, beta_07_ap, label='β=0.7', color='b', linewidth=2)
# axs[0].set_ylabel('β=0.7')
# axs[1].plot(epochs, beta_06_ap, label='β=0.6', color='g', linewidth=2)
# axs[1].set_ylabel('β=0.6')
# axs[2].plot(epochs, beta_05_ap, label='β=0.5', color='r', linewidth=2)
# axs[2].set_ylabel('β=0.5')
# axs[3].plot(epochs, beta_04_ap, label='β=0.4', color='c', linewidth=2)
# axs[3].set_ylabel('β=0.4')
# axs[4].plot(epochs, beta_03_ap, label='β=0.3', color='m', linewidth=2)
# axs[4].set_ylabel('β=0.3')
# axs[4].set_xlabel('Epoch')
#
# for ax in axs:
#     ax.grid(True)
#
# plt.show()
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # 设置字体为Times New Roman
# plt.rcParams['font.family'] = 'Times New Roman'
#
# # 读取保存ap值的Excel表格数据
# file_path = 'E:/SCAM_yolo/distiil/model.xls'
# df = pd.read_excel(file_path)
#
# # 将数据准备成适合绘制的格式
# df_long = df.melt(id_vars='epoch', var_name='β', value_name='AP50')
# df_long['AP50'] = df_long['AP50'] * 100  # 转换为百分比
#
# # 创建面板式图表
# g = sns.FacetGrid(df_long, col='β', col_wrap=3, height=4, sharex=True, sharey=True)
#
# # 使用不同的颜色绘制线条
# for i, β in enumerate(df_long['β'].unique()):
#     subset = df_long[df_long['β'] == β]
#     sns.lineplot(x='epoch', y='AP50', data=subset, ax=g.axes[i], color=sns.color_palette('tab10')[i])
#
# # 添加AP50 = 50的分界线
# for ax in g.axes.flat:
#     ax.axhline(y=50, color='purple', linestyle='--', linewidth=1.5)  # 修改分界线颜色为紫色
#
# # 添加标题和调整布局
# g.fig.suptitle('AP$_{50}$ Values for Different β Across Epochs', fontsize=16)
# plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局以避免标签被遮挡
#
# # 显示图表
# plt.show()








# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # 设置字体为Times New Roman
# plt.rcParams['font.family'] = 'Times New Roman'
#
# # 读取保存ap值的Excel表格数据
# file_path = 'E:/SCAM_yolo/distiil/model2.xls'
# df = pd.read_excel(file_path)
#
# # 提取列名中的 alpha 和 beta 值
# df.columns = [col if col == 'epoch' else f"{col.split(',')[0]}_{col.split(',')[1]}" for col in df.columns]
#
# # 将数据准备成适合绘制的格式
# df_long = df.melt(id_vars='epoch', var_name='alpha_beta', value_name='AP50')
# df_long['AP50'] = df_long['AP50'] * 100  # 转换为百分比
#
# # 提取 alpha 和 beta 值
# df_long['alpha'] = df_long['alpha_beta'].apply(lambda x: x.split('_')[0])
# df_long['beta'] = df_long['alpha_beta'].apply(lambda x: x.split('_')[1])
#
# # 创建面板式图表
# g = sns.FacetGrid(df_long, col='beta', col_wrap=3, height=4, sharex=True, sharey=True)
#
# # 使用不同的颜色绘制线条，并显示 alpha 和 beta 值
# for i, beta in enumerate(df_long['beta'].unique()):
#     subset = df_long[df_long['beta'] == beta]
#     sns.lineplot(x='epoch', y='AP50', data=subset, ax=g.axes[i],
#                  color=sns.color_palette('tab10')[i],
#                  label=f'alpha={subset["alpha"].iloc[0]}, beta={beta}')
#
# # 添加AP50 = 50的分界线及其文本
# for ax in g.axes.flat:
#     ax.axhline(y=50, color='purple', linestyle='--', linewidth=1.0)  # 修改分界线颜色为紫色，并设置宽度为1.0
#     # 在横线下方添加文本，颜色更改为 #704f94
#     ax.text(ax.get_xlim()[1] * 0.95, 46, 'AP$_{50}$ = 50.0%',
#             color='#704f94', ha='right', va='center', fontsize=14, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
#
# # 添加标题和调整布局
# g.fig.suptitle('AP$_{50}$ Values for Different Beta Across Epochs', fontsize=18)
#
# for ax in g.axes.flat:
#     ax.set_ylabel('AP$_{50}$ (%)', fontsize=14)  # 设置纵坐标标签的字体大小为14
#     ax.set_xlabel('Epoch', fontsize=14)  # 设置横坐标标签的字体大小为14
#
# # 调整图例字体大小
# for ax in g.axes.flat:
#     ax.legend(fontsize=12)  # 设置图例标签的字体大小为12
#
# # 删除顶部的β值标签
# for ax in g.axes.flat:
#     ax.set_title('')
#
# # 调整布局以避免标签被遮挡
# plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.2, hspace=0.4, wspace=0.3)
#
# # 使用tight_layout调整布局
# plt.tight_layout()
#
# # 保存图像
# plt.savefig('AP50_Values2.png', dpi=300, bbox_inches='tight')
#
# # 显示图表
# plt.show()
# #


# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # 设置字体为Times New Roman
# plt.rcParams['font.family'] = 'Times New Roman'
#
# # 读取保存ap值的Excel表格数据
# file_path = 'E:/SCAM_yolo/distiil/model2.xls'
# df = pd.read_excel(file_path)
#
# # 提取列名中的 alpha 和 beta 值
# df.columns = [col if col == 'epoch' else f"{col.split(',')[0]}_{col.split(',')[1]}" for col in df.columns]
#
# # 将数据准备成适合绘制的格式
# df_long = df.melt(id_vars='epoch', var_name='alpha_beta', value_name='AP50')
# df_long['AP50'] = df_long['AP50'] * 100  # 转换为百分比
#
# # 提取 alpha 和 beta 值
# df_long['alpha'] = df_long['alpha_beta'].apply(lambda x: x.split('_')[0])
# df_long['beta'] = df_long['alpha_beta'].apply(lambda x: x.split('_')[1])
#
# # 创建面板式图表
# g = sns.FacetGrid(df_long, col='beta', col_wrap=3, height=4, sharex=True, sharey=True)
#
# # 使用不同的颜色绘制线条，并显示 alpha 和 beta 值
# for i, beta in enumerate(df_long['beta'].unique()):
#     subset = df_long[df_long['beta'] == beta]
#     sns.lineplot(x='epoch', y='AP50', data=subset, ax=g.axes[i],
#                  color=sns.color_palette('tab10')[i],
#                  label=f'$\\alpha={subset["alpha"].iloc[0]}, \\beta={beta}$')
#
# # 添加AP50 = 50的分界线及其文本
# for ax in g.axes.flat:
#     ax.axhline(y=50, color='purple', linestyle='--', linewidth=1.0)  # 修改分界线颜色为紫色，并设置宽度为1.0
#     # 在横线下方添加文本，颜色更改为 #704f94
#     ax.text(ax.get_xlim()[1] * 0.95, 46, 'AP$_{50}$ = 50.0%',
#             color='#704f94', ha='right', va='center', fontsize=14, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
#
# # 添加标题和调整布局
# g.fig.suptitle('AP$_{50}$ Values for Different Beta Across Epochs', fontsize=18)
#
# for ax in g.axes.flat:
#     ax.set_ylabel('AP$_{50}$ (%)', fontsize=14)  # 设置纵坐标标签的字体大小为14
#     ax.set_xlabel('Epoch', fontsize=14)  # 设置横坐标标签的字体大小为14
#
# # 调整图例字体大小
# for ax in g.axes.flat:
#     ax.legend(fontsize=12)  # 设置图例标签的字体大小为12
#
# # 删除顶部的β值标签
# for ax in g.axes.flat:
#     ax.set_title('')
#
# # 调整布局以避免标签被遮挡
# plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.2, hspace=0.4, wspace=0.3)
#
# # 使用tight_layout调整布局
# plt.tight_layout()
#
# # 保存图像
# plt.savefig('AP50_Values2.png', dpi=300, bbox_inches='tight')
#
# # 显示图表
# plt.show()




#
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # 设置字体为Times New Roman
# plt.rcParams['font.family'] = 'Times New Roman'
#
# # 读取保存ap值的Excel表格数据
# file_path = 'E:/SCAM_yolo/distiil/model2.xls'
# df = pd.read_excel(file_path)
#
# # 将数据准备成适合绘制的格式
# df_long = df.melt(id_vars='epoch', var_name='α', value_name='AP50')
# df_long['AP50'] = df_long['AP50'] * 100  # 转换为百分比
#
# # 创建面板式图表
# g = sns.FacetGrid(df_long, col='α', col_wrap=3, height=4, sharex=True, sharey=True)
#
# # 使用不同的颜色绘制线条
# for i, α in enumerate(df_long['α'].unique()):
#     subset = df_long[df_long['α'] == α]
#     sns.lineplot(x='epoch', y='AP50', data=subset, ax=g.axes[i], color=sns.color_palette('tab10')[i], label=f'α={α}')
#
# # 添加AP50 = 50的分界线及其文本
# for ax in g.axes.flat:
#     ax.axhline(y=50, color='purple', linestyle='--', linewidth=1.0)  # 修改分界线颜色为紫色，并设置宽度为1.0
#     # 在横线下方添加文本，颜色更改为 #704f94
#     ax.text(ax.get_xlim()[1] * 0.95, 46, 'AP$_{50}$ = 50.0%',
#             color='#704f94', ha='right', va='center', fontsize=14, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
#
# # 修改每个子图的标题，添加 β=0.5 的信息
# for i, ax in enumerate(g.axes.flat):
#     ax.set_title(f'α={df_long["α"].unique()[i]} (β=0.5)', fontsize=14)
#
# # 添加标题和调整布局
# g.fig.suptitle('AP$_{50}$ Values for Different α Across Epochs', fontsize=18)
#
# for ax in g.axes.flat:
#     ax.set_ylabel('AP$_{50}$ (%)', fontsize=14)  # 设置纵坐标标签的字体大小为14
#     ax.set_xlabel('Epoch', fontsize=14)  # 设置横坐标标签的字体大小为14
#
# # 调整图例字体大小
# for ax in g.axes.flat:
#     ax.legend(fontsize=12)  # 设置图例标签的字体大小为12
#
# # 调整布局以避免标签被遮挡
# plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.2, hspace=0.4, wspace=0.3)
#
# # 使用tight_layout调整布局
# plt.tight_layout()
#
# # 保存图像
# plt.savefig('AP50_Values_with_alpha_beta.png', dpi=300, bbox_inches='tight')
#
# # 显示图表
# plt.show()



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 设置字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 读取保存ap值的Excel表格数据
file_path = 'E:/SCAM_yolo/distiil/model2.xls'
df = pd.read_excel(file_path)

# 将数据准备成适合绘制的格式
# df_long = df.melt(id_vars='epoch', var_name='β', value_name='AP50')
# df_long['AP50'] = df_long['AP50'] * 100  # 转换为百分比
#
# # 创建面板式图表
# g = sns.FacetGrid(df_long, col='β', col_wrap=3, height=4, sharex=True, sharey=True)
#
# # 使用不同的颜色绘制线条
# for i, β in enumerate(df_long['β'].unique()):
#     subset = df_long[df_long['β'] == β]
#     sns.lineplot(x='epoch', y='AP50', data=subset, ax=g.axes[i], color=sns.color_palette('tab10')[i], label=f'α=1.0,β={β}')

# # 将数据准备成适合绘制的格式
df_long = df.melt(id_vars='epoch', var_name='β', value_name='AP50')
df_long['AP50'] = df_long['AP50'] * 100  # 转换为百分比

# 创建面板式图表
g = sns.FacetGrid(df_long, col='β', col_wrap=3, height=4, sharex=True, sharey=True)

# 使用不同的颜色绘制线条
for i, β in enumerate(df_long['β'].unique()):
    subset = df_long[df_long['β'] == β]
    sns.lineplot(x='epoch', y='AP50', data=subset, ax=g.axes[i], color=sns.color_palette('tab10')[i], label=f'α={β},β=0.5')


# 添加AP50 = 50的分界线及其文本
for ax in g.axes.flat:
    ax.axhline(y=50, color='purple', linestyle='--', linewidth=1.0)  # 修改分界线颜色为紫色，并设置宽度为1.0
    # 在横线下方添加文本，颜色更改为 #704f94
    ax.text(ax.get_xlim()[1] * 0.95, 46, 'AP$_{50}$ = 50.0',
            color='#704f94', ha='right', va='center', fontsize=14, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

# 添加标题和调整布局
g.fig.suptitle('AP$_{50}$ Values for Different α and β Across Epochs', fontsize=18)

for ax in g.axes.flat:
    ax.set_ylabel('AP$_{50}$ (%)', fontsize=14)  # 设置纵坐标标签的字体大小为14
    ax.set_xlabel('Epoch', fontsize=14)  # 设置横坐标标签的字体大小为14

# 调整图例字体大小
for ax in g.axes.flat:
    ax.legend(fontsize=12)  # 设置图例标签的字体大小为12

# 删除顶部的β值标签
for ax in g.axes.flat:
    ax.set_title('')

# 调整布局以避免标签被遮挡
plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.2, hspace=0.4, wspace=0.3)

# 使用tight_layout调整布局
plt.tight_layout()

# 保存图像
plt.savefig('AP50_Values3-3.png', dpi=300, bbox_inches='tight')

# 显示图表
plt.show()


# 可用
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # 设置字体为Times New Roman
# plt.rcParams['font.family'] = 'Times New Roman'
#
# # 读取保存ap值的Excel表格数据
# file_path = 'E:/SCAM_yolo/distiil/model.xls'
# df = pd.read_excel(file_path)
#
# # 将数据准备成适合绘制的格式
# df_long = df.melt(id_vars='epoch', var_name='β', value_name='AP50')
# df_long['AP50'] = df_long['AP50'] * 100  # 转换为百分比
#
# # 创建面板式图表
# g = sns.FacetGrid(df_long, col='β', col_wrap=3, height=4, sharex=True, sharey=True)
#
# # 使用不同的颜色绘制线条
# for i, β in enumerate(df_long['β'].unique()):
#     subset = df_long[df_long['β'] == β]
#     sns.lineplot(x='epoch', y='AP50', data=subset, ax=g.axes[i], color=sns.color_palette('tab10')[i], label=f'β={β}')
#
# # 添加AP50 = 50的分界线
# for ax in g.axes.flat:
#     ax.axhline(y=50, color='purple', linestyle='--', linewidth=1.5)  # 修改分界线颜色为紫色
#
# # 添加标题和调整布局
# g.fig.suptitle('AP$_{50}$ Values for Different β Across Epochs', fontsize=16)
#
# for ax in g.axes.flat:
#     ax.set_ylabel('AP$_{50}$ (%)')  # 设置纵坐标标签为带下标的AP50
#     ax.set_xlabel('Epoch')  # 设置横坐标标签为Epoch
#     # 设置β值标签在子图内的适当位置
#     β_value = ax.get_title().split('=')[1]  # 获取β值
#     ax.text(0.5, 0.9, f'β={β_value}', horizontalalignment='center',
#             verticalalignment='center', transform=ax.transAxes,
#             fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
#
# # 删除顶部的β值标签
# for ax in g.axes.flat:
#     ax.set_title('')
#
# # 调整布局以避免标签被遮挡
# plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.35, hspace=0.4, wspace=0.4)
#
# # 手动调整每个子图的X轴标签位置
# for ax in g.axes.flat:
#     plt.setp(ax.get_xticklabels(), rotation=45, ha='right')  # 旋转X轴标签
#
# # 使用tight_layout调整布局
# plt.tight_layout()
#
# # 保存图像
# plt.savefig('AP50_Values.png', dpi=300, bbox_inches='tight')
#
# # 显示图表
# plt.show()

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # 设置字体为Times New Roman
# plt.rcParams['font.family'] = 'Times New Roman'
#
# # 读取保存ap值的Excel表格数据
# file_path = 'E:/SCAM_yolo/distiil/model.xls'
# df = pd.read_excel(file_path)
#
# # 将数据准备成适合绘制的格式
# df_long = df.melt(id_vars='epoch', var_name='β', value_name='AP50')
# df_long['AP50'] = df_long['AP50'] * 100  # 转换为百分比
#
# # 获取不同β值的数量
# unique_β = df_long['β'].unique()
# n = len(unique_β)  # 子图数量
#
# # 创建面板式图表
# g = sns.FacetGrid(df_long, col='β', col_wrap=3, height=4, sharex=True, sharey=True)
#
# # 使用不同的颜色绘制线条
# for i, β in enumerate(unique_β):
#     subset = df_long[df_long['β'] == β]
#     sns.lineplot(x='epoch', y='AP50', data=subset, ax=g.axes[i], color=sns.color_palette('tab10')[i])
#
# # 添加AP50 = 50的分界线
# for ax in g.axes.flat:
#     ax.axhline(y=50, color='purple', linestyle='--', linewidth=1.5)  # 修改分界线颜色为紫色
#
# # 调整每个子图的X轴标签和标题
# for i, ax in enumerate(g.axes.flat):
#     ax.set_xlabel('Epoch')  # 设置横坐标标签为Epoch
#     ax.set_ylabel('AP$_{50}$ (%)')  # 设置纵坐标标签为带下标的AP50
#     β_value = unique_β[i]  # 获取β值
#     ax.text(0.5, 0.9, f'β={β_value}', horizontalalignment='center',
#             verticalalignment='center', transform=ax.transAxes,
#             fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
#
# # 删除顶部的β值标签
# for ax in g.axes.flat:
#     ax.set_title('')
#
# # 添加全局标题
# g.fig.suptitle('AP$_{50}$ Values for Different β Across Epochs', fontsize=16, y=1.02)
#
# # 调整布局参数，确保子图之间有足够的间距
# plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.2, hspace=0.6, wspace=0.3)
#
# # 手动设置每个子图的X轴标签，避免标签重叠
# for ax in g.axes.flat:
#     plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
#
# # 使用tight_layout调整布局
# plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整tight_layout的rect参数避免覆盖suptitle
#
# # 保存图像
# plt.savefig('AP50_Values.png', dpi=300, bbox_inches='tight')
#
# # 显示图表
# plt.show()


















