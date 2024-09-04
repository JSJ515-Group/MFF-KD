# import pandas as pd
# # import seaborn as sns
# # import matplotlib.pyplot as plt
# # # 设置字体为Times New Roman
# # plt.rcParams['font.family'] = 'Times New Roman'
# #
# # # 读取保存ap值的Excel表格数据
# # file_path = 'E:/SCAM_yolo/distiil/model2.xls'
# # df = pd.read_excel(file_path)
# # # 假设 df 是你的原始数据框
# # # df_long = df.melt(id_vars='epoch', var_name='β', value_name='AP50')
# # df_long = df.melt(id_vars='epoch', var_name='β', value_name='AP50')
# # df_long['AP50'] = df_long['AP50'] * 100  # 转换为百分比
# #
# # # 创建一个新的图表
# # plt.figure(figsize=(12, 8))
# #
# # # 使用不同的颜色绘制每一列的折线
# # sns.lineplot(data=df_long, x='epoch', y='AP50', hue='β', palette='tab10')
# #
# # # 添加AP50 = 50的分界线及其文本
# # plt.axhline(y=50, color='purple', linestyle='--', linewidth=1.0)  # 修改分界线颜色为紫色，并设置宽度为1.0
# # plt.text(df_long['epoch'].max() * 0.95, 46, 'AP$_{50}$ = 50.0',
# #          color='#704f94', ha='right', va='center', fontsize=14, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
# #
# # # 设置x轴的刻度标签，每50个epoch显示一次
# # epoch_range = df_long['epoch'].unique()
# # xticks = range(0, max(epoch_range) + 50, 50)
# # plt.xticks(ticks=xticks, labels=[str(tick) for tick in xticks])
# #
# # # 添加标题和调整标签
# # plt.title('AP$_{50}$ Values for Different α and β Across Epochs', fontsize=18)
# # plt.ylabel('AP$_{50}$ (%)', fontsize=14)
# # plt.xlabel('Epoch', fontsize=14)
# #
# # # 调整图例字体大小
# # plt.legend(fontsize=12)
# #
# # # 调整布局以避免标签被遮挡
# # plt.tight_layout()
# #
# # # 保存图像
# # plt.savefig('AP50_Values_cha2.png', dpi=300, bbox_inches='tight')
# #
# # # 显示图表
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
# # 将数据转换为长格式
# df_long = df.melt(id_vars='epoch', var_name='β', value_name='AP50')
# df_long['AP50'] = df_long['AP50'] * 100  # 转换为百分比
#
# # 创建一个新的图表
# plt.figure(figsize=(12, 8))
#
# # 使用不同的颜色绘制每一列的折线
# sns.lineplot(data=df_long, x='epoch', y='AP50', hue='β', palette='tab10')
#
# # 添加AP50 = 50的分界线及其文本
# plt.axhline(y=50, color='purple', linestyle='--', linewidth=1.0)  # 修改分界线颜色为紫色，并设置宽度为1.0
# plt.text(df_long['epoch'].max() * 0.95, 46, 'AP$_{50}$ = 50.0',
#          color='#704f94', ha='right', va='center', fontsize=14, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
#
# # 设置x轴的范围从100到200
# plt.xlim(100, 200)
#
# # 设置x轴的刻度标签，每10个epoch显示一次
# epoch_range = range(100, 201, 10)
# plt.xticks(ticks=epoch_range, labels=[str(tick) for tick in epoch_range])
#
# # 设置y轴的范围从45到52
# plt.ylim(45, 52)
#
# # 设置y轴的刻度标签，每0.5显示一次
# plt.yticks(ticks=range(45, 53, 1), labels=[str(tick) for tick in range(45, 53, 1)])
#
# # 添加标题和调整标签
# plt.title('AP$_{50}$ Values for Different α and β Across Epochs', fontsize=18)
# plt.ylabel('AP$_{50}$ (%)', fontsize=14)
# plt.xlabel('Epoch', fontsize=14)
#
# # 调整图例字体大小
# plt.legend(fontsize=12)
#
# # 调整布局以避免标签被遮挡
# plt.tight_layout()
#
# # 保存图像
# plt.savefig('AP50_Values_cha2.png', dpi=300, bbox_inches='tight')
#
# # 显示图表
# plt.show()




#  画100epoch开始的
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # 设置字体为Times New Roman
# plt.rcParams['font.family'] = 'Times New Roman'
#
# # 读取保存AP值的Excel表格数据
# file_path = 'E:/SCAM_yolo/distiil/model2 - 副本.xls'
# df = pd.read_excel(file_path)
#
# # 将数据转换为长格式
# df_long = df.melt(id_vars='epoch', var_name='α', value_name='AP50')
# df_long['AP50'] = df_long['AP50'] * 100  # 转换为百分比
#
# # 创建一个新的图表
# plt.figure(figsize=(12, 8))
#
# # 使用不同的颜色绘制每一列的折线
# palette = sns.color_palette('tab10')
# sns.lineplot(data=df_long, x='epoch', y='AP50', hue='α', palette=palette)
#
# # 设置x轴的范围从95到205
# plt.xlim(95, 205)
#
# # 设置x轴的刻度标签，每10个epoch显示一次
# epoch_range = range(100, 201, 10)
# plt.xticks(ticks=epoch_range, labels=[str(tick) for tick in epoch_range])
#
# # 设置y轴的范围从48到52，并添加边距
# plt.ylim(47.5, 52.5)
#
# # 设置y轴的刻度标签，每0.5显示一次
# plt.yticks(ticks=[48, 48.5, 49, 49.5, 50, 50.5, 51, 51.5, 52], labels=[str(tick) for tick in [48, 48.5, 49, 49.5, 50, 50.5, 51, 51.5, 52]])
#
# # 找到所有线条中最高的AP50值
# max_row = df_long.loc[df_long['AP50'].idxmax()]
# max_alpha = max_row['α']
# max_color = palette[df_long['α'].unique().tolist().index(max_alpha)]
#
# # 四舍五入到一位小数
# max_ap50_rounded = round(max_row['AP50'], 1)
#
# # 在最高点添加AP50的值标注，并使用对应线条的颜色
# plt.text(max_row['epoch'], max_ap50_rounded + 0.2, f"AP$_{{50}}$={max_ap50_rounded:.1f}",
#          color=max_color, ha='center', va='bottom', fontsize=16, fontweight='bold')
#
# # 添加标题和调整标签
# plt.title('AP$_{50}$ Values for Different α Across Epochs with Fixed β = 0.5', fontsize=18)
# plt.ylabel('AP$_{50}$ (%)', fontsize=14)
# plt.xlabel('Epoch', fontsize=14)
#
# # 调整图例字体大小
# plt.legend(fontsize=12)
#
# # 调整布局以避免标签被遮挡
# plt.tight_layout()
#
# # 保存图像
# plt.savefig('AP50_Values_cha9.png', dpi=300, bbox_inches='tight')
#
# # 显示图表
# plt.show()








import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 设置字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 读取保存AP值的Excel表格数据
file_path = 'E:/SCAM_yolo/distiil/model2 - 副本.xls'
df = pd.read_excel(file_path)

# 转换为长格式
df_long = df.melt(id_vars='epoch', var_name='β', value_name='AP50')
df_long['AP50'] = df_long['AP50'] * 100  # 转换为百分比

# 创建一个新的图表
plt.figure(figsize=(12, 8))

# 使用不同的颜色绘制每一列的折线
sns.lineplot(data=df_long, x='epoch', y='AP50', hue='β', palette='tab10')

# 启用x轴方向和y轴方向的网格线（实线）
plt.grid(axis='both', linestyle='-', linewidth=0.7)

# 设置x轴的刻度标签，每50个epoch显示一次
epoch_range = df_long['epoch'].unique()
xticks = range(0, max(epoch_range) + 50, 50)
plt.xticks(ticks=xticks, labels=[str(tick) for tick in xticks])

# 添加标题和调整标签
plt.title('AP$_{50}$ Values for Different α Across Epochs with Fixed β = 0.5', fontsize=18)
plt.ylabel('AP$_{50}$ (%)', fontsize=14)
plt.xlabel('Epoch', fontsize=14)

# 调整图例字体大小
plt.legend(fontsize=12)

# 调整布局以避免标签被遮挡
plt.tight_layout()

# 保存图像
plt.savefig('b3AP50_Values_Combined.png', dpi=300, bbox_inches='tight')

# 显示图表
plt.show()

