import numpy as np
import matplotlib.pyplot as plt

# Data settings
categories = ["Empty", "Agree", "Anti", "Hypo"] * 6  # 6 models, each with 4 sub-results
models = ["LLaVA-7B", "LLaVA-13B", "InternVL-13B", "InternVL-34B", "Qwen-VL", "Qwen-Audio"]

origin_results = [32.6, 26.8, 42.8, 36.6, 23.8, 14.4, 26.8, 21.2, 18.8, 15.7, 18.8, 18.4,
                  14.6, 15.8, 16.0, 16.6, 10.8, 9.8, 16.2, 12.6, 18.6, 17.4, 19.2, 16.0]
con_results = [33.2, 48.6, 70.3, 76.5, 62.8, 78.8, 82.6, 86.6, 34.6, 37.2, 40.6, 43.4,
               36.6, 41.4, 48.0, 51.2, 60.2, 64.8, 72.4, 74.8, 64.6, 70.2, 72.6, 77.6]

# Generate angles (clockwise)
num_vars = len(categories)
angles = np.linspace(0, -2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # Close the radar chart

# Close the data loops
origin_results += origin_results[:1]
con_results += con_results[:1]

# Update categories to include model names
categories_with_models_wrapped = [
    f"{model}\n({category})"
    for model in models
    for category in ["Empty", "Agree", "Anti", "Hypo"]
]

# Create the radar chart
fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))

# Plot 'Text Instruction' data
ax.plot(angles, origin_results, label="Text Instruction", linewidth=2, color='#1f77b4')
ax.fill(angles, origin_results, alpha=0.25, color='#aec7e8')

# Plot 'Con Instruction' data
ax.plot(angles, con_results, label="Con Instruction", linewidth=2, color='#ff7f0e')
ax.fill(angles, con_results, alpha=0.25, color='#ffbb78')

# Configure categories and ticks
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories_with_models_wrapped, fontsize=15, fontweight='bold')

# Configure radial ticks
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_yticklabels(['0', '25', '50', '75', '100'], fontsize=14, color='gray')

# Set background and remove outer circle
ax.set_facecolor('#f0f0f0')
ax.spines['polar'].set_visible(False)

# Add legend
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.12), fontsize=18, frameon=True)

# Adjust layout to minimize whitespace
plt.subplots_adjust(top=0.9, bottom=0.1)

plt.show()



# import numpy as np
# import matplotlib.pyplot as plt
#
# # 数据设置
# categories = ["Empty", "Agree", "Anti", "Hypo"] * 6  # 6个模型，每个模型4个子结果
# models = ["LLaVA-7B", "LLaVA-13B", "InternVL-13B", "InternVL-34B", "Qwen-VL", "Qwen-Audio"]
#
#
# origin_results = [32.6, 26.8, 42.8, 36.6, 23.8, 14.4, 26.8, 21.2, 18.8, 15.7, 18.8, 18.4,
#                   14.6, 15.8, 16.0, 16.6, 10.8, 9.8, 16.2, 12.6, 18.6, 17.4, 19.2, 16.0]
# con_results = [33.2, 48.6, 70.3, 76.5, 62.8, 78.8, 82.6, 86.6, 34.6, 37.2, 40.6, 43.4,
#                36.6, 41.4, 48.0, 51.2, 60.2, 64.8, 72.4, 74.8, 64.6, 70.2, 72.6, 77.6, ]
# fontsize = 16
# # 创建角度
# num_vars = len(categories)
# angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
# angles += angles[:1]  # 闭合雷达图
#
# # 闭合数据环
# origin_results = np.append(origin_results, origin_results[0])
# con_results = np.append(con_results, con_results[0])
#
# # 绘制雷达图
# # 更新类别标签：将模型名称和类别结合
# categories_with_models_wrapped = [
#     f"{model}\n({category})"
#     for model in models
#     for category in ["Empty", "Agree", "Anti", "Hypo"]
# ]
#
# # 绘制雷达图，更新换行显示的类别标签
# fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
#
# # 绘制 Origin Instruction 的数据
# # ax.plot(angles, origin_results, label="Origin Instruction", linewidth=2)
# # ax.fill(angles, origin_results, alpha=0.25, color='skyblue')
# #
# # # 绘制 Con Instruction 的数据
# # ax.plot(angles, con_results, label="Con Instruction", linewidth=2)
# # ax.fill(angles, con_results, alpha=0.25, color='orange')
# ax.plot(angles, origin_results, label="Text Instruction", linewidth=2, color='#1f77b4')
# ax.fill(angles, origin_results, alpha=0.25, color='#aec7e8')
#
# # 绘制 Con Instruction 的数据
# ax.plot(angles, con_results, label="Con Instruction", linewidth=2, color='#ff7f0e')
# ax.fill(angles, con_results, alpha=0.25, color='#ffbb78')
# # 设置类别
# ax.set_xticks(angles[:-1])
# ax.set_xticklabels(categories_with_models_wrapped, fontsize=15, fontweight='bold')
# # ax.set_yticklabels([])  # 不显示径向标签
# ax.set_yticklabels(['0', '25', '50', '75', '100'], fontsize=14, color='gray')
#
# ax.set_facecolor('#f0f0f0')  # 设置雷达图内部背景色
#
# # 移除最外面的圆圈线条
# ax.spines['polar'].set_visible(False)
#
#
# # 添加图例，调整位置
# ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.12), fontsize=18, frameon=True)
#
# # 调整布局以减少上下空白
# plt.subplots_adjust(top=0.9, bottom=0.1)
#
# plt.show()

# 示例数据：为每个设定和模型生成随机值
# np.random.seed(42)  # 设置随机种子以保证结果可复现
# origin_results = np.random.rand(24) * 100
# con_results = np.random.rand(24) * 100

# origin_results = [15.6, 9.4, 26.6, 18.8,   2.7, 3.7, 22.3, 11.7,   8.5,8.7,14.7, 10.8,
#                   4.3, 4.5, 10.8, 7.9, 1.3, 1.2, 5.6, 3.7, 1.2, 1.3, 15.1, 11.3 ]
# con_results = [31.2, 48.6, 65.5, 74.5, 62.8, 76.0, 78.4, 86.8, 30.9, 42.5, 49.7, 53.5,
#                42.7, 45.5, 53.3, 60.6, 59.2, 62.5, 73.4, 76.3, 61.4, 72.3, 73.5, 76.2]
