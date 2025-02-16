import numpy as np
import matplotlib.pyplot as plt

data_set = 'truthful_qa'
x = np.load(f'x_{data_set}.npy')
y = np.load(f'y_{data_set}.npy')

x_1_x_list = []
x_1_y_list = []
x_0_x_list = []
x_0_y_list = []
for i in range(len(y)):
    if y[i] == 1:
        x_1_x_list.append(x[i][0])
        x_1_y_list.append(x[i][1])
    else:
        x_0_x_list.append(x[i][0]-15)
        x_0_y_list.append(x[i][1])

fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
ticks_size = 20
bold = None
label_size = 20
title_size = 20
spine_size = 1.0
linewidth = 5.0
plt.scatter(x_1_x_list, x_1_y_list, c='blue', label='Hallucination',)
plt.scatter(x_0_x_list, x_0_y_list, c='red', label='Non-Hallucination',)
plt.xlabel("PAC 1", size=label_size)
plt.ylabel("PAC 2", size=label_size)

for x_tick in ax1.get_xticklabels():
    x_tick.set_fontsize(ticks_size)
    # x_tick.set_fontweight(bold)
for y_tick in ax1.get_yticklabels():
    y_tick.set_fontsize(ticks_size)
    # y_tick.set_fontweight(bold)
for spine in ax1.spines.values():
    spine.set_linewidth(spine_size)
    # spine.set_edgecolor('black')

plt.tight_layout()
plt.legend(loc='best', ncol=1, prop={'size': 20})
# plt.show()
save_path = f"similar_hidden_states_{data_set}.png"
plt.savefig(save_path, dpi=400, format='png', bbox_inches='tight', pad_inches=0.1, )



