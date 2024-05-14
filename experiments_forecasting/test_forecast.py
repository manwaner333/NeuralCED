import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
import torchsde
# np.random.seed(42)
# torch.manual_seed(42)
# import mujoco_sde
# mujoco_sde.main(h_channels=16, hh_channels=16, layers=2,  lr=0.001, method="euler",
#                 missing_rate=0.0, time_seq=50, y_seq=10, intensity=False, epoch=500, step_mode='valloss', model="naivesde")


# import torch
# import torchcde
#
# # 示例数据：时间点和对应的数据点
# times = torch.linspace(0, 2, 6)  # 时间点从0到2，总共5个数据点
# data_points = torch.tensor([[2.0], [1.0], [3.0], [4.0], [4.3], [4.6]])
#
# # 增加一个批次维度
# # data_points = data_points.unsqueeze(0)
#
#
#
# coeff = torchcde.natural_cubic_coeffs(data_points, times)
#
#
# # 使用 CubicSpline 插值
# cubic_spline = torchcde.CubicSpline(coeff, times)
# # 在时间点 1.5 处计算插值结果
# interpolated_point = cubic_spline.evaluate(torch.tensor([1.5]))
#
# print("插值结果在时间点 1.5 处的值：", interpolated_point)



def plot(ts, samples, xlabel, ylabel, title=''):
    ts = ts.cpu()
    samples = samples.squeeze().t().cpu()
    plt.figure()
    for i, sample in enumerate(samples):
        plt.plot(ts, sample, marker='x', label=f'sample {i}')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

class SDE(nn.Module):

    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(0.1), requires_grad=True)  # Scalar parameter.
        self.noise_type = "diagonal"
        self.sde_type = "ito"

    def f(self, t, y):
        return torch.sin(t) + self.theta * y

    def g(self, t, y):
        return 0.3 * torch.sigmoid(torch.cos(t) * torch.exp(-y))

batch_size, state_size, t_size = 3, 1, 100
sde = SDE()
ts = torch.linspace(0, 1, t_size)
y0 = torch.zeros((batch_size, 1)).fill_(0.1)  # (batch_size, state_size).
bm = torchsde.BrownianInterval(t0=0.0, t1=1.0, size=(batch_size, state_size))

with torch.no_grad():
    ys = torchsde.sdeint(sde, y0, ts, method='milstein', bm=bm)
plot(ts, ys, xlabel='$t$', ylabel='$Y_t$', title='Solve SDE')

with torch.no_grad():
    ys = torchsde.sdeint(sde, y0, ts, method='milstein', bm=bm)
plot(ts, ys, xlabel='$t$', ylabel='$Y_t$',
     title='Solve SDE again (samples should be same as before)')

# Use a new BM sample, we expect different sample paths.
bm = torchsde.BrownianInterval(t0=0.0, t1=1.0, size=(batch_size, state_size))
with torch.no_grad():
    ys = torchsde.sdeint(sde, y0, ts, method='milstein', bm=bm)
plot(ts, ys, xlabel='$t$', ylabel='$Y_t$',
     title='Solve SDE (expect different sample paths)')





# 创建一个均值为 0，标准差为 1 的正态分布对象
normal_dist = torch.distributions.Normal(0, 1)

# 从分布中抽取一个样本
sample = normal_dist.sample()
print(sample)

# 计算概率密度
pdf_value = normal_dist.log_prob(torch.tensor(0.))
print(pdf_value.exp())


# 创建数据张量
data = torch.tensor([1, 2, 3, 4], dtype=torch.float32)

# 创建掩码张量
mask = torch.tensor([True, False, True, True])

# 使用掩码选择数据
masked_data = torch.masked_select(data, mask)

# 计算掩码部分的平均值
average = masked_data.mean()
print(average)
