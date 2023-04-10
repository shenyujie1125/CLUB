import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from mi_estimators import *


class GaussianSampler(nn.Module):
    def __init__(self, dim, para_list=None):
        super(GaussianSampler, self).__init__()
        self.dim = dim
        if para_list is None:
            para_list = [0.55] * dim
        self.p_theta_ = torch.nn.Parameter(torch.tensor(para_list, requires_grad=True))

    def get_trans_mat(self):
        p_theta = self.p_theta_.cuda().unsqueeze(-1)
        # p_theta = torch.softmax(p_theta, dim = 0)

        trans_row1 = torch.cat((torch.sin(p_theta), torch.cos(p_theta)), dim=-1).unsqueeze(-1)
        trans_row2 = torch.cat((torch.cos(p_theta), torch.sin(p_theta)), dim=-1).unsqueeze(-1)  # [dim, 2,1]
        return torch.cat((trans_row1, trans_row2), dim=-1)  # [dim,2,2]

    def gen_samples(self, num_sample, cuda=True):
        noise = torch.randn(self.dim, num_sample, 2).cuda()
        trans_mat = self.get_trans_mat()
        samples = torch.bmm(noise, trans_mat).transpose(0, 1)  # [dim, nsample, 2]
        if not cuda:
            samples = samples.cpu().detach().numpy()
        return samples[:, :, 0], samples[:, :, 1]

    def get_covariance(self):
        p_theta = self.p_theta_.cuda()
        return (2. * torch.sin(p_theta) * torch.cos(p_theta))

    def get_MI(self):
        rho = self.get_covariance()
        return -1. / 2. * torch.log(1 - rho ** 2).sum().item()
        # return -self.dim /2.*torch.log(1-rho**2 / 2).sum().item()

lr = 1e-4
batch_size = 100
num_iter = 5000
sample_dim = 2
hidden_size = 5
estimator_name = "CLUB"

sampler = GaussianSampler(sample_dim).cuda()
#print("The corvariance of Gaussian is {}".format(sampler.get_covariance().cpu().detach().numpy()))
x_sample, y_sample = sampler.gen_samples(1000, cuda = False)
plt.scatter(x_sample, y_sample)
plt.show()
mi_estimator = eval(estimator_name)(sample_dim, sample_dim, hidden_size).cuda()

sampler_optimizer = torch.optim.Adam(sampler.parameters(), lr = lr)
mi_optimizer = torch.optim.Adam(mi_estimator.parameters(), lr = lr)

mi_true_values = []
mi_est_values = []
min_mi = 100.

for i in range(num_iter):
    sampler.train()
    mi_estimator.eval()
    x_samples, y_samples = sampler.gen_samples(batch_size)
    sampler_loss = mi_estimator(x_samples, y_samples)
    sampler_optimizer.zero_grad()
    sampler_loss.backward() # retain_graph=True)
    sampler_optimizer.step()

    # sampler是可以训练的，而mi_estimator是不可以被训练的，先forward前向传播，然后计算损失，
    # 用该损失进行反向传播更新参数，此时的更新参数为采样模型参数，不是CLUB模型内部的参数
    for j in range(5):
        mi_estimator.train()
        x_samples, y_samples = sampler.gen_samples(batch_size)
        mi_loss = mi_estimator.learning_loss(x_samples, y_samples)
        mi_optimizer.zero_grad()
        mi_loss.backward()
        mi_optimizer.step()
        # mi_estimator设置为可以训练的，先进行learning_loss的前向传播，用learning_loss的损失进行反向传播
        # 更新参数，此时更新的参数为CLUB模型内部的参数，而不更新采样模型的参数
    mi_true_values.append(sampler.get_MI())
    mi_est_values.append(mi_estimator(x_samples, y_samples).item())
    if i % 100 ==0:
        print("step {}, true MI value {}".format(i, sampler.get_MI()))

plt.plot(np.arange(len(mi_est_values)), mi_est_values, label=estimator_name + " est")
plt.plot(np.arange(len(mi_true_values)), mi_true_values, label="True MI value")
plt.legend()
plt.show()

x_sample, y_sample = sampler.gen_samples(1000, cuda=False)
plt.scatter(x_sample, y_sample)
plt.show()

#这篇文章是互信息上限估计的文章，该文章可以为条件概率已知和条件概率未知两种情况做互信息上线估计，如何运用可以直接参考对应的pytorch代码，
# 原理部分大致可以看懂，本文将互信息上线估计运用在了两个地方，一个是变分编码器，用在信息瓶颈领域，在变分编码器的训练中，我们希望输入数据于
# 中间的隐藏状态之间的互信息变小，而中间隐藏状态与输出的互信息变大，这就要求分别进行互信息上限估计以及互信息下限估计，分别用于最小化输入与
# 中间状态的互信息和最大化输出与中间状态的互信息，发现使用club方法作为互信息上限的估计器，对于基于变分编码器的minist手写数字体识别任务，
# 可以提升识别准确率，说明在信息瓶颈领域，该互信息上限估计，能够提升模型得泛化性能，也就是在测试数据集上面得性能，关于信息瓶颈可以
# 参考https://zhuanlan.zhihu.com/p/51463628。 在域适应中，也就是对于几个手写数字体不一样的数据集的迁移性能上，将其损失函数分为
# 三个部分，第一部分就是对源域的手写体数字进行识别，为第一个内容编码器的任务：提取内容特征用于进行手写数字体识别，而在另一个域编码器中，
# 提取源域与目标域的特征信息，然后用一个判别器进行判别，域编码器中对源域与目标域的数据进行域特征编码，随着对模型的训练，可以不断使得域
# 编码器提取出来的信息具有更强得分辨效果，也就是提取出来得域信息差异性更大，提取差异化信息，为了使得内容编码器能够提取更多的关于两个域
# 的共同信息，需要在域编码器与内容编码器之间增加约束，也就相当于从总的信息中减去域编码器提取的差异化信息，其实也就是相当于两个编码器
# 分别提取共同特征以及特有特征，为了使得特有特征更加突出，加入判别器，为了使得共同特征更加突出，加入互信息约束，相当于从源域信息中
# 减去源域特有信息，然后从目标域信息中减去目标域特有信息，最终使得模型在两个数据集上同样具有识别效果，因为最终提取的用于识别的特
# 征来源于两个数据集的共同特征部分。
