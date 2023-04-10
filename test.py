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

    for j in range(5):
        mi_estimator.train()
        x_samples, y_samples = sampler.gen_samples(batch_size)
        mi_loss = mi_estimator.learning_loss(x_samples, y_samples)
        mi_optimizer.zero_grad()
        mi_loss.backward()
        mi_optimizer.step()

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