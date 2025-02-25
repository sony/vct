import torch
import torch.nn as nn

class VarianceExplodingKernel(nn.Module):
    def __init__(self, sigma_min, sigma_max, sigma_data):
        super(VarianceExplodingKernel, self).__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

    def forward(self, x, t, noise):
        return x + t * noise

    def get_scaling_factors_bc(self, sigma):
        c_in = 1 / torch.sqrt(self.sigma_data ** 2 + sigma ** 2)
        c_skip = self.sigma_data ** 2 / ((sigma - self.sigma_min) ** 2 + self.sigma_data ** 2)
        c_out = (sigma - self.sigma_min) * self.sigma_data * c_in
        return c_in, c_skip, c_out

    def get_scaling_factors(self, sigma):
        c_in = 1 / torch.sqrt(self.sigma_data ** 2 + sigma ** 2)
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data * c_in
        return c_in, c_skip, c_out