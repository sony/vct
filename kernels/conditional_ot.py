import torch
import torch.nn as nn

class ConditionalOTKernel(nn.Module):
    def __init__(self, sigma_min, sigma_max, sigma_data):
        super(ConditionalOTKernel, self).__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

    def forward(self, x, t, noise):
        return x * (1 - t / self.sigma_max) + t * noise

    def get_scaling_factors_bc(self, sigma):
        scaling_term = (1 - sigma / self.sigma_max)
        scaling_term_bc = (1 - (sigma - self.sigma_min) / (self.sigma_max - self.sigma_min))
        c_in = 1 / torch.sqrt(self.sigma_data ** 2 * scaling_term ** 2 + sigma ** 2)
        c_skip = (self.sigma_data ** 2 * scaling_term_bc) / (
                    (sigma - self.sigma_min) ** 2 + self.sigma_data ** 2 * scaling_term_bc ** 2)
        c_out = (sigma - self.sigma_min) * self.sigma_data * c_in
        return c_in, c_skip, c_out

    def get_scaling_factors(self, sigma):
        scaling_term = (1 - sigma / self.sigma_max)
        c_in = 1 / torch.sqrt(self.sigma_data ** 2 * scaling_term ** 2 + sigma ** 2)
        c_skip = (self.sigma_data ** 2 * scaling_term) / (
                    sigma ** 2 + self.sigma_data ** 2 * scaling_term ** 2)
        c_out = sigma * self.sigma_data * c_in
        return c_in, c_skip, c_out