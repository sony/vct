import math

import torch
import torch.nn as nn

from utils.optimal_transport import OTPlanSampler


class ConsistencyModelBase(nn.Module):
    def __init__(self,
                 net,
                 total_training_steps,
                 kernel,
                 coupling='independent',
                 kl_loss_scale=0.,
                 encoder=None,
                 sigma_min=0.002,
                 sigma_max=80,
                 p_mean=-1.1,
                 p_std=2.0,
                 sigma_data=0.5,
                 label_dim=0,
                 mid_t=[0.821],
                 ):

        super().__init__()

        self.net = net
        self.total_training_steps = total_training_steps
        self.kernel = kernel
        self.coupling = coupling
        self.kl_loss_scale = kl_loss_scale
        self.encoder = encoder

        self.total_training_steps = total_training_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.p_mean = p_mean
        self.p_std = p_std
        self.sigma_data = sigma_data
        self.label_dim = label_dim
        self.mid_t = mid_t

        if self.coupling == 'ot':
            self.ot_sampler = OTPlanSampler(method="exact")

    def _append_dims(self, x, target_dims):
        """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
        dims_to_append = target_dims - x.ndim
        if dims_to_append < 0:
            raise ValueError(
                f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
            )
        return x[(...,) + (None,) * dims_to_append]

    def coupling_fn(self, x, noise, class_labels):
        if self.coupling == 'ot':
            x, noise, class_labels, _ = self.ot_sampler.sample_plan_with_labels(x, noise * self.sigma_max, y0=class_labels, replace=False)
            noise = noise / self.sigma_max
            return x, noise, class_labels, None
        elif self.coupling == 'vae':
            mu, std = self.posterior_precond(x, class_labels)
            noise = noise * std + mu
            posterior = torch.distributions.Normal(mu, std + 1e-8)
            return x, noise, class_labels, posterior
        else:
            raise NotImplementedError

    def precond(self, x, sigma, class_labels, **model_kwargs):
        x = x.to(torch.float32)
        sigma = self._append_dims(sigma, x.ndim).to(torch.float32)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim],
                                                                    device=x.device) if class_labels is None else class_labels.to(
            torch.float32).reshape(-1, self.label_dim)
        c_in, c_skip, c_out = self.kernel.get_scaling_factors_bc(sigma)
        c_noise = sigma.log() / 4
        F_x = self.net((c_in * x), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def posterior_precond(self, x, class_labels, **model_kwargs):
        x = x.to(torch.float32)
        # here we always pass sigma as 0
        sigma = self._append_dims(torch.zeros(x.shape[0]), x.ndim).to(x).to(torch.float32)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim],
                                                                    device=x.device) if class_labels is None else class_labels.to(
            torch.float32).reshape(-1, self.label_dim)
        c_in = 1 / self.sigma_data
        # we still pass the time conditioning but it is always zero. could be removed to improve efficiency
        out = self.encoder((c_in * x), sigma.flatten(), class_labels=class_labels, **model_kwargs)
        mu = out[:, :x.shape[1]]
        std = out[:, x.shape[1]:]
        std = nn.functional.softplus(std)
        return mu.to(torch.float32), std.to(torch.float32)

    @torch.no_grad()
    def sample(self, sample_shape, n_iters, device, class_labels=None, mid_t=[]):
        z = torch.randn(sample_shape).to(device)

        if not mid_t:
            mid_t = [] if self.mid_t is None else self.mid_t
        t_steps = torch.tensor([self.sigma_max] + list(mid_t), dtype=torch.float64, device=device)[:n_iters]
        # Sampling steps
        x = z.to(torch.float64) * t_steps[0]
        x = self.precond(x, t_steps[0], class_labels).to(torch.float64)
        for i in range(1, n_iters):
            noise = torch.randn_like(x)
            if self.coupling != 'independent':
                x, noise, class_labels, _ = self.coupling_fn(x, noise, class_labels)
            x = self.kernel.forward(x, t_steps[i], noise)
            x = self.precond(x, t_steps[i], class_labels).to(torch.float64)
        return x
