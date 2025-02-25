import math

import torch
from models.cm_base import ConsistencyModelBase
from lightning.pytorch.utilities.seed import isolate_rng

class ImprovedConsistencyTraining(ConsistencyModelBase):
    def __init__(self,
                 scale_mode='exp',
                 start_scales=10,
                 end_scales=1280,  # s1
                 rho=7.,
                 noise_schedule='lognormal',  # 'uniform_time', 'lognormal'
                 loss_mode='huber',
                 **cm_kwargs
                 ):

        super().__init__(**cm_kwargs)

        self.scale_mode = scale_mode
        self.start_scales = start_scales
        self.end_scales = end_scales
        self.rho = rho

        self.noise_schedule = noise_schedule

        self.loss_mode = loss_mode
    def pseudo_huber_loss(self, input, target):
        c = 0.00054 * math.sqrt(input[0].numel())
        return torch.sqrt((input - target) ** 2 + c ** 2) - c

    def _step_schedule(self, step):
        if self.scale_mode == 'baseline':  # like in CM
            num_timesteps = (self.end_scales + 1) ** 2 - self.start_scales ** 2
            num_timesteps = step * num_timesteps / self.total_training_steps
            num_timesteps = math.ceil(math.sqrt(num_timesteps + self.start_scales ** 2) - 1) + 1

        elif self.scale_mode == 'exp':
            k_prime = math.floor(
                self.total_training_steps
                / (math.log2(math.floor(self.end_scales / self.start_scales)) + 1)
            )
            num_timesteps = self.start_scales * math.pow(2, math.floor(step / k_prime))
            num_timesteps = min(num_timesteps, self.end_scales) + 1

        elif self.scale_mode == 'none':
            num_timesteps = self.end_scales + 1

        else:
            raise NotImplementedError

        return int(num_timesteps)

    def _loss_fn(self, prediction, target):
        if self.loss_mode == 'l2':
            return (prediction - target) ** 2
        elif self.loss_mode == 'huber':
            return self.pseudo_huber_loss(prediction, target)
        else:
            raise NotImplementedError

    def _get_sigmas_karras(
            self,
            num_timesteps: int,
            device: torch.device = None,
    ):
        rho_inv = 1.0 / self.rho
        # Clamp steps to 1 so that we don't get nans
        steps = torch.arange(num_timesteps, device=device, dtype=torch.float32) / max(num_timesteps - 1, 1)
        sigmas = self.sigma_min ** rho_inv + steps * (
                self.sigma_max ** rho_inv - self.sigma_min ** rho_inv
        )
        sigmas = sigmas ** self.rho

        return sigmas.to(torch.float32)

    def _lognormal_timestep_distribution(self, num_samples, sigmas):
        pdf = torch.erf((torch.log(sigmas[1:]) - self.p_mean) / (self.p_std * math.sqrt(2))) - torch.erf(
            (torch.log(sigmas[:-1]) - self.p_mean) / (self.p_std * math.sqrt(2))
        )

        indices = torch.multinomial(pdf, num_samples, replacement=True)

        return indices

    def _get_sigmas(self, num_timesteps, device):
        sigmas = self._get_sigmas_karras(num_timesteps, device=device)
        return sigmas.to(torch.float32)

    def _get_indices(self, num_timesteps, sigmas, device, batch_size):
        if self.noise_schedule == 'uniform_time':
            indices = torch.randint(
                0, num_timesteps - 1, (batch_size,), device=device
            )
        elif self.noise_schedule == 'lognormal':
            indices = self._lognormal_timestep_distribution(batch_size, sigmas)
        else:
            raise ValueError(f'Unknown noise schedule')

        return indices

    def _get_loss_weights(self, sigmas, indices):
        return (1 / (sigmas[1:] - sigmas[:-1]))[indices]

    def loss(self, x, step, labels=None):

        log_dict = {}

        dims = x.ndim  # keeps track of data dimensionality to work with both images and tabular
        device = x.device
        batch_size = x.shape[0]

        # Augmentation if needed
        num_timesteps = self._step_schedule(step)

        sigmas = self._get_sigmas(num_timesteps, device=device)

        indices = self._get_indices(num_timesteps, sigmas, device=device, batch_size=batch_size)

        # sigma i + 1
        t = self._append_dims(sigmas[indices + 1], dims)
        r = self._append_dims(sigmas[indices], dims)

        loss_weights = self._append_dims(self._get_loss_weights(sigmas, indices), dims)

        noise = torch.randn_like(x)
        if self.coupling != 'independent':
            x, noise, labels, posterior = self.coupling_fn(x, noise, labels)

        x_t = self.kernel.forward(x, t, noise)
        x_r = self.kernel.forward(x, r, noise)
        # Shared Dropout Mask
        with isolate_rng():
            D_yt = self.precond(x_t, t, labels)

        with torch.no_grad():
            D_yr = self.precond(x_r, r, labels)

        loss = self._loss_fn(D_yt, D_yr.detach())
        log_dict['consistency_loss'] = loss.detach().view(batch_size, -1).sum(1).mean()
        loss = loss * loss_weights
        loss = loss.view(batch_size, -1).sum(1)

        if self.coupling == 'vae':
            prior = torch.distributions.Normal(torch.zeros_like(x_t), torch.ones_like(x_t))
            kl_loss = torch.distributions.kl_divergence(posterior, prior)
            log_dict['kl_divergence'] = kl_loss.detach().view(batch_size, -1).sum(1).mean()
            kl_loss_scale = (1 / (sigmas[-1] - sigmas[-2]) * self.kl_loss_scale).item()
            kl_loss = kl_loss.view(batch_size, -1).sum(1) * kl_loss_scale
            loss += kl_loss

        return loss.mean(), log_dict