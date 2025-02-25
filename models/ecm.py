import torch
import contextlib
from networks.networks_edm2 import MPConv
from kernels.conditional_ot import ConditionalOTKernel
from models.cm_base import ConsistencyModelBase
from lightning.pytorch.utilities.seed import isolate_rng

@contextlib.contextmanager
def disable_forced_wn(module):
    def set_force_wn(m):
        if isinstance(m, MPConv):
            m.force_wn = False

    def reset_force_wn(m):
            if isinstance(m, MPConv):
                m.force_wn = True

    module.apply(set_force_wn)
    try:
        yield
    finally:
        module.apply(reset_force_wn)


class EasyConsistencyModel(ConsistencyModelBase):
    def __init__(self,
                 n_stages=8,
                 q=2,
                 c=0.0,
                 k=8.0,
                 b=1.0,
                 adj='sigmoid',
                 loss_weighting='consistency',
                 **cm_kwargs,
                 ):

        super().__init__(**cm_kwargs)

        if adj == 'const':
            self.t_to_r = self.t_to_r_const
        elif adj == 'sigmoid':
            self.t_to_r = self.t_to_r_sigmoid
        else:
            raise ValueError(f'Unknow schedule type {adj}!')

        self.n_stages = n_stages
        self.q = q
        self.stage = 0
        self.ratio = 0.
        self.step = 0

        self.k = k
        self.b = b

        self.c = c
        self.loss_weighting = loss_weighting

    def _get_loss_weights(self, t, r):
        if self.loss_weighting == 'consistency':
            return 1 / (t - r).flatten()
        elif self.loss_weighting == 'karras':
            return ((t ** 2 + self.sigma_data ** 2) / (t * self.sigma_data) ** 2).flatten()
        else:
            raise ValueError(f'Unknow loss weighting {self.loss_weighting}')

    def update_schedule(self, step):
        # in the original code, the authors use the concept of stage, and have a parameter to define how often
        # the stage should be changed. In the defaults, the stage goes from 0 to 7 during training.
        # the code below reproduces this 8 stages behaviour using the current step and max training steps.
        self.stage = step // (self.total_training_steps / self.n_stages)  # strategy to adapt the original code
        self.ratio = 1 - 1 / self.q ** (self.stage + 1)
        self.step = step

    def t_to_r_const(self, t):
        decay = 1 / self.q ** (self.stage + 1)
        ratio = 1 - decay
        r = t * ratio
        return torch.clamp(r, min=0)

    def t_to_r_sigmoid(self, t):
        adj = 1 + self.k * torch.sigmoid(-self.b * t)
        decay = 1 / self.q ** (self.stage + 1)
        ratio = 1 - decay * adj
        r = t * ratio
        return torch.clamp(r, min=0)

    def precond(self, x, sigma, class_labels, **model_kwargs):
        x = x.to(torch.float32)
        sigma = self._append_dims(sigma, x.ndim).to(torch.float32)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim],
                                                                    device=x.device) if class_labels is None else class_labels.to(
            torch.float32).reshape(-1, self.label_dim)
        if self.training:
            c_in, c_skip, c_out = self.kernel.get_scaling_factors(sigma)
        else:
            c_in, c_skip, c_out = self.kernel.get_scaling_factors_bc(sigma)
        c_noise = sigma.log() / 4
        F_x = self.net((c_in * x), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def loss(self, x, step, labels=None):

        log_dict = {}
        dims = x.ndim  # keeps track of data dimensionality to work with both images and tabular
        device = x.device
        batch_size = x.shape[0]

        # t ~ p(t) and r ~ p(r|t, iters) (Mapping fn)
        self.update_schedule(step)
        rnd_normal = self._append_dims(torch.randn([batch_size], device=device), dims)
        t = (rnd_normal * self.p_std + self.p_mean).exp()
        if isinstance(self.kernel, ConditionalOTKernel):
            t = torch.clamp(t, max=self.sigma_max)
        r = self.t_to_r(t)

        # Shared noise direction
        noise = torch.randn_like(x)
        if self.coupling != 'independent':
            x, noise, labels, posterior = self.coupling_fn(x, noise, labels)

        x_t = self.kernel.forward(x, t, noise)
        x_r = self.kernel.forward(x, r, noise)
        with isolate_rng():
            D_yt = self.precond(x_t, t, labels)

        if r.max() > 0:
            with torch.no_grad():
                with disable_forced_wn(self.net):
                    D_yr = self.precond(x_r, r, labels)

            mask = r > 0
            D_yr = torch.nan_to_num(D_yr)
            D_yr = mask * D_yr + (~mask) * x
        else:
            D_yr = x

        # L2 Loss
        loss = (D_yt - D_yr.detach()) ** 2
        loss = torch.sum(loss.reshape(loss.shape[0], -1), dim=-1)

        # Producing Adaptive Weighting (p=0.5) through Huber Loss
        if self.c > 0:
            loss = torch.sqrt(loss + self.c ** 2) - self.c
        else:
            loss = torch.sqrt(loss)

        log_dict['consistency_loss'] = loss.detach().view(batch_size, -1).sum(1).mean()
        loss_weighting = self._get_loss_weights(t, r)
        loss = loss * loss_weighting

        if self.coupling == 'vae':
            prior = torch.distributions.Normal(torch.zeros_like(x_t), torch.ones_like(x_t))
            t_w = torch.tensor(self.sigma_max)
            r_w = self.t_to_r(t_w)
            if self.loss_weighting == 'consistency':
                kl_loss_scale = (1 / (t_w - r_w) * self.kl_loss_scale).item()
            else:
                kl_loss_scale = self.kl_loss_scale
            kl_loss = torch.distributions.kl_divergence(posterior, prior)
            log_dict['kl_divergence'] = kl_loss.detach().view(batch_size, -1).sum(1).mean()
            kl_loss = kl_loss.view(batch_size, -1).sum(1) * kl_loss_scale
            loss += kl_loss
        # Weighting fn
        return loss.mean(), log_dict
