
from omegaconf import DictConfig
import warnings
def get_run_name(cfg: DictConfig):
    name = f'{cfg.dataset.name}_{cfg.model.name}'

    if cfg.model.class_conditional:
        name += f'_cond'

    if cfg.network.reload_url:
        name += '_pretrained'

    assert cfg.model.name in ['ict', 'ecm']
    name += f'_{cfg.model.coupling}_ker_{cfg.model.kernel}'
    if cfg.model.coupling == 'vae':
        name += f'_kls_{cfg.model.kl_loss_scale}'
        name += f'_{cfg.model.encoder_size}_enc'

    name += f'_bs_{cfg.dataset.batch_size * cfg.batch_multiplier}_drop_{cfg.network.dropout}'

    if cfg.gradient_clip_val > 0:
        name += f'_gclip_{cfg.gradient_clip_val}'

    if cfg.extra_name:
        name += f'_{cfg.extra_name}'


    return name
