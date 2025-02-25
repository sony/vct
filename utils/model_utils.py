from omegaconf import DictConfig
from networks.edm_networks import SongUNet, DhariwalUNet
from networks.networks_edm2 import EDM2, Precond
from models.ecm import EasyConsistencyModel
from models.ict import ImprovedConsistencyTraining
from torch_utils import misc
import dnnlib
import pickle
import copy
from kernels.variance_exploding import VarianceExplodingKernel
from kernels.conditional_ot import ConditionalOTKernel

def get_kernel(cfg: DictConfig):
    if cfg.model.kernel == 've':
        kernel = VarianceExplodingKernel(cfg.model.sigma_min, cfg.model.sigma_max, cfg.model.sigma_data)
    elif cfg.model.kernel == 'cot':
        kernel = ConditionalOTKernel(cfg.model.sigma_min, cfg.model.sigma_max, cfg.model.sigma_data)
    return kernel

def get_neural_net(cfg: DictConfig):
    if cfg.model.class_conditional:
        label_dim = cfg.dataset.label_dim
    else:
        label_dim = 0
    if cfg.network.name in ['ddpmpp', 'ncsnpp']:
        net = SongUNet(
            img_resolution=cfg.dataset.img_resolution,
            in_channels=cfg.dataset.in_channels,
            out_channels=cfg.dataset.out_channels,
            label_dim=label_dim,
            embedding_type=cfg.network.embedding_type,
            encoder_type=cfg.network.encoder_type,
            decoder_type=cfg.network.decoder_type,
            channel_mult_noise=cfg.network.channel_mult_noise,
            resample_filter=list(cfg.network.resample_filter),
            model_channels=cfg.network.model_channels,
            channel_mult=list(cfg.network.channel_mult),
            dropout=cfg.network.dropout,
            num_blocks=cfg.network.num_blocks,
        )
    elif cfg.network.name == 'adm':
        net = DhariwalUNet(
            img_resolution=cfg.dataset.img_resolution,
            in_channels=cfg.dataset.in_channels,
            out_channels=cfg.dataset.out_channels,
            label_dim=label_dim,
            model_channels=cfg.network.model_channels,
            channel_mult=cfg.network.channel_mult,
            dropout=cfg.network.dropout,
            num_blocks=cfg.network.num_blocks,
        )
    elif cfg.network.name == 'edm2':
        if cfg.network.reload_url:
            net = Precond(
                img_resolution=cfg.dataset.img_resolution,
                img_channels=cfg.dataset.in_channels,
                label_dim=label_dim,
                use_fp16=False,
                model_channels=cfg.network.model_channels,
                channel_mult=cfg.network.channel_mult,
                dropout=cfg.network.dropout,
                dropout_res=cfg.network.dropout_res,
                num_blocks=cfg.network.num_blocks,

            )
        else:
            net = EDM2(
                img_resolution=cfg.dataset.img_resolution,
                in_channels=cfg.dataset.in_channels,
                out_channels=cfg.dataset.out_channels,
                label_dim=label_dim,
                model_channels=cfg.network.model_channels,
                channel_mult=cfg.network.channel_mult,
                dropout=cfg.network.dropout,
                dropout_res=cfg.network.dropout_res,
                num_blocks=cfg.network.num_blocks,
            )


    else:
        raise NotImplementedError

    if cfg.network.name == 'ddpmpp':
        if cfg.network.reload_url:
            with dnnlib.util.open_url(cfg.network.reload_url) as f:
                data = pickle.load(f)
                misc.copy_params_and_buffers(src_module=data['ema'].model, dst_module=net, require_all=False)
                del data
    elif cfg.network.name == 'adm':
        if cfg.network.reload_url:
            with dnnlib.util.open_url(cfg.network.reload_url) as f:
                data = pickle.load(f)
                misc.copy_params_and_buffers(src_module=data['ema'].model, dst_module=net, require_all=True)
                del data
    elif cfg.network.name == 'edm2':
        if cfg.network.reload_url:
            with dnnlib.util.open_url(cfg.network.reload_url) as f:
                data = pickle.load(f)
                misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=True)
                del data
            net = net.unet

    return net

def get_model(cfg: DictConfig):

    assert cfg.model.name in ['ict', 'ecm']
    assert cfg.model.coupling in ['independent', 'ot', 'vae']
    assert cfg.network.name in ['ddpmpp', 'ncsnpp', 'adm', 'edm2']
    assert cfg.model.kernel in ['ve', 'cot']
    kernel = get_kernel(cfg)
    net = get_neural_net(cfg)
    if cfg.model.coupling == 'vae':
        cfg_copy = copy.deepcopy(cfg)
        if cfg.network.name in ['ddpmpp', 'ncsnpp', 'adm', 'edm2']:
            assert cfg.model.encoder_size in ['small', 'big']
            if cfg.model.encoder_size == 'small':
                cfg_copy.network.num_blocks = 1
                cfg_copy.network.model_channels = 32
            elif cfg.model.encoder_size == 'big':
                cfg_copy.network.num_blocks = 2
                cfg_copy.network.model_channels = 32
            cfg_copy.dataset.out_channels = cfg_copy.dataset.out_channels * 2
            cfg_copy.network.reload_url = ''
            encoder = get_neural_net(cfg_copy)
    else:
        encoder = None
    if cfg.model.class_conditional:
        label_dim = cfg.dataset.label_dim
    else:
        label_dim = 0
    cm_kwargs = {
        'net': net,
        'total_training_steps': cfg.model.total_training_steps,
        'kernel': kernel,
        'sigma_min': cfg.model.sigma_min,
        'sigma_max': cfg.model.sigma_max,
        'p_mean': cfg.model.p_mean,
        'p_std': cfg.model.p_std,
        'sigma_data': cfg.model.sigma_data,
        'label_dim': label_dim,
        'mid_t': list(cfg.model.mid_t),
        'coupling': cfg.model.coupling,
        'kl_loss_scale': cfg.model.kl_loss_scale,
        'encoder': encoder,
    }

    if cfg.model.name == 'ict':
        model = ImprovedConsistencyTraining(**cm_kwargs,
                                            scale_mode=cfg.model.scale_mode,
                                            start_scales=cfg.model.start_scales,
                                            end_scales=cfg.model.end_scales,
                                            rho=cfg.model.rho,
                                            noise_schedule=cfg.model.noise_schedule,
                                            loss_mode=cfg.model.loss_mode,
                                            )
    elif cfg.model.name == 'ecm':
        model = EasyConsistencyModel(**cm_kwargs,
                                     n_stages=cfg.model.n_stages,
                                     q=cfg.model.q,
                                     c=cfg.model.c,
                                     k=cfg.model.k,
                                     b=cfg.model.b,
                                     adj=cfg.model.adj,
                                     loss_weighting=cfg.model.loss_weighting,
                                     )
    else:
        raise NotImplementedError
    return model