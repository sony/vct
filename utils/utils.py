import torch
from torch.utils.data import DataLoader
import numpy as np
def rescaling(x):
    """Rescale data to [-1, 1]. Assumes data in [0,1]."""
    return x * 2. - 1.


def rescaling_inv(x):
    """Rescale data back from [-1, 1] to [0,1]."""
    return .5 * x + .5

def adjust_channels(images):
    # Get the number of channels in the first image
    num_channels = images.size(1)

    # If the image has 1 channel, copy it three times
    if num_channels == 1:
        images = images.repeat(1, 3, 1, 1)

    # If the image has 3 channels, return it as is
    elif num_channels == 3:
        pass

    # If the number of channels is different, throw an error
    else:
        raise ValueError(f"Unexpected number of channels: {num_channels}. Expected 1 or 3.")

    return images

def get_all_data(dataloader):
    data_list = []
    for data in dataloader:
        data_list.append(data)
    return torch.cat(data_list, 0)

def std_to_exp(std):
    std = np.float64(std)
    tmp = std.flatten() ** -2
    exp = [np.roots([1, 7, 16 - t, 12 - t]).real.max() for t in tmp]
    exp = np.float64(exp).reshape(std.shape)
    return exp

def power_function_beta(std, t):
    beta = (1 - (1 / (t + 1))) ** (std_to_exp(std) + 1)
    return beta


class ResumableDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, *, prefetch_factor=2,
                 persistent_workers=False):
        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler,
                         num_workers, collate_fn, pin_memory, drop_last, timeout,
                         worker_init_fn, prefetch_factor=prefetch_factor,
                         persistent_workers=persistent_workers)
        self.iteration = 0

    def __iter__(self):
        for batch in super().__iter__():
            self.iteration += 1
            yield batch

    def state_dict(self):
        return {'iteration': self.iteration}

    def load_state_dict(self, state_dict):
        self.iteration = state_dict['iteration']
