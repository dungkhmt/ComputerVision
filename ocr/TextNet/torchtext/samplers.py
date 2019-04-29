from torch.utils.data.sampler import RandomSampler

def build_train_sampler(data_source,
                        train_sampler,
                        **kwargs):
    """Build sampler for training

    Args:
    - data_source (list): list of (img,...).
    """
    if train_sampler=='RandomSampler':
        sampler = RandomSampler(data_source)
    else:
        print('Suport RandomSampler only!')

    return sampler