from .total_text import TotalText


__detect_factory={
    'total-text': TotalText
}


def init_detect_dataset(name, **kwargs):
    if name not in list(__detect_factory.keys()):
        raise KeyError('Invalid dataset, got "{}", but expected to be one of {}'.format(name, list(__detect_factory.keys())))
    return __detect_factory[name](**kwargs)
