from .detect_net import DetectNet


__model_factory = {
    'resnet50': DetectNet,
    'vgg16': DetectNet,
    'se_resnext50_32x4d': DetectNet
}


def get_names():
    return list(__model_factory.keys())


def init_model(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError('Unknown model: {}'.format(name))
    return __model_factory[name](backbone=name, *args, **kwargs)
