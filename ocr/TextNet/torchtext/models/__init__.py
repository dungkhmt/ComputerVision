from .detect_net import DetectNet
from .craft_net import CRAFTNet
from .text_field_net import TextFieldNet


__model_factory = {
    'resnet50': CRAFTNet,
    'resnet101': CRAFTNet,
    'resnet152': CRAFTNet,
    'vgg16': TextFieldNet,
    # 'se_resnext50_32x4d': DetectNet,
    'se_resnext50_32x4d': CRAFTNet,
    'se_resnext101_32x4d': CRAFTNet
}


def get_names():
    return list(__model_factory.keys())


def init_model(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError('Unknown model: {}'.format(name))
    return __model_factory[name](backbone=name, *args, **kwargs)
