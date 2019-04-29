from .total_text import TotalText
from .pdf_text import PDFText
from .synth_text_2 import SynthText_2

__detect_factory = {
    'total-text': TotalText,
    'pdf-text': PDFText,
    'synth-text-2': SynthText_2
}


def init_detect_dataset(name, **kwargs):
    if name not in list(__detect_factory.keys()):
        raise KeyError('Invalid dataset, got "{}", but expected to be one of {}'.format(
            name, list(__detect_factory.keys())))
    return __detect_factory[name](**kwargs)
