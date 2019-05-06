import torch
from PIL import Image
from torchtext.transforms import build_transforms
import numpy as np
from torchtext.models import init_model
from torchtext.utils.misc import process_output, mkdirs
import pickle
from functools import partial
import cv2
import queue
import glob
from tqdm import tqdm
import os


def load_model(model, model_path):
    checkpoint = torch.load(model_path, pickle_module=pickle)
    pretrain_dict = checkpoint['state_dict']
    model.load_state_dict(pretrain_dict)
    print("Loaded pretrained weights from '{}'".format(model_path))
    return model


def write_to_file(contours, file_path):
    """
    :param contours: [[x1, y1], [x2, y2]... [xn, yn]]
    :param file_path: target file path
    """
    # according to total-text evaluation method, output file shoud be formatted to: y0,x0, ..... yn,xn
    with open(file_path, 'w') as f:
        for cont in contours:
            cont = np.stack([cont[:, 1], cont[:, 0]], 1)
            cont = cont.flatten().astype(str).tolist()
            cont = ','.join(cont)
            f.write(cont + '\n')


def test(path_input, path_output):
    mkdirs(path_output)
    model = init_model(name='se_resnext101_32x4d')
    load_model(
        model, 'log/se_resnext101_32x4d-final-text-net-total-text-768-2/quick_save_checkpoint_ep81.pth.tar')
    model = model.to('cuda')
    model.eval()
    transform = build_transforms(
        maxHeight=512, maxWidth=512, is_train=False)
    list_image = glob.glob(path_input+'/*.jpg')
    for idx in tqdm(range(len(list_image))):
        path_image = list_image[idx]
        image_id = path_image.split('/')[-1]
        im = Image.open(path_image)
        image = np.array(im)
        img, _ = transform(np.copy(image), None)
        im = img.transpose(2, 0, 1)
        im = torch.Tensor(im).to('cuda').unsqueeze(0)
        with torch.no_grad():
            output = model(im)
        contours = process_output(img, output[0].to(
            'cpu').numpy(), image, threshold=0.4, min_area=200)
        write_to_file(contours, os.path.join(
            path_output, image_id.replace('jpg', 'txt')))


if __name__ == "__main__":
    test('./data/total-text/Images/Test', 'output/total-text-768-81-0.4-200-512')
    os.system('python Deteval.py total-text-768-81-0.4-200-512')