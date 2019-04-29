import os
import os.path as osp
import numpy as np


class BaseDataset():

    def __init__(self, root):
        self.root = osp.expanduser(root)

    def get_imagedata_info(self, image_list,annotation_list):
        num_imgs = len(image_list)
        num_anno = len(annotation_list)
        return num_imgs,num_anno

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    def print_dataset_statistics(self, image_list, annotation_list):
        num_image, num_anno = self.get_imagedata_info(image_list,annotation_list)
        print('  ----------------------------------------')
        print('  #Images        | {:8d} |'.format(num_image))
        print('  ----------------------------------------')
        print('  #Annotation    | {:8d} |'.format(num_anno))
        print('  ----------------------------------------')
