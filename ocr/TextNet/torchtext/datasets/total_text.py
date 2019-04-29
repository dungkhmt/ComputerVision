from .bases import BaseImageDataset
import os
from scipy import io
import numpy as np
from tqdm import tqdm


class TotalText(BaseImageDataset):
    dataset_dir = 'total-text'

    def __init__(self, root='./data', ignore_list='./data/total-text/ignore_list.txt', is_training=True, verbose=True, **kwargs):
        super(TotalText, self).__init__(root)
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.is_training = is_training
        if ignore_list:
            with open(ignore_list) as f:
                ignore_list = f.readlines()
                ignore_list = [line.strip() for line in ignore_list]
        else:
            ignore_list = []
        self.image_dir = os.path.join(
            self.dataset_dir, 'Images', 'Train' if is_training else 'Test')
        self.annotation_dir = os.path.join(
            self.dataset_dir, 'gt', 'Train' if is_training else 'Test')

        self.check_before_run()

        self.image_list = os.listdir(self.image_dir)
        self.image_list = list(filter(lambda img: img.replace(
            '.jpg', '') not in ignore_list, self.image_list))
        self.annotation_list = []
        for i in tqdm(range(len(self.image_list))):
            image_index = self.image_list[i].replace('.jpg', '')
            self.annotation_list.append(os.path.join(
                self.annotation_dir, 'poly_gt_{}.mat'.format(image_index)))
            self.image_list[i] = os.path.join(
                self.image_dir, self.image_list[i])
        if verbose:
            print('=> TotalText loaded')
            self.print_dataset_statistics(
                self.image_list, self.annotation_list)
        self.train = self.process_dir(self.image_list, self.annotation_list)
        # image_path, annot = self.train[0]
        # print(annot[0][0],annot[0][1],annot[0][2])

    def check_before_run(self):
        """Check if all files are available before going deeper"""
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError(
                '"{}" is not available'.format(self.dataset_dir))
        if not os.path.exists(self.image_dir):
            raise RuntimeError('"{}" is not available'.format(self.image_dir))
        if not os.path.exists(self.annotation_dir):
            raise RuntimeError(
                '"{}" is not available'.format(self.annotation_dir))

    def process_dir(self, image_list, annotation_list):
        dataset = []
        for i in range(len(image_list)):
            dataset.append([image_list[i], annotation_list[i], self.parse_mat])
        return dataset

    def parse_mat(self, mat_path):
        """
        .mat file parser
        :param mat_path: (str), mat file path
        :return: (list), TextInstance
        """
        annot = io.loadmat(mat_path)
        polygon = []
        for cell in annot['polygt']:
            x = cell[1][0]
            y = cell[3][0]
            text = cell[4][0] if len(cell[4]) > 0 else '#'
            if len(x) < 4:  # too few points
                continue
            ori = cell[5][0] if len(cell[5]) > 0 else '#'
            pts = np.stack([x, y]).T.astype(np.int32)
            polygon.append([pts, ori, text])
        return polygon


if __name__ == "__main__":
    a = TotalText()
