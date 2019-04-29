from .bases import BaseImageDataset
import os
from scipy import io
import numpy as np
import glob
import pickle


class SynthText(BaseImageDataset):
    dataset_dir = 'SynthText'

    def __init__(self, root='./data', is_training=True, verbose=True, **kwargs):
        super(SynthText, self).__init__(root)
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.is_training = is_training
        self.image_dir = self.dataset_dir
        self.annotation_dir = os.path.join(
            self.dataset_dir, 'gt')

        self.check_before_run()
        self.annotation_list = []
        self.image_list = []
        # with open(os.path.join(self.dataset_dir, 'image_list.txt')) as f:
        # self.annotation_list = [line.strip() for line in f.readlines()]
        self.annotation_list = glob.glob(self.dataset_dir+'/gt/*.txt')
        # print(len(self.annotation_list))
        self.train = self.process_dir(self.image_list, self.annotation_list)

        if verbose:
            print('=> TotalText loaded')
            self.print_dataset_statistics(
                self.image_list, self.annotation_list)

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
        annotations = []
        pickle_path = self.dataset_dir+'/gt_pickle_dump.pkl'
        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as fp:
                dataset = pickle.load(fp)
            for i in range(len(dataset)):
                if i % 100000 == 0:
                    print('Synth-text load:', i, '/', len(self.annotation_list))
                self.image_list.append(dataset[i][0])
            return dataset
        for i, annotation_path in enumerate(self.annotation_list):
            image_id, annotation = self.parse_txt(annotation_path)
            annotations.append(annotation)
            image_path = os.path.join(self.image_dir, image_id)
            if i % 100000 == 0:
                print('Synth-text load:', i, '/', len(self.annotation_list))
            self.image_list.append(image_path)
        # for i in range(len(image_list)):
            dataset.append([self.image_list[i], annotations[i]])
        with open(pickle_path, 'wb') as fp:
            pickle.dump(dataset, fp)
        return dataset

    def parse_txt(self, annotation_path):

        with open(annotation_path) as f:
            lines = [line.strip() for line in f.readlines()]
            image_id = lines[0]
            polygons = []
            for line in lines[1:]:
                points = [float(coordinate) for coordinate in line.split(',')]
                points = np.array(points, dtype=int).reshape(4, 2)
                polygons.append([points, 'c', 'abc'])
        return image_id, polygons


if __name__ == "__main__":
    a = SynthText()
