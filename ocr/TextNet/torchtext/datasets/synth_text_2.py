from .bases import BaseImageDataset
import os
from scipy import io
import numpy as np
import glob
import pickle
import json
from tqdm import tqdm
import random


class SynthText_2(BaseImageDataset):
    dataset_dir = 'SynthText'

    def __init__(self, root='./data', is_training=True, verbose=True, **kwargs):
        super(SynthText_2, self).__init__(root)
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
        self.annotation_list = glob.glob(self.dataset_dir+'/json_gt/*.json')
        # print(len(self.annotation_list))
        self.train = self.process_dir(self.image_list, self.annotation_list)

        if verbose:
            print('=> SynthText loaded')
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

    # def process_dir(self, image_list, annotation_list):
    #     dataset = []
    #     # annotations = []
    #     pickle_path = self.dataset_dir+'/pickle_json'
    #     list_pickle = glob.glob(pickle_path+'/*.pkl')
    #     if len(list_pickle) == (len(self.annotation_list)//50000+1):
    #         for i in tqdm(range(len(list_pickle))):
    #             with open(list_pickle[i], 'rb') as fp:
    #                 a= pickle.load(fp)
    #                 a = random.choices(a,k=len(a)//2)
    #                 dataset+=a
    #         for i in tqdm(range(len(dataset))):
    #             self.image_list.append(dataset[i][0])
    #         return dataset
    #     tmp_dataset = []
    #     for i in tqdm(range(len(self.annotation_list))):
    #         annotation_path = self.annotation_list[i]
    #         image_id, annotation = self.parse_txt(annotation_path)
    #         # annotations.append(annotation)
    #         image_path = os.path.join(self.image_dir, image_id)
    #         self.image_list.append(image_path)
    #         # dataset.append([self.image_list[i], annotation])
    #         tmp_dataset.append([self.image_list[i], annotation])
    #         if (i > 0 and i % 50000 == 0) or i == (len(self.annotation_list)-1):
    #             with open(pickle_path+'/'+str(i+1)+'.pkl', 'wb') as fp:
    #                 pickle.dump(tmp_dataset, fp)
    #             tmp_dataset = []
    #     return dataset

    def process_dir(self, image_list, annotation_list):
        dataset = []
        annotations = []
        pickle_path = self.dataset_dir+'/pickle_json_ano_path'
        list_pickle = glob.glob(pickle_path+'/*.pkl')
        if len(list_pickle) == (len(self.annotation_list)//50000+1):
            for i in tqdm(range(len(list_pickle))):
                with open(list_pickle[i], 'rb') as fp:
                    dataset += pickle.load(fp)
            for i in tqdm(range(len(dataset))):
                self.image_list.append(dataset[i][0])
                dataset[i].append(self.parse_annotation)
            return dataset
        tmp_dataset = []
        for i in tqdm(range(len(self.annotation_list))):
            annotation_path = self.annotation_list[i]
            image_id, annotation = self.parse_json(annotation_path)
            annotations.append(annotation)
            image_path = os.path.join(self.image_dir, image_id)
            self.image_list.append(image_path)
            dataset.append(
                [self.image_list[i], annotation_path, self.parse_annotation])
            tmp_dataset.append(
                [self.image_list[i], annotation_path])
            if (i > 0 and i % 50000 == 0) or i == (len(self.annotation_list)-1):
                with open(pickle_path+'/'+str(i+1)+'.pkl', 'wb') as fp:
                    pickle.dump(tmp_dataset, fp)
                tmp_dataset = []
        return dataset

    def parse_json(self, annotation_path):

        with open(annotation_path, 'r') as fp:
            data = json.load(fp)
        image_id = data['imname']
        del data['imname']
        polygons = []
        for k in data.keys():
            wordBB = np.array(data[k]['wordBB']).reshape(4, 2)
            charBB = []
            char_list = []
            for c in data[k]['char_list']:
                bboxes = np.array(c['charBB']).reshape(4, 2)
                charBB.append(bboxes)
                char_list.append(c['char'])
            polygons.append([wordBB, 'c', ''.join(char_list)])
        return image_id, polygons

    def parse_annotation(self, annotation_path):

        with open(annotation_path, 'r') as fp:
            data = json.load(fp)
        _ = data['imname']
        del data['imname']
        polygons = []
        for k in data.keys():
            wordBB = np.array(data[k]['wordBB']).reshape(4, 2)
            charBB = []
            char_list = []
            for c in data[k]['char_list']:
                bboxes = np.array(c['charBB']).reshape(4, 2)
                charBB.append(bboxes)
                char_list.append(c['char'])
            polygons.append([wordBB, 'c', ''.join(char_list)])
        return polygons


if __name__ == "__main__":
    a = SynthText_2()
