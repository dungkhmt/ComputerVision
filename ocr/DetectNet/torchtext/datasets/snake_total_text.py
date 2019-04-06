import scipy.io as io
import numpy as np
import os

from .snake_dataloader import pil_load_img
from .snake_dataloader import TextDataset, TextInstance


class TotalText(TextDataset):

    def __init__(self, data_root, ignore_list=None, is_training=True, transform=None):
        super().__init__(transform)
        self.data_root = data_root
        self.is_training = is_training

        if ignore_list:
            with open(ignore_list) as f:
                ignore_list = f.readlines()
                ignore_list = [line.strip() for line in ignore_list]
        else:
            ignore_list = []

        self.image_root = os.path.join(
            data_root, 'Images', 'Train' if is_training else 'Test')
        self.annotation_root = os.path.join(
            data_root, 'gt', 'Train' if is_training else 'Test')
        self.image_list = os.listdir(self.image_root)
        self.image_list = list(filter(lambda img: img.replace(
            '.jpg', '') not in ignore_list, self.image_list))
        self.annotation_list = ['poly_gt_{}.mat'.format(
            img_name.replace('.jpg', '')) for img_name in self.image_list]

    def __getitem__(self, item):

        image_id = self.image_list[item]
        image_path = os.path.join(self.image_root, image_id)
        save_file = './data/tmp/'+str(image_id).split('.')[0]+'.npy'
        # Read image data
        image = pil_load_img(image_path)

        # Read annotation
        annotation_id = self.annotation_list[item]
        annotation_path = os.path.join(self.annotation_root, annotation_id)
        polygons = self.parse_mat(annotation_path)
        x_mask, y_mask = 0, 0
        load_save = False
        if load_save and os.path.exists(save_file):
            data = np.load(save_file)
            # print(data.shape)
            # print(data[3].shape)
            x_mask, y_mask = data[0].astype(
                np.float32), data[1].astype(np.float32)
            # print(image.shape, train_mask.shape, tr_mask.shape, tcl_mask.shape,
            #       radius_map.shape, sin_map.shape, cos_map.shape, x_mask.shape, y_mask.shape)
        if np.amax(x_mask)*np.amax(y_mask) != 0:
            image, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map,  _, _ = self.get_training_data(
                image, polygons, image_id=image_id, image_path=image_path, distance=False)
        else:
            image, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map,  x_mask, y_mask = self.get_training_data(
                image, polygons, image_id=image_id, image_path=image_path, distance=True)
            # data = np.concatenate(
            #     [np.expand_dims(x_mask, axis=0), np.expand_dims(y_mask, axis=0)], axis=0)
            # np.save(save_file, data)
        # print(image.dtype, train_mask.dtype, tr_mask.dtype, tcl_mask.dtype,
        #           radius_map.dtype, sin_map.dtype, cos_map.dtype, x_mask.dtype, y_mask.dtype)

        # import cv2
        # print(image.shape)
        # cv2.imshow('1',cv2.cvtColor(image.transpose(1,2,0),cv2.COLOR_RGB2BGR))
        # cv2.imshow('2',tr_mask*255)
        # for i in range(512):
        #     for j in range(512):
        #         if x_mask[i][j] > 0 :
        #             x_mask[i][j] = 255
        # cv2.imshow('3',x_mask)
        # for i in range(512):
        #     for j in range(512):
        #         if y_mask[i][j] > 0 :
        #             y_mask[i][j] = 255
        # cv2.imshow('4',y_mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # exit()

        return image, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map,  x_mask, y_mask

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    import os
    from util.augmentation import BaseTransform, Augmentation

    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)

    transform = Augmentation(
        size=512, mean=means, std=stds
    )

    trainset = TotalText(
        data_root='data/total-text',
        ignore_list='./ignore_list.txt',
        is_training=True,
        transform=transform
    )

    for idx in range(len(trainset)):
        img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta = trainset[
            idx]
        print(idx, img.shape)
