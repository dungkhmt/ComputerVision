import os
from PIL import Image
import numpy as np
import os.path as osp
import scipy.io as io
import copy

import torch
from torch.utils.data import Dataset
from skimage.draw import polygon as drawpoly
from torchtext.utils.misc import find_bottom, find_long_edges, split_edge_seqence, norm2, vector_cos, vector_sin, process_output
import cv2
import time


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError('{} does not exist'.format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print('IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'.format(
                img_path))
            pass
    return np.array(img)


class TextInstance():

    def __init__(self, points, orient, text):
        self.orient = orient
        self.text = text

        # remove_points = []

        # if len(points) > 4:
        #     # remove point if area is almost unchanged after removing it
        #     ori_area = cv2.contourArea(points)
        #     for p in range(len(points)):
        #         # attempt to remove p
        #         index = list(range(len(points)))
        #         index.remove(p)
        #         area = cv2.contourArea(points[index])
        #         if np.abs(ori_area - area) / ori_area < 0.017 and len(points) - len(remove_points) > 4:
        #             remove_points.append(p)
        #     self.points = np.array(
        #         [point for i, point in enumerate(points) if i not in remove_points])
        # else:
        self.points = np.array(points)

    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, item):
        return getattr(self, item)


class DetectDataset(Dataset):
    def __init__(self, model, raw_dataset, transform_image=None, **kwargs):
        self.model = model
        self.dataset = raw_dataset
        self.transform_image = transform_image
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, annotation_path, parse_annotation = self.dataset[index]
        annotation = parse_annotation(annotation_path)
        polygons = self.parse_annot(annotation)
        image = read_image(img_path)
        return self.get_training_data(image, polygons, img_path)

    def parse_annot(self, annotation):
        '''
        annotation type: list contain all text annot in image
        annotation[0] type: list containt one text annot
        annotation[0][0] np.array shape=(k,2) point (x,y) contour
        annotation[0][1] np.str shape=(1,) orientation text (c=curve; h=horizontal; m=multi-oriented; #=dont care)
        annotation[0][2] np.str shape(k,) text content 'ABCD', 'Hello'

        return list(TextInstance(points, ori, text))
        '''
        polygons = []
        for annot in annotation:
            polygons.append(TextInstance(annot[0], annot[1], annot[2]))
        return polygons

    def cal_vector(self, image, bboxes, hards):
        height, width = image.shape[0:2]
        accumulation = np.zeros((3, height, width), dtype=np.float32)
        for bboxi, points in enumerate(bboxes):
            points = points.astype(np.int32)
            left, top = points.min(axis=0)
            right, bottom = points.max(axis=0)
            if right < 0 or bottom < 0:
                continue
            if left > width-1 or top > height-1:
                continue
            left = max(0, left)
            top = max(0, top)
            right = min(width-1, right)
            bottom = min(height-1, bottom)
            new_points = points - points.min(axis=0)+1
            new_height = bottom-top+3
            new_width = right-left+3
            hard = hards[bboxi]
            new_seg = np.zeros((new_height, new_width), dtype=np.uint8)
            cv2.fillPoly(new_seg, [new_points], (1,))
            contours = np.array(
                [[0, 0], [new_seg.shape[1]-1, 0], [new_seg.shape[1]-1, new_seg.shape[0]-1], [0, new_seg.shape[0]-1]])
            cv2.drawContours(new_seg, [contours], -1, (0,), 1)
            new_img = new_seg.astype(np.uint8)
            dst, labels = cv2.distanceTransformWithLabels(
                new_img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL)
            index = np.copy(labels)
            index[new_img > 0] = 0
            place = np.argwhere(index > 0)
            nearCord = place[labels-1, :]
            # x height, y width
            x = nearCord[:, :, 0]
            y = nearCord[:, :, 1]
            nearPixel = np.zeros((2, new_height, new_width))
            nearPixel[0, :, :] = x
            nearPixel[1, :, :] = y
            grid = np.indices(new_img.shape)
            grid = grid.astype(np.float32)
            diff = grid - nearPixel
            dist = np.sqrt(np.sum(diff**2, axis=0))

            new_direction = np.zeros(
                (3, new_height, new_width), dtype=np.float32)
            new_direction[0, new_img > 0] = np.divide(
                diff[0, new_img > 0], dist[new_img > 0])
            new_direction[1, new_img > 0] = np.divide(
                diff[1, new_img > 0], dist[new_img > 0])

            direction = np.zeros(
                (3, height, width), dtype=np.float32)
            direction[:, top:bottom+1, left:right +
                      1] = new_direction[:, 1:new_height-1, 1:new_width-1]

            seg = np.zeros(image.shape[0:2], dtype=np.uint8)
            cv2.fillPoly(seg, [points], (1,))
            img = seg.astype(np.uint8)
            if hard == 0:
                direction[2, img > 0] = bboxi+1
            else:
                direction[2, img > 0] = -1

            accumulation[0, img > 0] = 0
            accumulation[1, img > 0] = 0
            accumulation[2, img > 0] = 0
            accumulation = accumulation + direction
        vec = np.stack((accumulation[0], accumulation[1]))
        # compute weight
        weight = np.zeros((height, width), dtype=np.float32)
        weight[accumulation[2] < 0] = -1
        posRegion = accumulation[2] > 0
        posCount = np.sum(posRegion)
        if posCount != 0:
            bboxRemain = 0
            for bboxi, polygon in enumerate(bboxes):
                overlap_bboxi = accumulation[2] == (bboxi+1)
                overlapCount_bboxi = np.sum(overlap_bboxi)
                if overlapCount_bboxi == 0:
                    continue
                bboxRemain = bboxRemain+1
            bboxAve = float(posCount)/bboxRemain
            for bboxi, polygon in enumerate(bboxes):
                overlap_bboxi = accumulation[2] == (bboxi+1)
                overlapCount_bboxi = np.sum(overlap_bboxi)
                if overlapCount_bboxi == 0:
                    continue
                pixAve = bboxAve/overlapCount_bboxi
                weight = weight*(~overlap_bboxi) + pixAve*overlap_bboxi
        return image, vec, weight.astype(np.float32)

    def make_word_vector(self, image, polygons, img_path):
        bboxes = []
        hards = []
        self.charBBs = []
        self.s_conf = 1
        for polygon in polygons:
            bboxes.append(polygon.points.astype(np.int32))
            hard = 1 if polygon.orient == '#' else 0
            hards.append(hard)

        return self.cal_vector(image, bboxes, hards)


    def get_training_data(self, image, polygons, image_path):
        # im_name = image_path.split('/')[-1].split('.jpg')[0]
        # cv2.imwrite('./trash/'+im_name+'img.jpg', image)
        if self.transform_image:
            image, polygons = self.transform_image(image, copy.copy(polygons))
        # cv2.imwrite('./trash/'+im_name+'img_tranpose.jpg', image*255)
        image, vec, weight = self.make_word_vector(image, polygons, image_path)

        # img = np.zeros(image.shape)
        # for i in range(img.shape[0]):
            # for j in range(img.shape[1]):
                # if vec[0][i][j] != 0 or vec[1][i][j] != 0:
                    # img[i][j] = 255
        # cv2.imwrite('./trash/'+im_name+'vec_word.jpg', img.astype(np.uint8))

        image = image.transpose(2, 0, 1)
        
        return image, vec, weight
