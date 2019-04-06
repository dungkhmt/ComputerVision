import os
from PIL import Image
import numpy as np
import os.path as osp
import scipy.io as io
import copy

import torch
from torch.utils.data import Dataset
from skimage.draw import polygon as drawpoly
from torchtext.utils.misc import find_bottom, find_long_edges, split_edge_seqence, norm2, vector_cos, vector_sin
import cv2
from torchtext.utils.config import config as cfg


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

        self.points = []

        # remove point if area is almost unchanged after removing
        ori_area = cv2.contourArea(points)
        for p in range(len(points)):
            index = list(range(len(points)))
            index.remove(p)
            area = cv2.contourArea(points[index])
            if np.abs(ori_area - area) / ori_area > 0.017:
                self.points.append(points[p])
        self.points = np.array(self.points)

    def find_bottom_and_sideline(self):
        # find two bottoms of this Text
        self.bottoms = find_bottom(self.points)
        self.e1, self.e2 = find_long_edges(
            self.points, self.bottoms)  # find two long edge sequence

    def disk_cover(self, n_disk=15):
        """
        cover text region with several disks
        :param n_disk: number of disks
        :return:
        """
        inner_points1 = split_edge_seqence(self.points, self.e1, n_disk)
        inner_points2 = split_edge_seqence(self.points, self.e2, n_disk)
        inner_points2 = inner_points2[::-1]  # innverse one of long edge

        center_points = (inner_points1 + inner_points2) / 2  # disk center
        # radii = norm2(inner_points1 - center_points, axis=1)  # disk radius

        return inner_points1, inner_points2, center_points  # , radii

    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, item):
        return getattr(self, item)


class DetectDataset(Dataset):
    def __init__(self, dataset, transform=None, **kwargs):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, annotation = self.dataset[index]
        image = read_image(img_path)

        polygons = self.parse_annot(annotation)

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

    def make_text_region(self, image, polygons):

        tr_mask = np.zeros(image.shape[:2], np.uint8)
        train_mask = np.ones(image.shape[:2], np.uint8)

        for polygon in polygons:
            cv2.fillPoly(tr_mask, [polygon.points.astype(
                np.int32)], color=(1,))  # fill text regions
            if polygon.text == '#':
                # fill train mask cal loss only region have text != #
                cv2.fillPoly(
                    train_mask, [polygon.points.astype(np.int32)], color=(0,))
        return tr_mask, train_mask

    def fill_polygon(self, mask, polygon, value):
        """
        fill polygon in the mask with value
        :param mask: input mask
        :param polygon: polygon to draw
        :param value: fill value
        """
        rr, cc = drawpoly(polygon[:, 1], polygon[:, 0],
                          shape=(cfg.input_size, cfg.input_size))
        mask[rr, cc] = value

    # def make_text_center_line(self, sideline1, sideline2, center_line, radius,
    #                           tcl_mask, radius_map, sin_map, cos_map, expand=0.5, shrink=0):
    def make_text_center_line(self, sideline1, sideline2, center_line, tcl_mask, expand=0.5, shrink=0):

        # TODO: shrink 1/2 * radius at two line end
        # shink = 0 => keep head and tail
        for i in range(shrink, len(center_line) - 1 - shrink):

            c1 = center_line[i]
            c2 = center_line[i + 1]
            top1 = sideline1[i]
            top2 = sideline1[i + 1]
            bottom1 = sideline2[i]
            bottom2 = sideline2[i + 1]

            # sin_theta = vector_sin(c2 - c1)
            # cos_theta = vector_cos(c2 - c1)

            p1 = c1 + (top1 - c1) * expand
            p2 = c1 + (bottom1 - c1) * expand
            p3 = c2 + (bottom2 - c2) * expand
            p4 = c2 + (top2 - c2) * expand
            polygon = np.stack([p1, p2, p3, p4])

            self.fill_polygon(tcl_mask, polygon, value=1)
            # self.fill_polygon(radius_map, polygon, value=radius[i])
            # self.fill_polygon(sin_map, polygon, value=sin_theta)
            # self.fill_polygon(cos_map, polygon, value=cos_theta)

    def make_text_vector(self, image, polygons):
        height, width = image.shape[0:2]
        accumulation = np.zeros((3, height, width), dtype=np.float32)
        for bboxi, polygon in enumerate(polygons):
            points = polygon.points.astype(np.int32)
            hard = 1 if polygon.orient == '#' else 0
            seg = np.zeros(image.shape[0:2], dtype=np.uint8)
            cv2.fillPoly(seg, [points], (1,))
            img = seg
            dst, labels = cv2.distanceTransformWithLabels(
                img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL)
            index = np.copy(labels)
            index[img > 0] = 0
            place = np.argwhere(index > 0)
            nearCord = place[labels-1, :]
            # x height, y width
            x = nearCord[:, :, 0]
            y = nearCord[:, :, 1]
            nearPixel = np.zeros((2, height, width))
            nearPixel[0, :, :] = x
            nearPixel[1, :, :] = y
            grid = np.indices(img.shape)
            grid = grid.astype(float)
            diff = grid - nearPixel
            dist = np.sqrt(np.sum(diff**2, axis=0))

            direction = np.zeros(
                (3, height, width), dtype=np.float32)
            direction[0, img > 0] = np.divide(diff[0, img > 0], dist[img > 0])
            direction[1, img > 0] = np.divide(diff[1, img > 0], dist[img > 0])
            if hard == 0:
                direction[2, img > 0] = bboxi+1

            accumulation[0, img > 0] = 0
            accumulation[1, img > 0] = 0
            accumulation[2, img > 0] = 0
            accumulation = accumulation + direction
        vec = np.stack((accumulation[0], accumulation[1]))

        # compute weight
        weight = np.zeros((height, width), dtype=np.float32)
        posRegion = accumulation[2] > 0
        posCount = np.sum(posRegion)
        if posCount != 0:
            bboxRemain = 0
            for bboxi, polygon in enumerate(polygons):
                overlap_bboxi = accumulation[2] == (bboxi+1)
                overlapCount_bboxi = np.sum(overlap_bboxi)
                if overlapCount_bboxi == 0:
                    continue
                bboxRemain = bboxRemain+1
            bboxAve = float(posCount)/bboxRemain
            for bboxi, polygon in enumerate(polygons):
                overlap_bboxi = accumulation[2] == (bboxi+1)
                overlapCount_bboxi = np.sum(overlap_bboxi)
                if overlapCount_bboxi == 0:
                    continue
                pixAve = bboxAve/overlapCount_bboxi
                weight = weight*(~overlap_bboxi) + pixAve*overlap_bboxi
        # weight = weight[np.newaxis, ...]

        return image, vec, weight.astype(np.float32)

    def get_training_data(self, image, polygons, image_path):

        # H, W, _ = image.shape

        # for i, polygon in enumerate(polygons):
        #     if polygon.text != '#':
        #         polygon.find_bottom_and_sideline()
        # import time
        # start = time.time()
        if self.transform:
            image, polygons = self.transform(image, copy.copy(polygons))
        # print('Time transpose: ',time.time()-start,'s')
        # print('image ',image.dtype)
        # tcl_mask = np.zeros(image.shape[:2], np.uint8)
        # # radius_map = np.zeros(image.shape[:2], np.float32)
        # # sin_map = np.zeros(image.shape[:2], np.float32)
        # # cos_map = np.zeros(image.shape[:2], np.float32)

        # for i, polygon in enumerate(polygons):
        #     if polygon.text != '#':
        #         # sideline1, sideline2, center_points, radius = polygon.disk_cover(
        #         #     n_disk=cfg.n_disk)
        #         sideline1, sideline2, center_points = polygon.disk_cover(
        #             n_disk=cfg.n_disk)
        #         self.make_text_center_line(
        #             sideline1, sideline2, center_points, tcl_mask)
        # tr_mask, train_mask = self.make_text_region(image, polygons)
        # start = time.time()
        image, vec, weight = self.make_text_vector(image, polygons)
        # print('Time make vec: ',time.time()-start,'s')
        # print(image_path.split('/')[-1].split('.jpg')[0])
        # np.save('./npy/'+image_path.split('/')[-1].split('.jpg')[0]+'.npy',vec)
        # to pytorch channel sequence
        image = image.transpose(2, 0, 1)

        return image, vec, weight
