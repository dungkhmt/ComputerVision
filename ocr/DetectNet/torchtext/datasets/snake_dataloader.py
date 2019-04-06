import copy
import cv2
import os
import torch.utils.data as data
import scipy.io as io
import numpy as np
from PIL import Image
from torchtext.utils.config import config as cfg
from skimage.draw import polygon as drawpoly
from torchtext.utils.misc import find_bottom, find_long_edges, split_edge_seqence, \
    norm2, vector_cos, vector_sin
import math


def pil_load_img(path):
    image = Image.open(path)
    image = np.array(image)
    return image


class TextInstance(object):

    def __init__(self, points, orient, text):
        self.orient = orient
        self.text = text

        remove_points = []

        if len(points) > 4:
            # remove point if area is almost unchanged after removing it
            ori_area = cv2.contourArea(points)
            for p in range(len(points)):
                # attempt to remove p
                index = list(range(len(points)))
                index.remove(p)
                area = cv2.contourArea(points[index])
                if np.abs(ori_area - area) / ori_area < 0.017 and len(points) - len(remove_points) > 4:
                    remove_points.append(p)
            self.points = np.array(
                [point for i, point in enumerate(points) if i not in remove_points])
        else:
            self.points = np.array(points)

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
        radii = norm2(inner_points1 - center_points, axis=1)  # disk radius

        return inner_points1, inner_points2, center_points, radii

    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, item):
        return getattr(self, item)


class TextDataset(data.Dataset):

    def __init__(self, transform):
        super().__init__()

        self.transform = transform

    def parse_mat(self, mat_path):
        """
        .mat file parser
        :param mat_path: (str), mat file path
        :return: (list), TextInstance
        """
        annot = io.loadmat(mat_path)
        polygons = []
        for cell in annot['polygt']:
            x = cell[1][0]
            y = cell[3][0]
            text = cell[4][0] if len(cell[4]) > 0 else '#'
            ori = cell[5][0] if len(cell[5]) > 0 else '#'

            if len(x) < 4:  # too few points
                continue
            pts = np.stack([x, y]).T.astype(np.int32)
            polygons.append(TextInstance(pts, ori, text))

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

    def make_text_center_line(self, sideline1, sideline2, center_line, radius,
                              tcl_mask, radius_map, sin_map, cos_map, expand=0.5, shrink=0):

        # TODO: shrink 1/2 * radius at two line end
        for i in range(shrink, len(center_line) - 1 - shrink):

            c1 = center_line[i]
            c2 = center_line[i + 1]
            top1 = sideline1[i]
            top2 = sideline1[i + 1]
            bottom1 = sideline2[i]
            bottom2 = sideline2[i + 1]

            sin_theta = vector_sin(c2 - c1)
            cos_theta = vector_cos(c2 - c1)

            p1 = c1 + (top1 - c1) * expand
            p2 = c1 + (bottom1 - c1) * expand
            p3 = c2 + (bottom2 - c2) * expand
            p4 = c2 + (top2 - c2) * expand
            polygon = np.stack([p1, p2, p3, p4])

            self.fill_polygon(tcl_mask, polygon, value=1)
            self.fill_polygon(radius_map, polygon, value=radius[i])
            self.fill_polygon(sin_map, polygon, value=sin_theta)
            self.fill_polygon(cos_map, polygon, value=cos_theta)

    def make_text_distance(self, tr_mask, polygons):
        x_mask = np.zeros(tr_mask.shape[:2], np.float32)
        y_mask = np.zeros(tr_mask.shape[:2], np.float32)
        count_mask = np.zeros(tr_mask.shape[:2], np.int8)
        for polygon in polygons:
            points = polygon.points.astype(np.int32)
            min_x = int(np.max([np.amin(points[:, 0]), 0]))
            max_x = int(np.min([np.amax(points[:, 0]), tr_mask.shape[1]]))
            min_y = int(np.max([np.amin(points[:, 1]), 0]))
            max_y = int(np.min([np.amax(points[:, 1]), tr_mask.shape[0]]))
            for yC in range(min_y, max_y):
                for xC in range(min_x, max_x):
                    point_B = points[-1]
                    min_dist = 10000
                    nearest_point = []
                    found = False
                    glob_min_dist = cv2.pointPolygonTest(
                        points.astype(np.int32), (xC, yC), True)
                    if cv2.pointPolygonTest(points.astype(np.int32), (xC, yC), False) != -1 and tr_mask[yC][xC] == 1:
                        # (xC, yC) inside polygon => find nearest point of (xC, yC) on contour polygon
                        # => find D in AB , CD perpendicular AB
                        # A(xA,yA) B(xB, yB) C(xC, yC)
                        for i in range(len(points)):
                            # continue
                            point_A = point_B
                            point_B = points[i]
                            xA = float(point_A[0])
                            yA = float(point_A[1])
                            xB = float(point_B[0])
                            yB = float(point_B[1])
                            # a = np.array([[xB-xA, yB-yA], [yA-yB, xB-xA]])
                            # b = np.array(
                            #     [xC*(xB-xA)+yC*(yB-yA), xA*(yA-yB)+yA*(xB-xA)])
                            # if xA != xB or yA != yB:
                            #     point_D = np.linalg.solve(a, b)
                            # else:
                            #     point_D = np.array([xA, yA])
                            #     print(self.image_id)
                            #     print('points ', points)
                            a = xB-xA
                            b = yB-yA
                            c = xC*a+yC*b
                            d = xA*(-b)+yA*a
                            # print(type(a),type(b),type(c),type(d))
                            if a == 0 and b == 0:
                                point_D = [xA, yA]
                            elif a == 0:
                                point_D = [-d/b, c/b]
                            elif b == 0:
                                point_D = [c/a, d/a]
                            else:
                                square = (a**2+b**2)
                                point_D = [(a*c-b*d)/square, (a*d+b*c)/square]

                            xD = point_D[0]
                            yD = point_D[1]
                            if (xD-xA)*(xD-xB)+(yD-yA)*(yD-yB) > 0:
                                distA = (xC-xA)**2+(yC-yA)**2
                                distB = (xC-xB)**2+(yC-yB)**2
                                point_D = [
                                    xA, yA] if distA < distB else [xB, yB]
                            dist = math.sqrt((xD-xC)**2+(yD-yC)**2)
                            # if np.sum((point_A-point_D)*(point_B-point_D)) > 0:
                            #     distA = np.sum((point_A-np.array([xC, yC]))**2)
                            #     distB = np.sum((point_B-np.array([xC, yC]))**2)
                            #     point_D = point_A if distA < distB else point_B
                            # dist = np.sqrt(
                            #     np.sum((point_D-np.array([xC, yC]))**2))
                            # continue
                            if dist < min_dist:
                                min_dist = dist
                                found = True
                                nearest_point = point_D
                                if min_dist == glob_min_dist:
                                    # print('aaaaaaaaaaaaaa')
                                    break
                        if found:
                            count_mask[yC][xC] += 1
                            x_distance = xC - nearest_point[0]
                            y_distance = yC - nearest_point[1]
                            if x_distance == 0 and y_distance == 0:
                                continue
                            if x_mask[yC][xC] == 0:
                                # x_mask[yC, xC] = x_distance * \
                                #     1.0/tr_mask.shape[1]
                                x_mask[yC, xC] = x_distance / \
                                    np.sqrt(x_distance**2+y_distance**2)
                                # x_mask[yC][xC] = x_distance
                            if y_mask[yC][xC] == 0:
                                # y_mask[yC, xC] = y_distance * \
                                #     1.0/tr_mask.shape[0]
                                y_mask[yC, xC] = y_distance / \
                                    np.sqrt(x_distance**2+y_distance**2)
                                # y_mask[yC][xC] = y_distance
                            # print(np.sqrt(x_distance**2+y_distance**2),x_distance,y_distance)
                        # else:
                        #     print('LOLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL')
        # print(np.amax(count_mask))
        # print(np.amax(x_mask),np.amax(y_mask))
        return x_mask, y_mask

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

        return image, vec, weight

    def get_training_data(self, image, polygons, image_id, image_path, distance=True):
        self.image_id = image_id
        H, W, _ = image.shape
        distance = True
        for i, polygon in enumerate(polygons):
            if polygon.text != '#':
                polygon.find_bottom_and_sideline()

        if self.transform:
            image, polygons = self.transform(image, copy.copy(polygons))

        tcl_mask = np.zeros(image.shape[:2], np.uint8)
        radius_map = np.zeros(image.shape[:2], np.float32)
        sin_map = np.zeros(image.shape[:2], np.float32)
        cos_map = np.zeros(image.shape[:2], np.float32)

        for i, polygon in enumerate(polygons):
            if polygon.text != '#':
                sideline1, sideline2, center_points, radius = polygon.disk_cover(
                    n_disk=cfg.n_disk)
                self.make_text_center_line(
                    sideline1, sideline2, center_points, radius, tcl_mask, radius_map, sin_map, cos_map)
        tr_mask, train_mask = self.make_text_region(image, polygons)
        if distance:
            import time
            start = time.time()
            x_mask, y_mask = self.make_text_distance(tr_mask, polygons)
            dis_T = time.time()-start
            start = time.time()
            image, vec, weight = self.make_text_vector(image, polygons)
            vec_T = time.time()-start
            print(np.sum(vec[0]**2+vec[1]**2))
            print(np.sum(x_mask**2+y_mask**2)-np.sum(vec[0]**2+vec[1]**2))
            print('Time make vector: ', vec_T, 's ', 'make distance: ',
                  dis_T, 's', ' Fastest: ', dis_T/vec_T,weight.shape)
        else:
            x_mask, y_mask = [], []
        
        # img1 = np.zeros((512, 512), np.uint8)
        # for i in range(512):
        #     for j in range(512):
        #         if x_mask[i][j] != 0 or y_mask[i][j] != 0:
        #             img1[i][j] = 255
        # cv2.imwrite('3.png',img1)
        # img2 = np.zeros((512, 512), np.uint8)
        # for i in range(512):
        #     for j in range(512):
        #         if vec[0][i][j] != 0 or vec[1][i][j] != 0:
        #             img2[i][j] = 255
        # cv2.imwrite('4.png',img)
        # cv2.imshow(image_id+'_1',img1)
        # cv2.imshow(image_id+'_2',img2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # exit()
        # exit()
        # cv2.imshow('1',img*255)
        # cv2.imwrite('4.png',tr_mask.astype(np.int32)*255)
        # cv2.imwrite('5.png',tcl_mask.astype(np.int32)*255)
        # cv2.imshow('2',tr_mask.astype(np.int32)*255)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # exit()
        # cv2.namedWindow('1',cv2.WINDOW_NORMAL)
        # cv2.imshow('1',x_mask)
        # cv2.namedWindow('2',cv2.WINDOW_NORMAL)
        # cv2.imshow('2',tr_mask*255)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # exit()
        # to pytorch channel sequence
        image = image.transpose(2, 0, 1)

        # meta = {
        #     'image_id': image_id,
        #     'image_path': image_path,
        #     'Height': H,
        #     'Width': W
        # }
        return image, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, x_mask, y_mask

    def __len__(self):
        raise NotImplementedError()
