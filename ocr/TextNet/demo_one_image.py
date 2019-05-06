import torch
from PIL import Image
from torchtext.transforms import build_transforms
import numpy as np
from torchtext.models import init_model
import pickle
from functools import partial
import cv2
import queue
import glob


class Coordinate():
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def copy(self):
        return Coordinate(self.x, self.y)


def load_model(model, model_path):
    checkpoint = torch.load(model_path, pickle_module=pickle)
    pretrain_dict = checkpoint['state_dict']
    # model_dict = model.state_dict()
    # pretrain_dict = {k: v for k, v in pretrain_dict.items(
    # ) if k in model_dict and model_dict[k].size() == v.size()}
    # model_dict.update(pretrain_dict)
    # model.load_state_dict(model_dict)
    model.load_state_dict(pretrain_dict)
    print("Loaded pretrained weights from '{}'".format(model_path))
    return model


def write_image_result(name, mask):
    # return
    print(name)
    height, width = mask.shape[0], mask.shape[1]
    img = np.zeros((height, width, 3), np.uint8)
    for i in range(height):
        for j in range(width):
            if mask[i][j] != 0:
                img[i][j] = np.array([255, 255, 255])*mask[i][j]
    cv2.imwrite(name, img)


def process_output(image, output, raw_img, threshold, min_area):
    height, width, _ = image.shape
    pred_x = output[0, :]
    pred_y = output[1, :]

    write_image_result('./trash/x_pred.png', pred_x)
    write_image_result('./trash/y_pred.png', pred_y)

    magnitude, angle = cv2.cartToPolar(pred_x, pred_y)

    thr = np.array([threshold])
    mask = cv2.compare(magnitude, thr, cv2.CMP_GT)
    mask = mask.astype(np.float32)

    write_image_result('./trash/magnitude.png', magnitude)
    write_image_result('./trash/mask.png', mask/255)

    parent = np.zeros((height, width, 2), np.float32)
    ending = np.zeros((height, width), np.float32)
    merged_ending = np.zeros((height, width), np.float32)
    PI = np.pi
    for row in range(0, height):
        mask_p = mask[row]
        angle_p = angle[row]
        parent_p = parent[row]
        ending_p = ending[row]
        for col in range(0, width):
            if mask_p[col] == 255:
                if angle_p[col] < PI/8 or angle_p[col] >= 15*PI/8:
                    parent_p[col][0] = 1
                    parent_p[col][1] = 0
                    if row+1 <= height-1:
                        mask_pn = mask[row+1]
                        if mask_pn[col] == 0:
                            ending_p[col] = 1
                elif angle_p[col] >= PI/8 and angle_p[col] < 3*PI/8:
                    parent_p[col][0] = 1
                    parent_p[col][1] = 1
                    if row+1 <= height-1 and col+1 <= width-1:
                        mask_pn = mask[row+1]
                        if mask_pn[col+1] == 0:
                            ending_p[col] = 1
                elif angle_p[col] >= 3*PI/8 and angle_p[col] < 5*PI/8:
                    parent_p[col][0] = 0
                    parent_p[col][1] = 1
                    if col+1 <= width-1:
                        mask_pn = mask[row]
                        if mask_pn[col+1] == 0:
                            ending_p[col] = 1
                elif angle_p[col] >= 5*PI/8 and angle_p[col] < 7*PI/8:
                    parent_p[col][0] = -1
                    parent_p[col][1] = 1
                    if row-1 >= 0 and col+1 <= width-1:
                        mask_pn = mask[row-1]
                        if mask_pn[col+1] == 0:
                            ending_p[col] = 1
                elif angle_p[col] >= 7*PI/8 and angle_p[col] < 9*PI/8:
                    parent_p[col][0] = -1
                    parent_p[col][1] = 0
                    if row-1 >= 0:
                        mask_pn = mask[row-1]
                        if mask_pn[col] == 0:
                            ending_p[col] = 1
                elif angle_p[col] >= 9*PI/8 and angle_p[col] < 11*PI/8:
                    parent_p[col][0] = -1
                    parent_p[col][1] = -1
                    if row-1 >= 0 and col-1 >= 0:
                        mask_pn = mask[row-1]
                        if mask_pn[col-1] == 0:
                            ending_p[col] = 1
                elif angle_p[col] >= 11*PI/8 and angle_p[col] < 13*PI/8:
                    parent_p[col][0] = 0
                    parent_p[col][1] = -1
                    if col-1 >= 0:
                        mask_pn = mask[row]
                        if mask_pn[col-1] == 0:
                            ending_p[col] = 1
                elif angle_p[col] >= 13*PI/8 and angle_p[col] < 15*PI/8:
                    parent_p[col][0] = 1
                    parent_p[col][1] = -1
                    if row+1 <= height-1 and col-1 >= 0:
                        mask_pn = mask[row+1]
                        if mask_pn[col-1] == 0:
                            ending_p[col] = 1

    write_image_result('./trash/parent.png',
                       np.abs(parent[:, :, 0]*parent[:, :, 1]))

    write_image_result('./trash/ending.png', ending)

    p = Coordinate()
    pc = Coordinate()
    pt = Coordinate()
    visited = np.zeros((height, width), np.float32)
    dict_id = np.zeros((height, width, 2), np.float32)
    # blob lableing to construct trees encoded by P
    # get depth each pixel in text instance
    sup_idx = 1
    for row in range(0, height):
        mask_p = mask[row]
        visited_p = visited[row]
        for col in range(0, width):
            if mask_p[col] == 255 and visited_p[col] == 0:
                p.x = row
                p.y = col
                Q = queue.Queue()
                Q.put(p)
                while not Q.empty():
                    pc = Q.get()
                    parent_pc = parent[pc.x]
                    visited_pc = visited[pc.x]
                    dict_pc = dict_id[pc.x]
                    dict_pc[pc.y][0] = sup_idx
                    visited_pc[pc.y] = 1
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            pt.x = pc.x + dx
                            pt.y = pc.y + dy
                            if pt.x >= 0 and pt.x <= height-1 and pt.y >= 0 and pt.y <= width-1:
                                parent_pt = parent[pt.x]
                                visited_pt = visited[pt.x]
                                dict_pt = dict_id[pt.x]
                                if visited_pt[pt.y] == 0 and (parent_pt[pt.y][0] != 0 or parent_pt[pt.y][1] != 0):
                                    if parent_pt[pt.y][0] == -1*dx and parent_pt[pt.y][1] == -1*dy:
                                        Q.put(pt.copy())
                                        dict_pc[pc.y][1] = max(
                                            dict_pc[pc.y][1], dict_pt[pt.y][1]+1)
                                    elif parent_pc[pc.y][0] == 1*dx and parent_pc[pc.y][1] == 1*dy:
                                        Q.put(pt.copy())
                                        dict_pt[pt.y][1] = max(
                                            dict_pt[pt.y][1], dict_pc[pc.y][1]+1)
                    # Q.remove(Q[0])
                sup_idx += 1

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # fill hole in ending
    merged_ending = cv2.dilate(
        ending, element, iterations=1).astype(np.float32)
    # dilate ending
    for row in range(0, height):
        ending_p = ending[row]
        parent_p = parent[row]
        dict_p = dict_id[row]
        for col in range(0, width):
            if ending_p[col] == 1:
                for dilDepth in range(1, min(int(1*dict_p[col][1]-16), 12)+1):
                    p.x = row+int(parent_p[col][0]*dilDepth)
                    p.y = col+int(parent_p[col][1]*dilDepth)
                    if p.x >= 0 and p.x <= height-1 and pt.y >= 0 and pt.y <= width-1:
                        merged_ending_p = merged_ending[p.x]
                        merged_ending_p[p.y] = 1

    write_image_result('./trash/ending_merge.png', merged_ending)

    # find connected Components
    cctmp = merged_ending.astype(np.uint8)
    # ccnum: num component, cctmp: mask component
    ccnum, cctmp = cv2.connectedComponents(
        cctmp, connectivity=8, ltype=cv2.CV_16U)
    label = cctmp.astype(np.float32)

    sup_map_cc = np.zeros((sup_idx), np.int32)
    stat = np.zeros((ccnum, 8), np.int32)
    # calculate num stat each label and assign label each sup_idx
    for row in range(0, height):
        ending_p = ending[row]
        parent_p = parent[row]
        label_p = label[row]
        dict_p = dict_id[row]
        for col in range(0, width):
            if ending_p[col] == 1:
                dx = int(parent_p[col][0])
                dy = int(parent_p[col][1])
                cc_idx = int(label_p[col])
                sup_map_cc[int(dict_p[col][0])] = cc_idx
                if dx == 1 and dy == 0:  # down
                    stat[cc_idx][0] += 1
                if dx == 1 and dy == 1:  # down right
                    stat[cc_idx][1] += 1
                if dx == 0 and dy == 1:  # right
                    stat[cc_idx][2] += 1
                if dx == -1 and dy == 1:  # up right
                    stat[cc_idx][3] += 1
                if dx == -1 and dy == 0:  # up
                    stat[cc_idx][4] += 1
                if dx == -1 and dy == -1:  # up left
                    stat[cc_idx][5] += 1
                if dx == 0 and dy == -1:  # left
                    stat[cc_idx][6] += 1
                if dx == 1 and dy == -1:  # down left
                    stat[cc_idx][7] += 1

    cc_map_filted = np.zeros((ccnum), np.int32)
    filted_idx = 1
    # Filter unblanced Text
    for cc_idx in range(1, ccnum):
        dif1 = np.abs(stat[cc_idx][0] - stat[cc_idx][4])  # abs(down - up)
        # abs(down_right - up_left)
        dif2 = np.abs(stat[cc_idx][1] - stat[cc_idx][5])
        dif3 = np.abs(stat[cc_idx][2] - stat[cc_idx][6])  # abs(right - left)
        # abs(down_left - up_right)
        dif4 = np.abs(stat[cc_idx][3] - stat[cc_idx][7])
        sum1 = stat[cc_idx][0]+stat[cc_idx][1]+stat[cc_idx][2]+stat[cc_idx][3]
        sum2 = stat[cc_idx][4]+stat[cc_idx][5]+stat[cc_idx][6]+stat[cc_idx][7]
        difsum = np.abs(sum1-sum2)
        sum_total = sum1 + sum2
        ratio1 = float(difsum) / float(sum_total)
        ratio2 = float(dif1+dif2+dif3+dif4) / float(sum_total)
        # keep candidate have low ratio (high opposite directions)
        if ratio1 <= 0.9 and ratio2 <= 0.9:
            cc_map_filted[cc_idx] = filted_idx
            filted_idx += 1

    # filter candidate
    for row in range(0, height):
        dict_p = dict_id[row]
        label_p = label[row]
        for col in range(0, width):
            if label_p[col] == 0:
                label_p[col] = cc_map_filted[int(
                    sup_map_cc[int(dict_p[col][0])])]
            else:
                label_p[col] = cc_map_filted[int(label_p[col])]

    res = np.zeros((height, width), np.float32)
    element_ = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    # get result mask
    for i in range(1, filted_idx):
        clstmp = cv2.compare(label, np.array([i]), cv2.CMP_EQ)
        clstmp = cv2.dilate(clstmp, element_, iterations=1)
        clstmp = cv2.erode(clstmp, element_, iterations=1)
        clstmp = cv2.compare(clstmp, np.array([0]), cv2.CMP_GT)
        clstmp = clstmp.astype(np.float32)
        res = cv2.multiply(res, 1-clstmp/255)
        res = cv2.add(res, clstmp/255*i)

    # print('resmax', np.amax(res), np.sum(res))
    # res = cv2.resize(res, (raw_w, raw_h), 0, 0, cv2.INTER_NEAREST)
    seg2bbox(res, cv2.resize(raw_img, (width, height)), raw_img, min_area)


def seg2bbox(seg, img_resize, raw_img, min_area):
    ratio_height, ratio_width = raw_img.shape[0] / \
        img_resize.shape[0], raw_img.shape[1]/img_resize.shape[1]
    min_area = min_area/ratio_height/ratio_width
    print('min area:', min_area)
    # seg = seg.astype(np.uint8)
    if np.amax(seg) == 0:
        # f = open(bbox_dir+seg_lst[num][:-4]+'.txt','w')
        # f.close()
        return
    filtered_seg = 0
    for idx in range(int(np.amax(seg))):
        seg_mask = (seg == (idx+1)).astype(np.uint8)
        if np.sum(seg_mask) <= int(min_area):
            filtered_seg += 1
            continue
    if np.amax(seg) == filtered_seg:
        # f = open(bbox_dir+seg_lst[num][:-4]+'.txt','w')
        # f.close()
        return
    print('max seg', np.amax(seg), 'filtered', filtered_seg)
    # f = open(bbox_dir+seg_lst[num][:-4]+'.txt','w')
    colors = []
    for i in range(2000):
        colors.append((int(np.random.randint(0, 255)), int(np.random.randint(
            0, 255)), int(np.random.randint(0, 255))))
    for idx in range(int(np.amax(seg))):
        seg_mask = (seg == (idx+1)).astype(np.uint8)
        # seg_mask = cv2.dilate(seg_mask, (5, 5), iterations=1)
        if np.sum(seg_mask) <= int(min_area):
            filtered_seg += 1
            continue
        _, contours, _ = cv2.findContours(
            seg_mask,  cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # if len(contours) > 1:
        #     contours.sort(key=lambda x: cv2.contourArea(x), reverse=True)
        # cnt = contours[0]
        maxc, maxc_idx = 0, 0
        # hull =[]
        for i in range(len(contours)):
            if len(contours[i]) > maxc:
                maxc = len(contours[i])
                maxc_idx = i
        cnt = contours[maxc_idx]
        # print(cnt.squeeze())
        cnt = cnt.squeeze()
        cnt = np.int0(cnt*[ratio_width, ratio_height])
        # rect = cv2.minAreaRect(cnt)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box*[ratio_width, ratio_height])
        cv2.drawContours(raw_img, [cnt], 0, colors[idx], 2)
        # cv2.rectangle(raw_img, left_top,
        #               right_bottom, colors[idx], 3)
        # exit()
        # hull = cv2.convexHull(cnt)
        # cv2.drawContours(img_resize, cnt, -1, colors[idx], 3)
        # bbox = np.transpose(cnt, (1, 0, 2))
        # bbox = bbox[0].astype(np.float64)
        # bbox = bbox[:, ::-1]
        # if bbox.shape[0]*bbox.shape[1] < 8:
        #     continue
        # bbox = bbox.reshape(1, bbox.shape[0]*bbox.shape[1])
        # print(bbox)
    # img_resize = cv2.resize(img_resize, (raw_img.shape[1], raw_img.shape[0]))
    list_result = glob.glob('./result/result_*.jpg')
    new_id = len(list_result)+1
    cv2.imwrite('./result/result_'+str(new_id)+'.jpg', raw_img)


def test(path_image):
    import cv2
    im = Image.open(path_image)
    image = np.array(im)
    cv2.imwrite('./trash/img.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    image = cv2.imread('./trash/img.jpg')
    raw_h, raw_w, _ = image.shape
    transform = build_transforms(
        maxHeight=768, maxWidth=768, is_train=False)
    # im = torch.Tensor(im)
    img, _ = transform(np.copy(image), None)
    im = img.transpose(2, 0, 1)
    im = torch.Tensor(im).to('cuda').unsqueeze(0)
    model = init_model(name='se_resnext101_32x4d')
    load_model(
        model, 'log/se_resnext101_32x4d-final-text-net-total-text-768-2/quick_save_checkpoint_ep43.pth.tar')
    model = model.to('cuda')
    import time
    # while True:
    start = time.time()
    model.eval()
    with torch.no_grad():
        output = model(im)
    process_output(img, output[0].to('cpu').numpy(),
                   image, threshold=0.4, min_area=200)
    print('Time: ', time.time()-start, ' s')


if __name__ == "__main__":
    # test('./data/pdf-text/Images/Train/img17
    # 68.jpg')
    test('./data/total-text/Images/Test/img1.jpg')
    # img = cv2.imread('img68.jpg')
    # raw_h, raw_w, _ = img.shape
    # im = cv2.resize(img, (512, 512))
    # vec = np.load('./npy/img68.npy')
    # process_output(im, vec, raw_h, raw_w)
