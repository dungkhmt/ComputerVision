import torch
from PIL import Image
from torchtext.transforms import build_transforms
import numpy as np
from torchtext.models import init_model
import pickle
from functools import partial
import cv2
import queue


class Coordinate():
    def __init__(self):
        self.x = 0
        self.y = 0


def load_model(model, model_path):
    checkpoint = torch.load(model_path, pickle_module=pickle)
    pretrain_dict = checkpoint['state_dict']
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items(
    ) if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    print("Loaded pretrained weights from '{}'".format(model_path))
    return model


def process_output(image, output, raw_h, raw_w):
    height, width, _ = image.shape
    pred_x = output[0, :]
    pred_y = output[1, :]
    # print(pred_x.max(),pred_y.max(),pred_x.shape)
    # img = np.zeros((512,512),np.uint8)
    # for i in range(512):
    #     for j in range(512):
    #         if pred_x[i][j]!=0 or pred_y[i][j]!=0:
    #             img[i][j] = 255

    # cv2.imwrite('3.png',img)
    magnitude, angle = cv2.cartToPolar(pred_x, pred_y)
    cv2.imwrite('10.png',magnitude*255)
    thr = np.array([0])
    mask = cv2.compare(magnitude, thr, cv2.CMP_GT)
    mask = mask.astype(np.float32)
    print(mask.max(),magnitude.max(),magnitude.min(),angle.max(),angle.min(),'aaaa',mask.dtype)
    cv2.imwrite('5.png',magnitude)
    cv2.imwrite('6.png',mask)
    # print(mask.shape, magnitude.shape, angle.shape, pred_x.shape, pred_y.shape)
    # print('h,w', height, width)
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
    img = np.zeros((512,512),np.uint8)    
    for i in range(512):
        for j in range(512):
            if parent[i][j][0]!=0 or parent[i][j][1]!=0:
                img[i][j] = 255

    cv2.imwrite('7.png',img)
    print('ending: ',ending.sum())
    img = np.zeros((512,512),np.uint8)    
    for i in range(512):
        for j in range(512):
            if ending[i][j]!=0 or ending[i][j]!=0:
                img[i][j] = 255

    cv2.imwrite('8.png',img)

    p = Coordinate()
    pc = Coordinate()
    pt = Coordinate()
    visited = np.zeros((height, width), np.float32)
    dic = np.zeros((height, width, 2), np.float32)

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
                    dict_pc = dic[pc.x]
                    dict_pc[pc.y][0] = sup_idx
                    visited_pc[pc.y] = 1
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            pt.x = pc.x + dx
                            pt.y = pc.y + dy
                            if pt.x >= 0 and pt.x <= height-1 and pt.y >= 0 and pt.y <= width-1:
                                parent_pt = parent[pt.x]
                                visited_pt = visited[pt.x]
                                dict_pt = dic[pt.x]
                                if visited_pt[pt.y] == 0 and (parent_pt[pt.y][0] != 0 or parent_pt[pt.y][1] != 0):
                                    if parent_pt[pt.y][0] == -1*dx and parent_pt[pt.y][1] == -1*dy:
                                        Q.put(pt)
                                        dict_pc[pc.y][1] = max(
                                            dict_pc[pc.y][1], dict_pt[pt.y][1]+1)
                                    elif parent_pc[pc.y][0] == 1*dx and parent_pc[pc.y][1] == 1*dy:
                                        Q.put(pt)
                                        dict_pt[pt.y][1] = max(
                                            dict_pt[pt.y][1], dict_pc[pc.y][1]+1)
                    # Q.remove(Q[0])
                sup_idx += 1
    
    # img = np.zeros((512,512),np.uint8)    
    # for i in range(512):
    #     for j in range(512):
    #         if dic[i][j][0]!=0 or dic[i][j][1]!=0:
    #             img[i][j] = 255

    # cv2.imwrite('4.png',img)
    
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    merged_ending = cv2.dilate(ending, element).astype(np.float32)

    for row in range(0, height):
        ending_p = ending[row]
        parent_p = parent[row]
        dict_p = dic[row]
        for col in range(0, width):
            if ending_p[col] == 1:
                for dilDepth in range(1, min(int(1*dict_p[col][1]-16), 12)+1):
                    p.x = row+int(parent_p[col][0]*dilDepth)
                    p.y = col+int(parent_p[col][1]*dilDepth)
                    if p.x >= 0 and p.x <= height-1 and pt.y >= 0 and pt.y <= width-1:
                        merged_ending_p = merged_ending[p.x]
                        merged_ending_p[p.y] = 1

    cctmp = merged_ending.astype(np.uint8)
    ccnum, cctmp = cv2.connectedComponents(
        cctmp, connectivity=8, ltype=cv2.CV_16U)
    label = cctmp.astype(np.float32)
    print(ccnum,'cccc')
    sup_map_cc = np.zeros((sup_idx), np.int32)
    stat = np.zeros((ccnum, 8), np.int32)
    for row in range(0, height):
        ending_p = ending[row]
        parent_p = parent[row]
        label_p = label[row]
        dict_p = dic[row]
        for col in range(0, width):
            if ending_p[col] == 1:
                dx = int(parent_p[col][0])
                dy = int(parent_p[col][1])
                cc_idx = int(label_p[col])
                sup_map_cc[int(dict_p[col][0])] = cc_idx
                if dx == 1 and dy == 0:
                    stat[cc_idx][0] += 1
                if dx == 1 and dy == 1:
                    stat[cc_idx][1] += 1
                if dx == 0 and dy == 1:
                    stat[cc_idx][2] += 1
                if dx == -1 and dy == 1:
                    stat[cc_idx][3] += 1
                if dx == -1 and dy == 0:
                    stat[cc_idx][4] += 1
                if dx == -1 and dy == -1:
                    stat[cc_idx][5] += 1
                if dx == 0 and dy == -1:
                    stat[cc_idx][6] += 1
                if dx == 1 and dy == -1:
                    stat[cc_idx][7] += 1

    cc_map_filted = np.zeros((ccnum), np.int32)
    filted_idx = 1
    for cc_idx in range(1, ccnum):
        dif1 = max(stat[cc_idx][0], stat[cc_idx][4]) - \
            min(stat[cc_idx][0], stat[cc_idx][4])
        dif2 = max(stat[cc_idx][1], stat[cc_idx][5]) - \
            min(stat[cc_idx][1], stat[cc_idx][5])
        dif3 = max(stat[cc_idx][2], stat[cc_idx][6]) - \
            min(stat[cc_idx][2], stat[cc_idx][6])
        dif4 = max(stat[cc_idx][3], stat[cc_idx][7]) - \
            min(stat[cc_idx][3], stat[cc_idx][7])
        sum1 = stat[cc_idx][0]+stat[cc_idx][1]+stat[cc_idx][2]+stat[cc_idx][3]
        sum2 = stat[cc_idx][4]+stat[cc_idx][5]+stat[cc_idx][6]+stat[cc_idx][7]
        difsum = max(sum1, sum2) - min(sum1, sum2)
        sum_total = sum1 + sum2
        ratio1 = float(difsum) / float(sum_total)
        ratio2 = float(dif1+dif2+dif3+dif4) / float(sum_total)
        print(ratio1,ratio2,'eeee',difsum,sum_total)
        if ratio1 <= 0.6 and ratio2 <= 0.6:
            #  cout<<ratio1<<","<<ratio2<<","<<sum<<endl;
            cc_map_filted[cc_idx] = filted_idx
            filted_idx += 1
    for row in range(0, height):
        dict_p = dic[row]
        label_p = label[row]
        for col in range(0, width):
            if label_p[col] == 0:
                label_p[col] = cc_map_filted[int(sup_map_cc[int(dict_p[col][0])])]
            else:
                label_p[col] = cc_map_filted[int(label_p[col])]
    print(label.max(),label.min(),'fffff')
    # Mat clstmp;
    # Mat res(height, width, CV_32FC1, Scalar(0));
    res = np.zeros((height, width), np.float32)
    element_ = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    print(filted_idx,'ddddd')
    for i in range(1, filted_idx):
        clstmp = cv2.compare(label, np.array([i]), cv2.CMP_EQ)
        print(clstmp.max(),clstmp.min(),'hhhhh')
        clstmp = cv2.dilate(clstmp, element_)
        clstmp = cv2.erode(clstmp, element_)
        clstmp = cv2.compare(clstmp, np.array([0]), cv2.CMP_GT)
        clstmp = clstmp.astype(np.float32)
        print(clstmp.max(),clstmp.min(),'kkkkk')
        res = cv2.multiply(res, 1-clstmp/255)
        res = cv2.add(res, clstmp/255*i)
    print(res.max(),res.min(),'lllllllllllllllll')
    res = cv2.resize(res, (raw_w, raw_h), 0, 0, cv2.INTER_NEAREST)
    print(res.max(),res.min(),'ggggg')
    cv2.imshow('1', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()


def test(path_image):
    import cv2
    im = Image.open(path_image)
    im = np.array(im)
    raw_h, raw_w, _ = im.shape
    transform = build_transforms(
        maxHeight=512, maxWidth=512, is_train=False)
    # im = torch.Tensor(im)
    img, _ = transform(im, None)
    im = img.transpose(2, 0, 1)
    im = torch.Tensor(im).to('cuda').unsqueeze(0)
    im = torch.cat([im, im, im], dim=0)
    model = init_model(name='resnet50')
    load_model(model, 'log/resnet50-textfield/quick_save_checkpoint_ep1.pth.tar')
    model = model.to('cuda')
    import time
    while True:
        start = time.time()
        with torch.no_grad():
            output = model(im)
        process_output(img, output[0].to('cpu').numpy(), raw_h, raw_w)
        print('Time: ', time.time()-start, ' s')


if __name__ == "__main__":
    # test('img11.jpg')
    img = cv2.imread('img126.jpg')
    raw_h, raw_w, _ = img.shape
    im = cv2.resize(img,(512,512))
    vec = np.load('./npy/img126.npy')
    process_output(im,vec,raw_h,raw_w)
