import numpy as np
import errno
import os
import cv2

def mkdirs(newdir):
    try:
        if not os.path.exists(newdir):
            os.makedirs(newdir)
    except OSError as err:
        # Reraise the error unless it's about an already existing directory
        if err.errno != errno.EEXIST or not os.path.isdir(newdir):
            raise


def fill_hole(input_mask):
    h, w = input_mask.shape
    canvas = np.zeros((h + 2, w + 2), np.uint8)
    canvas[1:h + 1, 1:w + 1] = input_mask.copy()

    mask = np.zeros((h + 4, w + 4), np.uint8)

    cv2.floodFill(canvas, mask, (0, 0), 1)
    canvas = canvas[1:h + 1, 1:w + 1].astype(np.bool)

    return (~canvas | input_mask.astype(np.uint8))


def regularize_sin_cos(sin, cos):
    # regularization
    scale = np.sqrt(1.0 / (sin ** 2 + cos ** 2))
    return sin * scale, cos * scale


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def norm2(x, axis=None):
    if axis:
        return np.sqrt(np.sum(x ** 2, axis=axis))
    return np.sqrt(np.sum(x ** 2))


def cos(p1, p2):
    return (p1 * p2).sum() / (norm2(p1) * norm2(p2))


def vector_sin(v):
    assert len(v) == 2
    # sin = y / (sqrt(x^2 + y^2))
    l = np.sqrt(v[0] ** 2 + v[1] ** 2)
    return v[1] / l


def vector_cos(v):
    assert len(v) == 2
    # cos = x / (sqrt(x^2 + y^2))
    l = np.sqrt(v[0] ** 2 + v[1] ** 2)
    return v[0] / l


def find_bottom(pts):

    if len(pts) > 4:
        e = np.concatenate([pts, pts[:3]])
        candidate = []
        for i in range(1, len(pts) + 1):
            v_prev = e[i] - e[i - 1]
            v_next = e[i + 2] - e[i + 1]
            if cos(v_prev, v_next) < -0.7:
                candidate.append((i % len(pts), (i + 1) %
                                  len(pts), norm2(e[i] - e[i + 1])))

        if len(candidate) != 2 or candidate[0][0] == candidate[1][1] or candidate[0][1] == candidate[1][0]:
            # if candidate number < 2, or two bottom are joined, select 2 farthest edge
            mid_list = []
            for i in range(len(pts)):
                mid_point = (e[i] + e[(i + 1) % len(pts)]) / 2
                mid_list.append((i, (i + 1) % len(pts), mid_point))

            dist_list = []
            for i in range(len(pts)):
                for j in range(len(pts)):
                    s1, e1, mid1 = mid_list[i]
                    s2, e2, mid2 = mid_list[j]
                    dist = norm2(mid1 - mid2)
                    dist_list.append((s1, e1, s2, e2, dist))
            bottom_idx = np.argsort(
                [dist for s1, e1, s2, e2, dist in dist_list])[-2:]
            bottoms = [dist_list[bottom_idx[0]]
                       [:2], dist_list[bottom_idx[1]][:2]]
        else:
            bottoms = [candidate[0][:2], candidate[1][:2]]

    else:
        d1 = norm2(pts[1] - pts[0]) + norm2(pts[2] - pts[3])
        d2 = norm2(pts[2] - pts[1]) + norm2(pts[0] - pts[3])
        bottoms = [(0, 1), (2, 3)] if d1 < d2 else [(1, 2), (3, 0)]
    assert len(bottoms) == 2, 'fewer than 2 bottoms'
    return bottoms


def split_long_edges(points, bottoms):
    """
    Find two long edge sequence of and polygon
    """
    b1_start, b1_end = bottoms[0]
    b2_start, b2_end = bottoms[1]
    n_pts = len(points)

    i = b1_end + 1
    long_edge_1 = []
    while (i % n_pts != b2_end):
        long_edge_1.append((i - 1, i))
        i = (i + 1) % n_pts

    i = b2_end + 1
    long_edge_2 = []
    while (i % n_pts != b1_end):
        long_edge_2.append((i - 1, i))
        i = (i + 1) % n_pts
    return long_edge_1, long_edge_2


def find_long_edges(points, bottoms):
    b1_start, b1_end = bottoms[0]
    b2_start, b2_end = bottoms[1]
    n_pts = len(points)
    i = (b1_end + 1) % n_pts
    long_edge_1 = []

    while (i % n_pts != b2_end):
        start = (i - 1) % n_pts
        end = i % n_pts
        long_edge_1.append((start, end))
        i = (i + 1) % n_pts

    i = (b2_end + 1) % n_pts
    long_edge_2 = []
    while (i % n_pts != b1_end):
        start = (i - 1) % n_pts
        end = i % n_pts
        long_edge_2.append((start, end))
        i = (i + 1) % n_pts
    return long_edge_1, long_edge_2


def split_edge_seqence(points, long_edge, n_parts):

    edge_length = [norm2(points[e1] - points[e2]) for e1, e2 in long_edge]
    point_cumsum = np.cumsum([0] + edge_length)
    total_length = sum(edge_length)
    length_per_part = total_length / n_parts

    cur_node = 0  # first point
    splited_result = []

    for i in range(1, n_parts):
        cur_end = i * length_per_part

        while(cur_end > point_cumsum[cur_node + 1]):
            cur_node += 1

        e1, e2 = long_edge[cur_node]
        e1, e2 = points[e1], points[e2]

        # start_point = points[long_edge[cur_node]]
        end_shift = cur_end - point_cumsum[cur_node]
        ratio = end_shift / edge_length[cur_node]
        new_point = e1 + ratio * (e2 - e1)
        # print(cur_end, point_cumsum[cur_node], end_shift, edge_length[cur_node], '=', new_point)
        splited_result.append(new_point)

    # add first and last point
    p_first = points[long_edge[0][0]]
    p_last = points[long_edge[-1][1]]
    splited_result = [p_first] + splited_result + [p_last]
    return np.stack(splited_result)


class Coordinate():
    def __init__(self,x=0,y=0):
        self.x = x
        self.y = y
    def copy(self):
        return Coordinate(self.x,self.y)

def process_output(image, output, raw_img, threshold=0.5):
    height, width, _ = image.shape
    pred_x = output[0, :]
    pred_y = output[1, :]

    magnitude, angle = cv2.cartToPolar(pred_x, pred_y)

    thr = np.array([threshold])
    mask = cv2.compare(magnitude, thr, cv2.CMP_GT)
    mask = mask.astype(np.float32)


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
        # <=0.6 mean >40% opposite directions
        if ratio1 <= 0.6 and ratio2 <= 0.6:
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

    return seg2bbox(res, cv2.resize(raw_img, (width, height)), raw_img)



def seg2bbox(seg, img_resize, raw_img):
    list_cnt = []
    min_area = 9
    ratio_height, ratio_width = raw_img.shape[0] / \
        img_resize.shape[0], raw_img.shape[1]/img_resize.shape[1]
    if np.amax(seg) == 0:
        return list_cnt
    filtered_seg = 0
    for idx in range(int(np.amax(seg))):
        seg_mask = (seg == (idx+1)).astype(np.uint8)
        if np.sum(seg_mask) <= int(min_area):
            filtered_seg += 1
            continue
    if np.amax(seg) == filtered_seg:
        return list_cnt
    # print('max seg', np.amax(seg), 'filtered', filtered_seg)
    for idx in range(int(np.amax(seg))):
        seg_mask = (seg == (idx+1)).astype(np.uint8)
        # seg_mask = cv2.dilate(seg_mask, (5, 5), iterations=1)
        if np.sum(seg_mask) <= int(min_area):
            filtered_seg += 1
            continue
        _, contours, _ = cv2.findContours(seg_mask, 1, 2)
        maxc, maxc_idx = 0, 0

        for i in range(len(contours)):
            if len(contours[i]) > maxc:
                maxc = len(contours[i])
                maxc_idx = i
        cnt = contours[maxc_idx]
        cnt = cnt.squeeze()
        cnt = np.int0(cnt*[ratio_width, ratio_height])
        list_cnt.append(cnt)
    return list_cnt

