import pylab as pl
import os
import time

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.optim import lr_scheduler

from torchtext.datasets.snake_total_text import TotalText
from torchtext.losses.snake_loss import TextLoss
from torchtext.models.snake_net import TextNet
from torchtext.utils.augmentation import BaseTransform, Augmentation
from torchtext.utils.config import config as cfg, update_config, print_config
from torchtext.utils.misc import AverageMeter
from torchtext.utils.misc import mkdirs, to_device
from torchtext.utils.option import BaseOptions
import torch.nn.functional as F

from PIL import Image
import numpy as np
import fiona
import shapely.geometry as geometry
from descartes import PolygonPatch
from scipy.spatial import Delaunay
from shapely.ops import cascaded_union, polygonize
# from torchtext.utils.visualize import visualize_network_output


def save_model(model, epoch, lr):

    save_dir = os.path.join(cfg.save_dir, cfg.exp_name)
    if not os.path.exists(save_dir):
        mkdirs(save_dir)

    save_path = os.path.join(
        save_dir, 'textsnake_{}_{}.pth'.format(model.backbone_name, epoch))
    print('Saving to {}.'.format(save_path))
    state_dict = {
        'lr': lr,
        'epoch': epoch,
        'model': model.state_dict()
    }
    torch.save(state_dict, save_path)


def train(model, train_loader, criterion, scheduler, optimizer, epoch):

    start = time.time()
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    model.train()

    for i, (img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, x_mask, y_mask) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if i > 5:
            break
        continue
        # start = time.time()
        # print('Time: ',end-start,'s')

        # print('img ', img.size(), 'train_mask ', train_mask.size(), 'tr_mask ', tr_mask.size(
        # ), 'tcl_mask ', tcl_mask.size(), 'radius_map ', radius_map.size(), 'sin ', sin_map.size(), 'cos ', cos_map.size())
        img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, x_mask, y_mask = to_device(
            img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, x_mask, y_mask)

        output = model(img)
        # print(output.size(),img.size())
        # tr_loss, tcl_loss, sin_loss, cos_loss, radii_loss = \
        #     criterion(output, tr_mask, tcl_mask, sin_map,
        #               cos_map, radius_map, train_mask, x_mask, y_mask)
        # loss = tr_loss + tcl_loss + sin_loss + cos_loss + radii_loss

        tr_loss, tcl_loss, x_loss, y_loss = \
            criterion(output, tr_mask, tcl_mask, sin_map,
                      cos_map, radius_map, train_mask, x_mask, y_mask)
        loss = tr_loss + tcl_loss + x_loss + y_loss

        # backward
        scheduler.step()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item())
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if cfg.viz and i < cfg.vis_num:
        #     visualize_network_output(output, tr_mask, tcl_mask, prefix='train_{}'.format(i))

        # if i % cfg.display_freq == 0:
        #     print('Epoch: [ {} ][ {:03d} / {:03d} ] - Loss: {:.4f} - tr_loss: {:.4f} - tcl_loss: {:.4f} - sin_loss: {:.4f} - cos_loss: {:.4f} - radii_loss: {:.4f} '.format(
        #         epoch, i, len(train_loader), loss.item(), tr_loss.item(), tcl_loss.item(), sin_loss.item(), cos_loss.item(), radii_loss.item())
        #     )

        if i % cfg.display_freq == 0:
            print('Epoch: [ {} ][ {:03d} / {:03d} ] - Loss: {:.4f} - tr_loss: {:.4f} - tcl_loss: {:.4f} - x_loss: {:.4f} - y_loss: {:.4f} - loss_avg: {:.4f}'.format(
                epoch, i, len(train_loader), loss.item(), tr_loss.item(), tcl_loss.item(), x_loss.item(), y_loss.item(), losses.avg)
            )
            # print(len(scheduler.get_lr()))
    print('Time use: ',time.time()-start,'s')
    if epoch % cfg.save_freq == 0 and epoch > 0:
        save_model(model, epoch, scheduler.get_lr())

    print('Training Loss: {}'.format(losses.avg))


def validation(model, valid_loader, criterion):

    model.eval()
    losses = AverageMeter()

    for i, (img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, x_mask, y_mask) in enumerate(valid_loader):

        img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map = to_device(
            img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map)

        output = model(img)

        tr_loss, tcl_loss, sin_loss, cos_loss, radii_loss, x_loss, y_loss = \
            criterion(output, tr_mask, tcl_mask, sin_map,
                      cos_map, radius_map, train_mask, x_mask, y_mask)
        loss = tr_loss + tcl_loss + sin_loss + cos_loss + radii_loss+x_loss+y_loss
        losses.update(loss.item())

        # if cfg.viz and i < cfg.vis_num:
        #     visualize_network_output(output, tr_mask, tcl_mask, prefix='val_{}'.format(i))

        if i % cfg.display_freq == 0:
            print(
                'Validation: - Loss: {:.4f} - tr_loss: {:.4f} - tcl_loss: {:.4f} - sin_loss: {:.4f} - cos_loss: {:.4f} - radii_loss: {:.4f} - x_loss: {:.4f} - y_loss: {:.4f}'.format(
                    loss.item(), tr_loss.item(), tcl_loss.item(), sin_loss.item(),
                    cos_loss.item(), radii_loss.item(), x_loss.item(), y_loss.item())
            )
    print('Validation Loss: {}'.format(losses.avg))


def load_model(model, model_path):
    print('Loading from {}'.format(model_path))
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict['model'])


def plot_polygon(polygon):
    fig = pl.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    margin = .3

    x_min, y_min, x_max, y_max = polygon.bounds

    ax.set_xlim([x_min-margin, x_max+margin])
    ax.set_ylim([y_min-margin, y_max+margin])
    patch = PolygonPatch(polygon, fc='#999999',
                         ec='#000000', fill=True, zorder=-1)
    ax.add_patch(patch)
    return fig


def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set
    of points.
    @param points: Iterable container of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull

    # coords = np.array([point.coords[0] for point in points])
    coords = np.array(points)
    tri = Delaunay(coords)
    triangles = coords[tri.vertices]
    a = ((triangles[:, 0, 0] - triangles[:, 1, 0]) ** 2 +
         (triangles[:, 0, 1] - triangles[:, 1, 1]) ** 2) ** 0.5
    b = ((triangles[:, 1, 0] - triangles[:, 2, 0]) ** 2 +
         (triangles[:, 1, 1] - triangles[:, 2, 1]) ** 2) ** 0.5
    c = ((triangles[:, 2, 0] - triangles[:, 0, 0]) ** 2 +
         (triangles[:, 2, 1] - triangles[:, 0, 1]) ** 2) ** 0.5
    s = (a + b + c) / 2.0
    areas = (s*(s-a)*(s-b)*(s-c)) ** 0.5
    circums = a * b * c / (4.0 * areas)
    filtered = triangles[circums < (1.0 / alpha)]
    edge1 = filtered[:, (0, 1)]
    edge2 = filtered[:, (1, 2)]
    edge3 = filtered[:, (2, 0)]
    edge_points = np.unique(np.concatenate(
        (edge1, edge2, edge3)), axis=0).tolist()
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    a = cascaded_union(triangles)
    # print(list(a.boundary.coords))
    # exit()
    return cascaded_union(triangles), edge_points


def test(path_image):
    import cv2
    im = Image.open(path_image)
    im = np.array(im)
    transform = BaseTransform(size=cfg.input_size,
                              mean=cfg.means, std=cfg.stds)
    # im = torch.Tensor(im)
    im, _ = transform(im, None)
    im = im.transpose(2, 0, 1)
    im = torch.Tensor(im).to('cuda').unsqueeze(0)
    model = TextNet()
    load_model(model, './save/ohem_l1_keep_distance/textsnake_vgg16_40.pth')
    model = model.to(cfg.device)
    with torch.no_grad():
        output = model(im)
    tr_pred = output[0][:2].to('cpu').softmax(dim=0).numpy()
    tcl_pred = output[0][2:4].to('cpu').softmax(dim=0).numpy()
    x_pred = output[0][4].to('cpu').numpy()
    y_pred = output[0][5].to('cpu').numpy()
    print(np.amax(x_pred),np.amin(x_pred))
    print(np.amax(y_pred),np.amin(y_pred))
    # exit()
    _, tr_pred = cv2.threshold(tr_pred[1], 0.4, 1, cv2.THRESH_BINARY)
    _, tcl_pred = cv2.threshold(tcl_pred[1], 0.6, 1, cv2.THRESH_BINARY)
    tcl_pred = tcl_pred*tr_pred
    tcl_pred = tcl_pred.astype(np.uint8)*255
    #tcl_pred = cv2.dilate(tcl_pred.astype(np.uint8)*255,(5,5))
    # tcl_pred = cv2.morphologyEx(tcl_pred.astype(np.uint8)*255,cv2.MORPH_CLOSE, (5,5))
    cv2.imshow('1', tcl_pred) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    _, contours, hierarchy = cv2.findContours(tcl_pred.astype(
        np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img = np.zeros((512, 512), np.uint8)
    # cv2.imshow('1',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print(len(contours))
    cnt = contours[0]
    cnt = cnt.squeeze()
    # print(cnt)
    print(np.amax(cnt, axis=0))
    hulls = []
    for cnt in contours:
        cnt = cnt.squeeze()
        min_x, min_y = np.amin(cnt, axis=0)
        max_x, max_y = np.amax(cnt, axis=0)
        # print(min_x, max_x, min_y, max_y)
        cand = []
        for i in range(min_x, max_x+1):
            for j in range(min_y, max_y+1):
                if cv2.pointPolygonTest(cnt, (i, j), False) != -1:
                    y_coord = int(j + y_pred[j][i])
                    x_coord = int(i + x_pred[j][i])
                    # img[y_coord][x_coord] = 255
                    cand.append(np.array([x_coord, y_coord]))
        print('cand1: ', len(cand))
        concave_hull, edge_points = alpha_shape(cand, 0.01)
        cand = list(concave_hull.boundary.coords)
        hulls.append(np.array(cand))
        # print(len(edge_points))
        # polygon = PolygonPatch(concave_hull)
        # plt = plot_polygon(concave_hull)
        # plt.savefig('10.png')
        # exit()
    print(hulls[0].shape)
    img = cv2.imread(path_image)
    img = cv2.resize(img,(512,512))
    for hull in hulls:
        for i in range(len(hull)-1):
            # print(hull[0].shape)
            cv2.line(img, tuple(hull[i].astype(np.uint16)), tuple(hull[i+1].astype(np.uint16)), (255, 0, 0), 3)
            # cv2.drawContours(img, [hull], -1, (255, 0, 0), 1)
    cv2.imshow('1', img)
    cv2.imwrite('1.png', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()
    # tcl_pred = F.softmax(tcl_pred).numpy()
    '''cv2.namedWindow('2', cv2.WINDOW_NORMAL)
    for i in range(512):
        for j in range(512):
            if tcl_pred[i][j]:
                pass
    cv2.imshow('2', x_pred)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
    tcl_pred = tcl_pred.numpy()*255
    # for i in range(512):
    #    for j in range(512):
    #        if tcl_pred[i][j] >255 or tcl_pred[i][j]<0:
    #            tcl_pred[i][j] = 0
    cv2.imwrite('1.jpg', tcl_pred)
    tcl_pred = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)
    tcl_pred = tcl_pred/255
    ret, tcl_pred = cv2.threshold(tcl_pred, 0.5, 1, cv2.THRESH_BINARY)
    cv2.imshow('2', tcl_pred*255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():

    trainset = TotalText(
        data_root='data/total-text',
        ignore_list='./data/total-text/ignore_list.txt',
        is_training=True,
        transform=Augmentation(size=cfg.input_size,
                               mean=cfg.means, std=cfg.stds)
    )

    valset = TotalText(
        data_root='data/total-text',
        ignore_list=None,
        is_training=False,
        transform=BaseTransform(size=cfg.input_size,
                                mean=cfg.means, std=cfg.stds)
    )

    train_loader = data.DataLoader(
        trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = data.DataLoader(
        valset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # Model
    model = TextNet()

    model = model.to(cfg.device)
    if cfg.cuda:
        cudnn.benchmark = True

    criterion = TextLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=5e-4,
                                 betas=(0.9, 0.999), amsgrad=True)
    # scheduler = lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[20, 40], gamma=0.1)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)

    print('Start training TextSnake.')

    for epoch in range(cfg.start_epoch, cfg.max_epoch):
        train(model, train_loader, criterion, scheduler, optimizer, epoch)
        # validation(model, val_loader, criterion)

    print('End.')


if __name__ == "__main__":
    # parse arguments
    option = BaseOptions()
    args = option.initialize()

    update_config(cfg, args)
    print_config(cfg)

    # main
    main()
    # test('img94.jpg')
