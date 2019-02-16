import cv2
import numpy as np
def themvien(path_image, size, padColor = 0):
    image = cv2.imread(path_image, cv2.IMREAD_COLOR)
    # image = cv2.imread(os.path.join(Datadir, img))
    #RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, weight = image.shape[:2]

    new_height, new_weight = size[:2]
    if height > new_height or weight > new_weight:
        interp = cv2.INTER_AREA
        # print("interp ", interp)
        # print("weight= ", weight)
    else:
        interp = cv2.INTER_CUBIC

    aspect = weight / height
    if aspect > 1:
        weight_ = new_weight
        height_ = np.round(weight_ / aspect).astype(int)
        pad_vert = (new_height - height_) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1:
        height_ = new_height
        weight_ = np.round(height_ * aspect).astype(int)
        pad_horz = (new_weight - weight_) / 2
        pad_left, pad_right = (
            np.floor(pad_horz).astype(int),
            np.ceil(pad_horz).astype(int),
        )

        pad_bot, pad_top = 0, 0
    else:
        height_ = new_height
        weight_ = new_weight
        pad_bot, pad_horz, pad_left, pad_top, pad_right = 0, 0, 0, 0, 0

    if len(image.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)):
        padColor = [padColor] * 3
    scaled_img = cv2.resize(image, (weight_, height_), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(
        scaled_img,
        pad_top,
        pad_bot,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=padColor,
    )
    return scaled_img

if __name__ == "__main__":
    import os
    DIR = ['./data/onepiece/raw_data/', './crysta1lee/', './ha/', './mabichtram97/']

    count = 0
    for dir_ in DIR:
        list_img = os.listdir(dir_)
        for i in list_img:
            try:
                print(dir_ + i)
                img = themvien(dir_ + i, (256, 256), 0)
                count += 1
                cv2.imwrite('./data/square/' + str(count) + '.jpg', img)
            except:
                print("Fail")
