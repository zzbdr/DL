import os
from PIL import Image
from torchvision import transforms as tfs


def get_crop_size(crop_size, upscale=2):
    return crop_size - (crop_size % upscale)


def input_transform(img, idx, boxes, crop_size, upscale_factor=2):
    x1, y1, w, h = list(map(int, boxes[idx].strip().split()[1:]))
    img = img.crop([x1, y1, x1+w, y1+h])
    return tfs.Compose([
        tfs.CenterCrop(crop_size),
        tfs.Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC)
    ])(img)


def target_transform(img, idx, boxes, crop_size):
    x1, y1, w, h = list(map(int, boxes[idx].strip().split()[1:]))
    img = img.crop([x1, y1, x1 + w, y1 + h])
    return tfs.Compose([
        tfs.CenterCrop(crop_size)
    ])(img)


def generate_data(row_path, save_path, file_path, upscale_factor=4, divide=0.95):
    all_data = os.listdir(row_path)
    data_length = 30000
    train_stop = int(data_length * divide)
    crop_size = get_crop_size(128, upscale_factor)
    f = open(file_path)
    boxes = f.readlines()[2:]
    if not os.path.exists(os.path.join(save_path, "train")):
        os.makedirs(os.path.join(save_path, "train"))
    f_train = open(os.path.join(save_path, "train.txt"), "w")
    if not os.path.exists(os.path.join(save_path, "val")):
        os.makedirs(os.path.join(save_path, "val"))
    f_val = open(os.path.join(save_path, "val.txt"), "w")
    for t in range(0, train_stop):
        img = Image.open(os.path.join(row_path, all_data[t].strip()))
        label = img.copy()
        img = input_transform(img, t, boxes, crop_size, upscale_factor)
        label = target_transform(label, t, boxes, crop_size)
        if not os.path.exists(os.path.join(save_path, "train", "img")):
            os.makedirs(os.path.join(save_path, "train", "img"))
        img.save(os.path.join(save_path, "train", "img", "{}.jpg".format(t)))
        if not os.path.exists(os.path.join(save_path, "train", "label")):
            os.makedirs(os.path.join(save_path, "train", "label"))
        label.save(os.path.join(save_path, "train", "label", "{}.jpg".format(t)))
        f_train.write(f"{t}.jpg\n")
        f_train.flush()

    for v in range(train_stop, data_length):
        img = Image.open(os.path.join(row_path, all_data[v].strip()))
        label = img.copy()
        img = input_transform(img, v, boxes, crop_size, upscale_factor)
        label = target_transform(label, v, boxes, crop_size)
        if not os.path.exists(os.path.join(save_path, "val", "img")):
            os.makedirs(os.path.join(save_path, "val", "img"))
        img.save(os.path.join(save_path, "val", "img", "{}.jpg".format(v - train_stop)))
        if not os.path.exists(os.path.join(save_path, "val", "label")):
            os.makedirs(os.path.join(save_path, "val", "label"))
        label.save(os.path.join(save_path, "val", "label", "{}.jpg".format(v - train_stop)))
        f_val.write(f"{v - train_stop}.jpg\n")
        f_val.flush()


if __name__ == '__main__':
    generate_data(r"F:\celebA\img_celeba", r"T:\srgan", r"F:\celebA\Anno\list_bbox_celeba.txt")




