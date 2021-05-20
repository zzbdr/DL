import os
from PIL import Image
from torchvision import transforms as tfs


def get_crop_size(crop_size, upscale):
    return crop_size - (crop_size % upscale)


def input_transform(crop_size, upscale_factor):
    return tfs.Compose([
        tfs.CenterCrop(crop_size),
        tfs.Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC)
    ])


def target_transform(crop_size):
    return tfs.Compose([
        tfs.CenterCrop(crop_size)
    ])


def generate_data(row_path, save_path, upscale_factor, divide=0.7):
    all_data = os.listdir(row_path)
    data_length = len(all_data)
    train_stop = int(data_length * divide)
    crop_size = get_crop_size(256, upscale_factor)
    img_transform = input_transform(crop_size, upscale_factor)
    label_transform = target_transform(crop_size)
    if not os.path.exists(os.path.join(save_path, "train")):
        os.makedirs(os.path.join(save_path, "train"))
    f_train = open(os.path.join(save_path, "train.txt"), "w")
    if not os.path.exists(os.path.join(save_path, "val")):
        os.makedirs(os.path.join(save_path, "val"))
    f_val = open(os.path.join(save_path, "val.txt"), "w")
    for t in range(0, train_stop):
        img = Image.open(os.path.join(row_path, all_data[t].strip()))
        label = img.copy()
        img = img_transform(img)
        label = label_transform(label)
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
        img = img_transform(img)
        label = label_transform(label)
        if not os.path.exists(os.path.join(save_path, "val", "img")):
            os.makedirs(os.path.join(save_path, "val", "img"))
        img.save(os.path.join(save_path, "val", "img", "{}.jpg".format(v - train_stop)))
        if not os.path.exists(os.path.join(save_path, "val", "label")):
            os.makedirs(os.path.join(save_path, "val", "label"))
        label.save(os.path.join(save_path, "val", "label", "{}.jpg".format(v - train_stop)))
        f_val.write(f"{v - train_stop}.jpg\n")
        f_val.flush()


if __name__ == '__main__':
    generate_data(r"F:\celebA\img_celeba", r"T:\espcn", 4)




