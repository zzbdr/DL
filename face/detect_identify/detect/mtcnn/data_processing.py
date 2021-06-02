import os
from PIL import Image
import numpy as np
import utils


def generate_data(row_path, save_path, data_size, divide=0.9, data_length=3000):
    all_data = os.listdir(os.path.join(row_path, "img_celeba"))
    train_stop = int(data_length * divide)
    with open(os.path.join(row_path, "Anno", "list_bbox_celeba.txt")) as f:
        boxes1 = f.readlines()[2:]
    with open(os.path.join(row_path, "Anno", "list_landmarks_celeba.txt")) as f:
        landmarks1 = f.readlines()[2:]
    if not os.path.exists(os.path.join(save_path, str(data_size), "train")):
        os.makedirs(os.path.join(save_path, str(data_size), "train"))
    f_train = open(os.path.join(save_path, str(data_size), "train.txt"), "w")
    if not os.path.exists(os.path.join(save_path, str(data_size), "val")):
        os.makedirs(os.path.join(save_path, str(data_size), "val"))
    f_val = open(os.path.join(save_path, str(data_size), "val.txt"), "w")
    for t in range(0, train_stop):
        img = Image.open(os.path.join(row_path, "img_celeba", all_data[t].strip()))
        boxes = boxes1[t].strip().split()
        x, y, w, h = int(boxes[1]), int(boxes[2]), int(boxes[3]), int(boxes[4])
        x1, y1, x2, y2 = x, y, x+w, y+h
        cx, cy = int(x+w/2), int(y+h/2)
        _cx, _cy = cx + np.random.randint(-0.2*w, 0.2*w+1), cy + np.random.randint(-0.2*h, 0.2*h+1)
        _w, _h = w + np.random.randint(-0.2*w, 0.2*w+1), h + np.random.randint(-0.2*h, 0.2*h+1)
        _x1, _y1, _x2, _y2 = int(_cx-_w/2), int(_cy-_h/2), int(_cx+_w/2), int(_cy+_h/2)
        landmarks = landmarks1[t].strip().split()
        ex1, ey1, ex2, ey2 = int(landmarks[1]), int(landmarks[2]), int(landmarks[3]), int(landmarks[4])
        nx1, ny1, mx1, my1 = int(landmarks[5]), int(landmarks[6]), int(landmarks[7]), int(landmarks[8])
        mx2, my2 = int(landmarks[9]), int(landmarks[10])
        nex1, ney1, nex2, ney2 = (ex1-_x1), (ey1-_y1), (ex2-_x1), (ey2-_y1)
        nnx1, nny1, nmx1, nmy1 = (nx1-_x1), (ny1-_y1), (mx1-_x1), (my1-_y1)
        nmx2, nmy2 = (mx2-_x1), (my2-_y1)
        crop_img = img.crop([_x1, _y1, _x2, _y2])
        crop_img = crop_img.resize([data_size, data_size])
        iou = utils.iou([x1, y1, x2, y2], np.array([[_x1, _y1, _x2, _y2]]))
        _x1_off, _y1_off, _x2_off, _y2_off = (x1-_x1)/_w, (y1-_y1)/_h, (x2-_x2)/_w, (y2-_y2)/_h
        if iou >= 0.65:
            if not os.path.exists(os.path.join(save_path, str(data_size), "train", r'positive')):
                os.makedirs(os.path.join(save_path, str(data_size), "train", r'positive'))
            crop_img.save(os.path.join(save_path, str(data_size), "train", r'positive', r'%s.jpg' % t))
            f_train.write(f"{t}.jpg 1 {_x1_off} {_y1_off} {_x2_off} {_y2_off} {nex1/_w} {ney1/_h} {nex2/_w} "
                          f"{ney2/_h} {nnx1/_w} {nny1/_h} {nmx1/_w} {nmy1/_h} {nmx2/_w} {nmy2/_h}\n")
        elif 0.35 < iou < 0.65:
            if not os.path.exists(os.path.join(save_path, str(data_size), "train", r'part')):
                os.makedirs(os.path.join(save_path, str(data_size), "train", r'part'))
            crop_img.save(os.path.join(save_path, str(data_size), "train", r'part', r'%s.jpg' % t))
            f_train.write(f"{t}.jpg 2 {_x1_off} {_y1_off} {_x2_off} {_y2_off} {nex1/_w} {ney1/_h} {nex2/_w} "
                          f"{ney2/_h} {nnx1/_w} {nny1/_h} {nmx1/_w} {nmy1/_h} {nmx2/_w} {nmy2/_h}\n")
        elif iou < 0.29:
            if not os.path.exists(os.path.join(save_path, str(data_size), "train", r'negative')):
                os.makedirs(os.path.join(save_path, str(data_size), "train", r'negative'))
            crop_img.save(os.path.join(save_path, str(data_size), "train", r'negative', r'%s.jpg' % t))
            f_train.write(f"{t}.jpg 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n")
        w, h = img.size
        _x1, _y1 = np.random.randint(0, w), np.random.randint(0, h)
        _w, _h = np.random.randint(0, w-x1), np.random.randint(0, h-y1)
        _x2, _y2 = x1+_w, y1+_h
        crop_img1 = img.crop([_x1, _y1, _x2, _y2])
        crop_img1 = crop_img1.resize((data_size, data_size))
        iou = utils.iou(np.array([x1, y1, x2, y2]), np.array([[_x1, _y1, _x2, _y2]]))
        if iou < 0.29:
            if not os.path.exists(os.path.join(save_path, str(data_size), "train", r'negative')):
                os.makedirs(os.path.join(save_path, str(data_size), "train", r'negative'))
            crop_img1.save(os.path.join(save_path, str(data_size), "train", r'negative', r'%s.jpg' % t))
            f_train.write(f"{t}.jpg 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n")

    for v in range(train_stop, data_length):
        img = Image.open(os.path.join(row_path, "img_celeba", all_data[v].strip()))
        boxes = boxes1[v].strip().split()
        x, y, w, h = int(boxes[1]), int(boxes[2]), int(boxes[3]), int(boxes[4])
        x1, y1, x2, y2 = x, y, x + w, y + h
        cx, cy = int(x + w / 2), int(y + h / 2)
        _cx, _cy = cx + np.random.randint(-0.2 * w, 0.2 * w + 1), cy + np.random.randint(-0.2 * h, 0.2 * h + 1)
        _w, _h = w + np.random.randint(-0.2 * w, 0.2 * w + 1), h + np.random.randint(-0.2 * h, 0.2 * h + 1)
        _x1, _y1, _x2, _y2 = int(_cx - _w / 2), int(_cy - _h / 2), int(_cx + _w / 2), int(_cy + _h / 2)
        landmarks = landmarks1[v].strip().split()
        ex1, ey1, ex2, ey2 = int(landmarks[1]), int(landmarks[2]), int(landmarks[3]), int(landmarks[4])
        nx1, ny1, mx1, my1 = int(landmarks[5]), int(landmarks[6]), int(landmarks[7]), int(landmarks[8])
        mx2, my2 = int(landmarks[9]), int(landmarks[10])
        nex1, ney1, nex2, ney2 = (ex1 - _x1), (ey1 - _y1), (ex2 - _x1), (ey2 - _y1)
        nnx1, nny1, nmx1, nmy1 = (nx1 - _x1), (ny1 - _y1), (mx1 - _x1), (my1 - _y1)
        nmx2, nmy2 = (mx2 - _x1), (my2 - _y1)
        crop_img = img.crop([_x1, _y1, _x2, _y2])
        crop_img = crop_img.resize([data_size, data_size])
        iou = utils.iou([x1, y1, x2, y2], np.array([[_x1, _y1, _x2, _y2]]))
        _x1_off, _y1_off, _x2_off, _y2_off = (x1 - _x1) / _w, (y1 - _y1) / _h, (x2 - _x2) / _w, (y2 - _y2) / _h
        if iou > 0.65:
            if not os.path.exists(os.path.join(save_path, str(data_size), "val", r'positive')):
                os.makedirs(os.path.join(save_path, str(data_size), "val", r'positive'))
            crop_img.save(os.path.join(save_path, str(data_size), "val", r'positive', r'%s.jpg' % v))
            f_val.write(f"{v}.jpg 1 {_x1_off} {_y1_off} {_x2_off} {_y2_off} {nex1 / _w} {ney1 / _h} {nex2 / _w} "
                          f"{ney2 / _h} {nnx1 / _w} {nny1 / _h} {nmx1 / _w} {nmy1 / _h} {nmx2 / _w} {nmy2 / _h}\n")
        elif iou < 0.29:
            if not os.path.exists(os.path.join(save_path, str(data_size), "val", r'negative')):
                os.makedirs(os.path.join(save_path, str(data_size), "val", r'negative'))
            crop_img.save(os.path.join(save_path, str(data_size), "val", r'negative', r'%s.jpg' % v))
            f_val.write(f"{v}.jpg 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n")
        w, h = img.size
        _x1, _y1 = np.random.randint(0, w), np.random.randint(0, h)
        _w, _h = np.random.randint(0, w - x1), np.random.randint(0, h - y1)
        _x2, _y2 = x1 + _w, y1 + _h
        crop_img1 = img.crop([_x1, _y1, _x2, _y2])
        crop_img1 = crop_img1.resize((data_size, data_size))
        iou = utils.iou(np.array([x1, y1, x2, y2]), np.array([[_x1, _y1, _x2, _y2]]))
        if iou < 0.29:
            if not os.path.exists(os.path.join(save_path, str(data_size), "val", r'negative')):
                os.makedirs(os.path.join(save_path, str(data_size), "val", r'negative'))
            crop_img1.save(os.path.join(save_path, str(data_size), "val", r'negative', r'%s.jpg' % v))
            f_val.write(f"{v}.jpg 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n")


if __name__ == '__main__':
    generate_data(r"F:\celebA", r"T:\mtcnn", 12)




