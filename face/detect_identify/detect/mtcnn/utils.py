import numpy as np


def iou(box, bbox, ismin=False):
    """
    :param box: true box
    :param bbox: other boxes
    :param ismin: Whether to use min mode
    :return: iou value
    """
    x1, y1, x2, y2 = box[0],  box[1],  box[2],  box[3]
    _x1, _y1, _x2, _y2 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    # the area
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (_x2 - _x1) * (_y2 - _y1)
    # find out the intersection
    xx1, yy1, xx2, yy2 = np.maximum(x1, _x1), np.maximum(y1, _y1), np.minimum(x2, _x2), np.minimum(y2, _y2)
    w, h = np.maximum(0, xx2-xx1), np.maximum(0, yy2-yy1)
    inter_area = w*h
    # the list to save the iou value
    iou_box = np.zeros([bbox.shape[0], ])
    zero_index = np.nonzero(inter_area == 0)
    no_zero = np.nonzero(inter_area)
    iou_box[zero_index] = 0
    if ismin:
        iou_box[no_zero] = inter_area[no_zero] / (np.minimum(area1, area2)[no_zero])
    else:
        iou_box[no_zero] = inter_area[no_zero] / ((area1 + area2 - inter_area)[no_zero])
    return iou_box


def nms(boxes, thresh=0.3, ismin=False):
    """
    :param boxes: 框
    :param thresh: 阈值
    :param ismin: 是否除以最小值
    :return: nms抑制后的框
    """
    if boxes.shape[0] == 0:  # 框为空时防止报错
        return np.array([])
    # 根据置信度从大到小排序(argsort默认从小到大，加负号从大到小）
    _boxes = boxes[(-boxes[:, 0]).argsort()]
    r_box = []  # 用于存放抑制后剩余的框
    while _boxes.shape[0] > 1:  # 当剩余框大与0个时
        r_box.append(_boxes[0])  # 添加第一个框
        abox = _boxes[0][11:]
        bbox = _boxes[1:][:, 11:]
        idxs = np.where(iou(abox, bbox, ismin) < thresh)  # iou小于thresh框的索引
        _boxes = _boxes[1:][idxs]  # 取出iou小于thresh的框
    if _boxes.shape[0] > 0:
        r_box.append(_boxes[0])  # 添加最后一个框
    return np.stack(r_box)


def to_square(boxes):
    """
    :param boxes: 长方形框
    :return: 正方形框
    """
    c_box = boxes.copy()  # 不改变原数组
    if c_box.shape[0] == 0:  # 没有盒子防止报错
        return np.array([])
    w = c_box[:, 3] - c_box[:, 1]  # 宽度
    h = c_box[:, 4] - c_box[:, 2]  # 高度
    max_len = np.maximum(w, h)  # 最长边
    # 计算正方形框
    c_box[:, 1] = c_box[:, 1] + w * 0.5 - max_len * 0.5
    c_box[:, 2] = c_box[:, 2] + h * 0.5 - max_len * 0.5
    c_box[:, 3] = c_box[:, 1] + max_len
    c_box[:, 4] = c_box[:, 2] + max_len
    return c_box


if __name__ == '__main__':
    box1 = [100, 100, 200, 200]
    bbox1 = np.array([[100, 90, 200, 200], [120, 120, 180, 180], [200, 200, 300, 300]])
    a = iou(box1, bbox1)
    print(a.shape)
    print(a)

