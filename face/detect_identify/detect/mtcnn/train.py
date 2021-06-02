import torch
import dataset
import os
import argparse
from torch.utils.data import DataLoader
import models
import time
import matplotlib.pyplot as plt


def net(name):
    if name == "pnet":
        return models.PNet()
    if name == "rnet":
        return models.RNet()
    if name == "onet":
        return models.ONet()


class Trainer:
    record = {"train_loss": []}
    x_epoch = []

    def __init__(self, args):
        self.args = args
        self.device = self.args.device
        self.net = net(self.args.net)
        batch = self.args.batch
        self.train_loader = DataLoader(dataset.FaceDataset(self.args.data_path, ty="train", size=self.args.size),
                                       batch_size=batch, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(dataset.FaceDataset(self.args.data_path, ty="val", size=self.args.size),
                                     batch_size=batch, shuffle=False, drop_last=True)
        self.criterion_box = torch.nn.MSELoss()
        self.criterion_conf = torch.nn.BCELoss()
        self.epoch = 0
        self.lr = 0.001
        self.best_miou = 0.
        if self.args.resume:
            if not os.path.exists(self.args.save_path):
                print("No params, start training...")
            else:
                param_dict = torch.load(self.args.save_path)
                self.epoch = param_dict["epoch"]
                self.lr = param_dict["lr"]
                self.net.load_state_dict(param_dict["net_dict"])
                self.best_miou = param_dict["best_miou"]
                print("Loaded params from {}\n[Epoch]: {}   [lr]: {}    [best_miou]: {}".format(self.args.save_path,
                                                                                                self.epoch, self.lr,
                                                                                                self.best_miou))
        self.net.to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    # @staticmethod
    # def iou(box, bbox):
    #     x1, y1, x2, y2 = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
    #     _x1, _y1, _x2, _y2 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    #     area1 = (x2 - x1).mul(y2 - y1)
    #     area2 = (_x2 - _x1).mul(_y2 - _y1)
    #     xx1, yy1, xx2, yy2 = torch.max(x1, _x1), torch.max(y1, _y1), torch.min(x2, _x2), torch.min(y2, _y2)
    #     w, h = (xx2 - xx1).clamp(min=0), (yy2 - yy1).clamp(min=0)
    #     inter_area = w.mul(h)
    #     iou_box = inter_area / (area1 + area2 - inter_area)
    #     return torch.mean(iou_box)

    def train(self, epoch):
        self.net.train()
        train_loss = 0.
        train_loss_all = 0.
        total = 0
        start = time.time()
        print("Start epoch: {}".format(epoch))
        for i, (img, boxes, landmarks, idx) in enumerate(self.train_loader):
            img = img.to(self.device)
            boxes = boxes.to(self.device)
            idx = idx.to(self.device)
            conf_idx = torch.as_tensor(idx < 2)
            conf_label = torch.masked_select(idx, conf_idx)
            boxes_idx = torch.as_tensor(idx > 0)
            boxes_label = boxes[boxes_idx]
            if self.args.net == "onet":
                landmarks = landmarks.to(self.device)
                off, land, conf = self.net(img)
                land_label = landmarks[boxes_idx]
                loss_off = self.criterion_box(off[boxes_idx].view(-1, 4), boxes_label.view(-1, 4))
                loss_land = self.criterion_box(land[boxes_idx].view(-1, 10), land_label.view(-1, 10))
                loss_conf = self.criterion_conf(conf[conf_idx].view(-1, 1), conf_label.float().view(-1, 1))
                loss = loss_off + loss_land + loss_conf
            else:
                off, conf = self.net(img)
                loss_off = self.criterion_box(off[boxes_idx].view(-1, 4), boxes_label.view(-1, 4))
                loss_conf = self.criterion_conf(conf[conf_idx].view(-1, 1), conf_label.float().view(-1, 1))
                loss = loss_off + loss_conf
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            train_loss_all += loss.item()
            total += 1

            if (i+1) % self.args.interval == 0:
                end = time.time()
                print("[Epoch]: {}[Progress: {:.1f}%]time:{:.2f} loss:{:.5f}".format(
                    epoch, (i+1)*100/len(self.train_loader), end-start, train_loss/self.args.interval
                ))
                train_loss = 0.
        print("Save params to {}".format(self.args.save_path))
        param_dict = {
            "epoch": epoch,
            "lr": self.lr,
            "best_miou": self.best_miou,
            "net_dict": self.net.state_dict()
        }
        torch.save(param_dict, self.args.save_path)
        return train_loss_all/len(self.train_loader)

    def draw_curve(self, fig, epoch, train_loss):
        ax0 = fig.add_subplot(title="loss")
        self.record["train_loss"].append(train_loss)
        self.x_epoch.append(epoch)
        ax0.plot(self.x_epoch, self.record["train_loss"], "ro-", label="train")
        if epoch == 0:
            ax0.legend()
        fig.savefig(r"./train_{}.jpg".format(self.args.net))


def main(args):
    t = Trainer(args)
    fig = plt.figure()
    for epoch in range(t.epoch, t.epoch + args.num_epochs):
        train_loss = t.train(epoch)
        t.draw_curve(fig, epoch, train_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training mtcnn with celebA")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--data_path", default=r"T:\mtcnn", type=str)
    parser.add_argument("--resume", default=False, type=bool)
    parser.add_argument("--num_epochs", default=100, type=int)
    parser.add_argument("--save_path", default=r"./rnet.pt", type=str)
    parser.add_argument("--interval", default=20, type=int)
    parser.add_argument("--batch", default=64, type=int)
    parser.add_argument("--net", default="rnet", type=str)
    parser.add_argument("--size", default=24, type=int)
    args1 = parser.parse_args()
    main(args1)




