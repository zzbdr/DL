import torch
import dataset
import os
import argparse
from torch.utils.data import DataLoader
import models
import time
import matplotlib.pyplot as plt


class Trainer:
    record = {"train_loss": [], "train_psnr": [], "val_loss": [], "val_psnr": []}
    x_epoch = []

    def __init__(self, args):
        self.args = args
        self.device = self.args.device
        self.net = models.ESPCN(3, 3)
        batch = self.args.batch
        self.train_loader = DataLoader(dataset.ESPCNDataset(self.args.data_path, "train"),
                                       batch_size=batch, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(dataset.ESPCNDataset(self.args.data_path, "val"),
                                     batch_size=batch, shuffle=False, drop_last=True)
        self.criterion = torch.nn.L1Loss()
        self.epoch = 0
        self.lr = 0.01
        self.best_psnr = 0.
        if self.args.resume:
            if not os.path.exists(self.args.save_path):
                print("No params, start training...")
            else:
                param_dict = torch.load(self.args.save_path)
                self.epoch = param_dict["epoch"]
                self.lr = param_dict["lr"]
                self.net.load_state_dict(param_dict["net_dict"])
                self.best_psnr = param_dict["best_psnr"]
                print("Loaded params from {}\n[Epoch]: {}   [lr]: {}    [best_psnr]: {}".format(self.args.save_path,
                                                                                                self.epoch, self.lr,
                                                                                                self.best_psnr))
        self.net.to(self.device)
        self.optimizer = torch.optim.Adam([
            {"params": self.net.conv.parameters()},
            {"params": self.net.up.parameters(), "lr": self.lr*0.1}
        ], lr=self.lr)
        # self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    @staticmethod
    def calculate_psnr(img1, img2):
        return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

    def train(self, epoch):
        self.net.train()
        train_loss = 0.
        train_loss_all = 0.
        psnr = 0.
        total = 0
        start = time.time()
        print("Start epoch: {}".format(epoch))
        for i, (img, label) in enumerate(self.train_loader):
            img = img.to(self.device)
            label = label.to(self.device)
            pre = self.net(img)
            loss = self.criterion(pre, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            train_loss_all += loss.item()
            psnr += self.calculate_psnr(pre, label).item()
            total += 1

            if (i+1) % self.args.interval == 0:
                end = time.time()
                print("[Epoch]: {}[Progress: {:.1f}%]time:{:.2f} loss:{:.5f} psnr:{:.4f}".format(
                    epoch, (i+1)*100/len(self.train_loader), end-start, train_loss/self.args.interval,
                    psnr/total
                ))
                train_loss = 0.
        print("Save params to {}".format(self.args.save_path1))
        param_dict = {
            "epoch": epoch,
            "lr": self.lr,
            "best_psnr": self.best_psnr,
            "net_dict": self.net.state_dict()
        }
        torch.save(param_dict, self.args.save_path)
        return train_loss_all/len(self.train_loader), psnr/total

    def val(self, epoch):
        self.net.eval()
        print("Test start...")
        val_loss = 0.
        psnr = 0.
        total = 0
        start = time.time()
        with torch.no_grad():
            for i, (img, label) in enumerate(self.train_loader):
                img = img.to(self.device)
                label = label.to(self.device)
                pre = self.net(img).clamp(0.0, 1.0)
                loss = self.criterion(pre, label)
                val_loss += loss.item()
                psnr += self.calculate_psnr(pre, label).item()
                total += 1

            mpsnr = psnr / total
            end = time.time()
            print("Test finished!")
            print("[Epoch]: {} time:{:.2f} loss:{:.5f} psnr:{:.4f}".format(
                epoch, end - start, val_loss / len(self.val_loader), mpsnr
            ))
            if mpsnr > self.best_psnr:
                self.best_psnr = mpsnr
                print("Save params to {}".format(self.args.save_path))
                param_dict = {
                    "epoch": epoch,
                    "lr": self.lr,
                    "best_psnr": self.best_psnr,
                    "net_dict": self.net.state_dict()
                }
                torch.save(param_dict, self.args.save_path1)
        return val_loss/len(self.val_loader), mpsnr

    def draw_curve(self, fig, epoch, train_loss, train_psnr, val_loss, val_psnr):
        ax0 = fig.add_subplot(121, title="loss")
        ax1 = fig.add_subplot(122, title="psnr")
        self.record["train_loss"].append(train_loss)
        self.record["train_psnr"].append(train_psnr)
        self.record["val_loss"].append(val_loss)
        self.record["val_psnr"].append(val_psnr)
        self.x_epoch.append(epoch)
        ax0.plot(self.x_epoch, self.record["train_loss"], "bo-", label="train")
        ax0.plot(self.x_epoch, self.record["val_loss"], "ro-", label="val")
        ax1.plot(self.x_epoch, self.record["train_psnr"], "bo-", label="train")
        ax1.plot(self.x_epoch, self.record["val_psnr"], "ro-", label="val")
        if epoch == 0:
            ax0.legend()
            ax1.legend()
        fig.savefig(r"./train_fig/train_{}.jpg".format(epoch))

    def lr_update(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr * 0.8
        self.lr = self.optimizer.param_groups[0]["lr"]
        print("===============================================")
        print("Learning rate has adjusted to {}".format(self.lr))


def main(args):
    t = Trainer(args)
    fig = plt.figure()
    for epoch in range(t.epoch, t.epoch + args.num_epochs):
        train_loss, train_psnr = t.train(epoch)
        val_loss, val_psnr = t.val(epoch)
        t.draw_curve(fig, epoch, train_loss, train_psnr, val_loss, val_psnr)
        if (epoch + 1) % 10 == 0:
            t.lr_update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training ESPCN with celebA")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--data_path", default=r"T:\3\espcn", type=str)
    parser.add_argument("--resume", default=False, type=bool)
    parser.add_argument("--num_epochs", default=100, type=int)
    parser.add_argument("--save_path", default=r"./weight01.pt", type=str)
    parser.add_argument("--save_path1", default=r"./weight00.pt", type=str)
    parser.add_argument("--interval", default=20, type=int)
    parser.add_argument("--batch", default=64, type=int)
    args1 = parser.parse_args()
    main(args1)




