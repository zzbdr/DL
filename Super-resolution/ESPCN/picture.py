import models
import torch
from PIL import Image
import torchvision.transforms as tfs
import os
import numpy as np
import argparse


class Tester:
    def __init__(self, args):
        self.args = args
        self.device = self.args.device
        self.net = models.ESPCN(3, 3)
        assert os.path.isfile(self.args.save_path), "Error: no params"
        param_dict = torch.load(self.args.save_path)
        print("Load params from {}\n[Epoch]: {}|[lr]: {}|[best_psnr]: {}".format(self.args.save_path,
                                                                                 param_dict["epoch"],
                                                                                 param_dict["lr"],
                                                                                 param_dict["best_psnr"]))
        self.net.load_state_dict(param_dict["net_dict"])
        self.net.to(self.device)
        self.net.eval()
        self.tfs = tfs.Compose([
            tfs.ToTensor(),
            # tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def run(self, img):
        with torch.no_grad():
            img = self.tfs(img).to(self.device)
            pre = self.net(img[None, ...])[0]
            # pre = self.UnNormalize(pre)
            # print(torch.max(pre))
            out_img = tfs.ToPILImage()(pre).convert("RGB")
            # output = pre.cpu().data.numpy()[0]
            # output = output.astype(np.uint8)
            # print(output.shape)
            # out_img = Image.fromarray(output.transpose(1, 2, 0))
            out_img.save(self.args.result_path)

    @staticmethod
    def UnNormalize(img, mean=None, std=None):
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        for i in range(3):
            img[i] = std[i]*img[i] + mean[i]
        return img

def main(args):
    t = Tester(args)
    img = Image.open(args.img_path)
    t.run(img)


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="Input a lr_img to get a hr_img")
    parse.add_argument("--device", default="cpu", type=str)
    parse.add_argument("--save_path", default=r"./weight1.pt", type=str)
    parse.add_argument("--result_path", default=r"./result.png", type=str)
    parse.add_argument("--img_path", default=r"./0.png", type=str)
    args1 = parse.parse_args()
    main(args1)


