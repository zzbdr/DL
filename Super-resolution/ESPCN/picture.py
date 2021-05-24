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

    @staticmethod
    def calculate_psnr(img1, img2):
        return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

    def init_model(self, img):
        init_img = torch.zeros([1, 3, img.shape[1], img.shape[0]])
        self.net(init_img)

    def run(self, img, bic=True):
        with torch.no_grad():
            if bic:
                bicubic = img.resize((img.width * 3, img.height * 3), resample=Image.BICUBIC)
                bicubic.save(self.args.bicubic_path)
            img = self.tfs(img).to(self.device)
            # label = self.tfs(label).to(self.device)
            pre = self.net(img[None, ...])[0].clamp(0.0, 1.0)
            out_img = tfs.ToPILImage()(pre)
            # psnr = self.calculate_psnr(pre, label).item()
            # psnr1 = self.calculate_psnr(tfs.ToTensor()(bicubic), label).item()
            # print("pre psnr: {}    bicubic psnr: {}".format(psnr, psnr1))
            # # pre = self.UnNormalize(pre)
            # print(torch.max(pre))
            # output = pre.cpu().data.numpy()[0]
            # output = output.astype(np.uint8)
            # print(output.shape)
            # out_img = Image.fromarray(output.transpose(1, 2, 0))
            return out_img

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
    img.show()
    # label = Image.open(args.label_path)
    out_img = t.run(img)
    out_img.show()
    # out_img.save(args.result_path)


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="Input a lr_img to get a hr_img")
    parse.add_argument("--device", default="cpu", type=str)
    parse.add_argument("--save_path", default=r"./weight01.pt", type=str)
    parse.add_argument("--result_path", default=r"./result.png", type=str)
    parse.add_argument("--img_path", default=r"./0.png", type=str)
    # parse.add_argument("--label_path", default=r"./0.jpg", type=str)
    parse.add_argument("--bicubic_path", default=r"./bicubic.jpg", type=str)
    args1 = parse.parse_args()
    main(args1)


