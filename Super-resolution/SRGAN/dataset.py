import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as tfs


class SRGANDataset(Dataset):
    def __init__(self, data_path, ty="train"):
        self.dataset = []
        self.path = data_path
        self.ty = ty
        f = open(os.path.join(data_path, "{}.txt".format(ty)))
        self.dataset.extend(f.readlines())
        f.close()
        self.tfs = tfs.Compose([
            tfs.ToTensor(),
            tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_name = self.dataset[index].strip()
        img = Image.open(os.path.join(self.path, self.ty, "img", img_name))
        label = Image.open(os.path.join(self.path, self.ty, "label", img_name))
        img = self.tfs(img)
        label = self.tfs(label)
        return img, label


if __name__ == '__main__':
    e = SRGANDataset(r"T:\srgan")
    a, b = e[0]
    print(a)












