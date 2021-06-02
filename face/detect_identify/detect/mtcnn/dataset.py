import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as tfs
import torch


class FaceDataset(Dataset):
    def __init__(self, data_path, size=12, ty="train"):
        self.dataset = []
        self.path = data_path
        self.ty = ty
        self.size = size
        f = open(os.path.join(data_path, str(size), "{}.txt".format(ty)))
        self.dataset.extend(f.readlines())
        f.close()
        self.tfs = tfs.Compose([
            tfs.ToTensor(),
            tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.choice = ["negative", "positive", "part"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_name = self.dataset[index].strip().split()[0]
        idx = int(self.dataset[index].strip().split()[1])
        boxes = list(map(float, self.dataset[index].strip().split()[2:6]))
        landmarks = list(map(float, self.dataset[index].strip().split()[6:]))
        boxes = torch.as_tensor(boxes, dtype=torch.float)
        landmarks = torch.as_tensor(landmarks, dtype=torch.float)
        img = Image.open(os.path.join(self.path, str(self.size),
                                      self.ty, self.choice[idx], img_name))
        img = self.tfs(img)
        return img, boxes, landmarks, idx


if __name__ == '__main__':
    e = SRGANDataset(r"T:\mtcnn")
    a, b, c = e[0]
    print(b)
    print(c)












