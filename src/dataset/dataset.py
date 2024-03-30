import os
import torch
import torchvision
import torchvision.transforms.functional as TF


class CatData:
    def __init__(self, root, train: bool = True, limit: int = None):
        if train:
            self.img_name = sorted(os.listdir(root))[: limit - 1]
        else:
            self.img_name = sorted(os.listdir(root))[-limit - 1 : -1]
        self.root = root

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, index):
        return TF.normalize(
            torchvision.io.read_image(f"{self.root}/{self.img_name[index]}").to(torch.float32) / 255.0,
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
        )


class BaseData:
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        if image.min() < 0:
            normalized_image = (image + 1) / 2
        else:
            normalized_image = image

        data = {'images': normalized_image}
        return data