from PIL import Image

import torch
import torch.utils.data as data


class ClsDataset(data.Dataset):
    def __init__(self, txt_path, root, transform=None):
        fh = open(txt_path, 'r')
        lists = []
        for line in fh:
            line = line.rstrip()
            words = line.split(' ')
            lists.append((words[0], int(words[1])))

        self.lists = lists
        self.transform = transform
        self.root = root

    def __getitem__(self, index):
        imagename, label = self.lists[index]
        imagename = self.root + imagename
        img = Image.open(imagename)

        if self.transform is not None:
            img = self.transform(img)
        return img, label, imagename.split('/')[-1]

    def __len__(self):
        return len(self.lists)


def deTransform(mean, std, tensor):
    mean = torch.as_tensor(mean, dtype=torch.float32, device=tensor.device)
    std = torch.as_tensor(std, dtype=torch.float32, device=tensor.device)
    tensor.mul_(std[:, None, None]).add_(mean[:, None, None])
    return tensor


if __name__ == '__main__':
    pass