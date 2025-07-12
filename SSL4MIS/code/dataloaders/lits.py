import os
import torch
import numpy as np
from torch.utils.data import Dataset
import h5py
from torchvision import transforms
from scipy import ndimage
from scipy.ndimage.interpolation import zoom

class LiTS(Dataset):
    """LiTS Dataset"""
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []

        if split == 'train':
            with open(os.path.join(self._base_dir, 'train.list'), 'r') as f:
                self.image_list = f.readlines()
        elif split == 'val':
            with open(os.path.join(self._base_dir, 'val.list'), 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(os.path.join(self._base_dir, 'test.list'), 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n', '') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(os.path.join(self._base_dir, 'data/slices', f'{image_name}.h5'), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label.astype(np.uint8)}
        if self.transform:
            sample = self.transform(sample)
        return sample

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if np.random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif np.random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {'image': image, 'label': label}
        return sample

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label