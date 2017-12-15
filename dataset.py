from os import listdir
from os.path import join

import torch.utils.data as data
import torchvision.transforms as transforms

from util import is_image_file, load_img


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir):
        super(DatasetFromFolder, self).__init__()
        self.content_path = join(image_dir, "content")
        self.style_path = join(image_dir, "style")
        self.target_path = join(image_dir, "target")
        self.image_filenames = [x for x in listdir(self.target_path) if is_image_file(x)]

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        # Load Image
        content = load_img(join(self.content_path, self.image_filenames[index]))
        content = self.transform(content)
        style = load_img(join(self.style_path, self.image_filenames[index]))
        style = self.transform(style)
        target = load_img(join(self.target_path, self.image_filenames[index]))
        target = self.transform(target)

        return content, style, target

    def __len__(self):
        return len(self.image_filenames)
