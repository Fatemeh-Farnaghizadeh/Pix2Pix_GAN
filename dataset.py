import os
import numpy as np
import config

from PIL import Image
from torch.utils.data import Dataset, DataLoader

class MapDataset(Dataset):

    def __init__(self, root_dir,input_transform, target_transform):
        self.root_dir = root_dir
        self.list_files = os.listdir(root_dir)
        self.len = len(self.list_files)
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __len__(self):
        return self.len
        
    def __getitem__(self, index):
        file_name = self.list_files
        file_path = os.path.join(self.root_dir, file_name[index])
        image = np.array(Image.open(file_path))
        input_image = Image.fromarray(image[:, :600, :])
        target_image = Image.fromarray(image[:, 600:, :])
       
        return self.input_transform(input_image), self.target_transform(target_image)
    

if __name__ == "__main__":
    dataset = MapDataset("data/maps/train/", config.input_transform, config.target_transform)
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        print(x.shape)
        
        import sys
        
        sys.exit()