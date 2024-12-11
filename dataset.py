import torch
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import pickle
import gzip
import cv2
import re
import glob
from torchvision.transforms import v2


class BrightfieldMicroscopyDataset(Dataset):
    '''
    This is the dataset class for the Brightfield Microscopy dataset. 
    It is used to load the images and labels for training and validation.

    When using this class, you should provide the root_dir_images and root_dir_labels based on where these are located
    on your machine. 
    
    The train and validation flags are used to determine which dataset to load. 
    
    The transform parameter takes a pytorch v2 transform. It is important to note that only v2 transforms will work. 
    Here is an example transform. Please do not include any ToImage or to ToPILImage transforms as this would cause the 
    the transform to fail. 

        transform = v2.Compose([v2.ToDtype(torch.float32, scale=True),
                                v2.Resize(512)
                                2.RandomHorizontalFlip(p=0.5)])

    You can also choose your favourite random seed and how many data points you'd like to use for validation.

    Finally this dataset also provides the possibility of specifying which channels should be kept for training. Default is all.

    Args:
        root_dir_images (str): The path to the directory containing the images.
        root_dir_labels (str): The path to the directory containing the labels.
        train (bool): Flag to determine if the training dataset should be loaded.
        validation (bool): Flag to determine if the validation dataset should be loaded.
        transform (torchvision.transforms.Compose): A pytorch v2 transform to apply to the images and labels.
        num_validation (float): The percentage of the dataset to use for validation.
        channels_to_use (list): A list of integers specifying which channels to use for training.
        seed (int): A random seed for reproducibility.              
    '''

    def __init__(self, root_dir_images='data/brightfield/', root_dir_labels='data/masks', train=True, validation=False, transform=None, num_validation=0.1, channels_to_use=[0,1,2,3,4,5,6,7,8,9,10], seed=42):
        
        if not seed:
            raise RuntimeError("Please provide a seed for reproducibility.")
        
        if transform and not isinstance(transform, v2.Compose):
            raise RuntimeError("Please provide a valid v2 transform.")
        
        if num_validation > 1 or num_validation < 0:
            raise RuntimeError("Invalid percentage for validation. Must be between 0 and 1.")
        
        if not os.path.exists(root_dir_images) or not os.path.exists(root_dir_labels):
            raise RuntimeError(f"Invalid path to images or labels. Images: {root_dir_images}, Labels: {root_dir_labels}")
        
        if len(channels_to_use) > 11 or len(channels_to_use) < 1 or max(channels_to_use) > 10 or min(channels_to_use) < 0:
            raise RuntimeError(f"Invalid channels to use. Must be between 0 and 10. You either specified an invalid index, provided too many or too little channels.")
        
        np.random.seed(seed)
        self.generator = np.random.default_rng(seed=seed)
        self.root_dir_images = root_dir_images
        self.root_dir_labels = root_dir_labels

        self.transform = transform

        if train or validation:

            subfolders = sorted([f.path for f in os.scandir(self.root_dir_images) if re.search(r'well[2-7]', f.path)])
            self.image_files = [sorted(glob.glob(f"{data_path}/*.tif"), key=lambda x: int(re.search(r's(\d+)', x).group(1))) for data_path in subfolders]
            self.image_sets = [sorted(set(re.search(r'(s\d+)', f).group(1) for f in glob.glob(f"{sf}/*.tif"))) for sf in subfolders]
            self.mask_files = sorted([f.path for f in os.scandir(self.root_dir_labels) if re.search(r'well[2-7]', f.path)])
        
        else:

            subfolders = sorted([f.path for f in os.scandir(self.root_dir_images) if re.search(r'well[1]', f.path)])
            self.image_files = [sorted(glob.glob(f"{data_path}/*.tif"), key=lambda x: int(re.search(r's(\d+)', x).group(1))) for data_path in subfolders]
            self.image_sets = [sorted(set(re.search(r'(s\d+)', f).group(1) for f in glob.glob(f"{sf}/*.tif"))) for sf in subfolders]
            self.mask_files = sorted([f.path for f in os.scandir(self.root_dir_labels) if re.search(r'well[1]', f.path)])
        
        image_files_and_sets = list(zip(self.image_files, self.image_sets))

        self.data = []

        for well, identifiers in image_files_and_sets:
            well_id = re.search(r'(well\d+)', well[0]).group(1)
            for id in identifiers:
                image_set_path = sorted([f for f in well if id in f], key=lambda x: int(re.search(r'z(\d+)', x).group(1)))
                image_mask_path = [f for f in self.mask_files if id in f and well_id in f]
                image_set_path = [f for i, f in enumerate(image_set_path) if i in channels_to_use]

                if not image_set_path:
                    raise RuntimeError(f"No valid channels for channels_to_use: {channels_to_use} at identifier {id}")
                
                combined = (image_set_path, image_mask_path)
                self.data.append(combined)
        
        if train:

            self.generator.shuffle(self.data)
            self.data = self.data[:int(len(self.data)*(1-num_validation))]

        elif validation:

            self.generator.shuffle(self.data)
            self.data = self.data[int(len(self.data)*(1-num_validation)):]
        
    def __len__(self):
        return len(self.data) 
    
    def __getitem__(self, idx):
        images_paths, label = self.data[idx]

        images = [np.array(Image.open(f)) for f in images_paths]
        images = np.stack(images, axis=0)
        label = Image.open(label[0])
        label = np.array(label)

        # if not images.any() or not label.any():
        #     raise RuntimeError(f"Missing images or labels for index {idx}. Images: {images}, Label: {label}")

        images = torch.from_numpy(images)
        label = torch.from_numpy(label)

        if self.transform:
            images, label = self.transform(images, label)
        
        return images, label, images_paths