from typing import Any, Tuple
from .utils.register import Register

import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

"""
format of defining transforms in config.yaml file

transforms:
  - name: resize
    params:
        size: [224, 224]
        
  - name: random_crop
    params:
        size: [150, 150]
        
  - name: random_flip
    params:
        probability_vertical: 0.5
        probability_horizontal: 0.5

  - name: normalize
    params:
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
        
  - name: to_tensor
"""

transforms_registry = Register()

def create_transform_from_config(config):
    transform_class = transforms_registry[config["name"]]
    if transform_class is None:
        raise ValueError(f"Transform {config['name']} not recognized")
    return transform_class(**config.get("params", {}))

def get_transforms_from_config(transforms_config: list):
    transforms_list = [create_transform_from_config(tc) for tc in transforms_config]
    return transforms.Compose(transforms_list)

""" Define your transforms """

@transforms_registry("resize")
class Resize:
    def __init__(self, size):
        self.size = size
        self.T = transforms.Resize(size)

    def __call__(self, sample):
        if (isinstance(sample, tuple) or isinstance(sample, list)) and len(sample) == 2:
            img, label = sample
            return (self.T(img), label)
        else:
            return self.T(sample)

@transforms_registry("random_flip")
class RandomFlip:
    def __init__(self, probability_vertical: float=0.5, probability_horizontal: float=0.5):
        self.T1 = transforms.RandomVerticalFlip(probability_vertical)
        self.T2 = transforms.RandomHorizontalFlip(probability_horizontal)

    def __call__(self, sample):
        if (isinstance(sample, tuple) or isinstance(sample, list)) and len(sample) == 2:
            img, label = sample
            return (self.T2(self.T1(img)), label)
        else:
            return self.T2(self.T1(sample))
        
@transforms_registry("random_crop_and_scale")
class RandomCropAndScale:
    def __init__(self, crop_size_ratio: Tuple[float, float, float, float]):
        """
        Initialize the transformer.
        Args:
        - max_crop_percent (dict): A dictionary containing the maximum crop percentage. 
                                   It has keys 'top', 'bottom', 'left', and 'right'.
        """
        self.crop_size_ratio = np.clip(crop_size_ratio, 0, 0.5)

    def __call__(self, sample):
        """
        Apply the random crop transformation.
        
        Args:
        - sample (PIL Image or tuple): The input sample containing the image (and optionally the label).
        
        Returns:
        - PIL Image or tuple: The transformed image (and optionally the label).
        """
        # Check if the sample is an image-label pair.
        if (isinstance(sample, tuple) or isinstance(sample, list)) and len(sample) == 2:
            img, label = sample
        else:
            img = sample
            label = None

        # Get the dimensions of the image.
        _, img_width, img_height = img.size()

        # Calculate the maximum pixels that can be cropped from each side.
        top_crop = int(img_height * self.crop_size_ratio[0])
        bottom_crop = int(img_height * self.crop_size_ratio[1])
        left_crop = int(img_width * self.crop_size_ratio[2])
        right_crop = int(img_width * self.crop_size_ratio[3])

        # Randomly choose the number of pixels to crop from each side.
        top = np.random.randint(0, top_crop + 1)
        bottom = np.random.randint(0, bottom_crop + 1)
        left = np.random.randint(0, left_crop + 1)
        right = np.random.randint(0, right_crop + 1)

        # Calculate the cropping coordinates.
        top_left_x = left
        top_left_y = top
        bottom_right_x = img_width - right
        bottom_right_y = img_height - bottom

        # Crop the image.
        img_cropped = F.crop(img, top_left_y, top_left_x, bottom_right_y - top_left_y, bottom_right_x - top_left_x)

        # If there was a label, return the cropped image and the label, otherwise just the image.
        if label is not None:
            return img_cropped, label
        else:
            return img_cropped

@transforms_registry("normalize")
class Normalize:
    def __init__(self, mean, std):
        self.T = transforms.Normalize(mean=mean, std=std)

    def __call__(self, sample):
        if (isinstance(sample, tuple) or isinstance(sample, list)) and len(sample) == 2:
            img, label = sample
            return (self.T(img), label)
        else:
            return self.T(sample)

@transforms_registry("to_tensor")
class ToTensor:
    def __init__(self):
        self.T = transforms.ToTensor()

    def __call__(self, sample):
        if (isinstance(sample, tuple) or isinstance(sample, list)) and len(sample) == 2:
            img, label = sample
            return (self.T(img), label)
        else:
            return self.T(sample)

if __name__ == '__main__':

    transforms_list = [
        {
            "name": "resize", 
            "params": {"size": (224, 224)}
        },
        {
            "name": "random_flip", 
            "params": {"probability_vertical": 0.5, "probability_horizontal": 0.5}
        },
        {
            "name": "random_crop", 
            "params": {"max_crop_ratio": {"top":0.25, "bottom":0.25, "left":0.25, "right":0.25}}
        },
        {
            "name": "normalize", 
            "params": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        },
        {
            "name": "to_tensor", 
            "params": {}
        },
    ]

    composed_transforms = get_transforms_from_config(transforms_list)