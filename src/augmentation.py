from typing import Any
from .utils.register import Register
import torchvision.transforms as transforms

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
        img, label = sample
        return (self.T(img), label)

@transforms_registry("random_flip")
class RandomFlip:
    def __init__(self, probability_vertical: float=0.5, probability_horizontal: float=0.5):
        self.T1 = transforms.RandomVerticalFlip(probability_vertical)
        self.T2 = transforms.RandomHorizontalFlip(probability_horizontal)

    def __call__(self, sample):
        img, label = sample
        img = self.T2(self.T1(img))
        return (img, label)

@transforms_registry("normalize")
class Normalize:
    def __init__(self, mean, std):
        self.T = transforms.Normalize(mean=mean, std=std)

    def __call__(self, sample):
        img, label = sample
        return (self.T(img), label)

@transforms_registry("to_tensor")
class ToTensor:
    def __init__(self):
        self.T = transforms.ToTensor()

    def __call__(self, sample):
        img, label = sample
        return (self.T(img), label)

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
            "name": "normalize", 
            "params": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        },
        {
            "name": "to_tensor", 
            "params": {}
        },
    ]

    composed_transforms = get_transforms_from_config(transforms_list)