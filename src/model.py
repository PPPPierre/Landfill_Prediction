import torch
import torch.nn as nn
import torchvision.models as models
from .utils.register import Register

model_register = Register()

def get_model_from_cfg(cfg: dict) -> torch.nn.Module:
    model_name = cfg['name']
    # Retrieve the model class from the registry
    model_cls = model_register[model_name]
    
    if model_cls is None:
        raise ValueError(f"Model {model_name} is not registered.")
    
    # Check if there are additional arguments provided in the config for model initialization
    model_args = cfg.get('params', {})
    if model_args is None:
        model_args = {}

    # Instantiate the model with provided arguments
    model_instance = model_cls(**model_args)
    
    return model_instance

@model_register("ResNet18")
class ResNet18(nn.Module):
    def __init__(self, freeze_backbone: bool=True) -> None:
        super(ResNet18, self).__init__()
        # Load pretrained resnet18
        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # remove last layer
        backbone = list(base_model.children())[:-1]
        self.backbone = nn.Sequential(*backbone)

        # Freeze the layers of backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # add a binary classification layer
        self.classifier = nn.Linear(in_features=512, out_features=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    from loss import BCEWithLogitsLoss
    from metrics import compute_metrics

    cfg = {
        "name": "ResNet18",
        "params": {}
    }

    model = get_model_from_cfg(cfg)
    BCEloss = BCEWithLogitsLoss()

    inputs = torch.randn(2, 3, 224, 224)
    labels = torch.randint(0, 2, (2, 1), dtype=torch.float32)
    outputs = model(inputs)
    loss = BCEloss(outputs, labels)

    print(loss)
