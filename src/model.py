import torch
import torch.nn as nn
import torchvision.models as models
from outils.register import Register

model_register = Register()

@model_register("ResNet18")
class ResNet18(nn.Module):
    def __init__(self) -> None:
        super(ResNet18, self).__init__()
        # Load pretrained resnet18
        base_model = models.resnet18(pretrained=True)
        
        # remove last layer
        backbone = list(base_model.children())[:-1]
        self.backbone = nn.Sequential(*backbone)
        
        # add a binary classification layer
        self.classifier = nn.Linear(in_features=512, out_features=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    from loss import BCEWithLogitsLoss

    BCEloss = BCEWithLogitsLoss()
    model = ResNet18()

    inputs = torch.randn(2, 3, 224, 224)
    labels = torch.randint(0, 2, (2, 1), dtype=torch.float32)
    outputs = model(inputs)
    loss = BCEloss(outputs, labels)

    print(loss)






        