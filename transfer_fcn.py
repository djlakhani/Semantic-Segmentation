import torch.nn as nn
import torchvision.models as models

class FCNTransfer(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class

        # loading pretrained ResNet34 model
        self.resnet34_model_no_out = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.resnet34_model_no_out = nn.Sequential(*list(self.resnet34_model_no_out.children())[:-2])

        # freezing pretrained layer weights
        for param in self.resnet34_model_no_out.parameters():
            param.requires_grad = False
        
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)

    def forward(self, x):
        
        x1 = self.resnet34_model_no_out(x)

        y1 = self.bn1(self.relu(self.deconv1(x1)))
        y2 = self.bn2(self.relu(self.deconv2(y1)))
        y3 = self.bn3(self.relu(self.deconv3(y2)))
        y4 = self.bn4(self.relu(self.deconv4(y3)))
        y5 = self.bn5(self.relu(self.deconv5(y4)))

        score = self.classifier(y5)

        return score  # size=(N, n_class, H, W)
