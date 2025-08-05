import torch.nn as nn
import torch

class FCN(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class

        # for max pooling
        self.max1 = nn.MaxPool2d(kernel_size=2)

        # moving downwards
        self.conv_block1 = self.conv_block(3, 64)
        self.conv_block2 = self.conv_block(64, 128)
        self.conv_block3 = self.conv_block(128, 256)
        self.conv_block4 = self.conv_block(256, 512)

        # hit rockbottom
        self.bottom_block = self.conv_block(512, 1024)

        # moving upwards
        self.convt1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.deconv1 = self.conv_block(1024, 512)

        self.convt2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.deconv2 = self.conv_block(512, 256)

        self.convt3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.deconv3 = self.conv_block(256, 128)

        self.convt4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.deconv4 = self.conv_block(128, 64)

        self.convt5 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        # Final output layer
        self.classifier = nn.Conv2d(64, self.n_class, kernel_size=1) # change back


    def conv_block(self, in_channels, out_channels):
        # keeping padding = 1 to preserve shape
        conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        relu1 = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        relu2 = nn.ReLU(inplace=True)
        return nn.Sequential(*[conv1, relu1, conv2, relu2])

   
    def forward(self, x):
        # encoding
        x1 = self.conv_block1(x)       
        x2 = self.conv_block2(self.max1(x1))
        x3 = self.conv_block3(self.max1(x2))
        x4 = self.conv_block4(self.max1(x3))
        b1 = self.bottom_block(self.max1(x4))

        # decoding
        y1 = self.convt1(b1)
        y1 = torch.cat((y1, x4), dim=1)
        y1 = self.deconv1(y1)
        y2 = self.convt2(y1)
        y2 = torch.cat((y2, x3), dim=1)
        y2 = self.deconv2(y2)
        y3 = self.convt3(y2)
        y3 = torch.cat((y3, x2), dim=1)
        y3 = self.deconv3(y3)
        y4 = self.convt4(y3)
        y4 = torch.cat((y4, x1), dim=1)
        y4 = self.deconv4(y4)

        score = self.classifier(y4)

        return score  # size=(N, n_class, H, W)
