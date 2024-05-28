import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class RainNet(nn.Module):
    def __init__(self, input_shape=[8, 100, 150, 150], mode="regression"):
        super(RainNet, self).__init__()
        self.mode = mode
        in_channels = input_shape[1]

        self.conv1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop4 = nn.Dropout(0.5)

        self.conv5 = ConvBlock(512, 1024)
        self.drop5 = nn.Dropout(0.5)

        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = ConvBlock(1024, 512)

        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = ConvBlock(512, 256)

        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = ConvBlock(256, 128)

        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = ConvBlock(128, 64)

        self.conv10 = nn.Conv2d(64, 2, kernel_size=3, padding=1)

        if mode == "regression":
            self.final = nn.Conv2d(2, 1, kernel_size=1)
        elif mode == "segmentation":
            self.final = nn.Conv2d(2, 1, kernel_size=1)
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        drop4 = self.drop4(conv4)
        pool4 = self.pool4(drop4)

        conv5 = self.conv5(pool4)
        drop5 = self.drop5(conv5)

        up6 = self.up6(drop5)
        up6 = self.crop_and_concat(up6, conv4)
        conv6 = self.conv6(up6)

        up7 = self.up7(conv6)
        up7 = self.crop_and_concat(up7, conv3)
        conv7 = self.conv7(up7)

        up8 = self.up8(conv7)
        up8 = self.crop_and_concat(up8, conv2)
        conv8 = self.conv8(up8)

        up9 = self.up9(conv8)
        up9 = self.crop_and_concat(up9, conv1)
        conv9 = self.conv9(up9)

        conv10 = self.conv10(conv9)
        
        if self.mode == "regression":
            final_output = self.final(conv10) # [8, 1, 144, 144]
            global_avg_pool = F.adaptive_avg_pool2d(final_output, (1, 1))
            return global_avg_pool.view(global_avg_pool.size(0), -1),final_output  # Flatten to [batch_size, 1]
        elif self.mode == "segmentation":
            return self.sigmoid(self.final(conv10))

    def crop_and_concat(self, upsampled, bypass):
        """
        Crop the bypass input tensor to the same size as the upsampled tensor and concatenate them.
        """
        _, _, H, W = upsampled.size()
        bypass = self.center_crop(bypass, H, W)
        return torch.cat((upsampled, bypass), 1)

    def center_crop(self, layer, max_height, max_width):
        _, _, h, w = layer.size()
        xy1 = (h - max_height) // 2
        xy2 = (w - max_width) // 2
        return layer[:, :, xy1:(xy1 + max_height), xy2:(xy2 + max_width)]

# Example usage:
# input_tensor = torch.randn(8, 100, 150, 150)  # 8 samples, 100 channels, 150x150 spatial dimensions
# model = RainNet(input_shape=input_tensor.shape, mode="regression")
# output,final = model(input_tensor)
# print(output.shape)  # Should print torch.Size([8, 1])
# print(output)
# print(final.shape)
# print(final)