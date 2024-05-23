from torch import nn
from layers import *
from PIL import Image
import matplotlib.pyplot as plt

class Fourcaster(nn.Module):
    def __init__(
            self,
            n_channels,
            n_classes,
            kernels_per_layer,
            args,
            bilinear = True

    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        kernels_per_layer = kernels_per_layer
        self.bilinear = bilinear
        reduction_ratio = 16
        self.args = args
        self.inner_size = 12


        self.inc = DoubleConvDS(self.n_channels, 64, kernels_per_layer=kernels_per_layer)
        self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam4 = CBAM(512, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam5 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.outc = OutConv(64, self.n_classes)
        self.apply(self.init_weights)

        self.regression_model = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
        )

        self.projection = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=2, padding=1),  # 64 input channels
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((50, 50))
        )
    @staticmethod
    def plot_image(image, flag=None):
        
        image = image.cpu().detach().permute(1,2,0).numpy()
        plt.imshow(image)
        if flag == 'R':
            plt.savefig('Real Image')
        else:
            plt.savefig('Generated Image')
        #Fourcaster.plot_image(x[0])

    def init_weights(self, module):
        """ Initialize the weights more appropriately based on layer type. """
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            nn.init.xavier_normal_(module.weight)
            if getattr(module, 'bias') is not None:
                nn.init.constant_(module.bias, 0)

        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=3.0, std=self.args.initializer_range)


    def forward(self, x, args):
        

        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4)
        x5Att = self.cbam5(x5)

        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)

        # shape 확인 - reconstruction image?
        x = self.up4(x, x1Att)
        generated_image = self.outc(x)
        generated_image = generated_image.permute(0,2,3,1)

        
        # regression_logits = self.regression_model(x)
        # generated_image = self.projection(x)

        # regression logits

        return generated_image # , regression_logits 
    

class RainfallPredictor(nn.Module):
    def __init__(self):
        super(RainfallPredictor, self).__init__()
        self.conv1 = nn.Conv2d(100, 64, 3, padding=1)  # 100은 입력 채널 수
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.fc1 = nn.Linear(256 * 18 * 18, 512)  # 18은 풀링 후의 크기
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.reshape(-1, 256 * 18 * 18)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x



if __name__ == "__main__":

    image = torch.randn(8,3,250,250) # (batch, 3, 250, 250)
    model=Fourcaster(n_channels=3,n_classes=1,kernels_per_layer=1) # n_classes는 n_channels와 같은 역할을 함. 
    classification_logits, regression_logits = model(image)
    print(classification_logits.size())
    print(regression_logits.size())
