import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad x1 to match the size of x2 if necessary
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, num_colors, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.num_colors = num_colors
        self.bilinear = bilinear

        # Color embedding layer
        self.color_embedding = nn.Linear(num_colors, 32)
        
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        
        # The bottleneck layer takes the original feature map + the color embedding
        self.down4 = Down(512, 1024)
        
        # --- CORRECTED LINES ---
        # Up1 input channels: 1024 (from bottleneck) + 32 (from color embedding) + 512 (from skip connection x4)
        self.up1 = Up(1024 + 32 + 512, 512, bilinear) 
        # Up2 input channels: 512 (from up1) + 256 (from skip connection x3)
        self.up2 = Up(512 + 256, 256, bilinear)
        # Up3 input channels: 256 (from up2) + 128 (from skip connection x2)
        self.up3 = Up(256 + 128, 128, bilinear)
        # Up4 input channels: 128 (from up3) + 64 (from skip connection x1)
        self.up4 = Up(128 + 64, 64, bilinear)
        # --- END OF CORRECTED LINES ---

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x, color_one_hot):
        # Forward pass for the color embedding
        color_emb = self.color_embedding(color_one_hot) # (batch_size, 32)
        
        # Encoder Path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Inject the color embedding at the bottleneck
        # The embedding needs to be tiled to match the spatial dimensions of x5
        batch_size, channels, H, W = x5.shape
        color_emb_tiled = color_emb.unsqueeze(2).unsqueeze(3).repeat(1, 1, H, W)
        x5_conditioned = torch.cat([x5, color_emb_tiled], dim=1)

        # Decoder Path
        x = self.up1(x5_conditioned, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# --- This block will only run when the script is executed directly ---
if __name__ == '__main__':
    print("--- Testing the UNet model architecture ---")
    try:
        # Define some dummy parameters for a test run
        in_channels = 1      # Grayscale input image
        out_channels = 3     # RGB output image
        num_colors = 8       # Assuming 8 unique colors in the dataset
        batch_size = 4
        image_size = (128, 128)

        # Instantiate the model
        model = UNet(n_channels=in_channels, n_classes=out_channels, num_colors=num_colors)
        
        # Create dummy input tensors
        # A batch of 4 grayscale images of size 128x128
        dummy_image_input = torch.randn(batch_size, in_channels, *image_size)
        # A batch of 4 one-hot color vectors
        dummy_color_input = F.one_hot(torch.arange(batch_size), num_classes=num_colors).float()

        print(f"Dummy Image Input shape: {dummy_image_input.shape}")
        print(f"Dummy Color Input shape: {dummy_color_input.shape}")

        # Perform a forward pass
        output = model(dummy_image_input, dummy_color_input)
        
        print(f"Model successfully ran a forward pass.")
        print(f"Output shape: {output.shape}")

        # Check if the output shape is as expected
        expected_shape = (batch_size, out_channels, *image_size)
        assert output.shape == expected_shape, f"Output shape mismatch! Expected {expected_shape}, got {output.shape}"
        print("Output shape is correct. Test successful!")

    except Exception as e:
        print(f"An error occurred during model test: {e}")

    print("--- Test complete ---")