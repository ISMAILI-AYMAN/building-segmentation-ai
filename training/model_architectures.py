#!/usr/bin/env python3
"""
Model Architectures for Building Segmentation
U-Net and ResNet-based segmentation models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import timm
import warnings
warnings.filterwarnings('ignore')

class DoubleConv(nn.Module):
    """Double convolution block for U-Net"""
    
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Output convolution"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """U-Net architecture for image segmentation"""
    
    def __init__(self, n_channels: int = 3, n_classes: int = 1, bilinear: bool = True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class ResNetUNet(nn.Module):
    """ResNet-based U-Net for building segmentation"""
    
    def __init__(self, n_channels: int = 3, n_classes: int = 1, 
                 backbone: str = 'resnet34', pretrained: bool = True):
        super(ResNetUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Load backbone
        self.backbone = timm.create_model(
            backbone, 
            pretrained=pretrained, 
            features_only=True,
            in_chans=n_channels
        )
        
        # Get feature dimensions
        feature_info = self.backbone.feature_info.get_dicts(keys=['num_chs', 'reduction'])
        self.feature_channels = [info['num_chs'] for info in feature_info]
        self.feature_reductions = [info['reduction'] for info in feature_info]
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        for i in range(len(self.feature_channels) - 1):
            in_ch = self.feature_channels[-(i+1)] + self.feature_channels[-(i+2)]  # Skip connection
            out_ch = self.feature_channels[-(i+2)]
            self.decoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Final output layer
        self.final_conv = nn.Conv2d(self.feature_channels[0], n_classes, kernel_size=1)
        
        # Upsampling layers
        self.upsample_layers = nn.ModuleList()
        for i in range(len(self.feature_channels) - 1):
            self.upsample_layers.append(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            )

    def forward(self, x):
        # Encoder
        features = self.backbone(x)
        
        # Decoder
        x = features[-1]
        for i, (decoder_layer, upsample_layer) in enumerate(zip(self.decoder_layers, self.upsample_layers)):
            # Upsample
            x = upsample_layer(x)
            
            # Skip connection
            skip_feature = features[-(i+2)]
            
            # Ensure same size
            if x.size() != skip_feature.size():
                x = F.interpolate(x, size=skip_feature.size()[2:], mode='bilinear', align_corners=True)
            
            # Concatenate
            x = torch.cat([x, skip_feature], dim=1)
            
            # Decode
            x = decoder_layer(x)
        
        # Final output
        x = self.final_conv(x)
        
        return x

class EfficientNetUNet(nn.Module):
    """EfficientNet-based U-Net for building segmentation"""
    
    def __init__(self, n_channels: int = 3, n_classes: int = 1, 
                 backbone: str = 'efficientnet_b0', pretrained: bool = True):
        super(EfficientNetUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Load backbone
        self.backbone = timm.create_model(
            backbone, 
            pretrained=pretrained, 
            features_only=True,
            in_chans=n_channels
        )
        
        # Get feature dimensions
        feature_info = self.backbone.feature_info.get_dicts(keys=['num_chs', 'reduction'])
        self.feature_channels = [info['num_chs'] for info in feature_info]
        
        # Decoder layers with attention
        self.decoder_layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        
        for i in range(len(self.feature_channels) - 1):
            in_ch = self.feature_channels[-(i+1)]
            out_ch = self.feature_channels[-(i+2)]
            
            # Attention mechanism
            self.attention_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, 1, kernel_size=1),
                    nn.Sigmoid()
                )
            )
            
            # Decoder layer
            self.decoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch + out_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Final output layer
        self.final_conv = nn.Conv2d(self.feature_channels[0], n_classes, kernel_size=1)
        
        # Upsampling layers
        self.upsample_layers = nn.ModuleList()
        for i in range(len(self.feature_channels) - 1):
            self.upsample_layers.append(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            )

    def forward(self, x):
        # Encoder
        features = self.backbone(x)
        
        # Decoder
        x = features[-1]
        for i, (decoder_layer, attention_layer, upsample_layer) in enumerate(
            zip(self.decoder_layers, self.attention_layers, self.upsample_layers)
        ):
            # Apply attention
            attention = attention_layer(x)
            x = x * attention
            
            # Upsample
            x = upsample_layer(x)
            
            # Skip connection
            skip_feature = features[-(i+2)]
            
            # Ensure same size
            if x.size() != skip_feature.size():
                x = F.interpolate(x, size=skip_feature.size()[2:], mode='bilinear', align_corners=True)
            
            # Concatenate
            x = torch.cat([x, skip_feature], dim=1)
            
            # Decode
            x = decoder_layer(x)
        
        # Final output
        x = self.final_conv(x)
        
        return x

class SegmentationModel(nn.Module):
    """Unified segmentation model with different backbones"""
    
    def __init__(self, model_type: str = 'unet', n_channels: int = 3, n_classes: int = 1, 
                 backbone: str = 'resnet34', pretrained: bool = True, **kwargs):
        super(SegmentationModel, self).__init__()
        
        if model_type == 'unet':
            self.model = UNet(n_channels=n_channels, n_classes=n_classes, **kwargs)
        elif model_type == 'resnet_unet':
            self.model = ResNetUNet(n_channels=n_channels, n_classes=n_classes, 
                                   backbone=backbone, pretrained=pretrained)
        elif model_type == 'efficientnet_unet':
            self.model = EfficientNetUNet(n_channels=n_channels, n_classes=n_classes, 
                                         backbone=backbone, pretrained=pretrained)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def forward(self, x):
        return self.model(x)
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_type': type(self.model).__name__,
            'input_channels': self.model.n_channels,
            'output_classes': self.model.n_classes
        }

def create_model(model_config: Dict) -> SegmentationModel:
    """Create model from configuration"""
    model_type = model_config.get('model_type', 'unet')
    n_channels = model_config.get('n_channels', 3)
    n_classes = model_config.get('n_classes', 1)
    backbone = model_config.get('backbone', 'resnet34')
    pretrained = model_config.get('pretrained', True)
    
    model = SegmentationModel(
        model_type=model_type,
        n_channels=n_channels,
        n_classes=n_classes,
        backbone=backbone,
        pretrained=pretrained
    )
    
    return model

def get_model_configs() -> Dict:
    """Get predefined model configurations"""
    configs = {
        'unet_basic': {
            'model_type': 'unet',
            'n_channels': 3,
            'n_classes': 1,
            'bilinear': True
        },
        'resnet34_unet': {
            'model_type': 'resnet_unet',
            'n_channels': 3,
            'n_classes': 1,
            'backbone': 'resnet34',
            'pretrained': True
        },
        'resnet50_unet': {
            'model_type': 'resnet_unet',
            'n_channels': 3,
            'n_classes': 1,
            'backbone': 'resnet50',
            'pretrained': True
        },
        'efficientnet_b0_unet': {
            'model_type': 'efficientnet_unet',
            'n_channels': 3,
            'n_classes': 1,
            'backbone': 'efficientnet_b0',
            'pretrained': True
        },
        'efficientnet_b3_unet': {
            'model_type': 'efficientnet_unet',
            'n_channels': 3,
            'n_classes': 1,
            'backbone': 'efficientnet_b3',
            'pretrained': True
        }
    }
    
    return configs

if __name__ == "__main__":
    # Test model creation
    configs = get_model_configs()
    
    for config_name, config in configs.items():
        print(f"\nTesting {config_name}...")
        model = create_model(config)
        
        # Test forward pass
        x = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            output = model(x)
        
        # Get model info
        info = model.get_model_info()
        
        print(f"Model: {info['model_type']}")
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Total parameters: {info['total_parameters']:,}")
        print(f"Trainable parameters: {info['trainable_parameters']:,}")
