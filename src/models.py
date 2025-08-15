import logging
import torch.nn as nn
import torchvision.models as tvm

def _adapt_first_conv_to_grayscale(conv: nn.Conv2d):
    """Replace first conv layer weights to accept 1 input channel (average RGB weights)."""
    if isinstance(conv, nn.Conv2d) and conv.in_channels == 3:
        w = conv.weight.data  # [out, 3, k, k]
        w_mean = w.mean(dim=1, keepdim=True)  # [out,1,k,k]
        new_conv = nn.Conv2d(in_channels=1, out_channels=conv.out_channels,
                             kernel_size=conv.kernel_size, stride=conv.stride,
                             padding=conv.padding, dilation=conv.dilation, bias=(conv.bias is not None))
        new_conv.weight.data = w_mean
        if conv.bias is not None:
            new_conv.bias.data = conv.bias.data.clone()
        return new_conv
    return conv

def build_model(arch: str, num_classes: int, pretrained: bool, grayscale_mode: str) -> nn.Module:
    """Create a classifier and adapt it to grayscale if requested."""
    if arch == "resnet18":
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
        if grayscale_mode == "single":
            m.conv1 = _adapt_first_conv_to_grayscale(m.conv1)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
        logging.info("Built ResNet18 (pretrained=%s) with num_classes=%d, grayscale_mode=%s.", pretrained, num_classes, grayscale_mode)
        return m
    elif arch == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        if grayscale_mode == "single":
            m.features[0][0] = _adapt_first_conv_to_grayscale(m.features[0][0])
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        logging.info("Built EfficientNet-B0 (pretrained=%s) with num_classes=%d, grayscale_mode=%s.", pretrained, num_classes, grayscale_mode)
        return m
    elif arch == "vit_b_16":
        m = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.DEFAULT if pretrained else None)
        in_f = m.heads.head.in_features
        m.heads.head = nn.Linear(in_f, num_classes)
        logging.info("Built ViT-B/16 (pretrained=%s) with num_classes=%d (requires 3-channel input).", pretrained, num_classes)
        return m
    else:
        raise ValueError(f"Unknown model arch: {arch}")
