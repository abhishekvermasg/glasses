import torch
import pandas as pd
from argparse import ArgumentParser
from torchsummary import summary
from glasses.nn.models import ResNet, ResNetXt, WideResNet, SEResNet, DenseNet, EfficientNet, MobileNetV2, FishNet
# from torchvision.models import *

parser = ArgumentParser()
parser.add_argument('-o', default='./table.md')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device={device}')
print(f'out={args.o}')


def row(model_factory):
    model = model_factory()
    total_params, _, param_size, total_size = summary(
        model.to(device), (3, 224, 224))

    del model

    return {
        'name': model_factory.__name__,
        'Parameters': f"{total_params.item():,}",
        'Size (MB)': f"{param_size.item():.2f}",
        # 'Total Size (MB)': int(total_size.item())
    }


models = [ResNet.resnet18, ResNet.resnet26, ResNet.resnet34, ResNet.resnet50, ResNet.resnet101, ResNet.resnet152,  ResNet.resnet200,
          ResNetXt.resnext50_32x4d, ResNetXt.resnext101_32x8d, WideResNet.wide_resnet50_2, WideResNet.wide_resnet101_2, 
          SEResNet.se_resnet18, SEResNet.se_resnet34, SEResNet.se_resnet50, SEResNet.se_resnet101, SEResNet.se_resnet152,
          DenseNet.densenet121, DenseNet.densenet161, DenseNet.densenet169, DenseNet.densenet201,
          MobileNetV2, FishNet.fishnet99, FishNet.fishnet150,
          EfficientNet.efficientnet_b0, EfficientNet.efficientnet_b1, EfficientNet.efficientnet_b2, EfficientNet.efficientnet_b3, 
          EfficientNet.efficientnet_b4, EfficientNet.efficientnet_b5, EfficientNet.efficientnet_b6, EfficientNet.efficientnet_b7,
           EfficientNet.efficientnet_b8,  EfficientNet.efficientnet_l2
          ]

res = list(map(row, models))

df = pd.DataFrame.from_records(res)
print(df['name'].values)
print(df)

mk = df.set_index('name', drop=True).to_markdown()

with open(args.o, 'w') as f:
    f.write(mk)