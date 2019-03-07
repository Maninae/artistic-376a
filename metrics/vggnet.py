from PIL import Image
import numpy as np
from torchvision.models.vgg import vgg16_bn
from torchvision import transforms
import torch.nn as nn




def get_vgg_model(pretrained=True, **kwargs):
    return vgg16_bn(pretrained, **kwargs)
"""
    0 Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    1 BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    2 ReLU(inplace)

    3 Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    4 BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    5 ReLU(inplace)
6 MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

    7 Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    8 BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    9 ReLU(inplace)

    10 Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    11 BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    12 ReLU(inplace)
13 MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

    14 Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    15 BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    16 ReLU(inplace)

    17 Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    18 BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    19 ReLU(inplace)

    20 Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    21 BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    22 ReLU(inplace)
23 MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

    24 Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    25 BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    26 ReLU(inplace)

    27 Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    28 BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    29 ReLU(inplace)

    30 Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    31 BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    32 ReLU(inplace)
33 MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

    34 Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    35 BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    36 ReLU(inplace)

    37 Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    38 BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    39 ReLU(inplace)

    40 Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    41 BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    42 ReLU(inplace)
43 MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
"""

def open_image(image_path):
    print("Creating a PIL image from:", image_path)
    im = Image.open(image_path) 


def prepare_input_VGG(img):
    """ input: ndarray, batch of images (M, H, W, 3). For single image, M=1
        :return: pytorch Variable of our image
    """
    transform_pipeline = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    img = transform_pipeline(img)
    img = img.unsqueeze(0)
    img = Variable(img)
    return img


# https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/12
# February '17'
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            x = module(x)
            if name in self.extracted_layers:
                outputs += [x]
        return outputs


if __name__ == "__main__":
    model = get_vgg_model()
    
    # activations after each of the MaxPools
    extracted_layers = ["6", "13", "23", "33", "43"]
    conv_features = model._modules["features"]
    fex = FeatureExtractor(conv_features, extracted_layers)

    # For every image: get features at designated layers
    pass

    
