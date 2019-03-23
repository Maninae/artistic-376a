import os
from PIL import Image

import numpy as np
from sklearn import manifold
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16_bn
from torchvision import transforms



"""
VGG16 with Batch Norm:

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




# Based off of:
# https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/12
# February '17'
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super().__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            x = module(x)
            if name in self.extracted_layers:
                outputs += [x]
        return outputs


def get_VGG_feature_extractor(extracted_layers):
    model = vgg16_bn(pretrained=True)
    conv_network = model._modules["features"] # The convolutional part
    return FeatureExtractor(conv_network, extracted_layers)


def prepare_input_VGG(img):
    """ input: ndarray, batch of images (M, H, W, 3). For single image, M=1
        :return: pytorch Variable of our image
    """
    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    img = transform_pipeline(img)
    if img.size()[0] == 4: # 4 channel RGBA, remove A to be 3 channel RGB
        img = img[:3,...]
    img = img.unsqueeze(0)
    return img


####################################################


def open_image(image_path):
    print("Creating a PIL image from:", image_path)
    im = Image.open(image_path)
    return im

def get_genre_images(image_dir):
    print("Opening dir:", image_dir)
    files = [os.path.join(image_dir, fn) for fn in os.listdir(image_dir) if fn.endswith(".jpg")]
    opened_files = [open_image(file) for file in files]
    return opened_files


# https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
def get_gram_matrix(x):
    a, b, c, d = x.size()  
    # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)
    assert a == 1

    features = x.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

def get_gram_matrix_from_image(img):
    print("computing a gram matrix...")
    img_tensor = prepare_input_VGG(img)
    activations = fex(img_tensor)[0]
    style_matrix = get_gram_matrix(activations)
    return style_matrix

def get_gram_matrix_from_directory_images(dirpath):
    images = get_genre_images(dirpath)
    gram_matrices = []
    for img in images:
        style_matrix = get_gram_matrix_from_image(img)
        gram_matrices.append(style_matrix)
    return gram_matrices


######################

def visualize(datapoints, list_of_genres):
    """ datapoints: (n,k) nparray of n datapoints, each k-dimensional. 
    """
    pca = PCA(n_components=50)
    datapoints = pca.fit_transform(datapoints)

    mds = manifold.MDS(n_components=2, max_iter=5000, eps=1e-9, dissimilarity="euclidean")
    positions = mds.fit_transform(datapoints)
    
    # assume 8 datapoints per genre, 10 genres
    # blue, green, red, cyan, magenta, yellow, black, lightgray, gray, lightgreen
    ten_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', tuple([0.3] * 3), tuple([0.7] * 3), (0,0.5,0,1)]
    
    for _ in range(len(list_of_genres)):
        start = 8 * _
        color = ten_colors[_]
        genre_name = list_of_genres[_]
        plt.scatter(positions[start : start+8, 0], positions[start : start+8, 1], c=color, label=genre_name)
    
    plt.legend()
    plt.show()



if __name__ == "__main__":
    fex = get_VGG_feature_extractor(["2"])
    # ["6", "13", "23", "33", "43"]

    
    genre_directories = [os.path.join("../different_genres", dirname) for dirname in os.listdir("../different_genres")]

    # Test: let's just do 2 genres
    # genre_directories = genre_directories[:2]

    list_of_genres = [] # To preserve ordering when coloring legends
    all_gram_matrices = []
    for genre_path in genre_directories:
        print("Doing genre path:", genre_path)
        list_of_genres.append(os.path.basename(genre_path))
        genre_gm = get_gram_matrix_from_directory_images(genre_path)
        assert len(genre_gm) == 8
        all_gram_matrices.extend(genre_gm)
        print("  Appended 8 images' gram matrices.")
    
    # Turn them into numpy matrices, without batch dimension
    all_gram_matrices_flattened = [gm.flatten().detach().numpy() for gm in all_gram_matrices]
    datapoints = np.array(all_gram_matrices_flattened)

    # center the data
    mean_sample = np.mean(datapoints, axis=0, keepdims=True)
    datapoints = datapoints - mean_sample
    
    visualize(datapoints, list_of_genres)

    
    