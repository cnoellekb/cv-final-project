# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt
import numpy
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import copy
import pdb

import os
from util import is_image_file, load_img, save_img
import argparse
######################################################################
# Cuda
use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

######################################################################
# Load images
parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
parser.add_argument('--action', required=True, help='train_or_test')
parser.add_argument('--dataset', required=True, help='facades')
opt = parser.parse_args()
# desired size of the output image
imsize = 256 if use_cuda else 128  # use small size if no gpu

loader = transforms.Compose([
    transforms.Scale(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    image = Variable(loader(image))
    # fake batch dimension required to fit network's input dimensions
    image = image.unsqueeze(0)
    return image

def image_loader_grayscale(image_name):
    image = Image.open(image_name)
    image = image.convert("L")
    image = Variable(loader(image))
    # fake batch dimension required to fit network's input dimensions
    image = image.unsqueeze(0)
    return image
# style_img = image_loader("images/pencil.jpg").type(dtype)
# # 

image_dir = "dataset/{}/{}/target/".format(opt.dataset,opt.action)
output_dir = "dataset/{}/{}/content/".format(opt.dataset,opt.action)
image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]

unloader = transforms.ToPILImage()  # reconvert tensor into PIL image


def imsave(tensor, path, imageName):
    image = tensor.clone().cpu()  # we clone the tensor to not do changes on it
    image = image.view(3, imsize, imsize)  # remove the fake batch dimension
    image = unloader(image)
    image.save( path + imageName, format=None)

class ContentLoss(nn.Module):

    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        self.target = target.detach() * weight
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion(input * self.weight, self.target)
        self.output = input
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss

class GramMatrix(nn.Module):

    def forward(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)
        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.output = input.clone()
        a,b,c,d = input.size()
        # input[:,:,int(c/5):int(c*4/5),int(d/5):int(d*4/5)] = 0
        self.G = self.gram(input)
        self.G.mul_(self.weight)

        self.loss = self.criterion(self.G, self.target)
        # self.loss = MMSELoss(self.target, self.G)
        # print(self.loss)

        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss

cnn = models.vgg19(pretrained=True).features

if use_cuda:
    cnn = cnn.cuda()

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_8']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# style_layers_default = ['conv_3','conv_4','conv_5']
def get_style_model_and_losses(cnn, style_img, content_img,
                               style_weight=1000, content_weight=1,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    model = nn.Sequential()  # the new Sequential module network
    gram = GramMatrix()  # we need a gram module in order to compute style targets

    # move these modules to the GPU if possible:
    if use_cuda:
        model = model.cuda()
        gram = gram.cuda()

    i = 1
    for layer in list(cnn):
        if isinstance(layer, nn.Conv2d):
            name = "conv_" + str(i)
            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(content_img).clone()
                # workspace.save_workspace('./target_conv.pkl',target)
                # print('target_conv')
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).clone()
                # a,b,c,d = target_feature.size()
                # target_feature[:,:,int(c/5):int(c*4/5),int(d/5):int(d*4/5)] = 0
                target_feature_gram = gram(target_feature)

                style_loss = StyleLoss(target_feature_gram, style_weight)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)

        if isinstance(layer, nn.ReLU):
            name = "relu_" + str(i)
            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)

            i += 1

        if isinstance(layer, nn.MaxPool2d):
            name = "pool_" + str(i)
            model.add_module(name, layer)  # ***
    
    return model, style_losses, content_losses

def get_input_param_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    input_param = nn.Parameter(input_img.data)
    # pdb.set_trace()
    optimizer = optim.LBFGS([input_param])
    return input_param, optimizer

######################################################################

def run_style_transfer(cnn, content_img, style_img, input_img, num_steps=300,
                       style_weight=500, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        style_img, content_img, style_weight, content_weight)
    # print(model)
    input_param, optimizer = get_input_param_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_param.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_param)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.backward()
            for cl in content_losses:
                content_score += cl.backward()

            run[0] += 1
            if run[0] in [51,101,151,201,251]:
                input_param.data = torch.FloatTensor(1,3,256,256).uniform_(-0.1, 0.1).cuda() + input_param.data

            if run[0] % 100 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.data[0], content_score.data[0]))
                print()

            return style_score + content_score


        optimizer.step(closure)
    # a last correction...
    input_param.data.clamp_(0, 1)

    return input_param.data

for image_name in image_filenames:

    content_img = image_loader(image_dir + image_name).type(dtype)
    style_img = Variable(torch.ones(content_img.data.size())).type(dtype)

    # input_img = Variable(torch.ones(content_img.data.size())).type(dtype)
    input_img = image_loader_grayscale(image_dir + image_name).type(dtype)
    input_img = torch.cat((input_img,input_img,input_img),1)
    ######################################################################
    # Finally, run the algorithm

    output = run_style_transfer(cnn, content_img, style_img, input_img)

    # plt.figure()
    # imshow(output, title='Output Image')
    imsave(output,output_dir, image_name)
    # sphinx_gallery_thumbnail_number = 4
    # plt.ioff()
    # plt.show()
