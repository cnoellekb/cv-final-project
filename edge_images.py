import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageFilter
import argparse
import os
from util import is_image_file, load_img, save_img


parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
parser.add_argument('--action', required=True, help='train_or_test')
parser.add_argument('--dataset', required=True, help='facades')
opt = parser.parse_args()


image_dir = "dataset/{}/{}/target/".format(opt.dataset,opt.action)
output_dir = "dataset/{}/{}/edge/".format(opt.dataset,opt.action)
image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]


def imsave(image, path, imageName):
    #image = tensor.clone().cpu()  # we clone the tensor to not do changes on it
    #image = image.view(3, imsize, imsize)  # remove the fake batch dimension
    #image = unloader(image)
    image.save( path + imageName, format=None)


def image_loader(image_name):
    image = Image.open(image_name)
    #image = Variable(loader(image))
    imageWithEdges = image.filter(ImageFilter.FIND_EDGES)
    # fake batch dimension required to fit network's input dimensions
    #image = image.unsqueeze(0)
    return imageWithEdges



for image_name in image_filenames:
	edge_image = image_loader(image_dir + image_name)
	imsave(edge_image,output_dir, image_name)



"""
img = cv2.imread('seagull-in-the-foreground-robert-moses-bridge-in-the-background-vicki-jauron.jpg',1)
edges = cv2.Canny(img,100,200)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()"""