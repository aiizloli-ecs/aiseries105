###### DO NOT MODIFY ANYTHING IN THIS FILE ######
import cv2
import matplotlib.pyplot as plt
import numpy as np

def resize(image, width=None, height=None):
    (h, w) = image.shape[:2]
    if width is None:
        r = height/float(h)
        dim = (int(w*r), height)
    else:
        r = width/float(w)
        dim = (width, int(h*r))
    return cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)

def apply_logo(image, logo, alpha, x, y, w, h):
    logo = resize(logo, width=w, height=h)
    for i in range(y, y+logo.shape[0]):
        for j in range(x, x+logo.shape[1]):
            if logo[i-y, j-x, 3] != alpha:
                image[i, j] = logo[i-y, j-x, :3]
    return image

def plot_gallery(images, titles):
    plt.figure(1)
    plt.subplots_adjust(bottom=.10, left=.10, right=.90, top=.90, hspace=0.2)
    for i in range(len(images)):
        plt.subplot(2, 4, i + 1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i], size=18)
        plt.xticks(())
        plt.yticks(())
    plt.show()