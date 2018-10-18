from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage import filters
from sklearn.preprocessing import normalize
import torch.nn as nn
import torch

class ImageGradient:
    def __init__(self, image):
        self.image = image
    def get_gradient(self):
        im = rgb2gray(imread('./dataset/Kim.PNG'))
        edges_x = filters.sobel_h(im)
        edges_y = filters.sobel_v(im)

        edges_x = normalize(edges_x)
        edges_y = normalize(edges_y)

        return edges_x, edges_y

class GradientLoss:
    def __init__(self, image, mask):
        self.image = image
        self.mask = mask
    def get_loss(self):
        image_grad_x, image_grad_y = ImageGradient(image=self.image).get_gradient()
        mask_grad_x, mask_grad_y = ImageGradient(image=self.mask).get_gradient()
        IMx = torch.mul(image_grad_x, mask_grad_x)
        IMy = torch.mul(image_grad_y, mask_grad_y)
        Mmag = torch.sqrt(torch.add(torch.pow(mask_grad_x, 2), torch.pow(mask_grad_y, 2)))
        IM = torch.add(1, torch.neg(torch.add(IMx, IMy)))
        numerator = torch.sum(torch.mul(Mmag, IM))
        denominator = torch.sum(Mmag)
        out = torch.div(numerator, denominator)
        return out

class HairMetteLoss(nn.CrossEntropyLoss):
    def __init__(self, image, mask, label):
        super(HairMetteLoss, self).__init__()
        criterion = nn.CrossEntropyLoss()
        criterion = criterion(image, label)
        grad_loss = ImageLoss(image, mask)

        return grad_loss + criterion

