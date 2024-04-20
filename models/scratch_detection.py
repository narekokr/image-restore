import os

import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from scipy import ndimage
import numpy as np
from skimage.color import rgb2lab, lab2rgb
import torch.nn.functional as F
from torchvision.io import read_image, ImageReadMode

from networks.autoencoder import ConvAutoencoder
from networks.network import Network
from networks.siggraph import SIGGRAPHGenerator
from util import preprocess_img, postprocess_tens

toTensor = transforms.ToTensor()

def merge(img, rgb):
    gray_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    rgb_lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    ab = rgb_lab[:, :, 1:]
    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))
    result = np.concatenate([gray_lab[:, :, 0][:, :, np.newaxis], ab], axis=-1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    return result


class ImageProcessor:
    def __init__(self, model_weights_path='checkpoints/network.pt', colorize_path='checkpoints/siggraph.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scratch_model = self.load_network(Network, False, model_weights_path)
        self.colorize_model = self.load_network(SIGGRAPHGenerator, False, colorize_path)

    def load_network(self, network, isParallel, path):
        # Instantiate your PyTorch model and load the weights
        model = network()
        checkpoint = torch.load(path, map_location=self.device)
        if isParallel:
            model = nn.DataParallel(model)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model

    def get_inpainted_image(self, image, mask, detect_automatically):
        orig = Image.open(image)

        width, height = orig.size

        # Calculate the new width and height that are divisible by 16
        new_width = (width // 16) * 16
        new_height = (height // 16) * 16

        # Resize the image using the calculated dimensions
        orig = orig.resize((new_width, new_height))

        img = orig.convert('L')

        if orig.size[0] < 256 or orig.size[1] < 256:
            img = img.resize((256, 256))
            should_resize_back = True
        else:
            should_resize_back = False

        if not detect_automatically:
            msk = Image.open(mask).convert('L').resize(orig.size)
            msk = cv2.cvtColor(np.array(msk), cv2.IMREAD_GRAYSCALE)
        else:
            output = self.scratch_detection(img)
            msk = cv2.cvtColor(output.astype(np.uint8), cv2.IMREAD_GRAYSCALE)

        img = cv2.cvtColor(np.array(orig), cv2.IMREAD_COLOR)
        if (should_resize_back):
            msk = cv2.resize(msk, orig.size)
        msk = msk[:, :, 0]

        result = cv2.inpaint(img, msk, 3, cv2.INPAINT_TELEA)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        if should_resize_back:
            result = cv2.resize(result, (orig.size[0], orig.size[1]))

        return result

    def colorize(self, image):
        with torch.no_grad():
            output = self.predict_siggraph(image)

        # result = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        return output

    def scratch_detection(self, image):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        with torch.no_grad():
            output = self.scratch_model(transform(image).reshape((1, 1, image.size[1], image.size[0])).to(self.device))
            output = output.cpu()
            mask = output[0].permute(1, 2, 0).numpy() > 0.5
            output = torch.sigmoid(output)
            mask = output[0].permute(1, 2, 0).numpy() > 0.5
            dilated = ndimage.binary_dilation(np.squeeze(mask, axis=2), iterations=2).astype(mask.dtype)

        return dilated

    def preprocess_colorize(self, image):
        image = image.unsqueeze(0)
        image = F.interpolate(image, (160, 160))
        image = image.squeeze(0)
        if image.shape[0] == 1:
            image = image.permute(1, 2, 0)
            image = image.repeat(1, 1, 3)
            image = image.permute(2, 0, 1)
        image = torch.tensor(rgb2lab(image.permute(1, 2, 0) / 255))
        image = (image + torch.tensor([0, 128, 128])) / torch.tensor([100, 255, 255])
        image = image.permute(2, 0, 1)
        # Use L channel from image to predict a,b channels of label
        image = image[:1, :, :]
        return image

    @torch.inference_mode()
    def predict(self, input_image):
        plt.imsave('a.png', input_image, cmap = plt.cm.gray)
        tensor = read_image('a.png', ImageReadMode.RGB)
        os.remove('a.png')
        image = self.preprocess_colorize(tensor)
        pred = self.colorize_model.forward(image[0].float().view(1, 1, 160, 160).to(self.device))
        lab_pred = torch.cat((image[0].view(1, 160, 160), pred[0].cpu()), dim=0)
        lab_pred_inv_scaled = lab_pred.permute(1, 2, 0) * torch.tensor([100, 255, 255]) - torch.tensor([0, 128, 128])
        rgb_pred = lab2rgb(lab_pred_inv_scaled.detach().numpy())
        rgb_pred = cv2.normalize(rgb_pred, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        out = merge(input_image, rgb_pred)
        a = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

        return a

    def predict_siggraph(self, img):
        (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
        out_img_siggraph17 = postprocess_tens(tens_l_orig, self.colorize_model(tens_l_rs.to(self.device)).cpu())
        return out_img_siggraph17
