import torch
from .layers import *
from torchvision import models
from torchvision import transforms
from states import BotStates
from PIL import Image
import numpy as np
import copy

WEIGHTS = {
    BotStates.CUBISM_STATE: 'models/weights/cubism.tar',
    BotStates.SUPER_RESOLUTION_STATE: 'models/weights/RRDB_ESRGAN_x4.pth',
    BotStates.RENAISSANCE_STATE: 'models/weights/renaissance.tar',
    BotStates.EXPRESSIONISM_STATE: 'models/weights/expressionism.tar'
}


class CycleGAN:
    RESCALE_SIZE = 512

    def __init__(self, weight_path):
        self.device = self._get_device()
        self.model = self._init_model(weight_path)

    def _get_device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device

    def _init_model(self, weight_path):
        model = self._get_model().to(self.device)
        checkpoint = torch.load(weight_path)
        model.load_state_dict(checkpoint['B2A_gen_state_dict'])
        model.eval()

        return model

    def _get_model(self):
        return Generator(32)

    async def predict(self, image):
        image = await self._prepare_img(image)
        with torch.no_grad():
            res = self.model(image)[0].float()
        res = await self._postprocessor(res)
        return res

    async def _prepare_img(self, image):
        width, height = image.size

        # Приводим изображения к такому виду,
        # чтобы меньшая сторана была ровна 512 а большая кратна 32
        if width > height:
            width = 32 * int((self.RESCALE_SIZE * width / height) // 32)
            height = self.RESCALE_SIZE
        else:
            height = 32 * int((self.RESCALE_SIZE * height / width) // 32)
            width = self.RESCALE_SIZE

        transform = transforms.Compose([
            transforms.Resize(self.RESCALE_SIZE, Image.BICUBIC),
            transforms.CenterCrop((height, width)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        transformed_image = transform(image)

        transformed_image = transformed_image.unsqueeze(0).to(self.device)

        return transformed_image

    async def _postprocessor(self, image):
        return np.rollaxis(await self._tensor2image(image), 0, 3)

    async def _tensor2image(self, tensor):
        image = 127.5 * (tensor.cpu().detach().numpy() + 1.0)
        return image.astype(np.uint8)


class ESRGAN:
    RESCALE_SIZE = 720

    def __init__(self, weight_path):
        self.weight_path = weight_path
        self.device = 'cpu'
        self.model = None

    async def _init_model(self, weight_path):
        model = await self._get_model()
        model = model.to(self.device)
        model.load_state_dict(torch.load(weight_path), strict=True)
        model.eval()

        return model

    async def _choice_device(self, sizes):
        if max(sizes) > self.RESCALE_SIZE:
            self.device = 'cpu'
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    async def _get_model(self):
        return RRDBNet(3, 3, 64, 23, gc=32)

    async def predict(self, image):
        await self._choice_device(image.size)
        self.model = await self._init_model(self.weight_path)

        image = await self._prepare_img(image)
        with torch.no_grad():
            res = self.model(image)[0].float().clamp_(0, 1)
        res = await self._postprocessor(res)

        return res

    async def _prepare_img(self, image):
        if min(image.size) > self.RESCALE_SIZE:
            transform = transforms.Resize(self.RESCALE_SIZE, Image.BICUBIC)
            image = transform(image)
        image = np.array(image)
        image = image * 1.0 / 255
        image_torch = torch.from_numpy(np.rollaxis(image, 2, 0))
        image_torch = image_torch.unsqueeze(0).float().to(self.device)
        return image_torch

    async def _postprocessor(self, image):
        return np.rollaxis(await self._tensor2image(image), 0, 3)

    async def _tensor2image(self, tensor):
        image = 255 * (tensor.cpu().numpy())
        image = image.round()
        return image.astype(np.uint8)


class StyleTransfer:
    RESCALE_SIZE = 512
    CONTENT_LAYERS = ['conv_4']
    STYLE_LAYERS = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    CNN_NORMALIZATION_MEAN = torch.tensor([0.485, 0.456, 0.406])
    CNN_NORMALIZATION_STD = torch.tensor([0.229, 0.224, 0.225])

    def __init__(self):
        self.device = self._get_device()

    def _get_device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device

    def predict(self, content_img, style_img,
                num_steps=500, style_weight=100000, content_weight=1):

        cnn = models.vgg19(pretrained=True).features.to(self.device).eval()
        input_img = content_img.clone()

        model, style_losses, content_losses = self.get_style_model_and_losses(cnn, self.CNN_NORMALIZATION_MEAN,
                                                                              self.CNN_NORMALIZATION_STD,
                                                                              style_img, content_img)

        optimizer = self.get_input_optimizer(input_img)

        run = [0]
        while run[0] <= num_steps:

            def closure():
                # correct the values
                # это для того, чтобы значения тензора картинки не выходили за пределы [0;1]
                input_img.data.clamp_(0, 1)

                optimizer.zero_grad()

                model(input_img)

                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1

                return style_score + content_score

            optimizer.step(closure)

        input_img.data.clamp_(0, 1)

        return input_img

    def _prepare_img(self, content_img, style_img):
        width, height = content_img.size

        # Приводим изображения к такому виду,
        # чтобы меньшая сторана была ровна 512 а большая кратна 32
        if width > height:
            width = 32 * int((self.RESCALE_SIZE * width / height) // 32)
            height = self.RESCALE_SIZE
        else:
            height = 32 * int((self.RESCALE_SIZE * height / width) // 32)
            width = self.RESCALE_SIZE

        content_transform = transforms.Compose([
            transforms.Resize(self.RESCALE_SIZE, Image.BICUBIC),
            transforms.CenterCrop((height, width)),
            transforms.ToTensor(),
        ])
        style_transform = transforms.Compose([
            transforms.Resize((height, width), Image.BICUBIC),
            transforms.ToTensor(),
        ])

        transformed_content_img = content_transform(content_img)
        transformed_style_img = style_transform(style_img)

        transformed_content_img = transformed_content_img.unsqueeze(0).to(self.device)
        transformed_style_img = transformed_style_img.unsqueeze(0).to(self.device)

        return transformed_content_img, transformed_style_img

    def get_style_model_and_losses(self, cnn, normalization_mean, normalization_std,
                                   style_img, content_img):
        cnn = copy.deepcopy(cnn)

        # normalization module
        normalization = Normalization(normalization_mean, normalization_std).to(self.device)

        # just in order to have an iterable access to or list of content/syle
        # losses
        content_losses = []
        style_losses = []

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(normalization)

        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                # Переопределим relu уровень
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in self.CONTENT_LAYERS:
                # add content loss:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in self.STYLE_LAYERS:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        # выбрасываем все уровни после последенего styel loss или content loss
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses

    def get_input_optimizer(self, input_img):
        optimizer = torch.optim.LBFGS([input_img.requires_grad_()])
        return optimizer


async def get_model(state):
    if state in (BotStates.CUBISM_STATE, BotStates.RENAISSANCE_STATE, BotStates.EXPRESSIONISM_STATE):
        model = CycleGAN(WEIGHTS[state])
    elif state == BotStates.SUPER_RESOLUTION_STATE:
        model = ESRGAN(WEIGHTS[state])
    elif state == BotStates.STYLE_TRANSFER_STATE_2:
        model = None
    else:
        model = None

    return model
