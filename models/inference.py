import torch
from .layers import Generator, RRDBNet
from torchvision import transforms
from states import BotStates
from PIL import Image
import numpy as np

WEIGHTS = {
    BotStates.CUBISM_STATE: 'models/weights/cubism.tar',
    BotStates.SUPER_RESOLUTION_STATE: 'models/weights/RRDB_ESRGAN_x4.pth',
    BotStates.RENAISSANCE_STATE: 'models/weights/renaissance.tar',
    BotStates.EXPRESSIONISM_STATE: 'models/weights/expressionism.tar'
}


class CycleGAN:
    RESCALE_SIZE = 512

    def __init__(self, weight_path):
        self.device = self.get_device()
        self.model = self.init_model(weight_path)

    def get_device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device

    def init_model(self, weight_path):
        model = self.get_model().to(self.device)
        checkpoint = torch.load(weight_path)
        model.load_state_dict(checkpoint['B2A_gen_state_dict'])
        model.eval()

        return model

    def get_model(self):
        return Generator(32)

    async def predict(self, image):
        image = await self.prepare_img(image)
        with torch.no_grad():
            res = self.model(image)[0].float().clamp_(-1, 1)
        res = await self.postprocessor(res)
        return res

    async def prepare_img(self, image):
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

    async def postprocessor(self, image):
        return np.rollaxis(await self.tensor2image(image), 0, 3)

    async def tensor2image(self, tensor):
        image = 127.5 * (tensor.cpu().detach().numpy() + 1.0)
        return image.astype(np.uint8)


class ESRGAN:
    def __init__(self, weight_path):
        self.weight_path = weight_path
        self.device = 'cpu'
        self.model = None

    async def init_model(self, weight_path):
        model = await self.get_model()
        model = model.to(self.device)
        model.load_state_dict(torch.load(weight_path), strict=True)
        model.eval()

        return model

    async def choice_device(self, sizes):
        if max(sizes) > 720:
            self.device = 'cpu'
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    async def get_model(self):
        return RRDBNet(3, 3, 64, 23, gc=32)

    async def predict(self, image):
        width, height = image.size
        await self.choice_device((width, height))
        self.model = await self.init_model(self.weight_path)

        image = await self.prepare_img(image)
        with torch.no_grad():
            res = self.model(image)[0].float().clamp_(0, 1)
        res = await self.postprocessor(res)

        return res

    async def prepare_img(self, image):
        image = np.array(image)
        image = image * 1.0 / 255
        image_torch = torch.from_numpy(np.rollaxis(image, 2, 0))
        image_torch = image_torch.unsqueeze(0).float().to(self.device)
        return image_torch

    async def postprocessor(self, image):
        return np.rollaxis(await self.tensor2image(image), 0, 3)

    async def tensor2image(self, tensor):
        image = 255 * (tensor.cpu().numpy())
        image = image.round()
        return image.astype(np.uint8)


class StyleTransfer:
    pass


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
