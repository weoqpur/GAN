import time
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from model import Generator

upscale_factor = 4
image_name = './images/test/image3.jpg'
model_name = 'netG_epoch_4_100.pth'

model = Generator(upscale_factor).eval()

model.load_state_dict(torch.load('epochs/' + model_name, map_location=lambda storage, loc: storage))

image = Image.open(image_name)
image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)

start = time.perf_counter()
out = model(image)
elapsed = (time.perf_counter() - start)
print('cost' + str(elapsed))
out_img = ToPILImage()(out[0].data.cpu())
out_img.save('test_images/out_srf_' + str(upscale_factor) + '_3_1.jpg')