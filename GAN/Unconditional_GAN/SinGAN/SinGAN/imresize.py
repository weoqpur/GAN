import numpy as np
from scipy.ndimage import filters, measurements, interpolation
from skimage import color
from math import pi
import torch

# 비정규화하기
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

# 정규화하기
def norm(x):
    out = (x - 0.5) * 2
    return out.clamp(-1, 1)

# device 구현
def move_to_gpu(t):
    if (torch.cuda.is_available()):
        t = t.to(torch.device('cuda'))
    return t


def np2torch(x, opt):
    if opt.nc_im == 3:
        x = x[:,:,:,None]
        x = x.transpose((3, 2, 0, 1))/255
    else:
        x = color.rgb2gray(x)
        x = x[:,:,None,None]
        x = x.transpose(3, 2, 0, 1)
    x = torch.from_numpy(x)
    if not (opt.not_cuda):
        x = move_to_gpu(x)
    x = x.type(torch.cuda.FloatTensor) if not(opt.not_cuda) else x.type(torch.FloatTensor)
    x = norm(x)
    return x


def torch2uint8(x):
    x = x[0,:,:,:]
    x = x.permute((1,2,0))
    x = 255*denorm(x)
    x = x.cpu().numpy()
    x = x.astype(np.uint8)
    return x


def fix_scale_and_size(input_shape, output_shape, scale_factor):
    # 표준화할 스케일 팩터가 제공된 경우 스케일 팩터를 고정한다
    # (입력 치수와 동일한 크기의 스케일 팩터 리스트)
    if scale_factor is not None:
        # 기본적으로 스케일 팩터가 스칼라인 경우 2d 크기 조정 및 복제합니다.
        if np.isscalar(scale_factor):
            scale_factor = [scale_factor, scale_factor]

        # 지정되지 않은 모든 척도에 1을 할당하여 척도 인자 목록의 크기를 입력 크기로 확장합니다.
        scale_factor = list(scale_factor)
        scale_factor.extend([1] * (len(input_shape) - len(scale_factor)))

    # 출력 모양 고정(지정된 경우): 원래의 입력 크기를 지정되지 않은 모든차원에 할당하여 입력 형상의 크기로 확장
    if output_shape is not None:
        output_shape = list(np.uint(np.array(output_shape))) + list(input_shape[len(output_shape):])

    # 출력 형태에 따라 계산되는 비기여 스케일 팩터의 경우 처리. 동일한 출력 모양에 서로 다른 척도가
    # 있을 수 있으므로 이 값은 차선입니다.
    if scale_factor is None:
        scale_factor = 1.0 * np.array(output_shape) / np.array(input_shape)

    # 누락된 출력 형상 처리 스케일 팩터에 따라 계산
    if output_shape is None:
        output_shape = np.uint(np.ceil(np.array(input_shape) * np.array(scale_factor)))

    return scale_factor, output_shape
