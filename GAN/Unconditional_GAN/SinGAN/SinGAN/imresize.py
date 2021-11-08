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


# Not define
def imresize(im,scale,opt):
    im = torch2uint8(im)
    #im = imresize_in(im, scale_factor=scale)

# not define
def imresize_to_shape():
    pass

# not define
def imresize_in(im, scale_fector=None, output_shape=None, kernel=None, antialiasing=True, kernel_shift_flag=False):

    scale_fector, output_shape = fix_scale_and_size(im.shape, output_shape, scale_fector)


def fix_scale_and_size(input_shape, output_shape, scale_factor):
    # 표준화할 스케일 팩터가 제공된 경우 스케일 팩터를 고정한다
    # (입력 치수와 동일한 크기의 스케일 팩터 리스트)
    if scale_factor is not None:
        # 기본적으로 스케일 팩터가 스칼라인 경우 2d 크기 조정 및 복제한다.
        if np.isscalar(scale_factor):
            scale_factor = [scale_factor, scale_factor]

        # 지정되지 않은 모든 척도에 1을 할당하여 척도 인자 목록의 크기를 입력 크기로 확장한다.
        scale_factor = list(scale_factor)
        scale_factor.extend([1] * (len(input_shape) - len(scale_factor)))

    # 출력 모양 고정(지정된 경우): 원래의 입력 크기를 지정되지 않은 모든차원에 할당하여 입력 형상의 크기로 확장
    if output_shape is not None:
        output_shape = list(np.uint(np.array(output_shape))) + list(input_shape[len(output_shape):])

    # 출력 형태에 따라 계산되는 비기여 스케일 팩터의 경우 처리. 동일한 출력 모양에 서로 다른 척도가
    # 있을 수 있으므로 이 값은 차선이다.
    if scale_factor is None:
        scale_factor = 1.0 * np.array(output_shape) / np.array(input_shape)

    # 누락된 출력 형상 처리 스케일 팩터에 따라 계산
    if output_shape is None:
        output_shape = np.uint(np.ceil(np.array(input_shape) * np.array(scale_factor)))

    return scale_factor, output_shape

def contributions(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    # 이 함수는 'filters' 집합과 나중에 적용될 field_of_view 집합을 계산하여
    # field_of_view의 각 위치에 보간 방법과 그 주변의 픽셀 중심으로부터의 하위
    # 픽셀 위치 거리를 기반으로 'weights'의 일치 필터를 곱한다.
    # 이 작업은 이미지의 한 차원에서만 수행된다.

    # antialiasing이 활성화되면(기본값 및 downscaling에만 해당) 수용 필드는 1/sf 크기로 늘어난다.
    # 즉, 필터링이 'low-pass filter' 임을 의미한다.
    fixed_kernel = (lambda arg: scale * kernel(scale * arg)) if antialiasing else kernel
    kernel_width *= 1.0 / scale if antialiasing else 1.0

    # 출력 이미지 좌표
    out_coordinates = np.arange(1, out_length+1)

    # 입력 영상 좌표에서 출력 좌표가 일치하는 위치이다.
    # 가장 좋은 예는 HR용 수평 픽셀이 4개이고 SF=2로 축소하여 2개의 픽셀을 얻는 경우이다.
    # [1,2,3,4] -> [1,2]. (각 픽셀 번호는 픽셀의 중간임을 기억하자.)
    # 스케일링은 픽셀 번호가 아닌 거리 사이에서 수행된다 (픽셀 4의 오른쪽 경계는 픽셀 2의 오른쪽 경계로 변환된다.)
    # 작은 이미지의 픽셀 1은 픽셀 2가 아닌 큰 이미지의 픽셀 1과 2 사이의 경계와 일치한다.
    # 즉, 위치가 단순히 기존 포스를 scale-factor로 곱한 것이 아니다.
    # 왼쪽 테두리로부터의 거리를 측정하면 픽셀 1의 중간은 d=0.5 거리이고, 1과 2 사이의 경계는 d=1 등 (d = p - 0.5)이다.
    # (d_new = d_old / sf)를 계산한다.
    # (p_new-0.5 = (p_old-0.5) / sf)    ->     p_new = p_old/sf + 0.5 * (1 - 1/sf)
    match_coordinates = 1.0 * out_coordinates / scale + 0.5 * (1 - 1.0 / scale)

    # 이것은 filter를 곱하기 시작하는 왼쪽 경계이다. filter size에 따라 다르다.
    left_boundary = np.floor(match_coordinates - kernel_width / 2)

    # 커버에 하위 픽셀 테두리가 있을 때 일부만 커버한 픽셀의 픽셀 중심을 'see' 하기 때문에 커널 너비를 확대할 필요가 있다.
    # 따라서 고려할 각 측면에 픽셀을 하나씩 추가한다. (weight는 0으로 만들 수 있음)
    expanded_kernel_width = np.ceil(kernel_width) + 2

    # 각 출력 위치에 대해 field_of_view 집합을 결정한다. 이는 출력 이미지의 픽셀이 '보이는' 입력 이미지의 픽셀이다.
    # 수평 dim이 출력 (큰)픽셀이고 수직 dim이 'sees' 픽셀(kernel_size + 2)인 매트릭스를 얻는다.
    field_of_view = np.squeeze(np.uint(np.expand_dims(left_boundary, axis=1) + np.arange(expanded_kernel_width) - 1))

    # 시야의 각 픽셀에 weight를 지정한다. 수평적 dim은 출력 픽셀이고 수직적 dim은 시야의 픽셀에 일치하는
    # 가중치 목록이다. ('field_of_view'에 지정됨)
    weights = fixed_kernel(1.0 * np.expand_dims(match_coordinates, axis=1) - field_of_view - 1)

    # weights를 1까지 합하도록 Normalize한다. 0으로 나누는 것을 주의하자
    sum_weights = np.sum(weights, axis=1)
    sum_weights[sum_weights == 0] = 1.0
    weights = 1.0 * weights / np.expand_dims(sum_weights, axis=1)

    # 우리는 이 거울 구조를 경계에서 반사 패딩을 위한 trick으로 사용한다.
    mirror = np.uint(np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))))
    field_of_view = mirror[np.mod(field_of_view, mirror.shape[0])]

    # weights가 0인 weights와 픽셀 위치를 제거합니다.
    non_zero_out_pixels = np.nonzero(np.any(weights, axis=0))
    weights = np.squeeze(weights[:, non_zero_out_pixels])
    field_of_view = np.squeeze(field_of_view[:, non_zero_out_pixels])

    # 최종 곱은 상대적인 위치와 일치하는 weights이며, 둘 다 output_size x fixed_kernel_size이다.
    return weights, field_of_view


