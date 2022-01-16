# This code was taken from: https://github.com/assafshocher/resizer by Assaf Shocher

import os
import torch
import torch.nn as nn
import numpy as np

from math import pi
from scipy.ndimage import filters, measurements, interpolation
from skimage import io
from utils import np2torch, torch2uint8, read_image

def img_resize(img, scale):
    img = torch2uint8(img)
    img = img_resize_in(img, scale_factor=scale)
    img = np2torch(img)
    return img

def img_resize_in(img, scale_factor=None, output_shape=None, kernel=None, antialiasing=True, kernel_shift_flag=False):
    scale_factor, output_shape = fix_scale_and_size(img.shape, output_shape, scale_factor)

    # 특정 수치 커널의 경우 conv과 sub sampling을 실행 (down scaling만)
    if type(kernel) == np.ndarray and scale_factor[0] <= 1:
        return numeric_kernel(img, kernel, scale_factor, output_shape, kernel_shift_flag)

    # interpolation method 선택
    method, kernel_width = {
        "cubic": (cubic, 4.0),
        "lanczos2": (lanczos2, 4.0),
        "lanczos3": (lanczos3, 6.0),
        "box": (box, 1.0),
        "linear": (linear, 2.0),
        None: (cubic, 4.0)
    }.get(kernel)

    # down scale시에만 사용
    antialiasing *= (scale_factor[0] < 1)

    # 각 차원에 scale에 따라 정렬
    sorted_dims = np.argsort(np.array(scale_factor)).tolist()

    out_img = np.copy(img)
    for dim in sorted_dims:
        # 1인 경우는 계산을 안 해도 됨
        if scale_factor[dim] == 1.0:
            continue

        weights, field_of_view = contributions(img.shape[dim], output_shape[dim], scale_factor[dim], method,
                                               kernel_width, antialiasing)

        out_img = resize_along_dim(out_img, dim, weights, field_of_view)

    return out_img

def resize_along_dim(out_img, dim, weights, field_of_view):
    tmp_img = np.swapaxes(out_img, dim, 0)

    weights = np.reshape(weights.T, list(weights.T.shape) + (np.ndim(out_img) - 1) * [1])

    tmp_out_img = np.sum(tmp_img[field_of_view.T] * weights, axis=0)

    return np.swapaxes(tmp_out_img, dim, 0)

def contributions(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    fixed_kernel = (lambda arg: scale * kernel(scale * arg)) if antialiasing else kernel
    kernel_width *= 1.0 / scale if antialiasing else 1.0

    out_coordinates = np.arange(1, out_length+1)

    match_coordinates = 1.0 * out_coordinates / scale + 0.5 * (1 - 1.0 / scale)

    left_boundary = np.floor(match_coordinates - kernel_width / 2)

    expanded_kernel_width = np.ceil(kernel_width) + 2

    field_of_view = np.squeeze(np.uint(np.expand_dims(left_boundary, axis=1) + np.arange(expanded_kernel_width) - 1))

    weights = fixed_kernel(1.0 * np.expand_dims(match_coordinates, axis=1) - field_of_view - 1)

    sum_weights = np.sum(weights, axis=1)
    sum_weights[sum_weights == 0] = 1.0
    weights = 1.0 * weights / np.expand_dims(sum_weights, axis=1)

    mirror = np.uint(np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))))
    field_of_view = mirror[np.mod(field_of_view, mirror.shape[0])]

    non_zero_out_pixels = np.nonzero(np.any(weights, axis=0))
    weights = np.squeeze(weights[:, non_zero_out_pixels])
    field_of_view = np.squeeze(field_of_view[:, non_zero_out_pixels])

    return weights, field_of_view


def fix_scale_and_size(input_shape, output_shape, scale_factor):
    if scale_factor is not None:
        # scale_factor를 2d로 변환
        if np.isscalar(scale_factor):
            scale_factor = [scale_factor, scale_factor]
        # scale_factor list의 크기를 input의 크기로 확장
        scale_factor = list(scale_factor)
        scale_factor.extend([1] * (len(input_shape) - len(scale_factor)))

    # output_shape를 input과 동일하게 확장
    if output_shape is not None:
        output_shape = list(np.uint(np.array(output_shape))) + list(input_shape[len(output_shape):])

    # scale_factor가 지정되지 않았을 때 output과 input으로 scale_factor계산
    if scale_factor is None:
        scale_factor = 1.0 * np.array(output_shape) / np.array(input_shape)

    # output_shape가 지정되지 않았을 때 scale_factor와 input으로 output계산
    if output_shape is None:
        output_shape = np.uint(np.ceil(np.array(input_shape) * np.array(scale_factor)))

    return scale_factor, output_shape

def numeric_kernel(img, kernel, scale_factor, output_shape, kernel_shift_flag):
    if kernel_shift_flag:
        kernel = kernel_shift(kernel, scale_factor)

    out_img = np.zeros_like(img)
    for channel in range(np.ndim(img)):
        out_img[:, :, channel] = filters.correlate(img[:, :, channel], kernel)

    return out_img[np.round(np.linspace(0, img.shape[0] - 1 / scale_factor[0], output_shape[0])).astype(int)[:, None],
                   np.round(np.linspace(0, img.shape[1] - 1 / scale_factor[1], output_shape[1])).astype(int), :]


def kernel_shift(kernel, scale_factor):
    # 먼저 커널의 현재 mass의 중심을 계산
    current_center_of_mass = measurements.center_of_mass(kernel)
    # 짝수 사이즈로 다른 shift size가 필요하여 계산
    wanted_center_of_mass = np.array(kernel.shape) / 2 + 0.5 * (scale_factor - (kernel.shape[0] % 2))

    # kernel shift에 대한 shift vector 정의 (x, y)
    shift_vec = wanted_center_of_mass - current_center_of_mass

    # shift를 적용하기 전에 먼저 shift에 의해 손실되는 일이 없도록 kernel을 pad로 고정
    kernel = np.pad(kernel, np.int(np.ceil(np.max(shift_vec))) + 1, 'constant')

    # shift 적용
    return interpolation.shift(kernel, shift_vec)


def cubic(x):
    absx = np.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return ((1.5 * absx3 - 2.5 * absx2 + 1) * (absx <= 1) +
            (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * ((1 < absx) & (absx <= 2)))

def lanczos2(x):
    return (((np.sin(pi * x) * np.sin(pi * x / 2) + np.finfo(np.float32).eps) /
             ((pi ** 2 * x ** 2 / 2) + np.finfo(np.float32).eps))
            * (abs(x) < 2))

def box(x):
    return ((-0.5 <= x) & (x < 0.5)) * 1.0

def lanczos3(x):
    return (((np.sin(pi * x) * np.sin(pi * x / 3) + np.finfo(np.float32).eps) /
             ((pi ** 2 * x ** 2 / 3) + np.finfo(np.float32).eps))
            * (abs(x) < 3))

def linear(x):
    return (x + 1) * ((-1 <= x) & (x < 0)) + (1 - x) * ((0 <= x) & (x <= 1))

