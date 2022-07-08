# ------------------------------------------------------------------------------------
# Modified from Differentiable Augmentation (https://github.com/odegeasslbc/FastGAN-pytorch)
# Copyright (c) 2020 Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu and Song Han. All Rights Reserved.
# ------------------------------------------------------------------------------------

from typing import Tuple

import torch
import torch.nn.functional as F


def DiffAugment(x: torch.FloatTensor, y: torch.FloatTensor, policy: str = '',
                channels_first: bool = True) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
            y = y.permute(0, 3, 1, 2)

        for p in policy.replace(' ', '').split(','):
            for f in AUGMENT_FNS[p]:
                x, y = f(x, y)

        if not channels_first:
            x = x.permute(0, 2, 3, 1)
            y = y.permute(0, 2, 3, 1)

    x = x.contiguous()
    y = y.contiguous()

    return x, y


def rand_brightness(x: torch.FloatTensor, y: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    rand_color = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)

    x = x + (rand_color - 0.5)
    y = y + (rand_color - 0.5)

    x = x.clamp(0, 1)
    y = y.clamp(0, 1)

    return x, y


def rand_saturation(x: torch.FloatTensor, y: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    rand_color = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)

    x_mean = x.mean(dim=1, keepdim=True)
    y_mean = y.mean(dim=1, keepdim=True)

    x = (x - x_mean) * (rand_color * 2) + x_mean
    y = (y - y_mean) * (rand_color * 2) + y_mean

    x = x.clamp(0, 1)
    y = y.clamp(0, 1)

    return x, y


def rand_contrast(x: torch.FloatTensor, y: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    rand_color = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)

    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    y_mean = y.mean(dim=[1, 2, 3], keepdim=True)

    x = (x - x_mean) * (rand_color + 0.5) + x_mean
    y = (y - y_mean) * (rand_color + 0.5) + y_mean

    x = x.clamp(0, 1)
    y = y.clamp(0, 1)

    return x, y


def rand_translation(x: torch.FloatTensor, y: torch.FloatTensor, ratio: float = 0.125) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)

    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)

    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )

    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)

    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    y_pad = F.pad(y, [1, 1, 1, 1, 0, 0, 0, 0])

    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    y = y_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)

    return x, y


def rand_cutout(x: torch.FloatTensor, y: torch.FloatTensor, ratio: float = 0.5) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)

    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)

    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )

    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)

    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0

    x = x * mask.unsqueeze(1)
    y = y * mask.unsqueeze(1)

    return x, y


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}
