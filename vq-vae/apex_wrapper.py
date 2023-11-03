import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import apex
    APEX_IS_AVAILABLE = True
except ImportError:
    print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")
    APEX_IS_AVAILABLE = False


def LayerNorm(hidden_size, eps=0.00001, elementwise_affine=True):
    if APEX_IS_AVAILABLE:
        return apex.normalization.FusedLayerNorm(hidden_size, eps, elementwise_affine)
    return nn.LayerNorm(hidden_size, eps, elementwise_affine)
