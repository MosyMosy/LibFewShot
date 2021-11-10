# -*- coding: utf-8 -*-
from .conv_four import Conv32F, Conv64F, R2D2Embedding
from .conv_four_no_affine import Conv64F_no_affine, Conv32F_no_affine, R2D2Embedding_no_affine
from .resnet_12 import resnet12
from .resnet_18 import resnet18
from .wrn import WRN
from .resnet_12_mtl_offcial import resnet12MTLofficial
from .vit import ViT

from core.model.backbone.utils.maml_module import convert_maml_module


def get_backbone(config):
    """Get the backbone according to the config dict.

    Args:
        config: The config dict.

    Returns: The backbone module.

    """
    kwargs = dict()
    kwargs.update(config["backbone"]["kwargs"])
    try:
        emb_func = eval(config["backbone"]["name"])(**kwargs)
    except NameError:
        raise ("{} is not implemented".format(config["backbone"]["name"]))

    return emb_func
