import os
from functools import partial
import torch
import time

from .vmamba import VSSM

from .mamba_config import get_config

def build_vssm_model(config, mamba_patch=2, ds_scale=[2,3,1], topconv_scale=1, inchans=1):
    model_type = config.MODEL.TYPE
    if model_type in ["vssm"]:
        model = VSSM(
            patch_size=mamba_patch,
            in_chans=inchans,
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.VSSM.DEPTHS, 
            dims=config.MODEL.VSSM.EMBED_DIM, 
            # ===================
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
            ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
            ssm_conv=config.MODEL.VSSM.SSM_CONV,
            ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
            ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
            ssm_init=config.MODEL.VSSM.SSM_INIT,
            forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
            # ===================
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            # ===================
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=False,
            # ===================
            posembed=config.MODEL.VSSM.POSEMBED,
            imgsize=config.MODEL.INPUTSIZE,
            downsample_scale=ds_scale,
            topconv_scale=topconv_scale
        )
        return model

    return None


def build_mamba_model(mamba_version, mamba_patch=4, ds_scale=[2,3,1], topconv_scale=1, inchans=1):
    config = get_config(mamba_version)
    model = build_vssm_model(config, mamba_patch, ds_scale, topconv_scale, inchans)
    return model
