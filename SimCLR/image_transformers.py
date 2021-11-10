import torch
import timm
from timm.models.helpers import overlay_external_default_cfg
from timm.models.vision_transformer import VisionTransformer, default_cfgs, build_model_with_cfg, checkpoint_filter_fn
from timm.models.registry import register_model
from typing import Dict
from copy import deepcopy

class Image2DTransformer(VisionTransformer):
    def __init__(self, remove_tokens_outputs=False, **kwargs):
      super(Image2DTransformer, self).__init__(**kwargs)
      self.remove_tokens_outputs = remove_tokens_outputs

    def forward_blocks(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        copied from timm Library

        function runs input image and returns outputs of transformer blocks 
        
        Args:
            x: tensor of input image shape (B, C, H, W)
        
        Returns
            dictionary map block number -> output of the block
        """
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        outputs = dict()
        for i, block in enumerate(self.blocks):
            x = block(x)
            if self.remove_tokens_outputs:
                if self.dist_token is None:
                    outputs[str(i)] = x[:, 1:, :] # remove class token output
                else:
                    outputs[str(i)] = x[:, 2:, :] # remove class and dist tokens outputs
            else:
                outputs[str(i)] = x
        return outputs


def _create_transformer_2d(variant, pretrained=False, default_cfg=None, **kwargs):
    """ copied from timm library """
    if default_cfg is None:
        default_cfg = deepcopy(default_cfgs[variant])
    overlay_external_default_cfg(default_cfg, kwargs)
    default_num_classes = default_cfg['num_classes']
    default_img_size = default_cfg['input_size'][-2:]

    num_classes = kwargs.pop('num_classes', default_num_classes)
    img_size = kwargs.pop('img_size', default_img_size)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        # Remove representation layer if fine-tuning. This may not always be the desired action,
        # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
        print("Removing representation layer for fine-tuning.")
        repr_size = None

    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = build_model_with_cfg(
        Image2DTransformer, variant, pretrained,
        default_cfg=default_cfg,
        img_size=img_size,
        num_classes=num_classes,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)

    return model

@register_model
def image_2d_transformer(pretrained=False, **kwargs):
    """
    modified copy from timm 
    DeiT base model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_transformer_2d('vit_deit_base_patch16_384', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def image_2d_distilled_transformer(pretrained=False, **kwargs):
    """ 
    modified copy from timm 
    DeiT-base distilled model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_transformer_2d(
        'vit_deit_base_distilled_patch16_384', pretrained=pretrained, distilled=True, **model_kwargs)
    return model