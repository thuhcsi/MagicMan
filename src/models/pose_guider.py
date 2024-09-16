from typing import Tuple, List, Union

import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.modeling_utils import ModelMixin

from src.models.motion_module import zero_module
from src.models.resnet_3d import InflatedConv3d
from src.models.transformer_3d import Transformer3DModel


class PoseGuider(ModelMixin):
    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Union[Tuple[int], List[int]] = (16, 32, 64, 128), 
        use_guidance_attention: bool = True, 
        attention_num_heads: int = 8,
    ):
        super().__init__()
        if not isinstance(block_out_channels, tuple):
            block_out_channels = tuple(block_out_channels)
            
        self.conv_in = InflatedConv3d(
            conditioning_channels, block_out_channels[0], kernel_size=3, padding=1
        )

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(
                InflatedConv3d(channel_in, channel_in, kernel_size=3, padding=1)
            )
            self.blocks.append(
                InflatedConv3d(
                    channel_in, channel_out, kernel_size=3, padding=1, stride=2
                )
            )
        
        if use_guidance_attention:
            self.guidance_attention = Transformer3DModel( 
                    attention_num_heads, 
                    block_out_channels[-1] // attention_num_heads,
                    block_out_channels[-1],
                    norm_num_groups=32,
                    unet_use_cross_frame_attention=False,
                    unet_use_temporal_attention=False,
                    use_clip_cross_attention=False, 
                )
        else:
            self.guidance_attention = None

        self.conv_out = zero_module(
            InflatedConv3d(
                block_out_channels[-1],
                conditioning_embedding_channels,
                kernel_size=3,
                padding=1,
            )
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        if self.guidance_attention is not None:
            embedding = self.guidance_attention(embedding).sample
        
        embedding = self.conv_out(embedding)

        return embedding
