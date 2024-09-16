# Adapted from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/models/unet_blocks.py

from collections import OrderedDict
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention_processor import AttentionProcessor
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME, BaseOutput, logging
from safetensors.torch import load_file

from .resnet_3d import InflatedConv3d, InflatedGroupNorm
from .unet_3d_blocks import UNetMidBlock3DCrossAttn, get_down_block, get_up_block
import copy

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class UNet3DConditionOutput(BaseOutput):
    sample: torch.FloatTensor


class UNet3DConditionModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D",
        ),
        mid_block_type: str = "UNetMidBlock3DCrossAttn",
        up_block_types: Tuple[str] = (
            "UpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
        ),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        use_inflated_groupnorm=False, 
        # Additional
        use_motion_module=False,
        use_camera_embedding=True,
        camera_dim=9,
        motion_module_resolutions=(1, 2, 4, 8),
        motion_module_mid_block=False,
        motion_module_decoder_only=False,
        motion_module_type=None,
        motion_module_kwargs={},
        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
        use_clip_cross_attention=True,
        down_block_attention_indices=[[0],[0],[0],[]],
        mid_block_attention_index=[0],
        up_block_attention_indices=[[],[0],[0],[0]],
        # Branch
        branch_num=1,
        copy_last_n_block=1,
        copy_first_n_block=1,
        fusion="avg",
    ):
        super().__init__()

        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4

        ### branch
        self.branch_num = branch_num
        self.copy_last_n_block = copy_last_n_block
        self.copy_first_n_block = copy_first_n_block
        self.fusion = fusion
        if self.fusion == "sum":
            pass
        elif self.fusion == "avg":
            pass
        elif self.fusion == "learn":
            self.fusion_conv = nn.Conv2d(block_out_channels[self.copy_first_n_block - 1] * (self.branch_num + 1), block_out_channels[self.copy_first_n_block - 1], kernel_size=3, padding=1)
        else:
            assert False

        
        # input bcfhw -> (bf)chw -> bcfhw
        self.conv_in = InflatedConv3d(
            in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1)
        )

        ### input branch
        self.conv_in_branch = nn.ModuleList([])
        for i in range(self.branch_num):
            self.conv_in_branch.append(copy.deepcopy(self.conv_in))
        
        # time
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        
        # camera
        if use_camera_embedding and camera_dim > 0:
            self.camera_embedding = TimestepEmbedding(camera_dim, time_embed_dim)
            
        # class embedding 
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        else:
            self.class_embedding = None

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        # down
        output_channel = block_out_channels[0]
        for i, (down_block_type, down_block_attention_index) in enumerate(zip(down_block_types,down_block_attention_indices)):
            res = 2**i
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                unet_use_temporal_attention=unet_use_temporal_attention,
                use_inflated_groupnorm=use_inflated_groupnorm,
                use_motion_module=use_motion_module
                and (res in motion_module_resolutions) 
                and (not motion_module_decoder_only),
                motion_module_type=motion_module_type,
                motion_module_kwargs=motion_module_kwargs,
                use_clip_cross_attention=use_clip_cross_attention,
                attention_index=down_block_attention_index,
            )
            self.down_blocks.append(down_block)

        ### down branch
        self.down_blocks_branch = nn.ModuleList([])
        for i in range(self.branch_num):
            copy_block_list = nn.ModuleList([])
            for j in range(self.copy_first_n_block):
                copy_block_list.append(copy.deepcopy(self.down_blocks[j]))
            self.down_blocks_branch.append(copy_block_list)

        # mid
        if mid_block_type == "UNetMidBlock3DCrossAttn":
            self.mid_block = UNetMidBlock3DCrossAttn(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[-1],
                resnet_groups=norm_num_groups,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,
                unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                unet_use_temporal_attention=unet_use_temporal_attention,
                use_inflated_groupnorm=use_inflated_groupnorm,
                use_motion_module=use_motion_module and motion_module_mid_block,
                motion_module_type=motion_module_type,
                motion_module_kwargs=motion_module_kwargs,
                use_clip_cross_attention=use_clip_cross_attention,
                attention_index=mid_block_attention_index,
            )
        else:
            raise ValueError(f"unknown mid_block_type : {mid_block_type}")

        # count how many layers upsample the videos
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_attention_head_dim = list(reversed(attention_head_dim))
        only_cross_attention = list(reversed(only_cross_attention))
        output_channel = reversed_block_out_channels[0]
        for i, (up_block_type, up_block_attention_index) in enumerate(zip(up_block_types, up_block_attention_indices)):
            res = 2 ** (3 - i)
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[
                min(i + 1, len(block_out_channels) - 1)
            ]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=reversed_attention_head_dim[i],
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                unet_use_temporal_attention=unet_use_temporal_attention,
                use_inflated_groupnorm=use_inflated_groupnorm,
                use_motion_module=use_motion_module
                and (res in motion_module_resolutions),
                motion_module_type=motion_module_type,
                motion_module_kwargs=motion_module_kwargs,
                use_clip_cross_attention=use_clip_cross_attention,
                attention_index = up_block_attention_index,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        ### up branch
        self.up_blocks_branch = nn.ModuleList([])
        for i in range(self.branch_num):
            copy_block_list = nn.ModuleList([])
            for j in range(self.copy_last_n_block, 0, -1):
                copy_block_list.append(copy.deepcopy(self.up_blocks[-j]))
            self.up_blocks_branch.append(copy_block_list)
        
        # out
        if use_inflated_groupnorm:
            self.conv_norm_out = InflatedGroupNorm(
                num_channels=block_out_channels[0],
                num_groups=norm_num_groups,
                eps=norm_eps,
            )
        else:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[0],
                num_groups=norm_num_groups,
                eps=norm_eps,
            )
        ### conv_norm_out branch
        self.conv_norm_out_branch = nn.ModuleList([])
        for i in range(self.branch_num):
            self.conv_norm_out_branch.append(copy.deepcopy(self.conv_norm_out))
        
        self.conv_act = nn.SiLU()
        ### conv_act banch
        self.conv_act_branch = nn.ModuleList([])
        for i in range(self.branch_num):
            self.conv_act_branch.append(copy.deepcopy(self.conv_act))
        
        self.conv_out = InflatedConv3d(
            block_out_channels[0], out_channels, kernel_size=3, padding=1
        )
        ### conv_out branch
        self.conv_out_branch = nn.ModuleList([])
        for i in range(self.branch_num):
            self.conv_out_branch.append(copy.deepcopy(self.conv_out))
            
    @property
    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(
            name: str,
            module: torch.nn.Module,
            processors: Dict[str, AttentionProcessor],
        ):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"] = module.processor

            for sub_name, child in module.named_children():
                if "temporal_transformer" not in sub_name:
                    fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            if "temporal_transformer" not in name:
                fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attention_slice(self, slice_size):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                `"max"`, maxium amount of memory will be saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_slicable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_slicable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_slicable_dims(module)

        num_slicable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_slicable_layers * [1]

        slice_size = (
            num_slicable_layers * [slice_size]
            if not isinstance(slice_size, list)
            else slice_size
        )

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(
            module: torch.nn.Module, slice_size: List[int]
        ):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(
        self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]
    ):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                if "temporal_transformer" not in sub_name:
                    fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            if "temporal_transformer" not in name:
                fn_recursive_attn_processor(name, module, processor)

    def forward(
        self,
        sample: torch.FloatTensor,
        sample_noisy_list: List[torch.FloatTensor], ### branch input
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        camera: Optional[torch.Tensor] = None, 
        class_labels: Optional[torch.Tensor] = None,
        pose_cond_fea: Optional[torch.Tensor] = None, # master smpl pose
        pose_cond_fea_list: Optional[List[torch.Tensor]] = None, # branch smpl pose
        # normal_cond_fea: Optional[torch.Tensor] = None,
        # depth_cond_fea: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet3DConditionOutput, Tuple]:
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, frame, height, width) noisy inputs tensor       
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states    
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0]) # b

        t_emb = self.time_proj(timesteps)  # b 320

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb) # b 1280
        
        # camera_emb
        camera = camera.expand(sample.shape[0], -1, -1) 
        camera_emb = self.camera_embedding(camera) if camera is not None else 0 # b f c=1280
        
        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError(
                    "class_labels should be provided when num_class_embeds > 0"
                )

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # 2. pre-process
        sample = self.conv_in(sample)
        if pose_cond_fea is not None:
            sample = sample + pose_cond_fea
        
        ### input branch
        sample_in_list = []
        for i in range(self.branch_num):
            sample_in_list.append(self.conv_in_branch[i](sample_noisy_list[i]))

        if len(pose_cond_fea_list) > 0: 
            for i in range(self.branch_num):
                if pose_cond_fea_list[i] is not None:
                    sample_in_list[i] = sample_in_list[i] + pose_cond_fea_list[i]
        

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks[:self.copy_first_n_block]:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    camera_emb=camera_emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    camera_emb=camera_emb,
                )

            down_block_res_samples += res_samples
            
        down_block_res_samples_list = []
        for i in range(self.branch_num): 
            down_block_res_samples_list.append((sample_in_list[i],))
        for j in range(self.branch_num):
            for i, downsample_block in enumerate(self.down_blocks_branch[j]):
                if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                    sample_in_list[j], res_samples = downsample_block(
                        hidden_states=sample_in_list[j],
                        temb=emb,
                        camera_emb=camera_emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                    ) 
                else:
                    sample_in_list[j], res_samples = downsample_block(
                        hidden_states=sample_in_list[j], temb=emb, camera_emb=camera_emb,
                    )

                down_block_res_samples_list[j] += res_samples

        sample_list = [sample]
        for i in range(self.branch_num):
            sample_list.append(sample_in_list[i])
            
        if self.fusion == "sum" or self.fusion == "avg":
            stacked_tensor = torch.stack(sample_list, dim=0)
            sample = torch.sum(stacked_tensor, dim=0)
            if self.fusion == "avg":
                sample = sample / (1 + self.branch_num)
        elif self.fusion == "learn":
            concat_tensor = torch.cat(sample_list, dim=1)
            sample = self.fusion_conv(concat_tensor)
        else:
            assert False
        
        for downsample_block in self.down_blocks[self.copy_first_n_block:]:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample, 
                    temb=emb, # time emb
                    camera_emb=camera_emb,
                    encoder_hidden_states=encoder_hidden_states, # image emb
                    attention_mask=attention_mask, # None
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample, temb=emb, camera_emb=camera_emb, 
                )

            down_block_res_samples += res_samples
            for i in range(self.branch_num): 
                down_block_res_samples_list[i] += res_samples   

        # mid
        sample = self.mid_block(
            sample,
            emb,
            camera_emb=camera_emb,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        )

        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks[:-self.copy_last_n_block]):
            is_final_block = False

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :] 
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)] 
            for j in range(self.branch_num):
                down_block_res_samples_list[j] = down_block_res_samples_list[j][: -len(upsample_block.resnets)] 

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    camera_emb=camera_emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    camera_emb=camera_emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )
        
        sample_list = []
        for j in range(self.branch_num):
            sample_copy = sample.clone()
            for i, upsample_block in enumerate(self.up_blocks_branch[j]):
                is_final_block = i == len(self.up_blocks_branch[j]) - 1

                res_samples = down_block_res_samples_list[j][-len(upsample_block.resnets) :]
                down_block_res_samples_list[j] = down_block_res_samples_list[j][: -len(upsample_block.resnets)]

                # if we have not reached the final block and need to forward the
                # upsample size, we do it here
                if not is_final_block and forward_upsample_size:
                    upsample_size = down_block_res_samples_list[j][-1].shape[2:]

                if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                    sample_copy = upsample_block(
                        hidden_states=sample_copy,
                        temb=emb,
                        camera_emb=camera_emb,
                        res_hidden_states_tuple=res_samples,
                        encoder_hidden_states=encoder_hidden_states,
                        upsample_size=upsample_size,
                        attention_mask=attention_mask,
                    )
                else:
                    sample_copy = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    camera_emb=camera_emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )
            sample_list.append(sample_copy)
        
        for i, upsample_block in enumerate(self.up_blocks[-self.copy_last_n_block:]):
            is_final_block = i == len(self.up_blocks[-self.copy_last_n_block:]) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    camera_emb=camera_emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    camera_emb=camera_emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )
            
        # 6. post-process
        ## master 
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample) # b,c,f,h,w 
        ## branch
        output_list = [sample]
        for i, sample_tmp in enumerate(sample_list):
            if self.conv_norm_out_branch[i]:
                sample_tmp = self.conv_norm_out_branch[i](sample_tmp)
                sample_tmp = self.conv_act_branch[i](sample_tmp)
            sample_tmp = self.conv_out_branch[i](sample_tmp)
            output_list.append(sample_tmp)
        sample = torch.cat(output_list, dim=1) # b,c*(branch_num+1),f,h,w 
        
        if not return_dict:
            return (sample,)

        return UNet3DConditionOutput(sample=sample)

    @classmethod
    def from_pretrained_2d(
        cls,
        pretrained_model_path: PathLike,
        motion_module_path: PathLike,
        subfolder=None,
        unet_additional_kwargs=None,
        mm_zero_proj_out=False,
        load_pe=True, # True in inference, False in training
    ):
        pretrained_model_path = Path(pretrained_model_path)
        motion_module_path = Path(motion_module_path)
        if subfolder is not None:
            pretrained_model_path = pretrained_model_path.joinpath(subfolder)
        logger.info(
            f"loaded temporal unet's pretrained weights from {pretrained_model_path} ..."
        )

        config_file = pretrained_model_path / "config.json"
        if not (config_file.exists() and config_file.is_file()):
            raise RuntimeError(f"{config_file} does not exist or is not a file")

        unet_config = cls.load_config(config_file)
        unet_config["_class_name"] = cls.__name__
        unet_config["down_block_types"] = [
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D",
        ]
        unet_config["up_block_types"] = [
            "UpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
        ]
        unet_config["mid_block_type"] = "UNetMidBlock3DCrossAttn"

        model = cls.from_config(unet_config, **unet_additional_kwargs)
        # load the vanilla weights
        if pretrained_model_path.joinpath(SAFETENSORS_WEIGHTS_NAME).exists():
            logger.debug(
                f"loading safeTensors weights from {pretrained_model_path} ..."
            )
            state_dict = load_file(
                pretrained_model_path.joinpath(SAFETENSORS_WEIGHTS_NAME), device="cpu"
            )

        elif pretrained_model_path.joinpath(WEIGHTS_NAME).exists():
            logger.debug(f"loading weights from {pretrained_model_path} ...")
            state_dict = torch.load(
                pretrained_model_path.joinpath(WEIGHTS_NAME),
                map_location="cpu",
                weights_only=True,
            )
        else:
            raise FileNotFoundError(f"no weights file found in {pretrained_model_path}")

        # initialize attn0 & norm0
        extra_state_dict = {} # use pretrained of attn1 & norm1
        for k in state_dict:
            if "transformer_blocks" in k:
                if "attn1" in k:
                    extra_k = k.replace("attn1", "attn0")
                    extra_state_dict[extra_k] = state_dict[k].clone()
                if "norm1" in k:
                    extra_k = k.replace("norm1", "norm0")
                    extra_state_dict[extra_k] = state_dict[k].clone()
        state_dict.update(extra_state_dict)
        
        # load the motion module weights 
        if motion_module_path.exists() and motion_module_path.is_file():
            if motion_module_path.suffix.lower() in [".pth", ".pt", ".ckpt"]:
                logger.info(f"Load motion module params from {motion_module_path}")
                motion_state_dict = torch.load(
                    motion_module_path, map_location="cpu", weights_only=True
                )
            elif motion_module_path.suffix.lower() == ".safetensors":
                motion_state_dict = load_file(motion_module_path, device="cpu")
            else:
                raise RuntimeError(
                    f"unknown file format for motion module weights: {motion_module_path.suffix}"
                )
            if mm_zero_proj_out:
                logger.info(f"Zero initialize proj_out layers in motion module...")
                new_motion_state_dict = OrderedDict()
                for k in motion_state_dict:
                    if "proj_out" in k:
                        continue
                    new_motion_state_dict[k] = motion_state_dict[k]
                motion_state_dict = new_motion_state_dict
            
            if load_pe: 
                motion_state_dict = motion_state_dict
            else: 
                new_motion_state_dict = {k: v for k, v in motion_state_dict.items() if 'pe' not in k}
                motion_state_dict = new_motion_state_dict
                
            # merge the state dicts
            state_dict.update(motion_state_dict)
                
        # use master weights to initialize branch
        branch_state_dict = {}
        for name, param in model.named_parameters():
            if ('branch' in name 
                and 'camera' not in name 
                and 'relative' not in name): 
                
                name_splits = name.split('.')
                master_name = '.'.join([name_splits[0]]+name_splits[2:]).replace("_branch", "") 
                if 'up_blocks' in name: 
                    name_splits = master_name.split('.')
                    block_id_in_branch = int(name_splits[1])
                    block_id_in_master = block_id_in_branch + 4 - model.copy_last_n_block
                    master_name = '.'.join([name_splits[0], str(block_id_in_master)]+name_splits[2:]) 
                assert master_name in state_dict.keys(), f"{master_name} missing"
                assert param.shape == state_dict[master_name].shape, f"{name} shape mismatch"
                branch_state_dict[name] = state_dict[master_name].clone()
        state_dict.update(branch_state_dict)

        # load the weights into the model
        m, u = model.load_state_dict(state_dict, strict=False)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        logger.debug(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")

        return model
