# Adapted from https://github.com/magic-research/magic-animate/blob/main/magicanimate/models/mutual_self_attention.py
from typing import Any, Dict, Optional

import torch
from einops import rearrange

from src.models.attention import TemporalBasicTransformerBlock

from .attention import BasicTransformerBlock


def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result

def torch_named_dfs(model: torch.nn.Module, name: str='unet'):
    result = [(name, model)]
    for child_name, child_model in model.named_children():
        result += torch_named_dfs(child_model, f'{name}.{child_name}')
    return result

class ReferenceAttentionControl:
    def __init__(
        self,
        unet,
        mode="write",
        do_classifier_free_guidance=False,
        reference_attn=True,
        reference_adain=False,
        fusion_blocks="midup",
        batch_size=1,
    ) -> None:
        self.unet = unet
        assert mode in ["read_cross_attn", "read_concat_attn", "write"]
        assert fusion_blocks in ["midup", "full"]
        self.reference_attn = reference_attn
        self.reference_adain = reference_adain
        self.fusion_blocks = fusion_blocks
        self.register_reference_hooks(
            mode,
            do_classifier_free_guidance,
            reference_attn,
            batch_size=batch_size,
        )

    def register_reference_hooks(
        self,
        mode,
        do_classifier_free_guidance,
        reference_attn=True,
        batch_size=1,
        num_images_per_prompt=1,
        device=torch.device("cpu"),
    ):
        MODE = mode
        do_classifier_free_guidance = do_classifier_free_guidance
        reference_attn = reference_attn # True
        num_images_per_prompt = num_images_per_prompt
        if do_classifier_free_guidance:
            uc_mask = (
                torch.Tensor(
                    [1] * batch_size * num_images_per_prompt * 16
                    + [0] * batch_size * num_images_per_prompt * 16
                )
                .to(device)
                .bool()
            )
        else: 
            uc_mask = (
                torch.Tensor([0] * batch_size * num_images_per_prompt * 2) #2
                .to(device)
                .bool()
            ) # [0,0]

        def hacked_basic_transformer_inner_forward(
            self, 
            hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
            video_length=None,
        ):
            # 1. Self-Attention
            cross_attention_kwargs = (
                cross_attention_kwargs if cross_attention_kwargs is not None else {}
            )
            if MODE == "write":
                # attn0(self attn)
                norm_hidden_states = self.norm0(hidden_states) # b l c
                hidden_states  = (
                    self.attn0(
                    norm_hidden_states,
                    encoder_hidden_states=None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                ) 
                    + hidden_states 
                )
                # attn1 (self attn)
                norm_hidden_states = self.norm1(hidden_states)
                self.bank.append(norm_hidden_states.clone())
                hidden_states  = (
                    self.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=None, # None
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                ) 
                    + hidden_states 
                )
            elif "read" in MODE: # ["read_concat_attn", "read_cross_attn"] 
                # attn0 single/multi-view attn
                norm_hidden_states = self.norm0(hidden_states) # (b f) (l=hw) c
                attention_index = self.attention_index * video_length //360 # angle to index
                attention_index_matrix = torch.arange(video_length).unsqueeze(1) + attention_index  # stage1: 1x1    stage2: num_views x selected_views
                attention_index_matrix = attention_index_matrix % video_length  
                norm_hidden_states = rearrange(norm_hidden_states, "(b f) l c -> b l f c", f=video_length)
                sample_norm_hidden_states = norm_hidden_states[:, :, attention_index_matrix, :] # b l f s c
                
                norm_hidden_states = rearrange(norm_hidden_states, "b l f c -> (b f) l c")
                sample_norm_hidden_states = rearrange(sample_norm_hidden_states, "b l f s c -> (b f) (l s) c")
                
                hidden_states  = (
                    self.attn0(
                    norm_hidden_states,
                    encoder_hidden_states=sample_norm_hidden_states, # 3D attention with selected views
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                ) 
                    + hidden_states 
                ) # (b f) l c
                
                # attn1 ref-attn
                norm_hidden_states = self.norm1(hidden_states)
                bank_fea = [
                    rearrange(
                        d.unsqueeze(1).repeat(1, video_length, 1, 1), 
                        "b t l c -> (b t) l c",                        
                    )
                    for d in self.bank
                ]
                modify_norm_hidden_states = torch.cat(
                    [norm_hidden_states] + bank_fea, dim=1
                ) # (bt) l c -> (bt) 2l c 
                hidden_states_uc = ( 
                    self.attn1( 
                        norm_hidden_states, 
                        encoder_hidden_states=modify_norm_hidden_states, 
                        attention_mask=attention_mask,
                    )
                    + hidden_states
                )
                if do_classifier_free_guidance: # only for inference
                    hidden_states_c = hidden_states_uc.clone() 
                    _uc_mask = uc_mask.clone()
                    if hidden_states.shape[0] != _uc_mask.shape[0]:
                        _uc_mask = (
                            torch.Tensor(
                                [1] * (hidden_states.shape[0] // 2)
                                + [0] * (hidden_states.shape[0] // 2)
                            )
                            .to(device)
                            .bool()
                        )
                    hidden_states_c[_uc_mask] = (
                        self.attn1(
                            norm_hidden_states[_uc_mask],
                            encoder_hidden_states=norm_hidden_states[_uc_mask], 
                            attention_mask=attention_mask,
                        )
                        + hidden_states[_uc_mask]
                    )
                    hidden_states = hidden_states_c.clone()
                else: # for training
                    hidden_states = hidden_states_uc

            # Feed-forward
            hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

            return hidden_states
            
        if self.reference_attn:
            if self.fusion_blocks == "midup":
                attn_modules = [
                    module
                    for module in (
                        torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks)
                    )
                    if isinstance(module, BasicTransformerBlock)
                    or isinstance(module, TemporalBasicTransformerBlock)
                ]
            elif self.fusion_blocks == "full":
                attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, BasicTransformerBlock)
                    or isinstance(module, TemporalBasicTransformerBlock)
                ]
            attn_modules = sorted(
                attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )

            for i, module in enumerate(attn_modules):
                module._original_inner_forward = module.forward
                if isinstance(module, BasicTransformerBlock):
                    module.forward = hacked_basic_transformer_inner_forward.__get__(
                        module, BasicTransformerBlock
                    )
                if isinstance(module, TemporalBasicTransformerBlock):
                    module.forward = hacked_basic_transformer_inner_forward.__get__(
                        module, TemporalBasicTransformerBlock
                    )

                module.bank = []
                module.attn_weight = float(i) / float(len(attn_modules))
                
                
    def update(self, writer, dtype=torch.float16):
        if self.reference_attn:
            if self.fusion_blocks == "midup":
                reader_attn_modules = [
                    module
                    for module in (
                        torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks)
                    )
                    if isinstance(module, TemporalBasicTransformerBlock)
                ]
                writer_attn_modules = [
                    module
                    for module in (
                        torch_dfs(writer.unet.mid_block)
                        + torch_dfs(writer.unet.up_blocks)
                    )
                    if isinstance(module, BasicTransformerBlock)
                ]
            elif self.fusion_blocks == "full": 
                reader_attn_modules = [
                    (name, module)
                    for name, module in torch_named_dfs(self.unet) # denoise_net
                    if isinstance(module, TemporalBasicTransformerBlock)
                ]
                writer_attn_modules = [
                    (name, module)
                    for name, module in torch_named_dfs(writer.unet) # ref_net
                    if isinstance(module, BasicTransformerBlock)
                ]
        
            reader_attn_modules = sorted( 
                reader_attn_modules, key=lambda x: -x[1].norm1.normalized_shape[0]
            )
            writer_attn_modules = sorted(
                writer_attn_modules, key=lambda x: -x[1].norm1.normalized_shape[0]
            )
            
            for r, w in zip(reader_attn_modules, writer_attn_modules):
                r[1].bank = [v.clone().to(dtype) for v in w[1].bank]

    def clear(self):
        if self.reference_attn:
            if self.fusion_blocks == "midup":
                reader_attn_modules = [
                    module
                    for module in (
                        torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks)
                    )
                    if isinstance(module, BasicTransformerBlock)
                    or isinstance(module, TemporalBasicTransformerBlock)
                ]
            elif self.fusion_blocks == "full":
                reader_attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, BasicTransformerBlock)
                    or isinstance(module, TemporalBasicTransformerBlock)
                ]
            reader_attn_modules = sorted(
                reader_attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            ) 
            for r in reader_attn_modules:
                r.bank.clear()
