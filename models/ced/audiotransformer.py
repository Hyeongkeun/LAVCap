# Copyright 2023-2024 Xiaomi Corporation and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
import torchaudio.transforms as audio_transforms
from einops import rearrange
from einops.layers.torch import Rearrange
import logging
from transformers import PretrainedConfig, PreTrainedModel

from timm.models.layers import to_2tuple, DropPath

from models.ced.layers import (Mlp, drop_patches, trunc_normal_)

class AudioPatchEmbed(nn.Module):
    def __init__(self, input_size=224, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        input_size = to_2tuple(input_size)
        patch_size = to_2tuple(patch_size)
        self.grid_size = (input_size[0] // patch_size[0], input_size[1] // patch_size[1],)
        num_patches = self.grid_size[0] * self.grid_size[1]

        self.img_size = input_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x)


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            attn_drop=0.0,
            proj_drop=0.0,
            causal: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.causal = causal

    def forward(self, x):
        B, N, C = x.shape
        q = (self.q_proj(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3))
        k = (self.k_proj(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3))
        v = (self.v_proj(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3))

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if self.causal:
            mask_value = -torch.finfo(attn.dtype).max
            i, j = attn.shape[-2:]
            mask = torch.ones(i, j, device=q.device, dtype=torch.bool).triu(j - i + 1)
            attn = attn.masked_fill(mask, mask_value)
        attn = attn.softmax(dim=-1)
        # Only for the case that a mask with all True entries on a row is passed.
        # attn = torch.nan_to_num(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer: Callable = nn.GELU, norm_layer: Callable = nn.LayerNorm,
                 attention_kwargs={}, **kwargs,):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, **attention_kwargs,)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.ls1 = nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.ls2 = nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,)

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

# class AudioTransformer(PreTrainedModel):
class AudioTransformer(nn.Module):
    def __init__(self, output_dim=527, n_mels=64, target_length=1024, patch_size=16,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True, 
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=None, 
                 act_layer=None, pad_last = True, pooling="mean", # "mean", "token", "dm", "logit
                 time_patch_out: Optional[float] = None, freq_patch_out: Optional[float] = None,
                 eval_avg="mean", **kwargs):
        super().__init__()

        assert pooling in ("mean", "token", "dm", "logit")

        self.output_dim = output_dim
        self.n_mels = n_mels
        self.target_length = target_length
        self.patch_size = patch_size

        self.pad_last = pad_last
        self.pooling = pooling
        self.time_patch_out = time_patch_out
        self.freq_patch_out = freq_patch_out
        self.eval_avg = eval_avg

        self.init_bn = nn.Sequential(Rearrange("b c f t -> b f c t"),
                                     torch.nn.BatchNorm2d(self.n_mels, momentum=0.01),
                                     Rearrange("b f c t -> b c f t"), )

        # Allowed length in number of frames, otherwise the positional embedding will throw an error
        self.maximal_allowed_length = self.target_length

        self.patch_embed = AudioPatchEmbed(input_size=(self.n_mels, target_length), patch_size=self.patch_size, embed_dim=embed_dim)

        if self.pooling == "token":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.token_pos_embed = nn.Parameter(torch.randn(1, embed_dim) * 0.02)

        self.time_pos_embed = nn.Parameter(torch.randn(1, embed_dim, 1, self.patch_embed.grid_size[1]) * 0.02)
        self.freq_pos_embed = nn.Parameter(torch.randn(1, embed_dim, self.patch_embed.grid_size[0], 1) * 0.02)

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.Sequential(
            *[Block(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, 
                         drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer,) for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)
        
        self.output_layer = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, output_dim))

        self.apply(self.init_weights)
        if hasattr(self, "cls_token"):
            nn.init.normal_(self.cls_token, std=1e-6)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"time_pos_embed", "cls_token", "freq_pos_embed", "token_pos_embed"}

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def load_state_dict(self, path):
        self_state = self.state_dict();
        loaded_state = torch.load(path, map_location=torch.device("cpu"))
        for origname, param in loaded_state.items():
            name = origname.replace("outputlayer","output_layer")
            if "front_end" in name:
                continue
            if "qkv" in name:
                qkv_size = param.shape[0] // 3
                q_name, q_param = name.replace("qkv", "q_proj"), param[:qkv_size]
                k_name, k_param = name.replace("qkv", "k_proj"), param[qkv_size:qkv_size*2]
                v_name, v_param = name.replace("qkv", "v_proj"), param[qkv_size*2:]
                self_state[q_name].copy_(q_param)
                self_state[k_name].copy_(k_param)
                self_state[v_name].copy_(v_param)
                logging.info("{} is loaded into q,k,v.".format(origname))
                continue

            if name not in self_state:
                logging.info("{} is not in the model.".format(origname))
                continue
            # else: logging.info("{} is loaded".format(name))

            if self_state[name].size() != loaded_state[origname].size():
                if "time_pos_embed" in name:
                    target_time_pos_embed_length, target_freq_pos_embed_length = self_state["time_pos_embed"].shape[-1], self_state["freq_pos_embed"].shape[-2]
                    pretrained_time_pos_embed, pretrained_freq_pos_embed = loaded_state["time_pos_embed"], loaded_state["freq_pos_embed"]

                    if target_time_pos_embed_length <= pretrained_time_pos_embed.shape[-1]:
                        self_state["time_pos_embed"].copy_(pretrained_time_pos_embed[...,:target_time_pos_embed_length])
                    else: 
                        self_state["time_pos_embed"].copy_(torch.nn.functional.interpolate(pretrained_time_pos_embed, size=(1, target_time_pos_embed_length),align_corners=False, mode="bilinear",))
                    
                    if target_freq_pos_embed_length <= pretrained_freq_pos_embed.shape[-1]:
                        self_state["freq_pos_embed"].copy_(pretrained_freq_pos_embed[:, :, :target_freq_pos_embed_length, :])
                    else: 
                        self_state["freq_pos_embed"].copy_(torch.nn.functional.interpolate(pretrained_freq_pos_embed, size=(target_freq_pos_embed_length, 1), align_corners=False, mode="bilinear",))
                    logging.info("Interpolate parameter {} length:, model: {} -> loaded: {}".format(name, self_state["time_pos_embed"].size(), loaded_state["time_pos_embed"].size()))
                    logging.info("Interpolate parameter {} length:, model: {} -> loaded: {}".format("freq_pos_embed", self_state["freq_pos_embed"].size(), loaded_state["freq_pos_embed"].size()))
                else:
                    logging.info("Wrong parameter length: {}, model: {}, loaded: {}".format(origname, self_state[name].size(), loaded_state[origname].size()))
                continue

            self_state[name].copy_(param)

        logging.info("Everything is alright, except above log.\n") 


    def forward(self, aud):
        aud = aud.unsqueeze(1).transpose(2, 3) # batch x 1 x temp x freq -> batch x 1 x freq x temp

        aud = self.init_bn(aud)
        aud = self.patch_embed(aud)

        B, C, F, T = aud.shape
        aud = aud + self.time_pos_embed[:, :, :, :T]
        aud = (aud + self.freq_pos_embed[:, :, :, :]) 

        if self.time_patch_out is not None:
            aud = drop_patches(aud, dim=-1, frac=self.time_patch_out)
        if self.freq_patch_out is not None:
            aud = drop_patches(aud, dim=-2, frac=self.freq_patch_out)

        aud = rearrange(aud, "b c f t -> b (f t) c")
        if self.pooling == "token":
            cls_token = self.cls_token.expand(B, -1, -1)
            cls_token = cls_token + self.token_pos_embed
            aud = torch.cat((cls_token, aud), dim=1)

        aud = self.pos_drop(aud)
        aud = self.blocks(aud)
        aud = self.norm(aud)

        return aud
    
