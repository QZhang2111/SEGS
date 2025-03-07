#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一后的辅助函数文件，包含以下功能：
  1. patch_target_conv_block：为指定卷积层打补丁，
     在执行 conv1 后保存中间特征。特征保存方式可通过参数控制是否 detach（以及是否移动到 CPU）。
  2. MySelfAttnProcessor：自定义 Attention Processor，
     在 self-attention 场景下提取 key 特征，并转换为 [B, C, H, W] 形状。
  3. prep_unet_attention：遍历 UNet 模型，替换所有 Attention 模块的处理器为 MySelfAttnProcessor，
     并可通过参数控制是否对 key 特征 detach。
  4. clean_attn_buffer：清理 Attention 模块中保存的中间变量（如 key）和卷积层保存的 feature。
  5. get_module_by_name：通过名称查找模型中的子模块。
  6. visualize_and_save_features_pca：对输入特征做 PCA 降维到 3 维后保存为图像。
  7. compute_pca_for_storage：对输入特征做 PCA，并返回均值、主成分和投影后的特征。
  8. classify_blocks：判断给定层名称是否包含关键字列表中的任一字符串。
  
该文件通过参数化方式统一了两份代码中的不同实现，既支持离线特征提取也支持梯度计算场景。
"""

import math
import os
import torch
import xformers
from diffusers.models.attention_processor import Attention

# ---------------------------
# 辅助函数：通过名称查找子模块
# ---------------------------
def get_module_by_name(model, module_name: str):
    """
    在 model.named_modules() 中根据完整名称检索子模块。
    如果找不到则返回 None。
    """
    for n, m in model.named_modules():
        if n == module_name:
            return m
    return None

# ---------------------------
# 统一后的 patch_target_conv_block 函数
# ---------------------------
def patch_target_conv_block(unet, module_name: str = None, detach_feature: bool = True):
    """
    为目标卷积层打补丁，修改其 forward 方法，在 conv1 后保存中间特征到 module.feature。
    
    参数:
      unet: 模型
      module_name: 若提供则通过 get_module_by_name 查找目标模块；
                   若为 None，则默认使用 unet.up_blocks[1].resnets[1]
      detach_feature: 若 True，则保存时使用 hidden.detach().cpu()（适用于离线特征提取）；
                      若 False，则直接保存 hidden（保留梯度）。
    """
    if module_name is not None:
        conv_module = get_module_by_name(unet, module_name)
        if conv_module is None:
            raise ValueError(f"未能在 unet 中找到名为 {module_name} 的卷积层模块，请检查名称是否正确。")
    else:
        try:
            conv_module = unet.up_blocks[1].resnets[1]
        except Exception as e:
            raise ValueError("无法通过默认路径 unet.up_blocks[1].resnets[1] 获取目标模块，请提供 module_name 参数。") from e

    original_forward = conv_module.forward  # 如果需要保留原始 forward 可保存

    def conv_forward(x, temb=None):
        hidden = conv_module.norm1(x)
        hidden = conv_module.nonlinearity(hidden)
        if hasattr(conv_module, "upsample") and conv_module.upsample is not None:
            x = conv_module.upsample(x)
            hidden = conv_module.upsample(hidden)
        # 执行 conv1 提取特征
        hidden = conv_module.conv1(hidden)
        if detach_feature:
            conv_module.feature = hidden.detach().cpu()
        else:
            conv_module.feature = hidden
        if temb is not None and hasattr(conv_module, "time_emb_proj"):
            temb_proj = conv_module.time_emb_proj(conv_module.nonlinearity(temb))[:, :, None, None]
            hidden = hidden + temb_proj
        hidden = conv_module.norm2(hidden)
        hidden = conv_module.nonlinearity(hidden)
        hidden = conv_module.dropout(hidden)
        hidden = conv_module.conv2(hidden)
        if hasattr(conv_module, "conv_shortcut") and conv_module.conv_shortcut is not None:
            x = conv_module.conv_shortcut(x)
        output = (x + hidden) / conv_module.output_scale_factor
        return output

    conv_module.forward = conv_forward
    return unet


def extract_and_visualize_features(features, step_id, save_dir, layer_name, batch_size, suffix="", num_save_basis=64):
    """
    提取特征并进行 PCA 可视化。
    
    参数:
        features: 输入特征，形状 [B, C, H, W] 或 [B, C, L]
        step_id: 当前步数，用于生成文件名
        save_dir: 保存图像的根目录
        layer_name: 图像命名中使用的层名称
        batch_size: 每批次样本数
        suffix: 文件名后缀
        num_save_basis: PCA 保留的主成分数量
    """
    if features.dim() == 4:  # 如果是 4D 特征 [B, C, H, W]
        features_2d = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
    elif features.dim() == 3:  # 如果是 3D 特征 [B, C, L]
        features_2d = features.permute(0, 2, 1).reshape(-1, features.shape[1])
    else:
        raise ValueError(f"Unsupported feature dimension: {features.dim()}")

    # 可视化并保存 PCA 结果
    # visualize_and_save_features_pca(
    #     features=features_2d,
    #     step_id=step_id,
    #     save_dir=save_dir,
    #     layer_name=layer_name,
    #     batch_size=batch_size,
    #     suffix=suffix
    # )

    # 计算 PCA 结果并返回
    return compute_pca_for_storage(features_2d, num_save_basis)


# ---------------------------
# 统一后的 visualize_and_save_features_pca 函数
# ---------------------------
def visualize_and_save_features_pca(features: torch.Tensor, step_id: int, save_dir: str, layer_name: str = "feature", batch_size: int = 4, suffix: str = ""):
    """
    将输入特征（形状 [N, d]）按 batch_size 划分，每个样本做 PCA 降维到 3 维，
    转换为 RGB 图像后保存。
    
    参数:
      features: 输入特征，形状 [N, d]（要求 N 能被 batch_size 整除）
      step_id: 当前步数，用于生成文件名
      save_dir: 保存图像的根目录
      layer_name: 图像命名中使用的层名称
      batch_size: 每批次样本数
      suffix: 文件名后缀
    """
    os.makedirs(save_dir, exist_ok=True)
    total_tokens, d = features.shape
    tokens_per_sample = total_tokens // batch_size
    sqrt_n = int(math.sqrt(tokens_per_sample))
    if sqrt_n * sqrt_n != tokens_per_sample:
        print(f"[Warning] 每个样本的 tokens 数 {tokens_per_sample} 不是完全平方数，取前 {sqrt_n * sqrt_n} 个 token")
    tokens_to_use = sqrt_n * sqrt_n
    for i in range(batch_size):
        sample_features = features[i * tokens_per_sample : i * tokens_per_sample + tokens_to_use, :]
        sample_features_centered = sample_features - sample_features.mean(dim=0, keepdim=True)
        sample_features_centered = sample_features_centered.float().cpu()
        try:
            U, S, V = torch.pca_lowrank(sample_features_centered, q=3)
        except Exception as e:
            print(f"[Error] 样本 {i} 的 PCA 计算失败: {e}")
            continue
        features_3d = sample_features_centered @ V[:, :3]
        min_vals = features_3d.amin(dim=0, keepdim=True)
        max_vals = features_3d.amax(dim=0, keepdim=True)
        features_norm = (features_3d - min_vals) / (max_vals - min_vals + 1e-5)
        img = features_norm.reshape(sqrt_n, sqrt_n, 3).numpy()
        sample_folder = os.path.join(save_dir, f"{layer_name}_sample{i}")
        os.makedirs(sample_folder, exist_ok=True)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.axis("off")
        filename = os.path.join(sample_folder, f"{layer_name}_sample{i}_step{step_id}{suffix}.png")
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()

# ---------------------------
# 统一后的 compute_pca_for_storage 函数
# ---------------------------
def compute_pca_for_storage(X: torch.Tensor, num_save_basis: int):
    """
    对 X 做 PCA，并返回包含均值、主成分和投影后特征的字典。
    
    参数:
      X: 输入特征，形状 [n, d]
      num_save_basis: 保留的主成分数量
    """
    X = X.to(torch.float32)
    mean = X.mean(dim=0)
    X_centered = X - mean
    U, S, V = torch.svd(X_centered)
    if num_save_basis > 0 and V.shape[1] > num_save_basis:
        V = V[:, :num_save_basis]
    basis = V
    projected = X_centered @ basis
    return {
        "mean": mean.cpu(),
        "basis": basis.cpu(),
        "teacher_feature": projected.cpu()
    }

# ---------------------------
# 统一后的 classify_blocks 函数
# ---------------------------
def classify_blocks(block_list, name: str):
    """
    判断给定层名称 name 是否包含 block_list 中的任意一个字符串。
    """
    for block in block_list:
        if block in name:
            return True
    return False

# ---------------------------
# 统一后的 MySelfAttnProcessor 类
# ---------------------------
class MySelfAttnProcessor:
    """
    自定义 Attention Processor：
      - 当 encoder_hidden_states 为 None（self-attention）时，提取 key 特征；
      - 将 key 特征从 [B, tokens, dim] 转换为 [B, dim, H, W]，其中 H, W 由 tokens 数（假设为完全平方）确定。
    
    参数:
      attention_op: 可传入 xformers 的 attention 操作
      detach_key: 若 True，则对提取的 key 特征进行 detach（适用于离线特征提取），
                  若 False，则保留计算图（适用于需要梯度的场景）。
    """
    def __init__(self, attention_op=None, detach_key: bool = True):
        self.attention_op = attention_op
        self.detach_key = detach_key
        self.is_self_attn = False
        self.k = None

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, scale: float = 1.0):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        if encoder_hidden_states is None:
            self.is_self_attn = True
        else:
            self.is_self_attn = False
        if encoder_hidden_states is None:
            batch_size, key_tokens, _ = hidden_states.shape
        else:
            batch_size, key_tokens, _ = encoder_hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
        if attention_mask is not None:
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query_ = attn.head_to_batch_dim(query).contiguous()
        key_ = attn.head_to_batch_dim(key).contiguous()
        value_ = attn.head_to_batch_dim(value)
        hidden_states_ = xformers.ops.memory_efficient_attention(
            query_, key_, value_, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
        )
        hidden_states_ = hidden_states_.to(query_.dtype)
        hidden_states_ = attn.batch_to_head_dim(hidden_states_)
        hidden_states_ = attn.to_out[0](hidden_states_)
        hidden_states_ = attn.to_out[1](hidden_states_)
        if input_ndim == 4:
            hidden_states_ = hidden_states_.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states_ = hidden_states_ + residual
        hidden_states_ = hidden_states_ / attn.rescale_output_factor
        if self.is_self_attn:
            b, tokens, dim = key.shape
            res = int(math.sqrt(tokens))
            key_4d = key.permute(0, 2, 1).reshape(b, dim, res, res)
            if self.detach_key:
                self.k = key_4d.detach()
            else:
                self.k = key_4d
        else:
            self.k = None
        return hidden_states_

# ---------------------------
# 统一后的 prep_unet_attention 函数
# ---------------------------
def prep_unet_attention(unet, detach_key=True, target_blocks=None):
    registered_attention_blocks = []
    for name, module in unet.named_modules():
        if isinstance(module, Attention):
            if target_blocks is not None and not classify_blocks(target_blocks, name):
                continue
            module.set_processor(MySelfAttnProcessor(detach_key=detach_key))
            registered_attention_blocks.append(name)
    unet.registered_attention_blocks = registered_attention_blocks
    return unet

# ---------------------------
# 统一后的 clean_attn_buffer 函数
# ---------------------------
def clean_attn_buffer(unet):
    """
    清除 UNet 中 Attention 模块 processor 内保存的中间变量，
    并清理所有卷积层的 feature 属性。
    """
    for name, module in unet.named_modules():
        if isinstance(module, Attention) and hasattr(module, "processor"):
            for attr in ["hidden_state", "query", "key", "value", "attention_mask", "attn", "k"]:
                if hasattr(module.processor, attr):
                    setattr(module.processor, attr, None)
        if hasattr(module, "feature"):
            module.feature = None
