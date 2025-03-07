#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.models.attention_processor import Attention
from tqdm import tqdm  # 导入 tqdm 进度条库

# 从统一后的辅助函数文件 help.py 引入相关函数
from help import (
    get_module_by_name,
    patch_target_conv_block,
    visualize_and_save_features_pca,
    classify_blocks,
    prep_unet_attention,
    clean_attn_buffer,
    compute_pca_for_storage,
    extract_and_visualize_features,
)


def generate_images_stepwise(pipe: StableDiffusionPipeline, args):
    device = pipe._execution_device
    generator = torch.Generator(device=device).manual_seed(args.seed)
    prompt_list = [args.prompt]
    do_cfg = args.guidance_scale > 1.0

    # 编码文本
    prompt_embeds = pipe._encode_prompt(
        prompt=prompt_list,
        device=device,
        num_images_per_prompt=args.num_images,
        do_classifier_free_guidance=do_cfg,
        negative_prompt=args.negative_prompt,
    )

    pipe.scheduler.set_timesteps(args.steps, device=device)
    
    timesteps = pipe.scheduler.timesteps.clone()
    #print("Original timesteps:", timesteps)


    height = pipe.unet.config.sample_size * pipe.vae_scale_factor
    width = pipe.unet.config.sample_size * pipe.vae_scale_factor
    latents = pipe.prepare_latents(
        batch_size=len(prompt_list) * args.num_images,
        num_channels_latents=pipe.unet.config.in_channels,
        height=height,
        width=width,
        dtype=prompt_embeds.dtype,
        device=device,
        generator=generator,
    )

    os.makedirs(args.save_folder, exist_ok=True)
    vis_folder = os.path.join(args.save_folder, "k_vis")
    os.makedirs(vis_folder, exist_ok=True)
    conv_vis_folder = os.path.join(args.save_folder, "conv_vis")
    os.makedirs(conv_vis_folder, exist_ok=True)

    pipe.unet.eval()
    pipe.vae.eval()

    pca_results = {}

    with torch.no_grad():
        # 使用 tqdm 包装 for 循环添加进度条
        for i, t in tqdm(enumerate(timesteps), total=len(timesteps), desc="Diffusion steps"):
            if do_cfg:
                latent_model_input = torch.cat([latents] * 2, dim=0)
            else:
                latent_model_input = latents

            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample

            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

            if i < args.max_vis_steps:
                pca_results[i] = {}

                # 遍历已注册的 Attention 模块
                for name in pipe.unet.registered_attention_blocks:
                    module = get_module_by_name(pipe.unet, name)
                    if module.processor.k is None:
                        continue
                    if "attentions.2" not in name:
                        continue

                    # 提取 key 特征
                    k_data = module.processor.k  # shape [B, C, H, W]
                    if do_cfg:
                        k_data = k_data.chunk(2)[1]  # 使用文本条件部分

                    # 可视化并保存 PCA 结果
                    pca_results[i][name] = extract_and_visualize_features(
                        k_data, int(t.item()), vis_folder, name.replace(".", "_"), args.num_images, "_k", args.num_save_basis
                    )

                # 对打补丁后的卷积层提取特征进行 PCA 可视化
                conv_module = get_module_by_name(pipe.unet, args.conv_block)
                if conv_module and hasattr(conv_module, "feature"):
                    feat = conv_module.feature  # shape [B, C, H, W]
                    if do_cfg:
                        feat = feat.chunk(2)[1]
                    pca_results[i][args.conv_block] = extract_and_visualize_features(
                        feat, int(t.item()), conv_vis_folder, args.conv_block, args.num_images, "", args.num_save_basis
                    )

            clean_attn_buffer(pipe.unet)
            torch.cuda.empty_cache()

        decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
        decoded = (decoded / 2 + 0.5).clamp(0, 1).detach().cpu().permute(0, 2, 3, 1).numpy()
        images = pipe.numpy_to_pil(decoded)

        for idx, img in enumerate(images):
            img_path = os.path.join(args.save_folder, f"final_image_{idx}.png")
            img.save(img_path)
            print(f"Saved image: {img_path}")

    pca_save_path = os.path.join(args.save_folder, "pca_results.pt")
    torch.save(pca_results, pca_save_path)
    print(f"PCA results saved at: {pca_save_path}")
    return images, pca_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="a baby dragon, back view")
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--steps", type=int, default=999, help="Diffusion steps")
    parser.add_argument("--num_images", type=int, default=20, help="Images per prompt")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--blocks", nargs="+", default=["up_blocks.1"], help="Keywords for attention blocks")
    parser.add_argument("--conv_block", type=str, default="up_blocks.1.resnets.1", help="Conv block name for patching")
    parser.add_argument("--num_save_basis", type=int, default=64, help="Number of PCA components")
    parser.add_argument("--max_vis_steps", type=int, default=999, help="Max steps for visualization")
    parser.add_argument("--save_folder", type=str, default="PCA_extraction/baby_dragon_Lucid", help="Output folder")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale for CFG")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "stabilityai/stable-diffusion-2-1-base"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    pipe.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as e:
        print("xformers memory efficient attention not available, using default.", e)

    # 预处理 UNet：注册需要处理的 Attention 模块和卷积层
    prep_unet_attention(pipe.unet, detach_key=True, target_blocks=args.blocks)
    patch_target_conv_block(pipe.unet, args.conv_block)

    generate_images_stepwise(pipe, args)

if __name__ == "__main__":
    main()
