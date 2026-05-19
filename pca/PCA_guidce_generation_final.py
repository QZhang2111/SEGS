#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from help import prep_unet_attention, patch_target_conv_block, clean_attn_buffer, get_module_by_name
from segs.structural_energy import mean_teacher_features

class PCACFGPipeline(StableDiffusionPipeline):
    def load_pca_info(self, pca_file: str):
        """加载 PCA 信息"""
        self.pca_info = torch.load(pca_file)

    def __call__(
        self,
        prompt,
        negative_prompt=None,
        num_inference_steps=50,
        guidance_scale=7.5,
        max_pca_steps=20,
        pca_guidance_scale=1000.0,
        generator=None,
        num_samples_per_group=3,
        feature_dim=64,
    ):
        device = self._execution_device
        do_guidance = guidance_scale > 1.0

        # 编码提示词
        prompt_embeds = self._encode_prompt(
            prompt, device, 1, do_guidance, negative_prompt
        )

        # 准备潜在变量
        latents = self.prepare_latents(
            1, self.unet.config.in_channels, 512, 512, prompt_embeds.dtype, device, generator
        )
        if do_guidance:
            latents = torch.cat([latents] * 2, dim=0)

        self.scheduler.set_timesteps(num_inference_steps, device=device)

        for i, t in enumerate(self.scheduler.timesteps):
            latents = latents.requires_grad_()
            latent_model_input = self.scheduler.scale_model_input(latents, t)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample

            if do_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # PCA 引导
            if hasattr(self, 'pca_info') and i < max_pca_steps:
                step_pca = self.pca_info.get(i, None)
                if step_pca:
                    pca_loss = 0.0
                    count = 0
                    for layer_name, pca_data in step_pca.items():
                        module = get_module_by_name(self.unet, layer_name)
                        if not module:
                            continue

                        # 提取特征
                        if "attentions" in layer_name and hasattr(module.processor, "k"):
                            feat = module.processor.k
                        elif "resnets" in layer_name and hasattr(module, "feature"):
                            feat = module.feature
                        else:
                            continue

                        if do_guidance:
                            feat = feat.chunk(2)[1]  # 使用文本条件部分

                        # 计算 PCA 损失
                        bs, c, h, w = feat.shape
                        feat_2d = feat.permute(0, 2, 3, 1).reshape(-1, c)
                        centered = feat_2d.to(device) - pca_data["mean"].to(device)
                        proj = centered @ pca_data["basis"].to(device)
                        
                        # 计算 teacher_feat_mean
                        teacher_feat_ = pca_data["teacher_feature"].to(device)
                        teacher_feat_mean = mean_teacher_features(
                            teacher_feat_,
                            topk=num_samples_per_group,
                            tokens_per_sample=proj.shape[0],
                        )
                        
                        # 计算损失（对于两种情况均使用 MSE 损失）
                        loss = F.mse_loss(proj, teacher_feat_mean)
                        pca_loss += loss
                        count += 1

                    if count > 0:
                        grad = torch.autograd.grad(pca_loss / count, latent_model_input)[0]
                        noise_pred = noise_pred + pca_guidance_scale * grad

            # 更新潜在变量
            with torch.no_grad():
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            clean_attn_buffer(self.unet)

        # 解码图像
        with torch.no_grad():
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image = (image / 2 + 0.5).clamp(0, 1).detach().cpu().permute(0, 2, 3, 1).numpy()
            image = self.numpy_to_pil(image)
        return image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="a corgi is running, back view")
    parser.add_argument("--pca_file", type=str, default="PCA_extraction/20_seed_image_pca64/pca_results.pt")
    parser.add_argument("--num_samples_per_group", type=int, default=3, help="每组样本数")
    parser.add_argument("--feature_dim", type=int, default=64, help="特征维度")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = PCACFGPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16).to(device)
    pipe.safety_checker = None

    # 预处理 UNet
    prep_unet_attention(pipe.unet, detach_key=False, target_blocks=["up_blocks.1"])
    patch_target_conv_block(pipe.unet, module_name="up_blocks.1.resnets.1")

    # 加载 PCA 信息
    pipe.load_pca_info(args.pca_file)

    # 生成图像
    generator = torch.Generator(device=device).manual_seed(0)
    image = pipe(
        prompt=args.prompt,
        negative_prompt="",
        num_inference_steps=200,
        guidance_scale=7.5,
        max_pca_steps=80,
        pca_guidance_scale=27.5,
        generator=generator,
        num_samples_per_group=args.num_samples_per_group,
        feature_dim=args.feature_dim,
    )

    # 保存结果
    os.makedirs("PCA_Guidance_Results", exist_ok=True)
    out_path = "PCA_Guidance_Results/20_seed_corgi_27.5_seed0.png"
    image[0].save(out_path)
    print("结果已保存到:", out_path)

if __name__ == "__main__":
    main()
