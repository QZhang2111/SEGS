#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import torch
from PIL import Image
import open_clip

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default=None, help="final_image_0.png 所在目录")
    parser.add_argument("--num_images", type=int, default=20, help="图片数量，默认20张")
    parser.add_argument("--batch_size", type=int, default=20, help="生成时的批大小")
    parser.add_argument("--topk", type=int, default=3, help="取背面差值最高的图片数")
    parser.add_argument("--pca_path", type=str, default=None, help="原始 pca_results.pt 路径")
    parser.add_argument("--save_path", type=str, default=None, help="处理后保存的文件目录")
    args = parser.parse_args()

    # 确保保存目录存在
    os.makedirs(args.save_path, exist_ok=True)

    # -----------------------
    # Part 1: 计算背面差值，选出 topk 的图片索引
    # -----------------------
    print("==> Loading OpenCLIP model...")
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
    tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')

    # 依次加载图片 [final_image_0.png ~ final_image_19.png]
    images = []
    filenames = []
    for i in range(args.num_images):
        filename = os.path.join(args.image_dir, f"final_image_{i}.png")
        if not os.path.exists(filename):
            print(f"Warning: {filename} 不存在，跳过。")
            continue
        try:
            image = Image.open(filename).convert("RGB")
        except Exception as e:
            print(f"加载图片 {filename} 失败: {e}")
            continue
        image = preprocess_val(image)
        images.append(image)
        filenames.append(filename)
    
    if len(images) == 0:
        print("未能加载任何图片，程序退出。")
        return

    # 堆叠图片张量 [num_images, C, H, W]
    image_tensor = torch.stack(images)

    # 提取图像特征
    print("==> Encoding images...")
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    # 定义三个视角描述
    texts = [
        "the object is viewed from the front",
        "the object is viewed from the side",
        "the object is viewed from the back"
    ]
    text_tokens = tokenizer(texts)

    print("==> Encoding texts...")
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # 计算相似度 [num_images, 3]
    similarity = image_features @ text_features.T
    front_scores = similarity[:, 0]
    side_scores  = similarity[:, 1]
    back_scores  = similarity[:, 2]

    # 计算背面差值: 背面得分 - max(正面, 侧面)
    front_side_max = torch.max(front_scores, side_scores)
    score_diff = back_scores - front_side_max

    # 取差值最高的 topk 图片
    topk_vals, topk_indices = torch.topk(score_diff, k=args.topk)
    topk_indices = topk_indices.tolist()
    topk_vals = topk_vals.tolist()

    # 打印结果
    print("\n最符合背面视角的图片：")
    for rank, (idx, val) in enumerate(zip(topk_indices, topk_vals), start=1):
        print(f"[Rank {rank}] Image Index = {idx}, File = {filenames[idx]}")
        print(f"  - front_score = {front_scores[idx].item():.4f}")
        print(f"  - side_score  = {side_scores[idx].item():.4f}")
        print(f"  - back_score  = {back_scores[idx].item():.4f}")
        print(f"  - score_diff  = {val:.4f}")
        print("-" * 40)

    # -----------------------
    # 新增：将前 topk 个图片的文件名和背面差值保存到 txt 文件中
    # -----------------------
    results_txt_path = os.path.join(args.save_path, f"top{args.topk}_results.txt")
    with open(results_txt_path, "w", encoding="utf-8") as f:
        f.write("最符合背面视角的图片：\n")
        for rank, (idx, val) in enumerate(zip(topk_indices, topk_vals), start=1):
            image_name = os.path.basename(filenames[idx])
            f.write(f"[Rank {rank}] File: {image_name}, score_diff: {val:.4f}\n")
    print(f"已保存 top{args.topk} 图片信息到 {results_txt_path}")

    # -----------------------
    # Part 2: 加载 pca_results.pt，保留 topk 索引
    # -----------------------
    print(f"\n==> Loading PCA results from {args.pca_path} ...")
    pca_results = torch.load(args.pca_path)

    # 如果你的图片和 batch 一一对应，则图片名称末尾 i 就是 batch_i
    # topk_indices 中保存的就是要保留的那几张图片在 batch 中的索引
    keep_indices = sorted(topk_indices)  # 建议先排序，便于后续查看

    print(f"==> Will keep teacher_feature for images indices: {keep_indices}")

    # 遍历 pca_results
    for step, step_data in pca_results.items():
        for layer_name, pca_info in step_data.items():
            teacher_feature = pca_info.get("teacher_feature", None)
            if teacher_feature is not None:
                # teacher_feature 原始形状: [batch_size * tokens_per_sample, feature_dim]
                total_tokens, feature_dim = teacher_feature.shape
                tokens_per_sample = total_tokens // args.batch_size

                # 重塑为 [batch_size, tokens_per_sample, feature_dim]
                teacher_feature = teacher_feature.view(args.batch_size, tokens_per_sample, feature_dim)

                # 只保留 keep_indices 对应的特征
                selected_feature = teacher_feature[keep_indices, :, :]  # shape: [topk, tokens_per_sample, feature_dim]

                # 再 reshape 回 [topk * tokens_per_sample, feature_dim]
                selected_feature = selected_feature.view(-1, feature_dim)

                # 更新 pca_info 中的 teacher_feature
                pca_info["teacher_feature"] = selected_feature
    
    # -----------------------
    # Part 3: 保存更新后的 pca_results
    # -----------------------
    pca_save_path = os.path.join(args.save_path, f"top{args.topk}_pca.pt")
    torch.save(pca_results, pca_save_path)
    print(f"\n处理完成，新的 PCA 结果已保存到 {pca_save_path}")

if __name__ == "__main__":
    main()
