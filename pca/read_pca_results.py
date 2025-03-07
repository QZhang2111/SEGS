#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca_path", type=str, default="/home/testusr_4/Desktop/Qing_WorkSpace/pca/PCA_extraction/20_seed_image_pca64_step1000/pca_results.pt",
                        help="pca_results.pt 文件路径（默认当前目录）")
    args = parser.parse_args()

    # 读取 pca_results.pt
    pca_results = torch.load(args.pca_path)
    
    # pca_results 结构: { step: { layer_name: {"mean":..., "basis":...} } }
    # 逐层打印其内容
    for step, step_data in pca_results.items():
        print(f"==== Step: {step} ====")
        for layer_name, pca_info in step_data.items():
            mean_tensor = pca_info["mean"]   # 张量形状: [d]
            basis_tensor = pca_info["basis"] # 张量形状: [num_save_basis, d]
            project_tensor = pca_info["teacher_feature"]
            
            print(f"Layer: {layer_name}")
            print(f"  Mean shape:  {mean_tensor.shape}")
            print(f"  Basis shape: {basis_tensor.shape}")
            print(f"  teacher_feature shape: {project_tensor.shape}")
            
            # 如果你想查看数值，也可以直接打印：
            # print("  Mean values:", mean_tensor)
            # print("  Basis values:", basis_tensor)
        print("")

if __name__ == "__main__":
    main()
