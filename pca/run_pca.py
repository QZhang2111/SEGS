import json
import subprocess
import os
from pathlib import Path

def sanitize_prompt(prompt):
    """将提示词转换为有效的文件夹名称"""
    sanitized = prompt.strip()
    sanitized = sanitized.replace(" ", "_").replace(",", "").replace("'", "").replace(":", "")
    sanitized = sanitized.replace("\\", "").replace("/", "")
    return sanitized

def main():
    # 读取JSON文件
    with open("simplified_prompts.json") as f:
        data1 = json.load(f)
        prompts1 = data1["selected_prompts"]
    
    with open("prompts_with_heads.json") as f:
        data2 = json.load(f)
        prompts2 = data2["selected_prompts"]
    
    assert len(prompts1) == len(prompts2), "两个JSON文件的提示词数量不一致"

    base_output_dir = "output"
    Path(base_output_dir).mkdir(parents=True, exist_ok=True)

    for idx, (prompt1, prompt2) in enumerate(zip(prompts1, prompts2)):
        # 处理第二个提示词为有效文件夹名
        folder_name = sanitize_prompt(prompt2)
        save_folder = os.path.join(base_output_dir, folder_name)
        modified_prompt1 = prompt1 + " back view"
        # 运行第一个脚本
        cmd1 = [
            "python",
            "PCA_extraction_final.py",
            "--prompt", modified_prompt1,
            "--save_folder", save_folder,
        ]
        print(f"\n运行第一个脚本: 提示词 '{prompt1}'")
        print("命令:", " ".join(cmd1))
        try:
            subprocess.run(cmd1, check=True)
        except subprocess.CalledProcessError as e:
            print(f"第一个脚本运行失败: {e}")
            continue
        
        # 检查PCA文件是否生成
        pca_path = os.path.join(save_folder, "pca_results.pt")
        if not os.path.exists(pca_path):
            print(f"错误: {pca_path} 未生成")
            continue
        
        # 运行第二个脚本
        pca_result_folder = os.path.join("pca_result", sanitize_prompt(prompt2))
        Path(pca_result_folder).mkdir(parents=True, exist_ok=True)
      
        cmd2 = [
            "python",
            "CLIP_Extract_PCA_Info.py",
            "--image_dir", save_folder,
            "--pca_path", pca_path,
            "--save_path", pca_result_folder
        ]
        print(f"\n运行第二个脚本: 使用前缀 '{folder_name}'")
        print("命令:", " ".join(cmd2))
        try:
            subprocess.run(cmd2, check=True)
        except subprocess.CalledProcessError as e:
            print(f"第二个脚本运行失败: {e}")
            continue

if __name__ == "__main__":
    main()