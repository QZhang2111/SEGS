import json
import subprocess
import os
from pathlib import Path
import yaml

def sanitize_prompt(prompt):
    """将提示词转换为有效的文件夹名称"""
    sanitized = prompt.strip()
    sanitized = sanitized.replace(" ", "_").replace(",", "").replace("'", "").replace(":", "")
    sanitized = sanitized.replace("\\", "").replace("/", "")
    return sanitized

def generate_yaml(base_yaml_path, output_yaml_path, text, initial_text, pca_result, init_prompt, workspace):
    """生成新的YAML配置文件"""
    with open(base_yaml_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # 更新参数
    base_config["GuidanceParams"]["text"] = text
    base_config["GuidanceParams"]["initial_text"] = initial_text
    base_config["GuidanceParams"]["pca_result"] = pca_result
    base_config["GenerateCamParams"]["init_prompt"] = init_prompt  # 添加 init_prompt
    base_config["ModelParams"]["workspace"] = workspace  # 添加 workspace
    
    # 保存新的YAML文件
    with open(output_yaml_path, 'w') as f:
        yaml.dump(base_config, f)

def main():
    # 读取JSON文件
    with open("simplified_prompts.json") as f:
        data1 = json.load(f)
        prompts1 = data1["selected_prompts"]
    
    with open("prompts_with_heads.json") as f:
        data2 = json.load(f)
        prompts2 = data2["selected_prompts"]
    
    assert len(prompts1) == len(prompts2), "两个JSON文件的提示词数量不一致"

    # 基础路径
    base_output_dir = "pca_output"
    pca_result_folder = "pca_result"
    base_yaml_path = "LucidDreamer/configs/baby_dragon.yaml"  # 基础YAML配置文件
    generated_yaml_dir = "LucidDreamer/configs/generated"  # 生成的YAML文件保存目录

    # 创建目录
    Path(base_output_dir).mkdir(parents=True, exist_ok=True)
    Path(generated_yaml_dir).mkdir(parents=True, exist_ok=True)

    for idx, (prompt1, prompt2) in enumerate(zip(prompts1, prompts2)):
        # 处理第二个提示词为有效文件夹名
        folder_name = sanitize_prompt(prompt2)
        save_folder = os.path.join(base_output_dir, folder_name)
        modified_prompt1 = prompt1 + " back view"

        # 运行第一个脚本（生成图片和PCA文件）
        cmd1 = [
            "python",
            "pca/PCA_extraction_final.py",
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
        
        # 运行第二个脚本（提取PCA信息）
        pca_result_subfolder = os.path.join(pca_result_folder, sanitize_prompt(prompt2))
        Path(pca_result_subfolder).mkdir(parents=True, exist_ok=True)
      
        cmd2 = [
            "python",
            "pca/CLIP_Extract_PCA_Info.py",
            "--image_dir", save_folder,
            "--pca_path", pca_path,
            "--save_path", pca_result_subfolder
        ]
        print(f"\n运行第二个脚本: 使用前缀 '{folder_name}'")
        print("命令:", " ".join(cmd2))
        try:
            subprocess.run(cmd2, check=True)
        except subprocess.CalledProcessError as e:
            print(f"第二个脚本运行失败: {e}")
            continue

        # 生成YAML配置文件
        yaml_filename = f"{folder_name}.yaml"
        yaml_path = os.path.join(generated_yaml_dir, yaml_filename)
        pca_result_pt = os.path.join(pca_result_subfolder, "top5_pca.pt")
        
        generate_yaml(
            base_yaml_path=base_yaml_path,
            output_yaml_path=yaml_path,
            text=prompt2,
            initial_text=prompt1,
            pca_result=pca_result_pt,
            init_prompt=prompt1,  # 添加 init_prompt
            workspace=folder_name  # 添加 workspace
        )
        print(f"生成YAML文件: {yaml_path}")

        # 运行训练脚本
        cmd_train = f'export CUDA_VISIBLE_DEVICES="0" && python LucidDreamer/train.py --opt "{yaml_path}"'
        print(f"运行训练脚本: {cmd_train}")
        try:
            subprocess.run(cmd_train, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"训练脚本运行失败: {e}")
            continue

if __name__ == "__main__":
    main()