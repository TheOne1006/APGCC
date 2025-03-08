import os
import sys
import argparse
import torch
import torchvision.transforms as standard_transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import cv2

# 添加自定义模块
from models import build_model
from config import cfg, merge_from_file, merge_from_list
from datasets.build import DeNormalize
from util.logger import vis

def parse_args():
    parser = argparse.ArgumentParser('APGCC预测脚本')
    parser.add_argument('-c', '--config_file', type=str, default="configs/SHHA.yaml", help='配置文件路径')
    parser.add_argument('-i', '--input', type=str, default="demo/demo1.jpg", help='输入图像路径')
    parser.add_argument('-o', '--output', type=str, default="output/prediction", help='输出目录')
    parser.add_argument('-w', '--weight', type=str, default="", help='模型权重路径')
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help='检测阈值')
    parser.add_argument('opts', help='命令行参数覆盖配置', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    
    # 加载配置
    if args.config_file != "":
        cfg_temp = merge_from_file(cfg, args.config_file)
    cfg_temp = merge_from_list(cfg, args.opts)
    cfg_temp.config_file = args.config_file
    return cfg_temp, args

def preprocess_image(image_path):
    """预处理图像"""
    # 读取图像
    img = Image.open(image_path).convert('RGB')
    
    # 应用标准变换
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
    ])
    
    img_tensor = transform(img)
    
    # 与测试代码保持一致的缩放逻辑
    max_size = max(img_tensor.shape[1:])
    upper_bound = 2560  # 与测试代码中的上限一致
    
    if max_size > upper_bound:
        scale = upper_bound / max_size
    elif max_size > 2560:
        scale = 2560 / max_size
    else:
        scale = 1.0
    
    if scale != 1.0:
        img_tensor = torch.nn.functional.interpolate(
            img_tensor.unsqueeze(0), 
            scale_factor=scale, 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        print(f"图像已缩放，缩放因子: {scale:.2f}")
    
    # 确保图像尺寸是32的倍数
    h, w = img_tensor.shape[1:]
    new_h = (h // 32) * 32
    new_w = (w // 32) * 32
    
    if h != new_h or w != new_w:
        img_tensor = torch.nn.functional.interpolate(
            img_tensor.unsqueeze(0),
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        print(f"图像已调整为32的倍数尺寸: {new_w}x{new_h}")
    
    # 转回PIL图像用于可视化
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
    
    return img_tensor.unsqueeze(0), img_pil

def visualize_prediction(image, points, output_path):
    """可视化预测结果"""
    plt.figure(figsize=(12, 8))
    
    # 转换PIL图像为numpy数组
    img_np = np.array(image)
    
    # 绘制原始图像
    plt.imshow(img_np)
    
    # 在图像上标记预测点
    if points:
        points_array = np.array(points)
        if len(points_array) > 0:
            plt.scatter(points_array[:, 0], points_array[:, 1], c='red', s=15, marker='o')
    
    # 添加计数信息
    plt.title(f'pred: {len(points)}', fontsize=18)
    plt.axis('off')
    
    # 保存图像
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=200)
    plt.close()
    
    print(f"可视化结果已保存至: {output_path}")
    return output_path

def main():
    # 解析参数并加载配置
    cfg, args = parse_args()
    
    # 设置设备
    device = torch.device(f'cuda:{cfg.GPU_ID}' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 构建模型
    model = build_model(cfg, training=False)
    
    # 加载模型权重
    checkpoint_path = args.weight
    
    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    pretrained_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = model.state_dict()
    param_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    model_dict.update(param_dict)
    model.load_state_dict(model_dict)
    
    model = model.to(device)
    model.eval()
    
    # 预处理图像
    image_path = os.path.join(os.path.dirname(__file__), args.input)
    
    # 使用32的倍数作为示例输入尺寸
    # img_tensor.shape = (1, 3, H, W)
    # original_img.shape = (H, W, 3)
    img_tensor, original_img = preprocess_image(image_path)
    
    # 预测
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        
        # 获取输入形状
        outputs = model(img_tensor)
        
        # 获取预测结果
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
        outputs_points = outputs['pred_points'][0]
        
        # 应用阈值过滤
        threshold = args.threshold
        mask = outputs_scores > threshold
        points = outputs_points[mask].detach().cpu().numpy().tolist()
        predict_cnt = int(mask.sum())
        
        print(f"预测人数: {predict_cnt}")
        
        # 可视化结果
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        
        visualize_prediction(original_img, points, output_path)
        
        # 保存带有点标记的图像
        marked_img = np.array(original_img.copy())
        for point in points:
            # 确保点坐标在合理范围内
            x, y = int(point[0]), int(point[1])
            if 0 <= x < marked_img.shape[1] and 0 <= y < marked_img.shape[0]:
                cv2.circle(marked_img, (x, y), 5, (255, 0, 0), -1)
        
        marked_output_path = os.path.join(output_dir, 'marked_' + os.path.basename(image_path))
        cv2.imwrite(marked_output_path, cv2.cvtColor(marked_img, cv2.COLOR_RGB2BGR))
        print(f"带标记的图像已保存至: {marked_output_path}")

if __name__ == "__main__":
    main()



# python pred.py -c configs/SHHA_test.yml -i demo/demo1.jpg -o output/prediction -w ../output/SHHA_best.pth
