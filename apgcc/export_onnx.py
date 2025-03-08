import os
import sys
import argparse
import torch
import numpy as np
import onnx
from pathlib import Path

# 添加自定义模块
from models import build_model
from config import cfg, merge_from_file, merge_from_list

def parse_args():
    parser = argparse.ArgumentParser('APGCC ONNX导出脚本')
    parser.add_argument('-c', '--config_file', type=str, default="configs/SHHA_basic.yml", help='配置文件路径')
    parser.add_argument('-w', '--weight', type=str, default="output/SHHA_best.pth", help='模型权重路径')
    parser.add_argument('-o', '--output', type=str, default="output/model.onnx", help='ONNX模型输出路径')
    parser.add_argument('--dynamic', action='store_true', help='是否使用动态尺寸')
    parser.add_argument('opts', help='命令行参数覆盖配置', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    
    # 加载配置
    if args.config_file != "":
        cfg_temp = merge_from_file(cfg, args.config_file)
    cfg_temp = merge_from_list(cfg, args.opts)
    cfg_temp.config_file = args.config_file
    return cfg_temp, args

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
    
    pretrained_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = model.state_dict()
    param_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    model_dict.update(param_dict)
    model.load_state_dict(model_dict)
    
    model = model.to(device)
    model.eval()
    
    # 创建示例输入张量
    # 使用标准的图像尺寸，如384x384
    dummy_input = torch.randn(1, 3, 384, 384, device=device)
    
    # 导出ONNX模型
    output_path = args.output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 动态尺寸设置
    if args.dynamic:
        dynamic_axes = {
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'pred_logits': {0: 'batch_size'},
            'pred_points': {0: 'batch_size'}
        }
    else:
        dynamic_axes = None
    
    # 定义模型的输入和输出名称
    input_names = ['input']
    output_names = ['pred_logits', 'pred_points']
    
    try:
        # 创建一个包装函数来处理模型的输出，确保它返回元组而不是字典
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super(ModelWrapper, self).__init__()
                self.model = model
                
            def forward(self, x):
                outputs = self.model(x)
                return outputs['pred_logits'], outputs['pred_points']
        
        wrapped_model = ModelWrapper(model)
        
        print(f"正在导出ONNX模型到: {output_path}")
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=16,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        
        print(f"ONNX模型导出成功: {output_path}")
        
        # 验证ONNX模型
        try:
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX模型验证通过！")
        except ImportError:
            print("未安装onnx包，跳过模型验证")
        except Exception as e:
            print(f"ONNX模型验证失败: {e}")
    
    except Exception as e:
        print(f"ONNX导出失败: {e}")

if __name__ == "__main__":
    main()


# python  export_onnx.py  -c configs/SHHA_test.yml  -w ../output/SHHA_best.pth -o ../output/onnx/m
