#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision.models as models
import argparse
import os
from models.Decoder import IFI_Decoder_Model


args =  {
        'num_classes': 2, 
        'inner_planes': 64, 
        'feat_layers': [3, 4], 
        'pos_dim': 32, 
        'ultra_pe': True, 
        'learn_pe': False, 
        'unfold': False, 
        'local': True, 
        'no_aspp': False, 
        'require_grad': True, 
        'out_type': 'Normal', 
        'head_layers':[1024, 512, 256, 256], 
        'in_planes': [128, 256, 512, 512], 
        'line': 2, 'row': 2, 
        'num_anchor_points': 4,
        'sync_bn': False, 'AUX_EN': True, 
        'AUX_NUMBER': [2, 2], 
        'AUX_RANGE': [2, 8], 
        'AUX_kwargs': {'pos_coef': 1.0, 'neg_coef': 1.0, 'pos_loc': 0.0002, 'neg_loc': 0.0002}
    }


def load_pretrained_model(pretrained_path):
    """
    加载预训练的IFI解码器
    """
    pretrained_dict = torch.load(pretrained_path, map_location='cpu')
    print(pretrained_dict.keys())


def build_decode_model():
    """
    加载预训练的IFI解码器

    """
    
    model = IFI_Decoder_Model(**args)
    
    return model

# def export_to_onnx(model, output_path, input_shape=None, dynamic_axes=None):
#     """
#     将PyTorch模型导出为ONNX格式
    
#     Args:
#         model: PyTorch模型
#         output_path: 输出ONNX文件路径
#         input_shape: 输入形状，默认为[1, 3, 224, 224]
#         dynamic_axes: 动态轴配置
#     """
#     if input_shape is None:
#         input_shape = [1, 3, 224, 224]
    
#     # 默认的动态轴配置（支持不同的图片宽高）
#     if dynamic_axes is None:
#         dynamic_axes = {
#             'input': {0: 'batch_size', 2: 'height', 3: 'width'},
#             'output': {0: 'batch_size'}
#         }
    
#     # 创建示例输入
#     dummy_input = torch.randn(*input_shape)
    
#     # 将模型设置为评估模式
#     model.eval()
    
#     # 确保输出目录存在
#     os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
#     # 导出模型
#     torch.onnx.export(
#         model,                   # 要导出的模型
#         dummy_input,             # 模型输入示例
#         output_path,             # 保存ONNX模型的路径
#         export_params=True,      # 存储训练好的参数权重
#         opset_version=12,        # ONNX操作集版本
#         do_constant_folding=True,# 是否执行常量折叠优化
#         input_names=['input'],   # 输入节点的名称
#         output_names=['output'], # 输出节点的名称
#         dynamic_axes=dynamic_axes # 动态轴（批大小和输入尺寸可变）
#     )
    
#     print(f"模型已成功导出到: {output_path}")
#     print(f"支持的动态尺寸: {dynamic_axes}")

# def verify_onnx_model(onnx_path, input_shape=None):
#     """
#     验证导出的ONNX模型
    
#     Args:
#         onnx_path: ONNX模型路径
#         input_shape: 输入形状，默认为[1, 3, 224, 224]
#     """
#     try:
#         import onnx
#         import onnxruntime as ort
#         import numpy as np
        
#         if input_shape is None:
#             input_shape = [1, 3, 224, 224]
        
#         # 加载ONNX模型
#         onnx_model = onnx.load(onnx_path)
#         onnx.checker.check_model(onnx_model)
#         print("ONNX模型结构验证通过！")
        
#         # 创建推理会话
#         ort_session = ort.InferenceSession(onnx_path)
        
#         # 准备输入数据
#         input_data = np.random.randn(*input_shape).astype(np.float32)
        
#         # 运行推理
#         ort_inputs = {ort_session.get_inputs()[0].name: input_data}
#         ort_outputs = ort_session.run(None, ort_inputs)
        
#         print("ONNX模型推理测试通过！")
        
#         # 测试不同尺寸输入
#         test_shapes = [
#             [1, 3, 320, 320],
#             [1, 3, 640, 480]
#         ]
        
#         for shape in test_shapes:
#             test_input = np.random.randn(*shape).astype(np.float32)
#             ort_inputs = {ort_session.get_inputs()[0].name: test_input}
#             ort_outputs = ort_session.run(None, ort_inputs)
#             print(f"动态尺寸测试通过: {shape}, ort_outputs shape: {ort_outputs[0].shape}")
            
        
#     except ImportError:
#         print("警告: 未安装onnx或onnxruntime，无法验证模型")
#     except Exception as e:
#         print(f"验证失败: {e}")



if __name__ == '__main__':
    # main()
    
    load_pretrained_model("./output/SHHA_best.pth")
    

    # python export_vgg.py --model vgg16_bn --output vgg16_bn.onnx --pretrained --verify
    
    
