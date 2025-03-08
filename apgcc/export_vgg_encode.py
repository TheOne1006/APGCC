#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision.models as models
import argparse
import os
from models.Encoder import Base_VGG


# args = {'last_pool': False, 'name': 'vgg16_bn'}

def build_vgg_model(model_name='vgg16_bn', last_pool=False):
    """
    加载预训练的VGG模型或自定义的VGG模型
    
    Args:
        model_name: 模型名称，可以是'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn'等
        pretrained: 是否使用预训练权重
    
    Returns:
        VGG模型实例
    """
    
    model = Base_VGG(model_name, last_pool=last_pool)
    
    return model

def export_to_onnx(model, output_path, input_shape=None, dynamic_axes=None):
    """
    将PyTorch模型导出为ONNX格式
    
    Args:
        model: PyTorch模型
        output_path: 输出ONNX文件路径
        input_shape: 输入形状，默认为[1, 3, 224, 224]
        dynamic_axes: 动态轴配置
    """
    if input_shape is None:
        input_shape = [1, 3, 224, 224]
    
    # 默认的动态轴配置（支持不同的图片宽高）
    if dynamic_axes is None:
        dynamic_axes = {
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size'}
        }
    
    # 创建示例输入
    dummy_input = torch.randn(*input_shape)
    
    # 将模型设置为评估模式
    model.eval()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # 导出模型
    torch.onnx.export(
        model,                   # 要导出的模型
        dummy_input,             # 模型输入示例
        output_path,             # 保存ONNX模型的路径
        export_params=True,      # 存储训练好的参数权重
        opset_version=12,        # ONNX操作集版本
        do_constant_folding=True,# 是否执行常量折叠优化
        input_names=['input'],   # 输入节点的名称
        output_names=['output'], # 输出节点的名称
        dynamic_axes=dynamic_axes # 动态轴（批大小和输入尺寸可变）
    )
    
    print(f"模型已成功导出到: {output_path}")
    print(f"支持的动态尺寸: {dynamic_axes}")

def verify_onnx_model(onnx_path, input_shape=None):
    """
    验证导出的ONNX模型
    
    Args:
        onnx_path: ONNX模型路径
        input_shape: 输入形状，默认为[1, 3, 224, 224]
    """
    try:
        import onnx
        import onnxruntime as ort
        import numpy as np
        
        if input_shape is None:
            input_shape = [1, 3, 224, 224]
        
        # 加载ONNX模型
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX模型结构验证通过！")
        
        # 创建推理会话
        ort_session = ort.InferenceSession(onnx_path)
        
        # 准备输入数据
        input_data = np.random.randn(*input_shape).astype(np.float32)
        
        # 运行推理
        ort_inputs = {ort_session.get_inputs()[0].name: input_data}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        print("ONNX模型推理测试通过！")
        
        # 测试不同尺寸输入
        test_shapes = [
            [1, 3, 320, 320],
            [1, 3, 640, 480]
        ]
        
        for shape in test_shapes:
            test_input = np.random.randn(*shape).astype(np.float32)
            ort_inputs = {ort_session.get_inputs()[0].name: test_input}
            ort_outputs = ort_session.run(None, ort_inputs)
            print(f"动态尺寸测试通过: {shape}, ort_outputs shape: {ort_outputs[0].shape}")
            
        
    except ImportError:
        print("警告: 未安装onnx或onnxruntime，无法验证模型")
    except Exception as e:
        print(f"验证失败: {e}")


def export_vgg_model(model, output, input_shape, verify):  
    # 加载模型
    model = build_vgg_model(model)
    
    # 导出为ONNX
    export_to_onnx(model, output, input_shape=input_shape)
    
    # 验证模型
    if verify:
        verify_onnx_model(output, input_shape=input_shape)



def main():
    parser = argparse.ArgumentParser(description="将VGG模型导出为ONNX格式")
    parser.add_argument('--model', type=str, default='vgg16_bn', 
                        help='要导出的VGG模型名称，例如vgg16_bn')
    parser.add_argument('--output', type=str, default='vgg16_bn.onnx',
                        help='输出ONNX文件路径')
    parser.add_argument('--verify', action='store_true', default=True,
                        help='是否验证导出的ONNX模型')
    parser.add_argument('--input-shape', type=int, nargs='+', default=[1, 3, 224, 224],
                        help='输入形状，默认为[1, 3, 224, 224]')
    
    args = parser.parse_args()
    
    export_vgg_model(args.model, args.pretrained, args.output, args.input_shape, args.verify)


if __name__ == '__main__':
    # main()
    
    export_vgg_model(model='vgg16_bn', output='./output/onnx/vgg16_bn.onnx', input_shape=[1, 3, 224, 224], verify=True)
    

    # python export_vgg.py --model vgg16_bn --output vgg16_bn.onnx --pretrained --verify
    
    
