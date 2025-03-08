import os
import sys
import argparse
import torch
import numpy as np
from torch import nn
import onnx
try:
    import onnxruntime as ort
    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False

import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# 添加自定义模块
from models import build_model
from config import cfg, merge_from_file
from pred import visualize_prediction, preprocess_image, init_model, predict, draw_points_with_origin

def parse_args():
    parser = argparse.ArgumentParser(description='导出APGCC模型为ONNX格式')
    parser.add_argument('-c', '--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('-w', '--weights', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('-o', '--output', type=str, required=True, help='输出ONNX模型路径')
    parser.add_argument('-i', '--input', type=str, default="demo/demo1.jpg", help='输入图像路径')
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help='检测阈值')
    parser.add_argument('--output-image', type=str, default="output/prediction", help='推理结果输出图像路径')
    args = parser.parse_args()
    return args




def visualize_prediction_with_pytorch(model, img_tensor, original_img, device, base_name, threshold=0.5, output_dir=None):
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        
        outputs_scores, outputs_points = predict(model, img_tensor, device)
        
        mask = outputs_scores > threshold
        points = outputs_points[mask].detach().cpu().numpy().tolist()
        predict_cnt = int(mask.sum())
            
        print(f"pytorch model预测人数: {predict_cnt}")
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"pytorch_result_{base_name}")
            
            visualize_prediction(original_img, points, output_path)


def visualize_prediction_with_onnx(
    onnx_file_path,
    img_tensor,
    original_img,
    base_name,
    threshold=0.5,
    output_dir=None
):
    onnx_model = onnx.load(onnx_file_path)
    onnx.checker.check_model(onnx_model)
    
    ort_session = ort.InferenceSession(onnx_file_path, providers=['CPUExecutionProvider'])
    
    # 准备输入
    ort_inputs = {ort_session.get_inputs()[0].name: img_tensor.cpu().numpy()}
    
    # 运行推理
    ort_outputs = ort_session.run(None, ort_inputs)
    
    # 处理结果
    onnx_points = get_predict_points_from_onnx(ort_outputs, threshold)
    onnx_count = len(onnx_points)
    print(f"ONNX模型预测人数: {onnx_count}")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"onnx_result_{base_name}")
        visualize_prediction(original_img, onnx_points, output_path)

# 添加可视化函数
def main():
    
    # 解析参数
    args = parse_args()
    
    # 设置设备
    device = torch.device(f'cuda:{cfg.GPU_ID}' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 构建模型
    model = init_model(cfg, args.weights, device)
    
        # 预处理图像
    image_path = os.path.join(os.path.dirname(__file__), args.input)
    img_tensor, original_img = preprocess_image(image_path)
    
    img_tensor_copy = img_tensor.clone()
    

    # 准备输出目录
    output_path = args.output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    
    # 导出ONNX模型
    # 动态尺寸设置
    dynamic_axes = {
        'input': {0: 'batch_size', 2: 'height', 3: 'width'},
        'pred_logits': {0: 'batch_size'},
        'pred_points': {0: 'batch_size'}
    }
    
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
            img_tensor,
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
        
        visualize_prediction_with_pytorch(
            model,
            img_tensor_copy,
            original_img,
            device,
            os.path.basename(args.input),
            args.threshold,
            args.output_image
        )
        
        visualize_prediction_with_onnx(
            output_path,
            img_tensor,
            original_img,
            os.path.basename(args.input),
            args.threshold,
            args.output_image
        )
    
    except Exception as e:
        print(f"ONNX导出失败: {e}")

# 从ONNX模型输出中获取预测点
def get_predict_points_from_onnx(onnx_outputs, threshold=0.5):
    # 这里需要根据您的模型输出格式来实现预测点的提取
    # 以下是一个示例实现，可能需要根据您的具体模型调整
    onnx_output = onnx_outputs[0]  # 假设第一个输出是预测结果
    
    # 获取预测的锚点
    anchor_points = onnx_output[:, 0]  # 假设outputs的形状为[points, 1]
    
    # 过滤掉可能性低的点
    valid_points = np.where(anchor_points > threshold)[0]
    
    # 获取有效点的坐标
    points = []
    if len(valid_points) > 0:
        # 假设模型中有anchor_points存储了预测点的坐标
        points = onnx_output[valid_points, 1:3]
    
    return points

if __name__ == "__main__":
    main()


# python  export_onnx.py  -c configs/SHHA_test.yml  -w ../output/SHHA_best.pth -o ../output/onnx/model.onnx
