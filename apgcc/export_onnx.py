import os
import sys
import argparse
import torch
import numpy as np
from torch import nn
import onnx
import onnxruntime as ort

from config import cfg, merge_from_file, merge_from_list
from pred import visualize_prediction, preprocess_image, init_model, predict, draw_points_with_origin

def parse_args():
    parser = argparse.ArgumentParser('APGCC导出ONNX模型脚本')
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




def visualize_prediction_with_pytorch(model, img_tensor, original_img, device, base_name, threshold=0.5, output_dir=None):
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        
        outputs_scores, outputs_points = predict(model, img_tensor, device)
        
            
        print(f"outputs_scores.shape: {outputs_scores.shape}")
        print(f"outputs_scores.std: {outputs_scores.std()}")
        print(f"outputs_scores.mean: {outputs_scores.mean()}")
        
        
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
    # 加载ONNX模型
    onnx_model = onnx.load(onnx_file_path)
    onnx.checker.check_model(onnx_model)
    
    # 创建推理会话
    ort_session = ort.InferenceSession(onnx_file_path, providers=['CPUExecutionProvider'])
    
    # 准备输入
    ort_inputs = {ort_session.get_inputs()[0].name: img_tensor.cpu().numpy()}
    
    # 运行推理
    (scores, points) = ort_session.run(None, ort_inputs)
    
    mask = scores > threshold
    points = points[mask].tolist()
    onnx_count = int(mask.sum())
    
    
    print(f"ONNX模型预测人数: {onnx_count}")
    
    # 可视化并保存结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"onnx_result_{base_name}")
        visualize_prediction(original_img, points, output_path)
        
        # # 在原图上直接画出点
        # marked_output_path = os.path.join(output_dir, f"onnx_marked_{base_name}")
        # draw_points_with_origin(original_img, points, marked_output_path)

# 添加可视化函数
def main():
    
    # 解析参数
    cfg, args = parse_args()
    
    # 设置设备
    device = torch.device(f'cuda:{cfg.GPU_ID}' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 构建模型
    model = init_model(cfg, args.weight, device)
    
        # 预处理图像
    image_path = os.path.join(os.path.dirname(__file__), args.input)
    img_tensor, original_img = preprocess_image(image_path)
    
    
    print(f"img_tensor shape: {img_tensor.shape}")
    # print(f"img_tensor mean: {img_tensor.mean()}")
    
    img_tensor_copy = img_tensor.clone()

    # 准备输出目录
    output_path = args.output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    
    # 导出ONNX模型
    # 动态尺寸设置
    dynamic_axes = {
        'input': {0: 'batch_size', 1: 'channel', 2: 'height', 3: 'width'},
        'scores': {0: 'batch_size'},
        'points': {0: 'batch_size'}
    }
    
    # 定义模型的输入和输出名称
    input_names = ['input']
    output_names = ['scores', 'points']
    
    try:
        # 创建一个包装函数来处理模型的输出，确保它返回元组而不是字典
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super(ModelWrapper, self).__init__()
                self.model = model
                
            def forward(self, x):
                outputs = self.model(x)
                # 获取预测结果
                outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
                outputs_points = outputs['pred_points'][0]
                return outputs_scores, outputs_points
        
        wrapped_model = ModelWrapper(model)
        apgcc_onnx_path = os.path.join(args.output, "apgcc.onnx")
        
        print(f"正在导出ONNX模型到: {apgcc_onnx_path}")

        torch.onnx.export(
            wrapped_model,
            img_tensor,
            apgcc_onnx_path,
            export_params=True,
            opset_version=16,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        
        print(f"ONNX模型导出成功: {apgcc_onnx_path}")
        
        # 验证ONNX模型
        model2 = init_model(cfg, args.weight, device)
        visualize_prediction_with_pytorch(
            model2,
            img_tensor_copy,
            original_img,
            device,
            os.path.basename(args.input),
            args.threshold,
            args.output
        )
        
        visualize_prediction_with_onnx(
            apgcc_onnx_path,
            img_tensor,
            original_img,
            os.path.basename(args.input),
            args.threshold,
            args.output
        )
    
    except Exception as e:
        print(f"ONNX导出失败: {e}")
        raise e
        

# 从ONNX模型输出中获取预测点
def get_predict_points_from_onnx(onnx_outputs, threshold=0.5):
    # 根据pred.py中的逻辑提取预测点
    # 假设ONNX模型输出顺序为：pred_logits, pred_points（与PyTorch模型输出一致）
    pred_logits = onnx_outputs[0]  # 第一个输出是logits
    pred_points = onnx_outputs[1]  # 第二个输出是点坐标
    
    # 计算softmax得分，与pred.py中的逻辑相同
    # 假设pred_logits的形状为 [batch_size, num_queries, 2]
    scores = softmax(pred_logits, axis=-1)[0, :, 1]  # 取第一个batch，所有queries的第二个类别概率
    
    # 过滤掉低于阈值的点
    valid_indices = np.where(scores > threshold)[0]
    
    # 提取有效的点坐标
    points = []
    if len(valid_indices) > 0:
        # 取出有效的点坐标
        points = pred_points[0, valid_indices].copy()  # 确保复制数据，防止引用问题
    
    return points

# 添加一个softmax函数，因为numpy没有直接提供softmax
def softmax(x, axis=None):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)



if __name__ == "__main__":
    main()


# python  export_onnx.py  -c configs/SHHA_test.yml  -w ../output/SHHA_best.pth -o ../output/onnx/model.onnx
