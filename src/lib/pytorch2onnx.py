import torch
import torchvision
import torch.onnx
from models.model import create_model, load_model

input = torch.randn(1, 3, 512, 512)
## mobilenetv2_10
# model = create_model('mobilenetv2_10', heads={'hm': 1, 'wh': 2, 'reg': 2}, head_conv=24)
## resnet18
model = create_model('res_18', heads={'hm': 1, 'wh': 2, 'reg': 2}, head_conv=64)
model = load_model(model, '/home/mario/Projects/Obj_det/CenterNet/exp/ctdet/face_res18/model_best.pth')


torch.onnx.export(model, input, "/home/mario/Projects/Obj_det/CenterNet/exp/ctdet/face_res18/res18.onnx", input_names=["input"], output_names=["hm","wh","reg"])
