
class Args:
    maxdisp = 192
    maxdisplist = [12, 3, 3]
    with_spn = False
    init_channels = 1
    nblocks = 2
    channels_3d = 4
    layers_3d = 4
    spn_init_channels = 8  # 添加这一行
    growth_rate = [32, 32, 32]  # 修改为列表形式
args = Args()
import torch
import torch.utils.data
from models import anynet
from dataloader.KITTIloader2015 import dataloader as DA
from PIL import Image
import numpy as np

# 参数设置
model_path = "/home/berlin/test/AnyNet-master/checkpoint/kitti2015_ck/checkpoint.tar"  # 提供的预训练模型路径
left_image_path = "/home/berlin/test/data_scene_flow/training/image_2/000000_10.png"  # 替换为左眼图像路径
right_image_path = "/home/berlin/test/data_scene_flow/training/image_3/000000_10.png"  # 替换为右眼图像路径
output_depth_map_path = "/home/berlin/test/results/depth_map.png"  # 替换为希望保存深度图的路径

# 加载模型
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model = anynet.AnyNet(args)  # 注意: 'args' 必须包含初始化AnyNet所需的参数
model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict['state_dict'].items()})
model.eval()

# 加载图像
left_img = Image.open(left_image_path).convert("RGB")
right_img = Image.open(right_image_path).convert("RGB")
# 注意: 以下转换可能需要根据您的数据预处理步骤进行调整
imgL = np.asarray(left_img).transpose(2, 0, 1)
imgR = np.asarray(right_img).transpose(2, 0, 1)

# 执行推理
with torch.no_grad():
    disp = model(imgL, imgR)  # 假设模型输出是深度图

# 保存深度图
depth_map = disp.squeeze().cpu().numpy()
Image.fromarray((depth_map * 255).astype(np.uint8)).save(output_depth_map_path)
