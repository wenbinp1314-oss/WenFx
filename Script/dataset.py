# 数据加载脚本
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
from config import IMAGE_SIZE
from utils import normalize_reagents


class FrothDataset(Dataset):
    def __init__(self, csv_file, initial_dir, target_dir):
        """
        参数:
            csv_file: 包含药剂数值和文件名索引的 CSV 文件路径
            initial_dir: 存放初始图 x(t) 的文件夹路径
            target_dir: 存放目标图 x(t+τ) 的文件夹路径
        """
        # 读取 CSV
        self.data_info = pd.read_csv(csv_file)
        self.initial_dir = initial_dir
        self.target_dir = target_dir

        # --- 核心修改：提前计算药剂列的极大值和极小值 ---
        # 假设第 2 到 7 列（索引 2:8）是药剂数据
        reagent_data = self.data_info.iloc[:, 2:8].values.astype('float32')
        self.u_min = np.min(reagent_data, axis=0)
        self.u_max = np.max(reagent_data, axis=0)

        # 图像预处理流水线 [cite: 224]
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # 缩放至 256x256 [cite: 224]
            transforms.ToTensor(),  # 转为 Tensor (0-1)
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到 [-1, 1] 以匹配 Tanh
        ])

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        # 1. 获取原始药剂数值
        reagents_raw = self.data_info.iloc[idx, 2:8].values.astype('float32')

        # --- 核心修改：执行动态归一化 ---
        reagents_norm = []
        for i in range(len(reagents_raw)):
            # 防止分母为 0（当最大值等于最小值时，直接设为 0，即归一化后的中心点）
            if self.u_max[i] == self.u_min[i]:
                reagents_norm.append(0.0)
            else:
                # 调用 utils.py 中的论文公式 (14)
                norm_val = normalize_reagents(reagents_raw[i], self.u_min[i], self.u_max[i])
                reagents_norm.append(norm_val)

        reagents_tensor = torch.tensor(reagents_norm, dtype=torch.float32)

        # 2. 获取图像文件名
        init_name = self.data_info.iloc[idx, 0]  # 初始图文件名
        target_name = self.data_info.iloc[idx, 1]  # 目标图文件名

        # 3. 读取图像
        init_path = os.path.join(self.initial_dir, init_name)
        target_path = os.path.join(self.target_dir, target_name)

        init_img = self.transform(Image.open(init_path).convert('RGB'))
        target_img = self.transform(Image.open(target_path).convert('RGB'))

        # 返回三元组：初始图、真实目标图、归一化后的药剂剂量 [cite: 92]
        return init_img, target_img, reagents_tensor