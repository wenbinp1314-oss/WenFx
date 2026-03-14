# 工具脚本（归一化、日志记录等）
import torch

def normalize_reagents(u, u_min, u_max):
    """
    根据论文公式 (14) 进行药剂归一化 [cite: 231]
    u_norm = -1 + (u - u_min) / (u_max - u_min) * 2
    """
    u_norm = -1 + ((u - u_min) / (u_max - u_min)) * 2
    return u_norm

def denormalize_reagents(u_norm, u_min, u_max):
    """将生成的预测值还原回真实药剂物理单位"""
    return ((u_norm + 1) / 2) * (u_max - u_min) + u_min