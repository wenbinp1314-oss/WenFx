# 网络架构脚本（Module G 和 F）
import torch
import torch.nn as nn
from config import REAGENT_DIM


# --- 基础组件：卷积块和反卷积块 ---
def conv_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, norm=True):
    """编码阶段的标准块：Conv + BN + ReLU [cite: 175]"""
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
    if norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def deconv_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, activation='relu'):
    """解码阶段的标准块：Deconv + BN + (ReLU/Tanh) [cite: 179]"""
    layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)]
    layers.append(nn.BatchNorm2d(out_channels))
    if activation == 'relu':
        layers.append(nn.ReLU(inplace=True))
    else:
        layers.append(nn.Tanh())  # 最后一层使用 Tanh [cite: 179]
    return nn.Sequential(*layers)


# --- 1. 生成器 Module G (Image-based reagent addition module) ---
class ModuleG(nn.Module):
    def __init__(self):
        super(ModuleG, self).__init__()
        # 编码阶段：6个卷积块，通道数从3逐步变为1024 [cite: 112, 113, 175]
        self.enc1 = conv_block(3, 32)
        self.enc2 = conv_block(32, 64)
        self.enc3 = conv_block(64, 128)
        self.enc4 = conv_block(128, 256)
        self.enc5 = conv_block(256, 512)
        self.enc6 = conv_block(512, 1024)

        # 解码阶段：6个反卷积块。注意：瓶颈层由于拼接了5种药剂，输入通道为1024+5=1029 [cite: 130, 132, 178, 179]
        self.dec1 = deconv_block(1024 + REAGENT_DIM, 512)
        self.dec2 = deconv_block(512, 256)
        self.dec3 = deconv_block(256, 128)
        self.dec4 = deconv_block(128, 64)
        self.dec5 = deconv_block(64, 32)
        self.dec6 = deconv_block(32, 3, activation='tanh')

    def forward(self, x, u):
        # x: 初始泡沫图 x(t) [cite: 89, 109]
        # u: 目标药剂剂量 u(t+) [cite: 89, 92]

        # 编码提取特征 [cite: 175]
        e6 = self.enc6(self.enc5(self.enc4(self.enc3(self.enc2(self.enc1(x))))))  # 4x4x1024 [cite: 130]

        # 瓶颈层融合药剂信息 [cite: 178]
        # 将药剂 u 扩展为 4x4 的特征图并拼接 [cite: 163, 178, 248]
        u_map = u.view(u.size(0), REAGENT_DIM, 1, 1).expand(-1, -1, 4, 4)
        combined = torch.cat([e6, u_map], dim=1)  # 4x4x1029 [cite: 130]

        # 解码重建预测图 x_hat(t+τ) [cite: 80, 133, 180]
        out = self.dec6(self.dec5(self.dec4(self.dec3(self.dec2(self.dec1(combined))))))
        return out


# --- 2. 感知模块 Module F (Feature difference perception module) ---
class ModuleF(nn.Module):
    def __init__(self):
        super(ModuleF, self).__init__()
        # 特征提取主体，与 G 的 Encoder 类似 [cite: 126, 173]
        self.feature_extractor = nn.Sequential(
            conv_block(3, 32), conv_block(32, 64), conv_block(64, 128),
            conv_block(128, 256), conv_block(256, 512), conv_block(512, 1024)
        )

        # 分支 A：特征感知 (用于计算对抗差异) [cite: 146, 184]
        self.perception_head = nn.Conv2d(1024, 1, kernel_size=4)  # 输出单值判断真假

        # 分支 B：互信息 (预测药剂剂量的均值和方差) [cite: 138, 143, 183, 185]
        # 使用 1x1 卷积实现 [cite: 162]
        self.mu_head = nn.Conv2d(1024, REAGENT_DIM, kernel_size=4)
        self.var_head = nn.Conv2d(1024, REAGENT_DIM, kernel_size=4)

    def forward(self, img):
        feat = self.feature_extractor(img)  # 4x4x1024

        # 输出特征感知值 [cite: 146]
        perception = self.perception_head(feat).view(feat.size(0), -1)

        # 输出均值和方差构建后验分布 [cite: 138, 143, 145, 185]
        mu = self.mu_head(feat).view(feat.size(0), -1)
        log_var = self.var_head(feat).view(feat.size(0), -1)

        return perception, mu, log_var