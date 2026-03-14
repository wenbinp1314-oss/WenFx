import torch
import os
import sys
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# --- 核心修正：将 Script 文件夹加入系统搜索路径 ---
# 获取当前脚本所在目录的 Script 子目录路径
script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Script')
if script_dir not in sys.path:
    sys.path.append(script_dir)

# 现在你可以直接导入了，不需要加 Script. 前缀
from config import *
from model import ModuleG, ModuleF
from dataset import FrothDataset


def test_inference():
    # --- 1. 路径配置 ---
    test_csv = r'F:\wenFX\data\test\reagents_train.csv'
    test_initial_dir = r'F:\wenFX\data\test\initial'
    test_target_dir = r'F:\wenFX\data\test\target'

    # 使用最佳模型权重
    g_weights = r'F:\wenFX\Script\checkpoint\netG_best.pth'
    f_weights = r'F:\wenFX\Script\checkpoint\netF_best.pth'

    # 创建推理结果存放目录
    save_dir = r'F:\wenFX\test_results'
    os.makedirs(save_dir, exist_ok=True)

    # --- 2. 加载模型 ---
    netG = ModuleG().to(DEVICE)
    netF = ModuleF().to(DEVICE)

    # 加载权重
    netG.load_state_dict(torch.load(g_weights, map_location=DEVICE))
    netF.load_state_dict(torch.load(f_weights, map_location=DEVICE))

    # 开启评估模式 [cite: 243]
    netG.eval()
    netF.eval()

    # --- 3. 准备测试数据 ---
    test_dataset = FrothDataset(csv_file=test_csv,
                                initial_dir=test_initial_dir,
                                target_dir=test_target_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f"开始预测，测试集样本数: {len(test_dataset)}")

    # --- 4. 执行预测 ---
    with torch.no_grad():
        for i, (init_img, target_img, reagents) in enumerate(test_loader):
            init_img = init_img.to(DEVICE)
            target_img = target_img.to(DEVICE)
            reagents = reagents.to(DEVICE)

            # A. 通过生成器 G 生成预测图像 [cite: 93, 234]
            fake_img = netG(init_img, reagents)

            # B. 通过判别器 F 计算评分 [cite: 101, 146]
            perception_score, _, _ = netF(fake_img)

            # C. 拼接对比图：初始图 | 预测生成的图 | 真实目标图
            comparison = torch.cat([init_img, fake_img, target_img], dim=3)

            save_path = os.path.join(save_dir, f'test_sample_{i}_score_{perception_score.item():.2f}.png')
            save_image(comparison, save_path, normalize=True)

            print(f"样本 {i} 处理完成，评分: {perception_score.item():.4f}")

    print(f"所有预测图已保存至: {save_dir}")


if __name__ == "__main__":
    test_inference()