import torch
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os

# 导入你项目中的模块
from config import *
from model import ModuleG, ModuleF
from dataset import FrothDataset


# --- 1. WGAN-GP 梯度惩罚函数 [cite: 68] ---
def compute_gradient_penalty(netF, real_samples, fake_samples):
    """计算梯度惩罚项以满足 Lipschitz 约束 [cite: 66, 68]"""
    alpha = torch.rand((real_samples.size(0), 1, 1, 1)).to(DEVICE)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

    # 获取特征感知分支的输出 [cite: 107]
    d_interpolates, _, _ = netF(interpolates)

    fake = torch.ones(d_interpolates.shape).to(DEVICE)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    # 强制梯度范数接近 K=1 [cite: 66, 68]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# --- 2. 初始化流程 ---
# 确保结果目录存在
os.makedirs('results', exist_ok=True)
os.makedirs('checkpoint', exist_ok=True)

dataset = FrothDataset(csv_file=r'F:\wenFX\data\train\reagents_train.csv',
                       initial_dir=r'F:\wenFX\data\train\initial',
                       target_dir=r'F:\wenFX\data\train\target')
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

netG = ModuleG().to(DEVICE)
netF = ModuleF().to(DEVICE)

# 论文设定优化器参数 [cite: 236]
optimizerG = optim.Adam(netG.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
optimizerF = optim.Adam(netF.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

best_loss_g = float('inf')  # 初始化为正无穷

# --- 3. 核心训练循环 [cite: 240] ---
for epoch in range(EPOCHS):
    epoch_loss_g = 0.0  # 用于计算全 Epoch 平均损失
    for i, (init_img, target_img, reagents) in enumerate(dataloader):
        init_img, target_img, reagents = init_img.to(DEVICE), target_img.to(DEVICE), reagents.to(DEVICE)

        ############################
        # (1) 更新 Module F：最大化特征感知差异 [cite: 191, 214]
        ############################
        optimizerF.zero_grad()

        # 生成预测图并脱离计算图（冻结G） [cite: 241]
        fake_img = netG(init_img, reagents).detach()

        # 对抗差异损失计算 [cite: 191]
        perception_real, _, _ = netF(target_img)
        perception_fake, _, _ = netF(fake_img)

        # WGAN-GP 梯度惩罚 [cite: 68]
        gp = compute_gradient_penalty(netF, target_img, fake_img)

        # 判别器总损失 [cite: 191]
        loss_F = torch.mean(perception_fake) - torch.mean(perception_real) + 10 * gp

        loss_F.backward()
        optimizerF.step()

        ############################
        # (2) 更新 Module G：复合损失最小化 [cite: 212, 213]
        ############################
        optimizerG.zero_grad()

        # 重新生成预测图
        fake_img = netG(init_img, reagents)

        # A. 对抗损失 [cite: 192]
        perception_fake, mu_fake, var_fake = netF(fake_img)
        loss_G_adv = -torch.mean(perception_fake)

        # B. 重构损失 (逐像素相似度) [cite: 196]
        loss_G_rec = F.mse_loss(fake_img, target_img)

        # C. 内容损失 (基于编码器 enc6 层特征) [cite: 201, 203]
        # 提取真实图和生成图在编码器最深层的特征差异
        real_feat = netG.enc6(netG.enc5(netG.enc4(netG.enc3(netG.enc2(netG.enc1(target_img))))))
        fake_feat = netG.enc6(netG.enc5(netG.enc4(netG.enc3(netG.enc2(netG.enc1(fake_img))))))
        loss_G_content = F.mse_loss(fake_feat, real_feat)

        # D. 互信息损失 (最大化药剂相关性) [cite: 186, 207]
        # 使用辅助后验分布 Q(u|x_hat) 计算高斯负对数似然 [cite: 209, 210]
        loss_G_info = F.gaussian_nll_loss(mu_fake, reagents, torch.exp(var_fake))

        # 最终总损失加权求和 [cite: 213, 215]
        total_loss_G = loss_G_adv + LAMBDA_REC * loss_G_rec + LAMBDA_CONT * loss_G_content + LAMBDA_INFO * loss_G_info

        total_loss_G.backward()
        optimizerG.step()

        # 【关键修正】：必须累加损失，否则平均损失永远是 0
        epoch_loss_g += total_loss_G.item()

        # 实时打印 Batch 进度
        if i % 10 == 0:
            print(f"Epoch [{epoch}/{EPOCHS}] Batch {i}/{len(dataloader)} "
                  f"Loss_F: {loss_F.item():.4f} Loss_G: {total_loss_G.item():.4f}")

    # --- 4. 周期性保存预测图与权重 [cite: 242] ---
    avg_epoch_loss = epoch_loss_g / len(dataloader)

    # A. 如果当前损失是历史最低，保存为“最佳模型”
    if avg_epoch_loss < best_loss_g:
        best_loss_g = avg_epoch_loss
        torch.save(netG.state_dict(), 'checkpoint/netG_best.pth')
        torch.save(netF.state_dict(), 'checkpoint/netF_best.pth')
        print(f"--- 发现更优模型 (Avg Loss: {best_loss_g:.4f})，已更新 best 权重 ---")

        # 每 10 个 Epoch 保存一次对比图
    if epoch % 10 == 0:
        viz_results = torch.cat([init_img[0:1], fake_img[0:1], target_img[0:1]], dim=0)
        save_image(viz_results, f'results/epoch_{epoch}_comparison.png', normalize=True)

# --- 5. 整个循环结束后，保存“最后一次模型” ---
torch.save(netG.state_dict(), 'checkpoint/netG_last.pth')
torch.save(netF.state_dict(), 'checkpoint/netF_last.pth')
print("训练正式完成！")