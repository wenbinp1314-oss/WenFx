# 配置文件（存放超参数）
import torch

# 图像相关
IMAGE_SIZE = 256  # 论文设定分辨率 [cite: 224]
CHANNELS = 3      # RGB图像 [cite: 220]

# 训练超参数 (源自论文 4.2 节)
LEARNING_RATE = 0.0001        # [cite: 236]
BETA1 = 0.5                  # Adam 优化器参数 [cite: 236]
BETA2 = 0.999                # [cite: 236]
BATCH_SIZE = 2              # [cite: 238]
EPOCHS = 200                 # [cite: 238]-

# 损失函数权重系数 (公式 12)
LAMBDA_REC = 10     # 重构损失权重 lambda1 [cite: 239]
LAMBDA_CONT = 1     # 内容损失权重 lambda2 [cite: 239]
LAMBDA_INFO = 10    # 互信息损失权重 lambda3 [cite: 239]

# 药剂相关
REAGENT_DIM = 6     # 6种药剂：黑药, 硫酸铜, 黄药, 2号油, 硝酸铅 [cite: 227]

# 设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")