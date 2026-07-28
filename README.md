# PyTorch Learning by Doing

## 一、教学模式架构

### 1. 双文件交付模式

- `编号_ans_主题.py`：参考版，完整代码 + 详细注释，用于对照学习
- `编号_pra_主题.py`：练习版，结构和注释与参考版完全一致，核心代码留白 `...` + TODO 线索，由学习者亲手补全
- 工作流：生成/调整文件对 → 一致性检查 → 学习者填写练习版 → 检查反馈（只读检查，不改学习者代码）

### 2. 七步固定链路（每一关都遵循）

- **Step 1 模块导入**
- **Step 2 全局初始化**：随机种子、device（cuda/mps/cpu）
- **Step 3 数据准备**：加载原始数据 → 划分训练/验证/测试（7:2:1）→ 特征工程
- **Step 4 模型/参数初始化**：参数直接在 device 上创建
- **Step 5 超参数设置**：lr、epochs + loss/评估指标函数定义
- **Step 6 训练**：epoch 级聚合模式（初始化累计量和列表 → batch 迁移 → 梯度清零 → 前向 → 损失 → 反向 → 更新 → 累计追加 → epoch 末聚合 → 验证评估 → 打印）
- **Step 7 推理**：no_grad 内迁移测试集 → 预测 → loss + 指标最终评估

### 3. 已沉淀的规范体系

- 注释在代码上方，张量形状具名标注 `[..]`，隐式广播说明
- 命名：指标名_数据集（如 `loss_test`）、batch 语义（如 `x_batch_train`）、valid 不缩写
- 工程习惯：loss 给机器看/指标给人看、按 batch 迁移 `to(device)`、统计量只在训练集算、测试集只用一次

### 4. 教学理念

- **干中学**：以完整可运行的闭环任务为起点，遇到卡点再回溯补基础
- **手写先行**：每个新架构先纯手工实现理解机制，再对照官方封装
- 每关聚焦升级 1~2 个环节，链路始终完整、可运行、可对比

## 二、关卡安排（三阶段 24 关）

### 第一阶段：吃透基础链路（01-07）

| 关卡 | 升级环节 | 核心内容 | 数据集 |
|---|---|---|---|
| 01 | 全链路 | 纯手工线性回归，七步基础链路 | 程序生成（合成回归数据） |
| 02 | Step 4 + Step 6 | nn.Module 基础 + nn.Linear、nn.MSELoss、optim.SGD，并引入 nn.Sequential 快速搭建 | 程序生成（合成回归数据） |
| 03 | Step 3 + Step 6 | Dataset/DataLoader、真正的 mini-batch 训练 | 程序生成（合成回归数据） |
| 04 | Step 3 + Step 5 | 真实数据集、特征工程落地、Adam + 学习率调度器 | sklearn 本地数据集（diabetes/wine 等，零下载） |
| 05 | 任务切换 | 分类任务：交叉熵 loss、准确率指标 | 程序生成（sklearn make_classification） |
| 06 | Step 4 | MLP + 激活函数 + 正则化（dropout/weight decay）；同一网络用 nn.Sequential/nn.ModuleList/nn.ModuleDict 三种方式组织，对比适用场景 | 程序生成（sklearn make_classification） |
| 07 | Step 6 + Step 7 | 模型保存/加载、早停、训练可视化 | 沿用之前关卡数据 |

### 第二阶段：图像任务（含图像生成，08-15）

| 关卡 | 升级环节 | 核心内容 | 数据集 |
|---|---|---|---|
| 08 | Step 3 + Step 4 | CNN：手写卷积滑窗理解机制，对照 nn.Conv2d，手写数字分类 | sklearn digits（8×8 手写数字，本地零下载） |
| 09 | Step 4 | 深化 CNN：池化、多层堆叠、彩色图像分类、数据增强 | 程序生成几何图形（彩色、可控分辨率）+ digits |
| 10 | Step 4 | ResNet：残差连接，理解深层网络退化问题与 shortcut 的作用，手写残差块再对照 torchvision 实现 | digits / 生成几何图形 |
| 11 | 任务切换 | 自编码器 AE：图像重建/去噪，理解编码器-解码器结构 | digits |
| 12 | Step 4 | U-Net：编码器-解码器 + 跳跃连接，图像去噪/分割任务；同时是后续 Diffusion 的骨干网络 | digits 去噪 / 生成几何图形分割 |
| 13 | Step 4 + Step 6 | VAE：从重建到生成，重参数化技巧、KL 散度 | digits |
| 14 | Step 4 + Step 6 | GAN：生成器/判别器对抗训练，双模型、双优化器交替训练范式 | digits / 生成几何图形 |
| 15 | Step 4 + Step 6 | Diffusion Model：简化版 DDPM，加噪/去噪过程、噪声预测网络复用 U-Net | digits / 生成几何图形 |

### 第三阶段：序列建模（含文本生成与预训练，16-24）

| 关卡 | 升级环节 | 核心内容 | 数据集 |
|---|---|---|---|
| 16 | Step 3 + Step 4 | 时序数据处理（滑动窗口）+ 手写 RNN cell，对照 nn.RNN | 程序生成（正弦波、AR 过程） |
| 17 | Step 4 | LSTM/GRU + 梯度裁剪，真实时序预测任务 | 程序生成时序 |
| 18 | Step 3 + Step 4 | 文本处理与 Embedding：分词、词表、nn.Embedding、padding 与 mask | Tiny Shakespeare（data/ 目录本地文件） |
| 19 | 任务切换 | 字符级语言模型：LSTM 文本生成，理解自回归 | Tiny Shakespeare |
| 20 | Step 4 | 注意力机制：手写 Q/K/V 缩放点积注意力 | 程序生成序列 / Tiny Shakespeare |
| 21 | Step 4 | 多头注意力 + 位置编码：手写实现 | 程序生成序列 / Tiny Shakespeare |
| 22 | Step 4 + Step 6 | Transformer Encoder：LayerNorm/残差/FFN 堆叠，序列分类任务，对照 nn.Transformer | Tiny Shakespeare |
| 23 | 综合 | mini-BERT：掩码语言模型（MLM）预训练 + 微调下游分类，Encoder 路线收官 | Tiny Shakespeare |
| 24 | 综合 | mini-GPT：Transformer Decoder 自回归文本生成，Decoder 路线收官 | Tiny Shakespeare |

### 设计说明

- 数据集总原则：尽量不额外下载数据集，优先使用程序生成数据和 sklearn 随包安装的本地数据集；文本语料 Tiny Shakespeare（约 1MB）为唯一例外，一次性下载后存放在项目 data/ 目录复用
- 每个新架构都沿用"先手写、再对照官方封装"的模式
- 七步链路骨架贯穿全程，每关只动升级的环节
- 图像生成支线形成完整演进链：AE（重建）→ U-Net（跳跃连接）→ VAE（概率生成）→ GAN（对抗生成）→ Diffusion（迭代去噪生成），四种主流生成范式全覆盖
- nn.Module 家族分两步：02 掌握 nn.Module + nn.Sequential 基本用法，06 系统对比 Sequential/ModuleList/ModuleDict 三种容器
- 23/24 双收官设计：BERT 走 Encoder + 判别式微调路线，GPT 走 Decoder + 自回归生成路线，覆盖现代大模型的两大源头
