"""
PyTorch 学习第一关：从零实现线性回归（完整训练推理基础链路版）

目标：不使用任何 PyTorch 封装好的模块，纯手工搭建完整链路，
     理解从训练到推理的完整链路。

本文件按"完整训练推理链路"七环节组织：
    Step 1：模块导入        —— 引入所需的库
    Step 2：全局初始化      —— 设置随机种子与计算设备
    Step 3：数据准备        —— 加载数据并划分训练集/验证集/测试集
    Step 4：模型/参数初始化 —— 手动创建可学习参数 w、b
    Step 5：超参数设置      —— 学习率、训练轮数、损失函数与评估指标
    Step 6：训练            —— 初始化累计量 → 数据迁移 → 梯度清零 → 前向 → 损失 → 反向 → 更新 → 累计统计 → 指标聚合 → 验证评估 → 结果展示
    Step 7：推理            —— 测试集最终评估

真实函数：y = 2x + 3
"""

# ============================================================
# Step 1：模块导入
# ============================================================

# torch 是 PyTorch 的核心库，提供张量运算、自动微分等基础能力。
# 本例只需要它一个：张量创建、随机数、no_grad 上下文都由它提供。
import torch

# ============================================================
# Step 2：全局初始化
# ============================================================

# 设置随机种子，保证每次运行结果一致（可复现性），方便调试
torch.manual_seed(42)

# device 表示张量存放和运算所在的计算设备（CPU / GPU 等）。
# 深度学习计算量大，通常把数据和模型放到 GPU 等加速设备上运算；
# 注意：数据和模型必须在同一设备上才能一起运算。
# 不同系统的配置方式可能不一样：
#   NVIDIA GPU 用 "cuda"，Mac M 系列芯片用 "mps"（Metal 加速），
#   没有加速设备时回退到 "cpu"。
# 当前系统使用的是 Mac M 系列芯片，所以采用下面的 mps 判断方式，
# 这里打印确认当前设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"当前 device: {device}")

# ============================================================
# Step 3：数据准备
# ============================================================

# Step 3.1：加载原始数据

# 真实项目中这一步是从文件/数据库/API 加载数据，
# 当前源文件用"生成"的方式得到原始数据

# 生成 100 个样本的输入，范围在 0~2 之间，x 形状 [100, 1]
x = torch.rand(100, 1) * 2

# 根据真实关系 y = 2x + 3 生成标签，加入随机噪声模拟真实数据的不确定性
# 噪声让模型不能靠记住数据来"作弊"，必须学到真正的规律
# noise 形状 [100, 1]
noise = torch.randn(100, 1) * 0.3
# 隐式广播：标量 2、3 的形状可视为 []，
# 与 x、noise（形状均 [100, 1]）运算时会被广播成 [100, 1]，再逐元素相乘/相加
# y 形状 [100, 1]
y = 2 * x + 3 + noise

# Step 3.2：划分训练集 / 验证集 / 测试集（70% / 20% / 10%）

# 三个集合的用途区别：
#   训练集（train）：训练时用来计算 loss、更新参数，是模型"见过"的数据
#   验证集（valid）：训练过程中定期评估，监控泛化能力、辅助调参决策
#   测试集（test） ：训练完全结束后只评估一次，模拟"真正没见过"的数据
# 为什么验证集和测试集要分开：
#   验证集参与调参决策（如选学习率、决定何时停止训练），
#   信息会间接"泄露"给模型，验证集上的表现会偏乐观；
#   测试集必须保持完全独立、只用一次，才能给出无偏的最终评估。

# 随机打乱样本索引，避免划分结果与数据生成顺序相关
n_samples = x.shape[0]
# indices 形状 [100]
indices = torch.randperm(n_samples)

# 按 70% / 20% / 10% 计算各集合的样本数量
n_train = int(n_samples * 0.7)
n_valid = int(n_samples * 0.2)

# 对打乱后的索引切片，得到三组互不重叠的索引
# train_idx 形状 [70]
train_idx = indices[:n_train]
# valid_idx 形状 [20]
valid_idx = indices[n_train:n_train + n_valid]
# test_idx 形状 [10]
test_idx = indices[n_train + n_valid:]

# 用索引取出三个集合的数据
# x_train、y_train 形状均 [70, 1]
x_train, y_train = x[train_idx], y[train_idx]
# x_valid、y_valid 形状均 [20, 1]
x_valid, y_valid = x[valid_idx], y[valid_idx]
# x_test、y_test 形状均 [10, 1]
x_test, y_test = x[test_idx], y[test_idx]

# Step 3.3：特征工程

# 特征工程在真实项目中是数据准备的关键一环，至少应包含：
#   类别型特征编码：如 one-hot、label encoding
#   数值型特征的归一化/标准化：如 min-max、z-score
# 注意：归一化/标准化的统计量（均值、方差等）必须只在训练集上计算，
#       然后应用到验证集和测试集，避免数据泄露。
# 本例特征是单一数值且范围可控，故跳过。

# ============================================================
# Step 4：模型/参数初始化
# ============================================================

# requires_grad=True 告诉 PyTorch：我需要对这个张量求导。
# 只有标记了 requires_grad 的张量，才会在计算图中被追踪，
# 之后调用 .backward() 时才能算出它的梯度。
# 这是 PyTorch 自动微分机制的入口开关。
# 注意：requires_grad=True 的参数必须直接在目标设备上创建（device=device）。
# 如果先在 cpu 上创建、再调用 .to(device)，
# .to() 返回的是计算图中的非叶子张量，
# 反向传播时梯度不会积累到它的 .grad 上，参数将无法更新。
# w 为标量参数，形状 [1]
w = torch.randn(1, requires_grad=True, device=device)
# b 为标量参数，形状 [1]
b = torch.zeros(1, requires_grad=True, device=device)

# ============================================================
# Step 5：超参数设置
# ============================================================

# 学习率决定每次参数更新的步长大小
# 太大会震荡发散，太小会收敛极慢
learning_rate = 0.1
epochs = 100

# 损失函数属于"训练配置"的一部分，
# 和学习率、epoch 数一样是训练前就确定的东西，所以放在超参数设置环节。
# 训练环节只做调用，职责更清晰。

# 损失函数：均方误差（MSE），训练时衡量预测与真实值的差距
# 这里用函数形式定义，训练环节只负责调用，职责分离
# 返回的 loss 为标量，形状 []
def mse_loss(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()

# loss 是给优化器"看"的：要求可导、对梯度友好；
# 评估指标是给人"看"的：要求直观可解释。两者职责不同。
# 本例中 MSE 既是 loss 也可以当评估指标，但工程上必须习惯把二者分开。

# 评估指标一：平均绝对误差（MAE），比 MSE 更直观，单位与 y 相同
# 返回的 MAE 为标量，形状 []
def mae_metric(y_pred, y_true):
    return (y_pred - y_true).abs().mean()

# 评估指标二：决定系数（R²），衡量模型解释数据变异的能力，越接近 1 越好
# 公式：R² = 1 - 残差平方和 / 总平方和
# 返回的 R² 为标量，形状 []
def r2_score(y_pred, y_true):
    # ss_res 为残差平方和，标量，形状 []
    ss_res = ((y_true - y_pred) ** 2).sum()
    # ss_tot 为总平方和，标量，形状 []
    # 隐式广播：y_true.mean() 是标量（形状 []），
    # 与 y_true 相减时会广播成与 y_true 相同的形状，再逐元素相减
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot

# ============================================================
# Step 6：训练
# ============================================================

for epoch in range(epochs):

    # Step 6.1：初始化 epoch 级累计量与列表
    # 工业界标准做法：每个 epoch 开始时先把累计量清零，
    # 每个 batch 结束后累计 loss 与样本数、收集预测值与真实值，
    # epoch 末尾统一计算整个训练集的 loss 和各指标。
    # loss_train_sum 为整个训练集的 loss 累计量
    loss_train_sum = 0.0
    # n_train_samples 为本 epoch 已处理的训练样本总数
    n_train_samples = 0
    # 整个训练集的各指标变量，先置初值，epoch 末尾统一计算
    loss_train = 0.0
    mae_train = 0.0
    r2_train = 0.0
    # 当前代码没有分 batch（全量数据等价一个 batch），
    # 但按有 batch 的通用写法组织：每个 batch 结束后
    # 把该 batch 的预测值和真实值追加进列表，
    # epoch 末尾统一计算整个训练集的指标
    y_pred_tensor_list = []
    y_true_tensor_list = []

    # Step 6.2：数据迁移到计算设备
    # 当前代码没有划分 batch，相当于默认把所有训练数据放进了一个 batch（全量批梯度下降）。
    # 正常有 batch 的训练中，每从 DataLoader 取出一个 batch，
    # 就要对这个 batch 的数据执行一次 to(device)，把它送上计算设备。
    # 这里把迁移放在循环内，就是为了跟这种"每取一个 batch 迁移一次"的工业界模式对齐。
    # 重复对已在 device 上的张量调用 .to(device) 是无害的空操作（no-op），不会重复拷贝。
    # x_batch_train / y_batch_train 表示当前 batch 的数据（本例即全量训练集），
    # 与 Step 3 划分出的数据集变量 x_train / y_train 区分开。
    # x_batch_train、y_batch_train 形状均 [70, 1]
    x_batch_train = x_train.to(device)
    y_batch_train = y_train.to(device)

    # Step 6.3：梯度清零
    # PyTorch 默认会累积梯度（而不是覆盖），
    # 这个设计是为了支持某些需要多次 backward 累加梯度的场景（如 RNN、梯度累积训练）。
    # 但在常规训练中，每个 batch/epoch 都是独立的，
    # 如果不清零，上一轮的梯度会叠加到这一轮，导致参数更新方向错误。
    # 放在每轮开头："先清空旧的，再算新的"，逻辑更清晰，也避免遗漏残留梯度。
    if w.grad is not None:
        w.grad.zero_()
        b.grad.zero_()

    # Step 6.4：前向传播
    # 用当前的 w 和 b 对训练集样本做预测
    # 只有训练集参与参数更新，验证集/测试集不进入训练
    # 隐式广播：w 形状 [1]、x_batch_train 形状 [70, 1]，
    # w 会广播成 [70, 1] 与 x_batch_train 逐元素相乘；
    # b 形状 [1]，同样广播成 [70, 1] 后逐元素相加
    # y_batch_pred 形状 [70, 1]
    y_batch_pred = w * x_batch_train + b

    # Step 6.5：计算损失
    # 调用超参数设置环节定义好的 MSE 损失函数，在训练集上计算
    # 训练环节只负责调用，损失的定义在训练前就已确定，职责分离
    # loss_batch 为标量，形状 []
    loss_batch = mse_loss(y_batch_pred, y_batch_train)

    # Step 6.6：反向传播
    # .backward() 从 loss_batch 出发，沿着计算图反向传播，
    # 自动计算 loss 对每一个 requires_grad=True 的张量的偏导数，
    # 结果存储在各张量的 .grad 属性中。
    # 这一步只算梯度，不更新参数。
    loss_batch.backward()

    # Step 6.7：参数更新
    # torch.no_grad() 的作用：临时关闭梯度追踪。
    # 如果不关闭，对 w 和 b 的赋值操作也会被记录到计算图中，
    # 导致计算图无限膨胀，还会影响下一轮的梯度计算。
    # 参数更新是"工程操作"而非"数学计算"，不需要求导。
    with torch.no_grad():
        # 梯度下降：沿梯度的反方向移动，减小损失
        # 更新后 w、b 形状仍为 [1]
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # Step 6.8：累计 epoch 级统计量（参数更新后、当前 batch 结束时）
    # mse_loss 是 mean 口径：loss_batch.item() 是当前 batch 的平均 loss，
    # 乘以 batch 样本数还原成该 batch 的 loss 总和后再累计，
    # epoch 末尾除以总样本数，即得到按样本数加权平均的整个训练集 loss，
    # 这样即使各 batch 大小不同（如最后一个 batch 不满），结果依然正确。
    # batch_size 为当前 batch 的样本数（本例全量数据即一个 batch，值为 70）
    batch_size = x_batch_train.shape[0]
    loss_train_sum += loss_batch.item() * batch_size
    n_train_samples += batch_size

    # 追加当前 batch 的预测值和真实值，供 epoch 末尾统一计算训练集指标
    # detach 的作用：返回一个与计算图断开的新张量，避免保留梯度信息，
    # 否则列表会一直持有各 batch 的计算图，白白占用内存
    y_pred_tensor_list.append(y_batch_pred.detach())
    y_true_tensor_list.append(y_batch_train)

    # Step 6.9：计算整个训练集的 loss 和各指标（所有 batch 处理完后）
    # loss_train = loss 总和 / 总样本数，与 mse_loss 的 mean 口径一致
    loss_train = loss_train_sum / n_train_samples

    # loss 用于优化器梯度更新（给机器看），MAE/R² 用于人工判断模型效果（给人看）。
    # 评估不参与梯度计算，需用 no_grad 避免计算图开销。
    with torch.no_grad():
        # torch.cat 沿第 0 维把列表中各 batch 的张量拼接成完整训练集
        # y_pred_tensor 形状 [70, 1]
        y_pred_tensor = torch.cat(y_pred_tensor_list, dim=0)
        # y_true_tensor 形状 [70, 1]
        y_true_tensor = torch.cat(y_true_tensor_list, dim=0)

        # MAE：平均偏差，单位与 y 相同
        # mae_train 为标量，形状 []
        mae_train = mae_metric(y_pred_tensor, y_true_tensor)

        # R²：越接近 1，模型解释能力越强
        # r2_train 为标量，形状 []
        r2_train = r2_score(y_pred_tensor, y_true_tensor)

    # Step 6.10：验证集评估与训练状态打印
    # 每个 epoch 都打印训练状态，观察收敛过程
    # 在验证集上算指标：验证集没参与参数更新，能反映模型对"没见过的数据"的泛化能力。
    # 评估不参与梯度计算，需用 no_grad 避免计算图开销。
    with torch.no_grad():
        # 验证集在评估前才迁移到计算设备，与训练集"使用前迁移"的模式一致
        # 重复调用 .to(device) 同样是无害的空操作（no-op）
        # x_valid、y_valid 形状均 [20, 1]
        x_valid = x_valid.to(device)
        y_valid = y_valid.to(device)

        # 用当前参数在验证集上做前向计算
        # 隐式广播同 Step 6.4（w、b 广播成 [20, 1]）
        # y_valid_pred 形状 [20, 1]
        y_valid_pred = w * x_valid + b

        # 验证集 loss：与训练 loss 同口径，便于对比训练/验证差距、监控过拟合
        # loss_valid 为标量，形状 []
        loss_valid = mse_loss(y_valid_pred, y_valid)

        # MAE：平均偏差，单位与 y 相同
        # mae_valid 为标量，形状 []
        mae_valid = mae_metric(y_valid_pred, y_valid)

        # R²：越接近 1，模型解释能力越强
        # r2_valid 为标量，形状 []
        r2_valid = r2_score(y_valid_pred, y_valid)

    # 一行内按数据集分组输出：先训练集、后验证集，最后是当前参数
    print(f"Epoch {epoch + 1:3d} | Train Loss: {loss_train:.4f} MAE: {mae_train.item():.4f} R²: {r2_train.item():.4f} | Valid Loss: {loss_valid.item():.4f} MAE: {mae_valid.item():.4f} R²: {r2_valid.item():.4f} | w: {w.item():.4f} | b: {b.item():.4f}")

# Step 6.11：训练结果展示
# 学到的参数与真实参数对比，确认训练是否学到了接近真实的规律，
# 属于训练环节的收尾
print("\n" + "=" * 40)
print("训练完成！")
print(f"学习到的参数：w = {w.item():.4f}, b = {b.item():.4f}")
print(f"真实参数：    w = 2.0000, b = 3.0000")
print("=" * 40)

# ============================================================
# Step 7：推理
# ============================================================

# 测试集最终评估：
# 测试集在训练和调参过程中从未被使用，
# 此时评估一次，得到的才是模型泛化能力的无偏估计。
# 评估不需要梯度，用 torch.no_grad() 包裹，省去计算图的构建开销。

with torch.no_grad():
    # 与训练时同样的惯例：测试集张量在使用前才迁移到计算设备
    # x_test、y_test 形状均 [10, 1]
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    # 用训练好的参数对测试集做前向计算
    # 隐式广播同 Step 6.4（w、b 广播成 [10, 1]）
    # y_test_pred 形状 [10, 1]
    y_test_pred = w * x_test + b

    # 复用 Step 5 定义的 mse_loss 计算测试集 loss：
    # 训练时看的 loss，推理/最终评估时也可以直接复用，口径完全一致。
    # 本例损失函数就是 MSE，所以 loss_test 与下面的 mse_test 数值相同，
    # 但工程上 loss 和评估指标要分开计算与打印，
    # 因为在其他任务中两者往往不同（如分类任务 loss 是交叉熵、指标是准确率）。
    # loss_test 为标量，形状 []
    loss_test = mse_loss(y_test_pred, y_test)

    # MSE：与训练时的 loss 同口径，作为参照
    # mse_test 为标量，形状 []
    mse_test = mse_loss(y_test_pred, y_test)

    # MAE：平均偏差，单位与 y 相同，更直观
    # mae_test 为标量，形状 []
    mae_test = mae_metric(y_test_pred, y_test)

    # R²：越接近 1，说明模型对数据变异的解释能力越强
    # r2_test 为标量，形状 []
    r2_test = r2_score(y_test_pred, y_test)

# 打印最终评估结果：loss 放最前面，评估指标跟在后面
print(f"\n测试集评估 | Loss: {loss_test.item():.4f} | MSE: {mse_test.item():.4f} | MAE: {mae_test.item():.4f} | R²: {r2_test.item():.4f}")
