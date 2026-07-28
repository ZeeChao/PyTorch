"""
PyTorch 学习第二关：框架化线性回归（nn.Module 完整训练推理链路版）

目标：把第一关纯手工搭建的组件逐个替换为 PyTorch 框架组件，
     理解"手写实现"与"框架封装"之间的一一对应关系。

本关升级点（对照第一关）：
    手写 w、b 参数          → nn.Module 子类 + nn.Linear
    手写 mse_loss 函数      → nn.MSELoss
    no_grad 中手动更新参数  → optim.SGD 的 optimizer.step()
    手动 w.grad.zero_()     → optimizer.zero_grad()
    手写 mae/r2 + 列表聚合  → torchmetrics 的 update/compute/reset

本文件按"完整训练推理链路"七环节组织：
    Step 1：模块导入        —— 引入所需的库
    Step 2：全局初始化      —— 设置随机种子与计算设备
    Step 3：数据准备        —— 加载数据并划分训练集/验证集/测试集
    Step 4：模型/参数初始化 —— nn.Module 子类替代手写 w、b
    Step 5：超参数设置      —— 学习率、训练轮数、损失函数、优化器与评估指标
    Step 6：训练            —— 初始化累计量 → 数据迁移 → 梯度清零 → 前向 → 损失 → 反向 → 更新 → 累计统计 → 指标聚合 → 验证评估 → 结果展示
    Step 7：推理            —— 测试集最终评估

真实函数：y = 2x + 3
"""

# ============================================================
# Step 1：模块导入
# ============================================================

# torch 是 PyTorch 的核心库，提供张量运算、自动微分等基础能力。
import torch

# torch.nn 是 PyTorch 的神经网络模块库，
# 提供 nn.Module（所有模型的基类）、nn.Linear（线性层）、
# nn.MSELoss（均方误差损失）等封装好的组件。
# 本关用它替代第一关手写的 w、b 参数和 mse_loss 函数。
# TODO: 导入 torch.nn 模块，并按惯例起别名 nn
...

# torch.optim 是 PyTorch 的优化器库，提供 SGD、Adam 等参数更新算法。
# 本关用它替代第一关在 no_grad 中手动执行的 w -= lr * w.grad。
# TODO: 导入 torch.optim 模块，并按惯例起别名 optim
...

# torchmetrics 是 PyTorch 生态的标准指标库，
# 提供 MeanAbsoluteError（MAE）、R2Score（R²）等封装好的评估指标。
# 本关用它替代第一关手写的 mae_metric、r2_score 函数
# 以及"列表追加 + torch.cat 统一计算"的聚合逻辑。
# TODO: 从 torchmetrics 中导入 MAE 和 R² 两个指标类（类名见上方注释）
...

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

# 第一关手写了 w = torch.randn(1, requires_grad=True, device=device) 和 b，
# 本关用 nn.Module 子类替代这套手写参数管理：
# nn.Module 是 PyTorch 所有模型的基类，负责登记参数、递归迁移设备、
# 切换训练/评估模式等"参数管理"的脏活累活，让我们只关注模型结构本身。
class LinearRegressionModel(nn.Module):

    # __init__ 中定义模型的"零件"（有参数的层都在这里创建）
    def __init__(self):
        # 必须先调用父类的 __init__，完成 nn.Module 内部登记机制的初始化，
        # 之后给 self 赋值的 nn.Linear 等子模块才会被自动登记为模型参数。
        # 漏掉这一行，参数不会被 model.parameters() 收集，优化器将拿不到参数。
        # TODO: 调用父类 nn.Module 的 __init__ 方法（提示：super()）
        ...
        # nn.Linear(in_features=1, out_features=1) 表示输入 1 个特征、输出 1 个值的线性层，
        # 内部计算 y = x @ weight.T + bias。
        # 第一关手写的 w/b 就藏在 nn.Linear 的 weight/bias 里：
        #   self.linear.weight 对应 w，形状 [1, 1]（out_features 行、in_features 列）
        #   weight 之所以存成 [out_features, in_features]：每一行 weight[i] 正好是
        #   第 i 个输出神经元连接所有输入的完整权重，想看某个输出神经元学到了什么，按行取即可；
        #   且 PyTorch 张量按行主序（row-major）存储，一行在内存中连续，
        #   初始化、逐神经元分析等按输出神经元进行的操作访问的都是连续内存；
        #   计算式里的 weight.T 只是视图、不复制数据，这种存法不损失性能。
        #   self.linear.bias   对应 b，形状 [1]
        # 它们创建时自动带 requires_grad=True，无需再手动指定。
        # TODO: 创建输入 1 个特征、输出 1 个值的线性层，赋值给 self.linear
        ...

    # forward 中定义前向传播的计算过程（数据怎么流过各个零件）。
    # 第一关的 y_batch_pred = w * x_batch_train + b 就对应这里的一行调用。
    # 调用模型时写 model(x) 而不是 model.forward(x)，
    # nn.Module 的 __call__ 会在调用 forward 前后做钩子处理。
    def forward(self, x):
        # 输入 x 形状 [N, 1]，输出形状 [N, 1]
        # 隐式广播：bias 形状 [1]，与 x @ weight.T（形状 [N, 1]）相加时
        # 会被广播成 [N, 1]，再逐元素相加
        # TODO: 把输入 x 传入 self.linear 并返回结果
        ...

# 创建模型实例
# TODO: 实例化 LinearRegressionModel，赋值给 model
model = ...

# nn.Sequential 等价写法作对照：按顺序把层串起来，无需自定义类。
# 两种方式的适用场景：
#   nn.Sequential：结构就是"一条直线"（层与层首尾相接）时最简洁，
#                  适合快速搭建简单模型；
#   nn.Module 子类：forward 可写任意 Python 逻辑（分支、多输入、跳跃连接等），
#                   适合复杂结构，是工业界的主流写法。
# 本关实际训练用上面的 nn.Module 子类版本，这里仅创建作对照展示。
# TODO: 用 nn.Sequential 包一个与上面等价的线性层，赋值给 model_sequential
model_sequential = ...

# 把模型迁移到计算设备（内部会把 weight、bias 一起迁移过去）。
# 与第一关的对应关系：第一关强调参数必须"创建时"就指定 device=device，
# 因为对普通张量调用 .to(device) 返回的是非叶子张量，梯度无法积累；
# 而 nn.Module 的 .to(device) 是特殊实现——它原地迁移登记过的所有参数，
# 迁移后参数依然是叶子张量，所以模型可以先在 CPU 上创建、再安全地搬家。
# TODO: 把 model 迁移到 device
...

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

# 损失函数：均方误差（MSE），训练时衡量预测与真实值的差距。
# 第一关手写的 mse_loss 函数（((y_pred - y_true) ** 2).mean()），
# 本关替换为框架组件 nn.MSELoss()：
# 它是一个可调用对象，用法与手写函数一致——loss_fn(y_pred, y_true)，
# 默认 reduction="mean"，口径与第一关的 .mean() 完全相同。
# 返回的 loss 为标量，形状 []
# TODO: 实例化 MSE 损失函数对象，赋值给 loss_fn
loss_fn = ...

# 优化器：随机梯度下降（SGD）。
# 第一关在 no_grad 中手动执行 w -= learning_rate * w.grad，
# 本关替换为 optim.SGD：把"更新哪些参数、步长多大"提前登记好，
# 训练时只需调用 optimizer.step()，它会在内部（自动关闭梯度追踪）
# 对登记过的每个参数执行同样的"参数 -= lr * 梯度"。
# model.parameters() 返回模型登记的全部可学习参数（本例即 weight 和 bias）。
# TODO: 创建 SGD 优化器，登记模型的全部参数，学习率用 learning_rate
optimizer = ...

# loss 是给优化器"看"的：要求可导、对梯度友好；
# 评估指标是给人"看"的：要求直观可解释。两者职责不同。
# 本例中 MSE 既是 loss 也可以当评估指标，但工程上必须习惯把二者分开。

# 评估指标：第一关手写了 mae_metric、r2_score 函数，
# 并用"列表追加 + torch.cat 统一计算"在 epoch 末聚合整个训练集的指标；
# 本关替换为 torchmetrics 的指标对象，其 update/compute 模式
# 与第一关的聚合逻辑同构：
#   metric.update(pred, true) ≈ 列表 append（逐 batch 累计到内部状态）
#   metric.compute()          ≈ torch.cat 后统一计算整个数据集的指标
#   metric.reset()            ≈ 清空列表，供下一个 epoch 重新累计
# 指标对象内部维护累计状态张量，必须 .to(device) 与数据放在同一设备；
# 每个数据集用独立的指标对象，避免训练/验证/测试的累计状态互相污染。

# 训练集指标对象：MAE（平均绝对误差）与 R²（决定系数）
# TODO: 实例化 MAE 和 R² 指标对象并迁移到 device，
#       分别赋值给 mae_metric_train、r2_metric_train
mae_metric_train = ...
r2_metric_train = ...

# 验证集指标对象
# TODO: 同上，分别赋值给 mae_metric_valid、r2_metric_valid
mae_metric_valid = ...
r2_metric_valid = ...

# 测试集指标对象
# TODO: 同上，分别赋值给 mae_metric_test、r2_metric_test
mae_metric_test = ...
r2_metric_test = ...

# ============================================================
# Step 6：训练
# ============================================================

for epoch in range(epochs):

    # Step 6.1：切换训练模式
    # train()/eval() 切换的是 Dropout、BatchNorm 这类"训练与推理行为不同的层"的模式，
    # 本关只有 nn.Linear，两种模式行为无差别，但这是标准工程习惯，
    # 与验证/推理时的 eval() 成对出现。
    # 上一个 epoch 验证前切到了 eval()，每轮开头切回 train() 才能正确训练。
    # TODO: 把 model 切换到训练模式
    ...

    # Step 6.2：初始化 epoch 级累计量
    # 工业界标准做法：每个 epoch 开始时先把累计量清零，
    # 每个 batch 结束后累计 loss 与样本数、update 指标状态，
    # epoch 末尾统一计算整个训练集的 loss 和各指标。
    # 第一关这里还要初始化 y_pred_tensor_list / y_true_tensor_list 两个列表，
    # 本关列表聚合已由 torchmetrics 指标对象的内部状态替代，无需再建列表。
    # loss_train_sum 为整个训练集的 loss 累计量
    loss_train_sum = 0.0
    # n_train_samples 为本 epoch 已处理的训练样本总数
    n_train_samples = 0
    # 整个训练集的各指标变量，先置初值，epoch 末尾统一计算
    loss_train = 0.0
    mae_train = 0.0
    r2_train = 0.0

    # Step 6.3：数据迁移到计算设备
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

    # Step 6.4：梯度清零
    # PyTorch 默认会累积梯度（而不是覆盖），
    # 这个设计是为了支持某些需要多次 backward 累加梯度的场景（如 RNN、梯度累积训练）。
    # 但在常规训练中，每个 batch/epoch 都是独立的，
    # 如果不清零，上一轮的梯度会叠加到这一轮，导致参数更新方向错误。
    # 放在每轮开头："先清空旧的，再算新的"，逻辑更清晰，也避免遗漏残留梯度。
    # 第一关手写 if w.grad is not None: w.grad.zero_()，还要逐个参数处理；
    # 本关替换为优化器的清零方法：
    # 一行清零优化器登记的全部参数梯度，首轮梯度不存在时也无需判空。
    # TODO: 调用 optimizer 的梯度清零方法
    ...

    # Step 6.5：前向传播
    # 第一关手写 y_batch_pred = w * x_batch_train + b；
    # 本关直接调用模型：model(x_batch_train) 会触发 forward，
    # 内部由 nn.Linear 完成同样的线性计算。
    # 只有训练集参与参数更新，验证集/测试集不进入训练
    # y_batch_pred 形状 [70, 1]
    # TODO: 调用 model 对 x_batch_train 做前向计算，赋值给 y_batch_pred
    y_batch_pred = ...

    # Step 6.6：计算损失
    # 调用超参数设置环节定义好的 nn.MSELoss 实例，在训练集上计算，
    # 用法与第一关调用手写 mse_loss 函数完全一致。
    # 训练环节只负责调用，损失的定义在训练前就已确定，职责分离
    # loss_batch 为标量，形状 []
    # TODO: 调用 loss_fn，传入 y_batch_pred 和 y_batch_train
    loss_batch = ...

    # Step 6.7：反向传播
    # .backward() 从 loss_batch 出发，沿着计算图反向传播，
    # 自动计算 loss 对每一个 requires_grad=True 的张量的偏导数，
    # 结果存储在各张量的 .grad 属性中。
    # 这一步只算梯度，不更新参数。
    loss_batch.backward()

    # Step 6.8：参数更新
    # 第一关手写：with torch.no_grad(): w -= learning_rate * w.grad（b 同理）；
    # 本关替换为优化器的更新方法：
    # 对登记过的每个参数执行同样的"参数 -= lr * 梯度"，
    # 内部自动关闭梯度追踪，无需再手写 no_grad 上下文。
    # TODO: 调用 optimizer 的参数更新方法
    ...

    # Step 6.9：累计 epoch 级统计量（参数更新后、当前 batch 结束时）
    # nn.MSELoss 默认 mean 口径：loss_batch.item() 是当前 batch 的平均 loss，
    # 乘以 batch 样本数还原成该 batch 的 loss 总和后再累计，
    # epoch 末尾除以总样本数，即得到按样本数加权平均的整个训练集 loss，
    # 这样即使各 batch 大小不同（如最后一个 batch 不满），结果依然正确。
    # batch_size 为当前 batch 的样本数（本例全量数据即一个 batch，值为 70）
    batch_size = x_batch_train.shape[0]
    loss_train_sum += loss_batch.item() * batch_size
    n_train_samples += batch_size

    # 第一关这里把预测值和真实值追加进列表，供 epoch 末尾 torch.cat 统一计算；
    # 本关替换为 metric.update()：逐 batch 把预测值和真实值累计进指标内部状态。
    # detach 的作用：返回一个与计算图断开的新张量，避免指标状态持有计算图，
    # 否则各 batch 的计算图会被一直保留，白白占用内存
    # TODO: 对 mae_metric_train 和 r2_metric_train 分别调用 update，
    #       传入 y_batch_pred.detach() 和 y_batch_train
    ...
    ...

    # Step 6.10：计算整个训练集的 loss 和各指标（所有 batch 处理完后）
    # loss_train = loss 总和 / 总样本数，与 nn.MSELoss 的 mean 口径一致
    loss_train = loss_train_sum / n_train_samples

    # loss 用于优化器梯度更新（给机器看），MAE/R² 用于人工判断模型效果（给人看）。
    # 评估不参与梯度计算，需用 no_grad 避免计算图开销。
    with torch.no_grad():
        # 第一关这里 torch.cat 拼接列表后调用手写指标函数；
        # 本关替换为 metric.compute()：基于 update 累计的内部状态，
        # 一次性算出整个训练集的指标。
        # MAE：平均偏差，单位与 y 相同
        # mae_train 为标量，形状 []
        # TODO: 调用 mae_metric_train 的 compute，赋值给 mae_train
        mae_train = ...

        # R²：越接近 1，模型解释能力越强
        # r2_train 为标量，形状 []
        # TODO: 调用 r2_metric_train 的 compute，赋值给 r2_train
        r2_train = ...

        # metric.reset()：清空内部累计状态（对应第一关"下一个 epoch 重建空列表"），
        # 否则下一个 epoch 的 update 会叠加在本 epoch 的状态上，指标被污染。
        # TODO: 对 mae_metric_train 和 r2_metric_train 分别调用 reset
        ...
        ...

    # Step 6.11：验证集评估与训练状态打印
    # 每个 epoch 都打印训练状态，观察收敛过程
    # 在验证集上算指标：验证集没参与参数更新，能反映模型对"没见过的数据"的泛化能力。
    # 验证前先切换到评估模式。eval() 与 no_grad 是两件独立的事：
    # eval() 切换层的行为模式，no_grad 关闭梯度计算，
    # 工业界验证/推理时两者同时使用。
    # TODO: 把 model 切换到评估模式
    ...

    # 评估不参与梯度计算，需用 no_grad 避免计算图开销。
    with torch.no_grad():
        # 验证集在评估前才迁移到计算设备，与训练集"使用前迁移"的模式一致
        # 重复调用 .to(device) 同样是无害的空操作（no-op）
        # x_valid、y_valid 形状均 [20, 1]
        x_valid = x_valid.to(device)
        y_valid = y_valid.to(device)

        # 用当前模型在验证集上做前向计算（对应第一关的 w * x_valid + b）
        # y_valid_pred 形状 [20, 1]
        # TODO: 调用 model 对 x_valid 做前向计算，赋值给 y_valid_pred
        y_valid_pred = ...

        # 验证集 loss：与训练 loss 同口径，便于对比训练/验证差距、监控过拟合
        # loss_valid 为标量，形状 []
        # TODO: 调用 loss_fn，传入 y_valid_pred 和 y_valid
        loss_valid = ...

        # 验证集指标同样走 update → compute → reset 三步：
        # 验证集一次性全量评估，update 只调用一次，但保持与训练集相同的模式，
        # 之后引入 DataLoader 分 batch 验证时，这套写法不需要任何改动。
        # MAE：平均偏差，单位与 y 相同
        # mae_valid 为标量，形状 []
        # TODO: 对 mae_metric_valid 依次执行 update（传入 y_valid_pred 和 y_valid）、
        #       compute（结果赋值给 mae_valid）、reset
        ...
        mae_valid = ...
        ...

        # R²：越接近 1，模型解释能力越强
        # r2_valid 为标量，形状 []
        # TODO: 对 r2_metric_valid 依次执行 update（传入 y_valid_pred 和 y_valid）、
        #       compute（结果赋值给 r2_valid）、reset
        ...
        r2_valid = ...
        ...

    # 一行内按数据集分组输出：先训练集、后验证集，最后是当前参数
    # 第一关直接打印 w.item() / b.item()；本关参数在模型内部，
    # 用 model.linear.weight.item()、model.linear.bias.item() 取出
    # （weight 形状 [1, 1]、bias 形状 [1]，都只含一个元素，可直接 .item()）
    print(f"Epoch {epoch + 1:3d} | Train Loss: {loss_train:.4f} MAE: {mae_train.item():.4f} R²: {r2_train.item():.4f} | Valid Loss: {loss_valid.item():.4f} MAE: {mae_valid.item():.4f} R²: {r2_valid.item():.4f} | w: {model.linear.weight.item():.4f} | b: {model.linear.bias.item():.4f}")

# Step 6.12：训练结果展示
# 学到的参数与真实参数对比，确认训练是否学到了接近真实的规律，
# 属于训练环节的收尾
print("\n" + "=" * 40)
print("训练完成！")
print(f"学习到的参数：w = {model.linear.weight.item():.4f}, b = {model.linear.bias.item():.4f}")
print(f"真实参数：    w = 2.0000, b = 3.0000")
print("=" * 40)

# ============================================================
# Step 7：推理
# ============================================================

# model.eval() 把模型切换到评估模式：
# 它会改变 Dropout（评估时不再随机丢弃）、BatchNorm（评估时用全局统计量）
# 等"训练/评估行为不同"的层的行为。
# 本关模型只有一个 nn.Linear，训练和评估行为完全一样，
# 所以调用与否看不出任何差别；但"推理前先 eval()"是必须养成的工程习惯，
# 后续关卡引入 Dropout/BatchNorm 后，漏掉这一行会直接导致评估结果错误。
# 注意：eval() 只切换层的行为模式，不关梯度；关梯度仍需下面的 no_grad。
# TODO: 把 model 切换到评估模式
...

# 测试集最终评估：
# 测试集在训练和调参过程中从未被使用，
# 此时评估一次，得到的才是模型泛化能力的无偏估计。
# 评估不需要梯度，用 torch.no_grad() 包裹，省去计算图的构建开销。

with torch.no_grad():
    # 与训练时同样的惯例：测试集张量在使用前才迁移到计算设备
    # x_test、y_test 形状均 [10, 1]
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    # 用训练好的模型对测试集做前向计算（对应第一关的 w * x_test + b）
    # y_test_pred 形状 [10, 1]
    # TODO: 调用 model 对 x_test 做前向计算，赋值给 y_test_pred
    y_test_pred = ...

    # 复用 Step 5 定义的 loss_fn 计算测试集 loss：
    # 训练时看的 loss，推理/最终评估时也可以直接复用，口径完全一致。
    # 工程上 loss 和评估指标要分开计算与打印，
    # 因为在其他任务中两者往往不同（如分类任务 loss 是交叉熵、指标是准确率）。
    # loss_test 为标量，形状 []
    # TODO: 调用 loss_fn，传入 y_test_pred 和 y_test
    loss_test = ...

    # 测试集指标用 torchmetrics 计算，模式与训练/验证一致：update → compute。
    # 测试集只评估这一次，之后不再复用，可以省去 reset。
    # MAE：平均偏差，单位与 y 相同，更直观
    # mae_test 为标量，形状 []
    # TODO: 对 mae_metric_test 依次执行 update（传入 y_test_pred 和 y_test）、
    #       compute（结果赋值给 mae_test）
    ...
    mae_test = ...

    # R²：越接近 1，说明模型对数据变异的解释能力越强
    # r2_test 为标量，形状 []
    # TODO: 对 r2_metric_test 依次执行 update（传入 y_test_pred 和 y_test）、
    #       compute（结果赋值给 r2_test）
    ...
    r2_test = ...

# 打印最终评估结果：loss 放最前面，评估指标跟在后面
print(f"\n测试集评估 | Loss: {loss_test.item():.4f} | MAE: {mae_test.item():.4f} | R²: {r2_test.item():.4f}")
