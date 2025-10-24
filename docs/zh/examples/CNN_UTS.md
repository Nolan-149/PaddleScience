# Predicting the Strength of Composites

=== "模型训练命令"

    ``` sh
    python main.py mode=train
    ```

=== "模型评估命令"

    ``` sh
    python main.py mode=eval
    ```

## 下载预训练模型

| [resnet18-v5-fold1](https://paddle-org.bj.bcebos.com/paddlescience/models/CNN_UTS/resnet18-v5-fold1.pdparams) |
 [resnet18-v5-fold2](https://paddle-org.bj.bcebos.com/paddlescience/models/CNN_UTS/resnet18-v5-fold1.pdparams) |
 [resnet18-v5-fold3](https://paddle-org.bj.bcebos.com/paddlescience/models/CNN_UTS/resnet18-v5-fold1.pdparams) |
 [resnet18-v5-fold4](https://paddle-org.bj.bcebos.com/paddlescience/models/CNN_UTS/resnet18-v5-fold1.pdparams) |
 [resnet18-v5-fold5](https://paddle-org.bj.bcebos.com/paddlescience/models/CNN_UTS/resnet18-v5-fold1.pdparams) ||

## 下载模型必要参数

| [Saved_Output](https://paddle-org.bj.bcebos.com/paddlescience/models/CNN_UTS/Saved_Output.tar.gz) |

## 背景简介

材料的极限抗拉强度（UTS）是衡量复合材料抗拉伸破坏的核心指标，直接决定其应用安全性与可靠性。它是结构设计的关键依据，确保构件在拉伸载荷下不失效；也是材料选型的重要标准，匹配不同场景的强度需求，最终保障复合材料制品的性能上限。但由于复杂的形态-性能关系，预测其机械性能仍然较为困难，使用传统机器学习方法很难对其做出有效的预测。

针对材料科学领域中材料结构强度预测这一问题，通过X射线CT图像预测聚合物-陶瓷复合材料的极限抗拉强度（UTS）。相较于传统材料强度预测方法对于数据和模型的需求严苛，且需要耗费较长的时间成本，本项目通过深度学习技术，在小样本数据集的条件下，实现了较高精度的UTS值预测，提供了更快速且准确的工具。帮助研究人员快速了解材料的特性，并优化材料设计

本研究中使用卷积神经网络（CNN） 来分析冷烧结聚合物-陶瓷复合材料的 X 射线计算机断层扫描 （CT） 图像来应对这一问题。以形态特征作为输入的传统机器学习模型产生的准确性有限，而使用预训练的卷积神经网络，并使用集成学习进一步优化了模型。使用小型数据集来揭示复合材料中形态-结构-性能关系的替代机器学习方法，为衡量复合材料的性能提供了更精确且高效的解决方案。

## 目录结构

```
CNN_UTS/
│
├─ conf/  
│    └─ resnet.yaml
├─ data_utils.py  
├─ model_utils.py  
├─ main.py  
├─ requirements.txt  
├─ readme.md  
├─ resnet18-v5-finetune/  
├─ outputs/  
├─ Saved_Output/  
└─ Dataset/  
     ├─ Train_val/  
     └─ Test/  
```

## 2. 模型原理

本章节对基于卷积神经网络的材料拉伸强度预测模型的原理进行介绍。

该方法的主要思想是通过卷积神经网络建立材料微观结构图像与拉伸强度（UTS）之间的非线性映射关系。模型采用ResNet架构，能够有效提取图像中的深层特征信息。

本案例采用ResNet-18作为基础模型架构，主要包括以下几个部分：

1. 输入层：接收 224×224×3 的RGB图像数据
2. 卷积层：多个卷积块，包含残差连接
3. 池化层：最大池化操作，降低特征图尺寸
4. 全连接层：将特征映射到最终的预测值
5. 输出层：输出预测的UTS值（MPa）

通过这种方式，我们可以自动学习材料微观结构图像中的关键特征，建立图像与性能之间的映射关系，实现准确的拉伸强度预测。

## 3. 模型实现

本章节我们讲解如何基于 PaddleScience 代码实现材料拉伸强度预测模型。本案例使用5折交叉验证进行模型训练和评估，并使用 PaddleScience 内置的各种功能模块。

### 3.1 数据格式说明

数据集下载链接:<https://paddle-org.bj.bcebos.com/paddlescience/datasets/CNN_UTS/Dataset.zip>

本案例使用的数据集包含材料微观结构图像和对应的拉伸强度标签。数据集分为以下几个部分：

1. **训练集**：`Dataset/Train_val/` - 包含2600张图像和对应的CSV标签文件
2. **测试集**：`Dataset/Test/` - 包含500张图像和对应的CSV标签文件

**数据集特点**：
- 每个样本包含RGB图像和对应的UTS标签
- 图像经过预处理，统一调整为224×224尺寸
- 使用ImageNet预训练权重的标准化参数进行归一化
- UTS值范围：0.46 - 3.9 MPa，平均值为1.924 MPa，标准差为1.154 MPa

**数据格式示例**：
```
Dataset/
├── Train_val/
│   ├── IPP_10__40060.jpg
│   ├── IPP_15__10_1.19_1.057_1.1697.jpg
│   └── samples.csv
└── Test/
    ├── IPP_15__10_1.19_1.057_1.1697/
    │   ├── *.jpg (100张图像)
    │   └── *.csv (标签文件)
    └── ...
```

为了方便数据处理，我们使用了 `make_dataset` 函数来创建数据集：

``` py linenums="77" title="examples/CNN_UTS/main.py"
--8<--
examples/CNN_UTS/main.py:77:78
--8<--
```

### 3.2 模型构建

本案例使用 PaddlePaddle 内置的 `paddle.vision.models.resnet18` 构建ResNet-18模型。模型的主要参数包括：

1. 网络结构：ResNet-18 (2,2,2,2)
2. 输入通道：3（RGB图像）
3. 输出维度：1（UTS预测值）
4. 预训练权重：ImageNet

模型定义代码如下：

``` py linenums="116" title="examples/CNN_UTS/main.py"
--8<--
examples/CNN_UTS/main.py:116:119
--8<--
```

### 3.3 数据增强

为了提高模型的泛化能力，我们实现了多种数据增强策略：

1. 随机水平翻转
2. 随机垂直翻转
3. 中心裁剪到224×224
4. 标准化处理

数据增强配置如下：

``` py linenums="58" title="examples/CNN_UTS/main.py"
--8<--
examples/CNN_UTS/main.py:58:74
--8<--
```

### 3.4 训练策略

本案例采用5折交叉验证策略进行模型训练，具体流程如下：

#### 3.4.1 5折交叉验证流程

1. **数据分割**：将训练数据按样本ID进行分层分组，确保同一样本的所有图像在同一fold中
2. **模型训练**：每个fold训练一个独立的ResNet-18模型
3. **模型保存**：保存每个fold的最佳模型权重
4. **集成预测**：使用所有fold的预测结果进行集成

#### 3.4.2 预训练模型权重的使用

**ImageNet预训练权重的加载**：
```python
model = paddle.vision.models.resnet18(pretrained=True)
```
- 在模型初始化时自动加载ImageNet预训练权重
- 这些权重提供了强大的特征提取能力
- 适用于图像分类任务，为UTS回归任务提供良好的初始化

**模型适配**：
```python
model.fc = paddle.nn.Linear(model.fc.weight.shape[0], 1)
```
- 将最后的全连接层从1000个输出（ImageNet类别数）改为1个输出（UTS回归值）
- 保持预训练的特征提取层不变
- 只训练新添加的回归层

#### 3.4.3 训练过程

每个fold的训练过程包括：

1. **数据加载**：使用StratifiedGroupKFold进行数据分割
2. **模型初始化**：加载ImageNet预训练权重并适配回归任务
3. **训练循环**：
   - 前向传播：图像 → ResNet-18 → UTS预测值
   - 损失计算：MSE损失
   - 反向传播：Adam优化器更新参数
   - 验证评估：在验证集上评估性能
4. **模型保存**：保存验证损失最低的模型权重

#### 3.4.4 预训练模型权重的使用步骤

**训练阶段**：
1. **模型初始化**：每个fold开始时，使用 `paddle.vision.models.resnet18(pretrained=True)` 加载ImageNet预训练权重
2. **模型适配**：修改最后一层为回归层 `model.fc = paddle.nn.Linear(model.fc.weight.shape[0], 1)`
3. **训练过程**：在训练过程中，预训练的特征提取层会进行微调，新的回归层从头开始训练
4. **模型保存**：每个fold训练完成后，保存最佳模型权重到 `resnet18-v5-fold{fold_number}.pdparams`

**评估阶段**：
1. **模型加载**：使用 `paddle.load()` 加载对应fold的预训练权重
2. **模型恢复**：将权重加载到相同结构的模型中
3. **推理预测**：使用加载的模型进行预测

**预训练权重文件说明**：
- `resnet18-v5-fold1.pdparams`：第1折的最佳模型权重
- `resnet18-v5-fold2.pdparams`：第2折的最佳模型权重
- `resnet18-v5-fold3.pdparams`：第3折的最佳模型权重
- `resnet18-v5-fold4.pdparams`：第4折的最佳模型权重
- `resnet18-v5-fold5.pdparams`：第5折的最佳模型权重

**训练配置**：
- 训练轮数：可配置（默认值）
- 批次大小：可配置（默认32）
- 学习率：可配置（默认值）
- 优化器：Adam
- 损失函数：MSE Loss

**训练监控**：
- 每个epoch记录训练损失、验证损失和测试损失
- 当验证损失达到新低时，自动保存模型权重
- 支持GPU内存不足时的自动处理机制

``` py linenums="89" title="examples/CNN_UTS/main.py"
--8<--
examples/CNN_UTS/main.py:89:102
--8<--
```

### 3.5 损失函数和优化器

使用均方误差损失函数进行回归任务：

``` py linenums="120" title="examples/CNN_UTS/main.py"
--8<--
examples/CNN_UTS/main.py:120:120
--8<--
```

使用Adam优化器进行参数更新：

``` py linenums="121" title="examples/CNN_UTS/main.py"
--8<--
examples/CNN_UTS/main.py:121:123
--8<--
```

### 3.6 模型评估

评估过程包括：

1. 计算MSE和R²指标
2. 生成parity plot和violin plot
3. 进行集成预测

评估器构建代码如下：

``` py linenums="162" title="examples/CNN_UTS/main.py"
--8<--
examples/CNN_UTS/main.py:162:194
--8<--
```

## 4. 训练结果与性能指标

### 4.1 数据集统计

- **训练集样本数**：2600张图像
- **测试集样本数**：500张图像
- **UTS值范围**：0.46 - 3.9 MPa
- **UTS平均值**：1.924 MPa
- **UTS标准差**：1.154 MPa

### 4.2 单模型性能（5折交叉验证）

| Fold | MSE | R² | 说明 |
|------|-----|----|----|
| Fold 1 | 0.2202 | 0.8347 | 良好性能 |
| Fold 2 | 0.1275 | 0.9043 | 优秀性能 |
| Fold 3 | 0.1209 | 0.9092 | **最佳性能** |
| Fold 4 | 0.2716 | 0.7961 | 相对较低 |
| Fold 5 | 0.2154 | 0.8383 | 良好性能 |

**统计结果**：
- **平均MSE**：0.1911 ± 0.0581
- **平均R²**：0.8565 ± 0.0436
- **最佳R²**：0.9092 (Fold 3)
- **最差R²**：0.7961 (Fold 4)

### 4.3 集成学习性能

| 方法 | MSE | R² | 说明 |
|------|-----|----|----|
| **均值集成** | 0.1052 | 0.9210 | 推荐使用 |
| **中位数集成** | 0.0928 | 0.9303 | **最佳性能** |
| **单模型平均** | 0.1911 | 0.8565 | 基准对比 |

**集成学习效果分析**：
- 中位数集成相比单模型平均提升了 **7.4%** 的R²
- 均值集成相比单模型平均提升了 **6.4%** 的R²
- 集成学习显著提高了模型的稳定性和预测精度

## 5. 完整代码

``` py linenums="1" title="examples/CNN_UTS/main.py"
--8<--
examples/CNN_UTS/main.py
--8<--
```

## 参考文献

- [Predicting the Strength of Composites with Computer Vision Using Small Experimental Datasets](<https://pubs.acs.org/doi/10.1021/acsmaterialslett.4c02424>)
