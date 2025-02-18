---
layout: post
title: deep-learning
date: 2024-09-11
---

[cs231n](https://cs231n.stanford.edu/schedule.html)

[toc]

深度学习关系

机器学习概念：一般性规律，传统学习：浅层

语义鸿沟：核心问题

反向传播：概念

Lenet:贡献(反向传播引入神经网络)

imagenet: 1000 百万张图片

alexNet : 2012 端到端：gpu relu dropout

深度学习关键点：数据，算法，算力

不同任务的共性



图像分类：地位；语义鸿沟：来源（人和计算机看到的不同）；挑战；分类器不能硬编码：会过拟合

基于数据驱动的数据集：代码不考



k-nn: 

度量函数

时间复杂度

1近邻存在的问题：离群点不能很好的分类，对于噪声比较敏感

问题：k个点属于k个类

线性分类器

形状：输入输出

代数视角理解：线性代数的乘法和加法

视觉：学习模板，对输入进行匹配

几何：分类超平面

难以处理的情况：映射到高维空间（非线性的变换）

记损失函数进行计算

问题：car降低0.5, 最小值和最大值，...

softmax公式



正则化：偏好简单的模型

正则化项：需要除以N求平均

L1和L2正则化的偏好

模型优化

梯度下降流程：数值法（极限）方法太慢了，解析法，之间的关系（优点和缺点），数值法检查解析法的梯度

SGD: 流程，问题 动量：原因；梯度变化过大，学习率难以调整，对梯度进行归一化

AdaGrad & MRSprop

学习率曲线图

一阶二阶优化的概念，时间复杂度



全连接层的特点：参数量大,没有空间信息

激活函数：4

二层三层：ppt

计算图：画和计算

梯度流的计算公式：门 copy_gate : 梯度相加

向量矩阵的形状：

dL/dx的形状总是与x相同



CNN

滤波器学到的东西：浅层（局部特征：边缘，颜色等纹理） 深层（更高层次的特征，形状，模式）

输出计算：

感受野计算：不同步长（前一层还是输入图像）前一层就是卷积核大小

参数量公式

池化层：独立操作（每个通道单独处理），没有参数

1*1：增加

转置卷积，空洞卷积，分组卷积（提高计算效率）... 概念

带计算器



BN层

输入不以0为中心，进行缩放（学习可学习的缩放参数）

找找资料

BN层的好处，与其他的norm有什么区别

经典神经网络

alexnet：使用relu, drop_out

VGG：更小的滤波器，更深的网络 $3*3卷积,2*2pooling$, 16,19, 为什么使用3*3？非线性，感受野，参数量

显存和参数量计算：浅层卷积和参数量

googlenet: inception 模块：并联算子（问题？使用1*1卷积）

resnet：ilsvrc15 冠军，解决问题，残差模块，bottleneck模块（增加非线性）

网络总结

seNet: 通道的注意力（对每个通道进行缩放）

迁移学习：原因



训练神经网络

感受野：当前输入 & 网络输入 FC层的感受野

输出向下取整

激活函数的问题

输出以0为中心的好处：收敛更快

预处理：常用方法：每个像素逐通道计算均值除以标准差

drop-out：作用：学习冗余表示，测试时想求期望，输出乘drop_out_rate

超参数选择：步骤（检查初始损失，过拟合少量数据，选择合适的lr, 粗调1-5,精调10-20个）



Rnn

公式，特性：时间步共享参数，优点和缺点

梯度计算

lstm

公式+计算图

梯度消失问题解决了吗？



transformer

注意力的类型

CNN,RNN，全连接...的特点

self-attention: 计算过程（图和公式）

multi-head:怎么分成不同的head  channel/h 分组  res+norm -> feed_forward:mlp

transformer: 图（k,v来自encoder）

VIT: norm层



目标检测和图像分割

语义分割：滑窗计算的问题：计算低效等

全卷积：问题，解决方案：下采样...

Unet: 大概做法（下采样，上采样）

encoder-only: 没有上采样，模型属于哪个类别

目标检测：region Proposal: selective search: cpu, 2000多个区域在几秒内

r-cnn:流程（调成相同大小，可以利用batch进行处理），问题：效率低，计算慢 f-cnn相似

fast r-cnn：卷积整张图像 rol pooling不考，问题：region 生成慢

faster r-cnn：rpn预测proposal

region proposal：概念，是否有物体，如果有调整锚框大小

faster r-cnn：4个损失，两阶段检测器

FPN了解

实例分割：mask R-cnn : fast r-cnn加了mask head



视频理解

问题：任务流程

late fusion: 大致做法

early fusion: 大概做法

3D-conv: 对比不需要看，特性

2d-conv: 特性

双流网络：motion,appearance

recurrent conv network: 概念

I3D: 替换原则：使得2d和3d的预测结果相同，重用2d-cnn来处理3d任务



可视化：不考

gram 矩阵：用法



自监督

定义：pretext downstream task

评估：

自监督方式：旋转角度预测，图像快位置，图像块拼图，像素，重建，着色

pretext_task:做法，关心的问题

对比学习的形式化定义



多模态

预训练方式：

全监督：做法，缺点

语言图像对比学习

clip做法，形式

通用视觉模型的尝试

视觉语言的gap,粒度，饼图

图像分割：分割任务

任务粒度：图像，区域，像素，举一些例子



## 绪论

<font size = 5 >深度学习的关系</font>

![image-20241229220531934](https://s2.loli.net/2024/12/29/1jhzo5taQrKNUbD.png)

机器学习

- 从有限的观测数据中学习出具有一般性的规律，并利用这些规律对未知数据进行预测

- 浅层学习（传统机器学习）关注学习预测模型，不涉及特征学习，其特征主要靠特征工程来提取

  ![image-20241229220754677](https://s2.loli.net/2024/12/29/RSWeYVX35NboUfc.png)

表示学习

- 自动学习出有效的特征
- 关键问题是解决语义鸿沟
- **语义鸿沟**：输入数据的底层特征和高层语义信息之间的不一致性和差异性
- 核心问题：什么是好的表示，如何学习到好的表示

好的表示：

- 具有很强的表示能力，
- 需要包含更高层的语义信息
- 具有一般性

学习好的表示

- 构建具有一定深度的多层次特征表示

深度学习

- 特征学习+预测模型
- 关键问题是**贡献度分配问题**
- 关键点是：数据，算法，算力



### 不同任务间的共性

- 输入和输出之间具有非常复杂的关系
- 输入输出遵循一定的规则



Lenet:贡献(反向传播引入神经网络)

imagenet: 1000 百万张图片

alexNet : 2012 端到端：gpu relu dropout

深度学习关键点：数据，算法，算力

不同任务的共性



## 数据集和模型

- 视频数据集：Sports-1M(100万个487中不同运动标签的视频)
- imagenet：1000个类别，百万张图片
- Lenet: 引入反向传播算法
- alexnet： GPU并行训练，relu,dropout, 数据增强



## 图像分类

- 挑战
  - 视角差异: 视角改变，所有像素都会变化
  - 杂乱背景：
  - 语义鸿沟：图像与$H*W*3$的矩阵是有一定差距的
  - 光照变化
  - 形变，遮挡
  - 类内差异
  - 类间相似
  - 语义干扰

图像分类器无法对图像识别算法进行**硬编码**

### **k-nn**: 

选出k个邻居，进行投票

```python
def predice(self,X):
    num_test = X.shape[0]
    Ypred = np.zeros(num_test,dtype=self.ytr.dtype)
    for i in range(num_test):
        distances = np.sum(np.abs(self.Xtr-X[i,:]),axis=1) #L1距离度量
        #distances = np.sqrt(np.sum((self.Xtr-X[i,:])**2,axis=1)) #L2距离度量
        min_index = np.argmin(distances)
        Ypred[i]=self.ytr[min_index]
        
    return Ypred
```

时间复杂度：训练$O(1)$,预测$O(M*N*D+M*Nlogk)$

问题：超参数（或模型）的选择，距离度量$L_1,L_2$

- 1近邻有哪些问题：有可能过拟合，对于噪声比较敏感
- k个邻居属于k个类：加权投票，距离较近的权重更高；增加k值；选择频率较高的类别

- 在验证集上选择超参数

- 交叉验证：训练集拆分成多个部分，分别对每个部分作为验证，将结果取平均，整个过程中测试集不参与训练和验证。

  ![image-20241219221313969](https://s2.loli.net/2024/12/19/MX7FqmGr9lNPiWo.png)

- 不要直接使用像素间的距离进行度量（泛化性与稳定性较差）

### 线性分类器
$$
\text{f}^{3 \times 1}(x,W) = W^{3 \times 3072}x^{3072 \times 1}
$$

x是$32*32*3$的矩阵

![image-20241219223511433](https://s2.loli.net/2024/12/19/h6CBLFsUj8IZaJr.png)

![image-20241219223826119](https://s2.loli.net/2024/12/19/YA6aO8VCRUbitgz.png)

低维不可分，高维可分，映射到高维空间，仍然是可分的。

优化找到**最小化损失函数**的参数

**损失函数**：
$$
L = \frac{1}{N}\sum_{i}L_i(f(x_i,y_i),y_i)
$$
**多分类SVM loss**

![image-20241219230433413](https://s2.loli.net/2024/12/19/MRaTWDYpHkf7SlQ.png)

1是松弛条件，可以换成别的。

线性分类器 softmax loss:

**交叉熵损失**
$$
H(P||Q) = H(p)+ D_{\text{KL}}(P||Q)
$$
然而这里的$H(P)$是真实分布本身的不确定性，和预测无关，因此直接使用KL散度

$$
D_{\text{KL}}(P||Q) = \sum_{y}P(y)\text{log}\frac{\text{P}(y)}{\text{Q}(y)}\\=-\text{log}\text{Q}(y_k)\\=-\text{log}(\frac{e^{s_{y_i}}}{\sum_j e^{s_j}})
$$

分类问题中，Q(y)是one-hot分布，与类别相同就等于1,其余情况就为0。因此交叉熵损失如上

## <font color = pink >正则化与优化</font>



<font size = 5 >Occam's Razor :如无必要，勿增实体</font>

### 正则化

<font size =5 font color = red>**正则化的作用**</font>

- 表达对模型参数的偏好
- 使得模型简单，使其适用与测试数据
- 改进模型优化

正则化：偏好简单的模型，降低模型对数据的过拟合，降低训练噪声对模型的影响
$$
L(W) = \frac{1}{N}\sum_{i=1}^{N}L_i(f(x_i,W),y_i)+\lambda R(W)
$$
损失：预测结果应该和训练标签一致

正则化：避免模型过拟合训练 （$\lambda = 正则化强度超参数$）

L1 regularization: 系数矩阵W的L1范数，更加偏好需要稀疏解的应用场景，倾向于让一些权重归零,

L2 regularization: 系数矩阵W的L2范数，更加偏好分散的权重，适合梯度下降，更加偏好较小且均匀的权重值，会使得权重更加平滑

| 特性         | L1 正则化                            | L2 正则化                                  |
| ------------ | ------------------------------------ | ------------------------------------------ |
| **惩罚项**   | ( \sum                               | w_i                                        |
| **权重偏好** | 稀疏解，许多权重为零                 | 平滑解，所有权重接近零                     |
| **特征选择** | 能进行特征选择，自动去除不重要的特征 | 不会自动去除特征，而是平滑化所有特征的权重 |
| **适用场景** | 高维数据，特征选择重要               | 当所有特征都很重要时，或特征之间高度相关   |
| **数学性质** | 产生稀疏的解决方案                   | 产生较平滑的解决方案，避免极端权重值       |

Elastic net(L1+L2): $R(W)=\beta L2+L1$

**其他正则化方法：**

- Dropout：训练过程中对于神经元有dropoutrate概率被隐藏，为了弥补期望带来的变化，需要在训练时对于剩余神经元的输出进行缩放，$\text{rescalerate}=\frac{1}{1-\text{dropoutrate}}$

  或者在训练时不进行缩放，但是在测试时需要进行缩放```layer = layer*(1-dropoutrate)```

- **<font color  = red>Batch normalization</font>：**BN被建议插入在（每个）ReLU激活层前面

  [算法详细解释](https://www.cnblogs.com/shine-lee/p/11989612.html)

  ![image-20241222095611212](https://s2.loli.net/2024/12/22/C13HXETpkw8vit7.png)

  具体做法是对于每个节点的输出进行normalization, 并进行缩放和平移

  ![image-20241222100722499](https://s2.loli.net/2024/12/22/hOIfyTNcmPELdSW.png)
  $$
  y^{(b)}_i = BN(x_i)^{(b)} = \gamma(\frac{x_i^{b}-\mu(x_i)}{\sqrt{\sigma(x_i)^2+\epsilon}}) + \beta
  $$
  $\gamma,\beta$是可以学习的超参数。在推理阶段，使用几次训练中的mini-batch来固定$\mu,\sigma$。

  算法如下

  <img src="https://s2.ax1x.com/2019/12/03/QMlxc8.png" alt="算法步骤" style="zoom: 80%;" />

  

- Stochastic depth: 随机深度。在**训练**阶段，对于每个层或者残差块，使用一个概率p来决定是否跳过它，当层被跳过时，输出直接传递给下一个层，不进行计算和梯度更新。在**推理**时，如果该层有p的概率被跳过，推理时会乘一个放缩因子p，来补偿训练过程中跳过的层。

- fractional pooling：分数阶最大值池化。池化的步长或者窗口是分数值。通过一些策略（如切割、插值等）计算池化结果

### 模型优化

$$
\theta^\star \leftarrow \arg\min_{\theta}\ -\sum_{i} \log p_\theta(y_i \mid x_i)
$$

对于每个维度，沿着该维度梯度的反方向更新模型参数，更新的幅度和梯度的大小有关
$$
\theta^*=\theta + \alpha v=\theta-\alpha\nabla_{\theta} \mathcal{L}(\theta) =
\begin{pmatrix}
\frac{\partial \mathcal{L}(\theta)}{\partial \theta_1} \\
\frac{\partial \mathcal{L}(\theta)}{\partial \theta_2} \\
\vdots \\
\frac{\partial \mathcal{L}(\theta)}{\partial \theta_n}
\end{pmatrix}
$$
数值法 : 计算慢
$$
\lim_{h \rightarrow 0}\frac{f(x+h)-f(x)}{h}
$$
解析梯度：实现复杂

使用小梯度

```python
while True:
    batch = sample_training_data(data,256)
    weight_grad = evaluate_gradient(loss_fun,batch,weights)
    weights += -step_size*weight_grad
```



<font color = red>梯度检查</font>: 解析法计算梯度，数值法检查梯度

<font color = blue>随机梯度下降（Stochastic Gradient Descent）</font>

和批量梯度下降计算整个数据集对于某个参数的平均梯度不同，SGD每次只根据一个样本更新参数

![image-20241222115253378](https://s2.loli.net/2024/12/22/r4Tu82IjAyS1kdV.png)



**问题**

鞍点：附近梯度都很小或为0，使用梯度下降时跳出需要很长时间

局部最优值：梯度为0,通常大型网络的局部最优值和全局最优值比较接近，随着参数量的增加，局部最优值的影响会减小。

- 如果在一个方向上变化的很快，在另一个方向上变化的很慢，会有什么表现？

在w1方向缓慢下降，沿着w2抖动（有点运动的分解的意思），所以当前最陡的方向并不总是最优的方向

- 局部最优怎么办？

  随着参数量的增加，影响会变小，通常局部最优和全局最优比较接近。

- <font color = red>**鞍点**</font>：梯度很小，跳出需要很长时间

  和局部最优的区别：Hessian矩阵特征值有正有负，局部最优特征值全为正或者全为负

  ![image-20241222165747621](https://s2.loli.net/2024/12/22/IcpuQt4aW8DLzhA.png)

<font size =4>**SGD改进**</font>

- 对连续梯度优化的方向进行修正

  SGD+Momentum: 将连续时间步的梯度进行加权平均，再进行参数更新

  ![image-20241222170451680](https://s2.loli.net/2024/12/22/XjtbeYco34OIkwl.png)

- 对梯度大小进行“归一化”

  RMSProp(root mean Squared propagation): 根据历史梯度值的平方和进行梯度缩放，有衰减率。将陡峭的方向优化变慢，平缓的方向优化变快，相当于引入的<font color = oran>**学习率的动态调整**</font>： 更适合深度学习，非凸问题

  ```python
  grad_squared = 0
  while True:
      dx = compute_grad(x)
      grad_squared = decay_rate * grad_squared + (1-decay_rate)*dx*dx
      x = x-(lr*dx/(np.sqrt(grad_squared+1e-7)))
  ```

  AdaGrad: 没上面的高级，平方和没有衰减，更**适合解决凸优化问题**，学习率会快速减小。

<font size = 6;font color = orange>Adam</font>

![image-20241222172905703](https://s2.loli.net/2024/12/22/zxKJmFfr8yhvlgt.png)

bias correction: 最开始更新时first_moment, second_moment值都接近0，将值放大进行参数更新，随着时间步的进行，这一步系数会接近1

标准adam对于第一步梯度进行L2正则化

<font size = 6;font color = orange>AdamW</font>

在参数更新时加入权重衰减

```python
x =x-learning_rate*(first_unbias /(np.sqrt(second_unbias)+1e-7)+decay_rate*x)
```

<font size =4>**学习率**</font>

<img src="https://s2.loli.net/2024/12/22/WeioYLsOyHXSnwB.png" alt="image-20241222191035506" style="zoom: 50%;" />

<img src="/home/oxidation/.config/Typora/typora-user-images/image-20241222191053526.png" alt="image-20241222191053526" style="zoom:50%;" />

因此要随着时间改变学习率

- 在固定的epoch后降低学习率（resnet 30，60,90时*0.1）
- consione: $\alpha_t = \frac{1}{2}\alpha_0(1+\cos(t\pi/T)),T:\text{Total number of epochs}$ 
- linear: $\alpha_t = \alpha_0(1-t/T)$
- lnverse: $\alpha_t = \alpha_0/ \sqrt t$

linear warmup: 高学习率会使得损失激增，前5000次迭代中，从0开始线性递增学习率

如果batch_size增加N, 学习率也按照N缩放

![image-20241229172549310](https://s2.loli.net/2024/12/29/QWMcbtdTe6VnRI2.png)

- 一阶优化（First-Order Optimization）
  $$
  f(x)= f(x_0)+f'(x_0)(x-x_0)
  $$
  $O(n)$

- 二阶优化：牛顿法（Newton’s method）
  $$
  f(x)=f(x_0)+f(x_0)+f'(x_0)(x-x_0)+\frac{1}{2}f''(x_0)(x-x_0)^2
  $$

  $$
  \mathcal{L}(\theta) \approx \mathcal{L}(\theta_0) + \underbrace{\nabla_\theta \mathcal{L}(\theta_0)(\theta - \theta_0)}_{\text{gradient}} 
  + \frac{1}{2} (\theta - \theta_0)^T \underbrace{\nabla_\theta^2 \mathcal{L}(\theta_0)}_{\text{Hessian}} (\theta - \theta_0)
  
  
  \\
  \nabla_\theta^2 \mathcal{L}(\theta_0) =
  \begin{bmatrix}
  \frac{\partial^2 \mathcal{L}}{\partial \theta_1^2} & \frac{\partial^2 \mathcal{L}}{\partial \theta_1 \partial \theta_2} & \frac{\partial^2 \mathcal{L}}{\partial \theta_1 \partial \theta_3} \\
  \frac{\partial^2 \mathcal{L}}{\partial \theta_2 \partial \theta_1} & \frac{\partial^2 \mathcal{L}}{\partial \theta_2^2} & \frac{\partial^2 \mathcal{L}}{\partial \theta_2 \partial \theta_3} \\
  \frac{\partial^2 \mathcal{L}}{\partial \theta_3 \partial \theta_1} & \frac{\partial^2 \mathcal{L}}{\partial \theta_3 \partial \theta_2} & \frac{\partial^2 \mathcal{L}}{\partial \theta_3^2}
  \end{bmatrix}
  \\
  
  
  \theta^* \leftarrow \theta_0 - (\nabla_\theta^2 \mathcal{L}(\theta_0))^{-1} \nabla_\theta \mathcal{L}(\theta_0)
  $$

  $O(n^3)$

- 二阶方法的计算代价太大，所以我们一般选择一阶方法 

- 拟牛顿法，不求Hessian的逆，求Hessian矩阵逆的近似解（$O(n^2)$）

- L-BGFS

  <font color = red> **实战选择**</font>
  ◼在许多情况下，Adam(W)是一个很好的默认选择；即使学
  习率保持不变，它通常也能正常工作
  ◼ SGD+Momentum 有时优于 Adam，但可能需要对 LR 和
  LR schedule policy 进行更多调整
  ◼ 如果可以进行全批量更新，那么可以使用高阶优化方法

## 神经网络与反向传播

激活函数

没有激活函数会得到线性分类器

![image-20241222200639509](https://s2.loli.net/2024/12/22/2I1xSdJAsUbHW7n.png)
$$
\text{tanh}=\frac{e^x-e^{-x}}{e^x+e^{-x}}
$$
前向传播

![image-20241222201048637](https://s2.loli.net/2024/12/22/sV7GwKxelSkFRz2.png)

```python
import numpy as np
from numpy.random import randn

N, D_in, H, D_out = 64, 1000, 100, 10
x, y = randn(N, D_in), randn(N, D_out)
w1, w2 = randn(D_in, H), randn(H, D_out)

for t in range(2000):
    h = 1 / (1 + np.exp(-x.dot(w1)))
    y_pred = h.dot(w2)
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h.T.dot(grad_y_pred)
    grad_h = grad_y_pred.dot(w2.T)
    grad_w1 = x.T.dot(grad_h * h * (1 - h))

    w1 -= 1e-4 * grad_w1
    w2 -= 1e-4 * grad_w2

```

$$
\frac{d\sigma(x)}{dx} = (1-\sigma(x))\sigma(x)
$$

![image-20241222211558957](https://s2.loli.net/2024/12/22/tRAXw2agWhiqrbm.png)

多元函数的导数是梯度，向量函数的导数是雅可比矩阵

![image-20241222214239278](https://s2.loli.net/2024/12/22/QOEnVoUR4XNW2Ci.png)

<font color = pink font size =7>向量的反向传播</font>

**变量的梯度和原始变量具有相同的维度**

![image-20241222222225915](https://s2.loli.net/2024/12/22/NapfVbHnPweCl65.png)

<font color = pink font size =7>矩阵的反向传播</font>

![image-20241222223207937](https://s2.loli.net/2024/12/22/4bvjxPYfZow12in.png)
$$
\begin{aligned}
&\text{[N × D]} \quad \text{[N × M]} \quad \text{[M × D]} \\
&\frac{\partial L}{\partial x} = \left( \frac{\partial L}{\partial y} \right) w^T \\
&\text{[D × M]} \quad \text{[D × N]} \quad \text{[N × M]} \\
&\frac{\partial L}{\partial w} = x^T \left( \frac{\partial L}{\partial y} \right)
\end{aligned}
$$
和矩阵形状匹配就好了，和标量是相同的

## 卷积神经网络

图像分类是最核心的计算机视觉任务

- 基于图像像素的分类

- 基于图像特征的分类

  特征提取

  - 颜色直方图
  - 方向梯度直方图
  - 词带：聚类图像块

<font color = brown font size = 6.5>**ps: 请注意！滤波器深度始终与输入的深度相同**</font>

![image-20241222231808135](https://s2.loli.net/2024/12/22/lA4d7gcO31W6iVL.png)



<center><font size = 7>Conv+ Relu</font>

线性分类器：每个类别的模板

全连接神经网络：整张图像的模板库

第一层卷积：局部图像模板

后面的卷积层：提取到高层次特征（形状，模式等）

卷积就是对应位置相乘再相加
$$
\text{output} = \frac{(N+2*P-F)}{S}+1
$$

eg: 输入$3*32*32$, filter: $10*5*5$, stride: 1, padding: 2

参数量：$(5*5*3+1)*10$ <font color =red font size = 2>这里1是bias的数量</font>

感受野计算
$$
R_l = R_{l-1} + (k_l-1)*\prod_i^{l-1} s_i
$$
对于大图像，需要许多卷积层才能看到整个图像

**下采样**

- 大步长
- 池化层

卷积层+池化层+全连层+激活函数
◼ 趋向于更小的滤波器和更深的层
◼ 趋向于舍弃池化层和全连接层

<font size = 5>**转置卷积**</font>：将较小的特征图放大，对低分辨率的图像进行上采样，提高分辨率

<font size = 5>**空洞卷积**</font>：对卷积核周围补零，扩大卷积核，扩大感受野

<font size = 5>**分组卷积**</font>: 分组卷积是将标准卷积中的通道分为多个组，每个组单独执行卷积操作，从而减少计算量和参数量。

 Alexnet网络中每一层对输入图像的感受野（包含卷积层，池化层，全连接层）是多少？

输入$227*227*3$

>  cov1: 11*11
>
> max pool1: $11+(3-1)*4=19$, 19*19
>
> conv2: $19+(5-1)*2*4=51$, 51*51
>
> max pool2 $51+(3-1)*1*2*4=67$, 67*67
>
> conv3: $67+(3-1)*2*1*2*4=99$,99*99
>
> conv4: $99+(3-1)*1*2*1*2*4=131$, 131*131
>
> conv5: $131+(3-1)*1*1*2*1*2*4=163$, 163*163
>
> max pool3: $163+(3-1)*1*1*1*2*1*2*4=195$, 195*195
>
> 这里的全连接实际上是$(6\times6\times256)*4096$的卷积来进行的，因此具有感受野
>
> FC6: $195+ (6-1)*2*1*1*1*2*1*2*4=355$, 355*355
>
> FC7: 355*355
>
> FC8: 355*355



## 卷积神经网络架构

![image-20241223170446115](https://s2.loli.net/2024/12/23/GI378tAuYRXWLOw.png)

输入不以0为中心/输入元素具有不同的缩放比例 -> 网络难以优化 -> 对输入进行缩放

逐个通道进行归一化
$$
x: N \times D \\
\mu_j = \frac{1}{N}\sum_{i=1}^{N}x_{i,j}\\
\sigma_j^{2}=\frac{1}{N}\sum_{i=1}^N(x_{i,j}-\mu_j)^2\\
\hat x_{i,j}= \frac{x_{i,j}-\mu_j}{\sqrt{\sigma_j^2+\epsilon}}
$$
<img src="https://s2.loli.net/2024/12/23/BfRCA2aIPseoj3U.png" alt="image-20241223171605683" style="zoom:33%;" />

归一化之后丢失了大量的数据信息（均值，方差）

对$\hat x_{i,j}$进行~缩放~ $y_{i,j}= \gamma_j \hat x_{i,j} +\beta_j$, $\gamma,\beta(\text{D})$是可以学习的参数

 ![image-20241223172637548](https://s2.loli.net/2024/12/23/b2EupqlVBf4ycoj.png)

<font color = blue>ps: 在训练过程中会记录下一个$\mu_t =\lambda \mu_b +(1-\lambda)\mu_{t-1}$ 这里的$\mu_b$是训练过程中当前的mini-batch中计算出的。相当于求出了每个批次均值和方差的加权平均。$\sigma$也是同理，在测试过程中使用的是最终得出的方差与均值，不会再进行计算。</font>

![image-20241223173333978](https://s2.loli.net/2024/12/23/gVOiSen9uvkFPQ6.png)

 **对比概览表**

| 特性                 | **Batch Normalization**              | **Layer Normalization**                    | **Instance Normalization**         | **Group Normalization**            |
| -------------------- | ------------------------------------ | ------------------------------------------ | ---------------------------------- | ---------------------------------- |
| **归一化范围**       | **每个通道**，跨**样本批量**计算     | **每个样本**，跨**通道**计算               | **每个样本**，每**通道**独立计算   | **每个样本**，每**组通道**计算     |
| **计算的均值和方差** | 对**批量样本的某个通道**求均值和方差 | 对**整个样本的所有通道**求均值和方差       | 对**单个样本每个通道**求均值和方差 | 对**单个样本每组通道**求均值和方差 |
| **适用场景**         | **大批量训练**任务（如分类任务）     | NLP、**小批量或序列任务**，transformer,RNN | 图像生成任务、风格迁移             | 小批量任务（如目标检测、分割任务） |
| **通用性**           | 适合批量大、特定任务                 | 更通用，尤其是序列数据                     | 专门用于风格迁移等任务             | 小批量训练更优                     |

------

**<font color = green>适用场景总结</font>**

| **任务场景**              | **推荐归一化方法**                         |
| ------------------------- | ------------------------------------------ |
| **大批量图像分类**        | Batch Normalization                        |
| **小批量图像任务**        | Group Normalization                        |
| **图像风格迁移**          | Instance Normalization                     |
| **小批量目标检测或分割**  | Group Normalization                        |
| **NLP、RNN、Transformer** | Layer Normalization                        |
| **动态批量或动态输入**    | Layer Normalization 或 Group Normalization |

### BN层的好处

- 使得深度网络更容易训练
- 改善梯度流
- 允许更高的学习率，更快的收敛
- 网络对初始化变得稳健
- 训练中起到正则化的作用

### 经典CNN模型

#### alexnet

![image-20241223190857439](https://s2.loli.net/2024/12/23/zmIxG2vcLF5OPA4.png)

#### zfnet

![image-20241223191339638](https://s2.loli.net/2024/12/23/AcM94adUXr8GhBo.png)

<font size = 5>更深的网络</font>

![image-20241223191448931](https://s2.loli.net/2024/12/23/9tofDBKmbWM4SGT.png)

只使用$3*3$卷积的原因：

- 三个$3 \times 3$卷积层的感受野相当于一个$7 \times 7$的卷积层，而这只需要更少的参数量。
- 同时有更深，更多的非线性变换

#### **VGGNet**

![image-20241223192534180](https://s2.loli.net/2024/12/23/TrxPQyechks6OSC.png)

最后的FC_1层也是通过卷积来实现的

- <font color = gray>**显存占用主要在浅层，参数瓶颈主要在FC层**</font>

#### **GoogleNet**

Inception模块

没有FC隐藏层
$$
\text{Input: }H \times W \times C_{in}\\
\text{output:} H \times W \times (C_{1 \times1}+ C_{3 \times3}+C_{5 \times5+C_{pool}})
$$
![image-20241223202130252](https://s2.loli.net/2024/12/23/NavQ9Up52j7ZJfT.png)

这里的计算量只计算了乘法的次数(<font color = red font size = 5>**ps:请注意，不要忽略卷积核的厚度**</font>)

- 通道数也持续变大

解决方法：使用$1 \times 1$卷积降低计算量，通道数，参数量

![image-20241223200308372](https://s2.loli.net/2024/12/23/Z7p2xOjkSbrBg4a.png)

通过$1 \times1$卷积来改变通道数，然后进行不同的卷积操作，提取不同的特征，最后根据通道维度将feature map拼接在一起

googlenet 网络结构

![image-20241223212110042](https://s2.loli.net/2024/12/23/TPHSrtVANbOK9w5.png)

中间是堆叠的inception模块，周围是分类器与辅助分类器，只有在分类器上才有FC层



#### **resnet**

- 深层网络性能比浅层更差
- 但不是由过拟合造成的
- 关键问题：让深层网络的表达能力至少与浅层网络一样好（恒等映射）

<img src="https://s2.loli.net/2024/12/23/DF2KS31h9cnqsBX.png" alt="image-20241223215606546" style="zoom:50%;" />

使用bottleneck部分提升计算效率

<img src="https://s2.loli.net/2024/12/23/kf1IboCLMig9Grn.png" alt="image-20241223215823561" style="zoom:50%;" />

小结：

- alexnet: cnn可以用于计算机视觉任务
- VGG: 网络深度的重要性
- google-net: 网络宽度的重要性
- resnet：训练极深的卷积网络，CNN的表现优于人类

##### 基于resnet的改进

**Squeeze-and-Excitation Networks (SENet, 2017)**

- 为 ResNet 加上注意力

![image-20241223220337894](https://s2.loli.net/2024/12/23/S8cRNkntml1YQeX.png)

右边部分相当于是attention_weight, 还有缩放的部分

![image-20241223220937765](https://s2.loli.net/2024/12/23/OzCwFfNX8e1gQMx.png)



**Identity Mappings in Deep Residual Networks (2016)**

- 改进网络设计，让信息能更容易传播

- 深度1001层

  <img src="https://s2.loli.net/2024/12/23/7LyVl6ERcvCYspf.png" alt="image-20241223221338021" style="zoom:50%;" />

  

**Wide Residual Networks (2016）**

- 增加宽度，减少深度，计算效率更高

![image-20241223221436948](https://s2.loli.net/2024/12/23/Yu6lrpxMXK7yVNW.png)

加宽了通道数



**Aggregated Residual Transformations for Deep Neural**
**Networks (ResNeXt, 2016)**

- 增加宽度，保持深度，不增加参数量

![image-20241223221810306](https://s2.loli.net/2024/12/23/lbcXxHyP1AhsYjm.png)

(<font color = blue font size = 6>**ps:请注意，不要忽略卷积核的厚度**</font>



**Densely Connected Convolutional Networks (DenseNet，**
**CVPR 2017 Best Paper)**

- 密集使用残差模块，每一层都与前面的所有层连接

### 迁移学习

将在分类任务上训练好的网络迁移到其他网络

- 分类任务数据量大，标注难度低，标注成本低
- 其他任务数据量小，标注难度高，标注成本高
- 小数据集上直接训练模型，性能很差

小数据集上迁移时可以只训练分类头，数据集越大训练层数越多

![image-20241223224705396](https://s2.loli.net/2024/12/23/3ayU8ZsDnWh5jQ4.png)

- 预训练并不是必须的，使用:smile:**合适的预训练模型**一定不会得到更差的效果



## 训练神经网络

[感受野计算](https://distill.pub/2019/computing-receptive-
fields/#return-from-solving-receptive-field-size)
$$
r_0 = \sum_{l=1}^{L} \left( (k_l - 1) \prod_{i=1}^{l-1} s_i \right) + 1
$$
计算输出的feature_map是向下取整

### 激活函数

##### 常见的激活函数

<font color = blue>**sigmoid**: $\sigma (x)= 1/(1+e^{-x})$</font>

- 梯度消失（饱和时）
- 指数计算成本高
- 输出不是以零为中心
- 很好的建模真实生物神经元

<font color = blue>**tanh**</font>

- 以零为中心
- 梯度消失问题

<font color = blue>**relu**</font>

- 正数梯度不会消失
- 计算效率高
- 收敛速度快
- 输出不是0为中心
- 负数区域梯度为0

<font color = blue>**leaky relu**: $\max(0.01x,x)$</font>

- 计算效率高
- 收敛速度快

- **prelu**: $\max(\alpha x,x)$

<font color = blue>**GELU**: $f(x)= x\Phi(x)$ </font>

![image-20241223231739756](https://s2.loli.net/2024/12/23/3ne9VrMwoGyEDOI.png)

$\Phi(x)$: 在图片中的公式中，
$$
\Phi(x) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^x e^{-\frac{t^2}{2}} dt
$$
$\Phi(x)$ 表示从负无穷到 x的区域，标准正态分布曲线下的面积。它反映了一个随机变量 X（服从标准正态分布）的取值小于或等于 x 的概率。
$$
f(x) = x \cdot \Phi(x)
$$
$\Phi(x)$ 用来对输入 x 进行“概率加权”，将输入非线性地映射到输出空间。其导数为标准正态分布的概率密度函数
$$
\Phi'(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}}
$$

- 0附近的导数可计算
- 平滑函数有助于训练
- 更高的计算成本
- 较大的负值梯度为0

<font color = blue>**ELU(exponential Linear Units)**</font>
$$
f(x) =
\begin{cases} 
x, & \text{if } x > 0 \\
\alpha (\exp(x) - 1), & \text{if } x \leq 0
\end{cases}
$$

- 具有ReLU所有优点
- 相对于Leaky ReLU,负数区域的稳定性较高
- 计算成本较高

<font color = blue>**Scaled Exponential Linear Units (SELU)**</font>
$$
f(x) =
\begin{cases} \lambda
x, & \text{if } x > 0 \\ \lambda
\alpha (\exp(x) - 1), & \text{if } x \leq 0
\end{cases}
$$

- ELU的扩展版本，适合**深度网络**
- 具有自我规范的属性: 正值部分线性，负值部分通过$\alpha (\text{exp}(x)-1)$确保对称性，从而在正负方向上均值趋于0, 通过$\lambda$进行缩放，将方差调整到合理的范围，因此可以不需要BN层就能保证输出的分布稳定。
- 可以在没有BN的情况下训练深度网络

![image-20241224000340419](https://s2.loli.net/2024/12/24/AYwLeKHaGF5v7yd.png)

<font color = blue>**MaxOut**: $\max{(\omega_1^{T}x+b_1,\omega_2^{T}x+b_2})$</font>

- 非线性
- 具有leaky relu的优点
- 参数量翻倍

##### 问题

输出以0为中心的好处

- 模型收敛更快
  $$
  \frac{\partial L}{\partial w_i} = \frac{\partial L}{\partial f}\frac{\partial f}{\partial z}\frac{\partial z}{\partial w_i}=\frac{\partial L}{\partial f}\frac{\partial f}{\partial z} x_i
  $$
  正如上面的式子，如果不是以0为中心的，即上一层的输出是全负或者全正的，那么梯度的正负就只由前面的部分决定，所以对于权重来说，更新的方向只能是相同的。

  ![image-20241224090747473](https://s2.loli.net/2024/12/24/7MC5fD24KsUZrxB.png)

  如上图所示，希望$w_1$向负方向更新，$w_0$向正方向更新，这需要梯度的符号相反，所以需要上一层的输出，即这一层的输入符号不同，模型因此会收敛的更快

<font color = red font size =5>**使用建议：**</font>

- 使用 ReLU
- 可以尝试 Leaky ReLU / PReLU / GELU
- 尽量避免使用 sigmoid / tanh

### 数据预处理

![image-20241224091909867](https://s2.loli.net/2024/12/24/COqZrohScuJlnDb.png)

假设输入图像大小为[32,32,3]

- 减去数据集中的平均图像（[32,32,3） AlexNet
- 减去通道的均值（[1,1,3）VGGNet
- 减去通道的均值，除以通道的标准差（[1,1,3]）现在最常用的方法

### 权重初始化

如果将所有的权重初始化为同一个值，而此时的输入又相同，那么所有神经元的输出都是相同的，因此在进行梯度更新时会更新相同的值，丧失了多样性。

- 随机初始化为一个小数值 ```0.01*np.random.randn(Din,Dout)```

  - 在浅层网络中效果不错，在深层网络中较差
  - 深层网络的输出趋向于0，这导致梯度很小，难以训练

- 初始化过大的权重值```0.05*np.random.randn(Din,Dout)```

- 合适的权重初始化

  ![](https://s2.loli.net/2024/12/24/U3Jkconf9iZwMxN.png)
  
  ![](https://s2.loli.net/2024/12/24/eli1wWBZxJoKpS4.png)
  
  <font color = red>**Xavier” initialization**</font>
  
  ```python
  dims = [4096] * 7
  hs = []
  x = np.random.randn(16, dims[0])
  
  for Din, Dout in zip(dims[:-1], dims[1:]):
      W = np.random.randn(Din, Dout) / np.sqrt(Din)
      x = np.tanh(x.dot(W))
      hs.append(x)
  
  ```
  
  $$
  \text{Var}(y) = \text{Var}(W)\text{Var}(x)\text{Din}
  $$
  
  为了让输入和输出的方差一致
  $$
  \text{Var}(W) = \frac{1}{\text{Din}}
  $$

![image-20241224154818899](https://s2.loli.net/2024/12/24/Fqb7eCD5wYXphTE.png)

<font color = red>**Kaiming Initialization**</font>

![image-20241224155035025](https://s2.loli.net/2024/12/24/jBoplmWTkwFngIe.png)

### 正则化



问题：如何减少训练正确率和测试集上正确率的gap

#### Earlystop

当验证集上准确率降低时就停止训练

或者一直训练，但储存在验证集上效果最好的模型快照

![image-20241224155421317](https://s2.loli.net/2024/12/24/VKqUrM2dzXRIDjv.png)

#### 模型集成

集成学习的思想，训练多个独立模型，测试时使用所有模型的预测结果（去均值），可以带来可观的效果提升。

#### Dropout

在每次前向传播的计算中，将一些神经元随机设置为0,常用0.5。

```python
p =0.5
def train_step(X):
    H1 = np.maximum(0,np.dot(W1,X)+b1)
    U1 = np.random.rand(*H1.shape) < p #dropout
    H1*=U1
    H2 = np.maximum(0,np.dot(H1,X)+b2)
    U2 = np.random.rand(*H2.shape) < p #dropout
    H2*=U2
    out = np.dot(W3,H2)+b3
```

- 让网络学习冗余表示，防止特征的协同适应（让模型被迫学习特征的独立性，而不是依赖特征的组合）

问题：如果在测试时也使用dropout,模型的输出会具有随机性，因此，我们想将随机性的期望作为确定的输出
$$
y = f(x) = E_z[f(x,z)] = \int p(z)f(x,z)dz
$$

但是计算这个积分有些困难，因此对测试时的dropoutout积分进行近似

![image-20241224165233787](https://s2.loli.net/2024/12/24/faIj9pMWGJOZs3t.png)

```python
def predict(X):
    H1 = np.maximum(0,np.dot(W1,X)+b1)*p
    H2 = np.maximum(0,np.dot(W2,H1)+b2)*p
    out = np.dot(W3,H2)+b3
```

训练时的预期输出 = 测试时的输出

**inverted dropout**

![image-20241224165606892](https://s2.loli.net/2024/12/24/nYqfU4TxFKm1usP.png)

相当于是模型集成的方法每一个随机掩码都对应了一个模型

#### dropoconnect

<img src="https://s2.loli.net/2024/12/24/woJHhXnaEjA3WMD.png" alt="image-20241224165710791" style="zoom:50%;" />

#### Stochastic Depth

训练时随机跳过某些层

测试时使用所有层

**训练时增加随机性**

- Batch Normalization: 在训练时，对于**随机的mini-batch**中的统计量进行归一化。记录衰减的历史均值和方差，测试时不再使用

#### 数据增强

- 图像变换：水平翻转，随机裁剪和缩放，颜色扰动，

- Cutout: 删除图像中的一部分，在小数据集上效果很好，大数据集上不常用

- Mixup: 随机将多张图片混合为一张图片，测试时使用原始图像

  ![image-20241224170557145](https://s2.loli.net/2024/12/24/TxXulLknqYiCwht.png)

### 超参数选择

- 检查初始损失（关闭权重衰减）

  C个类别，softmax_loss初始损失为$-\log (1/C)= \log C$

- 过拟合少量样本：准确率达到99%~100%

- 选择合适的LR: 让损失在100 iterations内显著下降（lr: 1e-1, 1e-2, 1e-3, 1e-4）

- 粗调1-5 epochs

- 精调 10-20 epochs (从上一步开始选择最好的模型)

- 检测损失和准确率曲线

- 如果出现问题，返回精调的步骤

  如果准确率持续上升，需要训练更多轮次

  ![image-20241224171253902](https://s2.loli.net/2024/12/24/hAmfx9nyKkNtg5G.png)

  train & val gap过大意味着过拟合，需要增加正则化/数据

  ![image-20241224171343917](https://s2.loli.net/2024/12/24/qSiJ68dOatl2fI4.png)

  train & val gap 过小意味着欠拟合，需要训练更久/更大的模型

  ![image-20241224171444329](https://s2.loli.net/2024/12/24/4N9YjFwSOelzMLk.png)

- 参数搜素方法：Grid Layout, Random Layout

## 循环神经网络

![image-20241224172446062](https://s2.loli.net/2024/12/24/z97pRrOoGlFu46k.png)

**Many to Many**

![image-20241224172518087](https://s2.loli.net/2024/12/24/xVO5KjabdCfQpcP.png)

权重是共享的
$$
h_t = f_W(h_{t-1},x_t) = \text{tanh}(W_{hh}h_{t-1}+W_{xh}x_t)\\
y_t = W_{hy}h_t
$$
**Many to One**

![image-20241224172813215](https://s2.loli.net/2024/12/24/c6dJKiYbMCTokn8.png)

**One to Many** 

![image-20241224172847894](https://s2.loli.net/2024/12/24/Qu1lYiRIV9ghTrE.png)

隐藏层的计算
$$
h_t = \text{tanh}(w_{hx}x+w_{hh}h_{t-1})
$$




损失计算

分块计算损失

- 计算所有时间步的前向传播
- 但只在每个时间块内进行反向传播

![image-20241224214041788](https://s2.loli.net/2024/12/24/SLuvBFsReG81kyd.png)

<font size = 5>**RNN优势**</font>

- 可以处理任何长度的输入
- 步骤t的计算可以使用之前许多步骤的信息
- 输入时间越长，模型尺寸越大
- 每个时间步使用相同的权重

<font size = 5>**RNN缺点**</font>

- 循环计算速度慢
- 很难从多个步骤中获取信息（遗忘）

图像描述

在cnn的基础上去除了最后的FC层，加入rnn进行图像描述

![image-20241224223959686](https://s2.loli.net/2024/12/24/IHGN2Ccvqsdwe6Y.png)

- 视觉问答

![image-20241224224330450](https://s2.loli.net/2024/12/24/GTp8Jszn6qbvVD3.png)

point-wise-multiplication: 结合视觉特征和文本特征，实现跨模态的信息融合

- 视觉对话（visual-Dialog）
- 视觉语言导航（visual Language Navigation）

多层rnn

![image-20241224225630531](https://s2.loli.net/2024/12/24/5VojgObHFpzns1R.png)

#### RNN梯度计算

$$
h_t = \text{tanh}(W_{hh}h_{t-1}+W_{xh}x)\\
\frac{\partial h_t}{\partial h_{t-1}} = \text{tanh}'(W_{hh}h_{t-1}+W_{xh}x)W_{hh}
$$

![image-20241229200956057](https://s2.loli.net/2024/12/29/61k4FsZ9jIOfcUP.png)

这里可能存在一些问题

$h_t$梯度会总是小于1,因此容易造成梯度消失的问题

去除tanh会造成梯度爆炸，需要进行梯度裁剪

![image-20241229201113674](https://s2.loli.net/2024/12/29/P17X8fbnjRrYQhD.png)

### LSTM

<img src="https://s2.loli.net/2024/12/24/hnauLZf6rCgWl3T.png" alt="image-20241224231825777" style="zoom:50%;" />

![image-20241224232107045](https://s2.loli.net/2024/12/24/XwjUyuc7IZkAMFb.png)

- $f_t = \sigma(W_f[h_{t-1},x_t]+b_f)$
- $i_t = \sigma(W_i[h_{t-1},x_t]+b_i)$
- $\tilde C_t = \text{tanh}(W_C[h_{t-1},x_t]+b_C)$
- $C_t = f_t * C_{t-1} + i_t *\tilde C_{t}$
- $o_t = \sigma(W_o[h_{t-1},x_t]+b_o)$
- $h_t = o_t*\text{tanh}(C_t)$

![image-20241224233354995](https://s2.loli.net/2024/12/24/hYJXSKBEZl5gAku.png)

- LSTM 架构使 RNN 更容易在多个时间步长内保存信息
- 例如，如果f=1且i=0，则该单元的信息将无限期保留
- 相比之下，vanilla RNN更难学习循环权重矩阵 Wh 以保存信息
- LSTM不能保证没有消失/爆炸梯度，但它确实为模型学习长距离依赖关系提供了一种更简单的方法

<center><font color = red font size = 5>连续的梯度流</font></center>

### 其他网络

双向循环网络

![image-20241224233905574](https://s2.loli.net/2024/12/24/DIZT8h7A5kCBnzJ.png)

编码器-解码器结构

- 输入，输出长度可变，隐藏状态形状固定

![image-20241224233944573](https://s2.loli.net/2024/12/24/IWgmhZCUjNTsHyf.png)

![image-20241224234055456](https://s2.loli.net/2024/12/24/LNZfaD8ozmWjBRu.png)

## Transformer

self-attention：依赖一个输入

cross-attention: 依赖两个输入

CNN -> Attention -> CNN -> predict -> (loss)  

![image-20241225150334989](https://s2.loli.net/2024/12/25/q9NXdsHWE6lVneZ.png)

注意力机制分为两步

- 计算权重分布$\alpha$
  $$
  \alpha_n = p(z=n|X,q)\\=\text{softmax}(s(x_n,q))\\=\frac{\text{exp}(s(x_n,q))}{\sum_{j=1}^{N}\text{exp}(s(x_j,q))}
  $$

- 根据$\alpha$计算输入信息的加权平均
  $$
  \text{att}(X,q) = \sum_{n=1}^{N}\alpha_nx_n\\=\mathbb{E}_{\mathbf{Z} \sim p(z \mid \mathbf{X}, \mathbf{q})} \left[\mathbf{x}_{\mathbf{Z}}\right]
  $$

<font size = 5>**注意力的类型**</font>

![image-20241225151937704](https://s2.loli.net/2024/12/25/aMfZqnlxHvCGyEg.png)

![image-20241225152008130](https://s2.loli.net/2024/12/25/Jhz4dbaqGPoTOlF.png)

特点：能处理变长的向量序列

与其他模型相比

| CNN & RNN                                                    | 全连接层                                                     | self-attention                                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20241225153134013](https://s2.loli.net/2024/12/25/KS4bZXicDo2YWud.png) | ![image-20241225153147028](https://s2.loli.net/2024/12/25/Ai6Kd9wXZYSkeDa.png) | ![image-20241225153200602](https://s2.loli.net/2024/12/25/xMqrpf63aABnhjF.png) |
| <font color = red>**只能建模输入信息的局部依赖关系**</font>  | 可以处理全局信息，<font color = red>**但是无法处理变长问题**</font> | 连接权重$\alpha_{ij}$由注意力机制<font color = blue>**动态生成**</font> |

### self-attention

![image-20241225160231174](https://s2.loli.net/2024/12/25/E1KXpxmfUARaSVs.png)
$$
\begin{align*}
Q &= W_q X \in \mathbb{R}^{D_k \times N}, \\
K &= W_k X \in \mathbb{R}^{D_k \times N}, \\
V &= W_v X \in \mathbb{R}^{D_v \times N}, \\
h_n &= \text{att}(K, V, q_n), \\
H &= V \, \text{softmax} \left( \frac{K^\top Q}{\sqrt{D_k}} \right),
\end{align*}
$$
attention_weight中第n列其实是$h_n$的权重

### multi-head Attention

计算多次self-attention,然后将输出H矩阵拼接在一起，最后再用$W_o$再次线性变换得到最终的结果

![image-20241225161509303](https://s2.loli.net/2024/12/25/O9QY31z7bIrJGnB.png)

### Transformer

- encoder：全局语义建模 $\text{Attention}(Q,K,V)=\text{softmax}(\frac{QK^T}{\sqrt d_k})V$

  - input(input embedding + position encoding)

    ![image-20241225164602323](https://s2.loli.net/2024/12/25/r7k9bqM1LscWgV2.png)

  - Multi-head attention

    <font size = 5>**Q, K, V的由来**</font>

    ![image-20241225170800540](https://s2.loli.net/2024/12/25/FENp4j2heHraqsM.png)

    <font size = 5>**Muti-head选择**</font>

    ![image-20241225171209866](https://s2.loli.net/2024/12/25/njsrpCekdmO5Ro2.png)

    按照channel/h进行分组，分别计算attention, 然后进行拼接

    计算attention weight时要注意一个细节

    ![image-20241225172326308](https://s2.loli.net/2024/12/25/tYQzu5V8HACbKON.png)

    ~~这里是除以$\sqrt d_{model}$，因为在最后Z是直接进行拼接的，因此不能按照正常multi-head attention来进行理解，因此需要直接除以$\sqrt d_{model}$~~

    ps: 这里还是需要除以$\sqrt {d_{model}/h}$, 因为这里处理的动机是为了将$QK^T$的方差标准差化成1, 如果$Q,K$都是服从正态分布的，那么$QK^T$的方差是通道数
  
  - add & norm 
    $$
    \text{input} + \text{norm}[Z]
    $$
    mean 0,std = 1 稳定训练，正则化

    add residuals：避免梯度消失，可以训练更深的网络

  - feed forward: mlp
  
  - add & norm: reasiduals

encoders: input to $encoder_{i+1}$  output form $Encoder_i$

- decoder

  - targets

    ![image-20241225174137810](https://s2.loli.net/2024/12/25/epGgomS32FRUCAv.png)

  - masked multi head attention

    测试时只能看到t-1时刻及之前的信息，因此将之后的tokenizer进行masked 变成$-\infty$ , 这会使得softmax计算时变为0

    ![image-20241225174528009](https://s2.loli.net/2024/12/25/CmRrvN2FujXg79k.png)

  - encoder decoder attention

    ![image-20241225174634363](https://s2.loli.net/2024/12/25/yiZ3NbalDQWeuAk.png)

    这里使用来自前n个encoder最终的k和v, 由自己提供q进行multi-head attention操作

    ![image-20241230112234850](https://s2.loli.net/2024/12/30/EABw9Wngy4Ha52P.png)

  - linear
  
  - softmax

![image-20241225192855053](https://s2.loli.net/2024/12/25/4IiNPJEc7UuLmwq.png)

**<font color = blue font size = 5>encoder可以看到所有的时间步，decoder只能看到t-1及之前的时间步</font>**

#### VIT

![image-20241225191843582](https://s2.loli.net/2024/12/25/jiRld5e8vmIL3su.png)

#### 多模态

![image-20241225193209385](https://s2.loli.net/2024/12/25/Ghbka2KqTYoVdSt.png)

### PEFT

![image-20241225193244533](https://s2.loli.net/2024/12/25/q1h2cFHoDfSVv4P.png)



## 目标检测与图像分割

### 语义分割：图像的每个像素进行分类

- 对图像的每个像素进行分类？

  - 问题：如果没有语义信息，很难正确分类每个像素
  - 解决：感知像素周围的语义，帮助正确分类像素-**滑窗计算**
  - 问题：计算非常低效。图像块重叠的部分会被重复计算很多次。

- 全卷积

  问题：分类模型会大幅降低特征的分辨率，难以满足分割所需的高分辨率**输出**

  解决方案：先逐步下采样降低计算量（大步长的卷积，池化），再上采样恢复图像分辨率（unpooling,转置卷积）

<font color = green>**Unet(encoder-decoder)**</font>

![image-20241225201211156](https://s2.loli.net/2024/12/25/4mbeCSKEH9lBP8u.png)

**<font color = green>DeepLab(Encoder-noly)</font>**

![image-20241225201349164](https://s2.loli.net/2024/12/25/SYictvwufX9H7gG.png)

<font color = green>**PSPNet**</font>（encoder-only）

![image-20241225202201819](https://s2.loli.net/2024/12/25/lUun2LBaVZ1TCJK.png)

<font size = 5 font color = red> **对像素进行分类，不区分不同物体!**</font>

![image-20241225202447088](https://s2.loli.net/2024/12/25/5LNJuOQqrK9niUX.png)

例如上面这张图中只关心是不是牛，而在实例分割中需要区分牛1和牛2

### 目标检测：对物体进行分类+定位（建模成回归任务）

- 单个物体：分类+定位（建模成回归任务）位置信息是连续值

  ![image-20241225203335892](https://s2.loli.net/2024/12/25/UQykdMuPWEAJ75H.png)

- 多个物体

  - 问题：同一个物体可能会有多个输出
  - 将图像分为不同的区域，对每个图像区域进行单个物体检测
  - 如何快速确定图像位置？位置区域有不同的位置，大小，比例

**region proposals**(区域候选框)：selective search

找到可能存在物体的区域

selective search：在cpu上运行，几秒内可以生成2000多个区域



**<font size = 5>R-CNN</font>**

流程

- 输入图像 区域生成网络生成（RoI）
- wrap进行处理：cnn处理需要固定的尺寸，进行调整以便进一步处理
- 每个区域利用cnn（imagenet预训练过的cnn）进行处理
- 使用svm进行分类，同时通过bboxreg修正候选框

**<font color = red>问题: 效率很低，非常慢，同一块区域可能被多次计算</font>**

- 使用卷积网络处理过整张图像的feature_map进行区域提取

![image-20241225205435181](https://s2.loli.net/2024/12/25/q5cM1bVNrGE8hjL.png)



**<font size = 5>Fast R-CNN</font>**

在feature_map上进行候选区域的生成，然后进行crop+resize处理，这样还可以使用batch进行处理

![image-20241225210019194](https://s2.loli.net/2024/12/25/go68Vjwa3MsWACy.png)

将RoI特征与原始proposal中的特征存在差异

<font color = red>**问题：使用传统的selective search方法生成region of interest速度太慢了**</font>

**<font size = 5> Faster R-CNN</font>**

![image-20241225211428690](https://s2.loli.net/2024/12/25/ZO4nUQCYsGEjwLS.png)

<font size = 4> **Region Proposal Network** </font>

- 假设每个点都有固定大小的锚框
- 预测锚框内是否存在物体
- 对具有物体的锚框进行调整

![image-20241225212933682](https://s2.loli.net/2024/12/25/guZhKGySMsT4cO7.png)

<font color = blue font size = 5> **联合训练四个损失**</font>

- RPN分类损失：是否存在物体
- RPN回归损失：矩形框(锚框)
- 检测分类损失：物体类别
- 检测回归损失：矩形框

**两阶段的检测器**

- 第一阶段：在每张图上运行一次(RPN & Backbone)
- 第二阶段：每个region上运行一次(RoI Pooling/RoI Align, Detector)

**FPN**

构建自顶向下的路径，将高层语义特征逐步向低层传播，同时融合多尺度特征。

![image-20241225222721459](https://s2.loli.net/2024/12/25/gmOhAJSIVre9Bd2.png)

![image-20241225223638553](https://s2.loli.net/2024/12/25/uWLPqg1rixzm36N.png)

![image-20241225223619475](https://s2.loli.net/2024/12/25/SAEHPkXt8dyv3i1.png)

对于不同的预测头进行整合，得出最终的结果。

 ![image-20241225223853745](https://s2.loli.net/2024/12/25/khsgB3FdAmwR82t.png)

有点像RNN，经过一次预测得出的box,送入下一次检测中，产生新的box, 再送到新的检测中

#### 单阶段检测器（YOLO/SSD/RetinaNet）

- 直接对每个锚框进行分类和回归
- 可以看作有类别的RPN

### 实例分割：对检测到的每个实例物体进行分割

mask R-cnn : fast r-cnn加了mask head

![image-20241225224346724](https://s2.loli.net/2024/12/25/slLMeoQFH2daYvi.png)



## 视频理解

### 视频分类

问题：视频太大了

解决方案：在clips上进行训练，低fps,低分辨率

训练：在低FPS的clips上进行训练

测试：在不同的clips上测试模型，取平均预测结果（在一些模型中输入的时序是确定的，因此需要固定时间长度）

#### 经典模型

**<font size = 5>单帧CNN</font>**

通常是一个比较强的baseline,一般只比考虑时序的方法低3～5个点

- 对视频帧进行独立分类，再使用softmax进行平均结果

<font size = 5> **Late Fusion**</font>

- late fusion(FC): 将每一帧的cnn的特征并联并送入MLP中

![image-20241226165525332](https://s2.loli.net/2024/12/26/Agz5t2y6pPkJ8KI.png)

- late fusion(Pooling): 将每一帧的CNN特征池化并送入线性层

![image-20241226165831384](https://s2.loli.net/2024/12/26/xrFJKD3YtQO6CNz.png)

<font color = red> **问题：虽然具有了位置特征，但是由于帧数过多，并不容易并联，并且很难设置mlp的大小；难以捕捉帧之间 low-level motion**</font>

<font size = 5> **Early Fusion**</font>

将input（$T \times 3 \times H \times W$）进行reshape （$3T \times H \times W$），然后进行2dCnn卷积进行分类

![image-20241226170628569](https://s2.loli.net/2024/12/26/gl4ItDbAYhE9UCy.png)

问题：一层cnn对于视频处理可能能力有些不足

2D-CNN存在的问题：对于将时间加入通道中的2D CNN,每一帧都是由不同的滤波器进行处理的，例如一个$3T \times H \times W$的输入，一个卷积核的大小就有$3T \times F \times F$, 因此每一帧都是不同的filter进行进行处理的，并没有在时间的维度上进行卷积核滑动来特征提取，没有所谓的时间平移不变性。

![image-20241226171639978](https://s2.loli.net/2024/12/26/gXEGMZojxhfVLIp.png)

<font size = 5 font color = blue> **3D CNN**</font>

![image-20241226170805515](https://s2.loli.net/2024/12/26/f6z2HvBaixeGohn.png)

![image-20241226170958539](https://s2.loli.net/2024/12/26/HviKt93E8mUTCJ6.png)

![image-20241226172125745](https://s2.loli.net/2024/12/26/siGrTxICQ6VwZ3u.png)



- C3D：3D中的VGG

问题：计算量很大，大概是VGG的2.9倍

<font size = 5> **双流网络**</font>

Motion Stream(运动流)：捕捉视频中随着时间变化的运动信息。

接受视频的光流（一种时间序列特征，表示像素点在相邻帧之间的运动情况）作为输入

可以直接捕捉运动方向和速度，帮助网络理解动作和动态特征，专注于描述运动。

Appearance Stream (外观流): 提取视频中每一帧的静态空间特征，通常是视频中单帧或者多个采样帧(RGB图像)作为输入

较好的保留了场景和目标的空间特征，对基于外观的动作识别任务很重要

![image-20241226195238018](https://s2.loli.net/2024/12/26/O1ol8pd5ruS9k3T.png)



### **<font size = 5>建模长时间的时序结构</font>**

上述的方法主要关注2-5秒的非常短片段中模拟帧之间的 局部运动，如何处理长时间的时序结构？

**<font size = 4 font color = blue>RCN(Recurrent Convolutional Network)</font>**

![image-20241226200031865](https://s2.loli.net/2024/12/26/i5RDUIxp4YAzOKQ.png)

这里使用类似于rnn的结构，只是将所有的矩阵乘法替换为2D卷积

![image-20241226200228766](https://s2.loli.net/2024/12/26/HsVzmLS4l5iADXu.png)

<font size =4.5  font color =red>**问题：RNN对于长序列很慢，无法进行并行化**</font>

<font size = 5>**I3D**</font>

将每个2Dkernel替换为3Dkernel，重用2D-CNN来处理视频任务

替换原则：将2D conv的权重在空间中复制$k_t$次，然后将结果除以$k_t$，因为输出的维度和2d是相同的，因此如果是同一个输入复制了$k_t$次，最终输出的结果是2d的$k_t$倍，因此需要除以$k_t$



### 视频任务

<font color = pink>**分类**</font>

1. Action recognition
2. Action Prediction

<font color = gray>**搜索**</font>

1. temporal action proposal: 长视频中生成候选时间区间，这些区间可能包含动作
2. spatiotemporal action proposal：视频中生成动作候选区域，包含空间位置和时间边界

Temporal Action Localization/Detection：给定一个长序列，识别不同的动作对应的帧

可以先使用类似faster R-cnn的架构，先生成temporal proposals,然后进行分类



Spatiotemporal Action Localization/Detection：给定一个长序列，检测时间和空间的所有人，并对他们正在进行的活动进行分类

![image-20241226202845781](https://s2.loli.net/2024/12/26/MPYhpJZyd2z7L49.png)

其他任务

![image-20241226210211401](https://s2.loli.net/2024/12/26/yKSfj6ph9BD27xO.png)

![image-20241226210258788](https://s2.loli.net/2024/12/26/NxikCjltEa4LPef.png)

![image-20241226210320000](https://s2.loli.net/2024/12/26/Qy9z3XjZnmfYrkB.png)



### 可视化

风格迁移

Gram Matrix: 对于某一层的feature_map, Gram矩阵通过计算各通道之间的内积，反映不同特征之间的相关性。

计算步骤：

- 将通道的特征图展平，表示成一个$C \times (H \times W)$的矩阵
- 计算这个矩阵的内积

![image-20241226212759141](https://s2.loli.net/2024/12/26/QeiXcdVI2uFMrEO.png)
$$
G^\phi_j(x)_{c,c'} = \frac{1}{C_jH_jW_j}\sum_{h=1}^{H_j}\sum_{w=1}^{W_j} \phi_j(x)_{h,w,c}\phi_{j}(x)_{h,w,c'}
$$
这里的j表示特征图的层数, $\phi_{j}(x)_{h,w,c}$表示第j层的特征图在(h,w,c)处的具体取值

这里在原论文中进行了归一化

问题：因为是直接对y进行梯度更新，所以在每次生成都需要多次的正向反向传播。

改进：能不能在生成过程中将风格学习到网络中

![image-20241226214617680](https://s2.loli.net/2024/12/26/zeDZ2tVrAGpubk1.png)

这样在训练过程中可以只更新前馈网络的权重，训练完成后不需要再训练就可以生成固定风格的输出

适合Instance Norm

![image-20241226214837332](https://s2.loli.net/2024/12/26/gUE7jqYFQfV5O36.png)

对通道进行正则化







## 自监督学习

pretext Task: 根据数据本身定义任务，无需注释的无监督任务

Downstream Task：想要应用的任务，没有大规模的数据集，数据集有标注



<font size = 5 font color = blue>**如何进行评估？**</font>

- pretext性能：但是应用范围比较窄，重点还是在下游任务的性能上
- **特征质量**（in-domain上的性能--同分布的数据集）：评测时把自监督学习好的模型作为特征提取器，训练一层线性分类器，根据线性分类器的效果来评价自监督学习性能
- **鲁棒性和泛化性**（在不同数据集上的效果）
- 计算效率：训练时间和训练所需的资源
- 迁移学习和下游任务性能

使用少量有标签的数据在目标任务上训练浅层网络

![image-20241226230143401](https://s2.loli.net/2024/12/26/HkSF6Gg8uBrIKOe.png)

### Pretext Task

- 预测旋转角度
- 预测图像块位置
- 预测图像拼图
- 预测缺失的像素
- 图像着色：根据灰度图进行着色，
- 视频着色：建模视频中颜色的时间连贯性（假设可以使模型学习跟踪没有标签的区域或对象）

<font size = 5 font color = red> **关心的问题**</font>

- pretext task通常侧重于“视觉常识”，例如旋转角度，修复，重新排列和着色

- 模型需要学习自然图像的良好特征，例如对象类别的语义表示（例如能够学到猫的图像中的毛发，耳朵形状等，而不是简单的像素）

- 我们通常不关心pretext task任务的性能，而是关心学习到的特征对于Downstream Task(分类，检测，分割)的效果

pretext Task学到的表征可能与特定的pretext Task有关

更通用的pretext Task



### 对比学习

对比学习的形式化定义
$$
\text{score}(f(x),f(x^+)) >> \text{score}(f(x),f(x^-))\\
L = -\mathbb{E}_X[\log \frac{\text{exp}(s(f(x),f(x^+)))}{\text{exp}(s(f(x),f(x^+)))+\sum_{j=1}^{N-1}\text{exp}(s(f(x),f(x_j^{-})))}]
$$
InfoNCE Loss(信息噪声对比估计损失)

SimCLR

![image-20241227102140390](https://s2.loli.net/2024/12/27/qxODuN5ywrFYIHA.png)

数据增强形成正样本对

![image-20241227102316049](https://s2.loli.net/2024/12/27/4VsaOlFQUf5ByPS.png)

![image-20241227102332511](https://s2.loli.net/2024/12/27/8YJgB4uP91XFAaK.png)

Moco

![image-20241227103223315](https://s2.loli.net/2024/12/27/NwBTGvP6XacWf48.png)

首先对样本进行数据增强得到x_q, x_k作为一个正样本对，在每一次训练完成后，将这一轮编码好的正样本压入队列中，然后根据动量缓慢更新key编码器的权重（相当于历史q编码器权重的加权和）

## 视觉-语言模型

### 多模态模型

- Vision Transformer(ViT)

  ![image-20241227104434590](https://s2.loli.net/2024/12/27/6rhIMD7bPBoGRmv.png)

### 预训练方式

<font size = 5 font color = red>**全监督训练**</font>

- 将图像映射到与视觉概念相关的离散标签（resnet, alexnet, swin,vit）
- 人工标注很昂贵，标签可能也有限

<font size = 5 font color = red>**图像对比学习**：</font>

<font size = 5 font color = red>**语言-图像对比学习（clip）**</font >从海量文本对中学习，简单的对比学习训练方式，但是数据海量

![image-20241227104753648](https://s2.loli.net/2024/12/27/wgExu9oyVfCS3JY.png)

CLIP的关键设计

- 海量数据
- 模型设计
- 目标函数
  - **推理**：
    - CLIP 可以直接应用于零样本分类任务：
    - 为每个类别生成一组描述文本（如“a photo of a dog”，“a photo of a cat”）。
    - 将图像编码后，计算其表示与这些文本描述的表示的相似度。
    - 选择最相似的类别作为预测结果。

![image-20241227105158149](https://s2.loli.net/2024/12/27/hNipIBcr76lQ9gm.png)<font size = 5>模型设计</font>

**图像端**

FLIP：通过掩遮自编码器（MAE）扩展CLIP的视觉编码器训练

- 训练：依旧使用CLIP Loss, 不使用MIMloss
- 方法：随机掩遮图像区域，只编码可见的图像区域用于训练，从而提升训练效率

**语言端**

K-Lite: 使用外部知识（字典定义）对语言端进行增强

**更多模态**

ImageBind: 使用一个特征空间来连接所有的模态

<font size = 5 font color = red>**掩蔽自编码器**</font>

- MAE: 使用原始图像块作为掩码图像恢复的目标
- MaskFeat: 使用图像特征作为掩码图像块恢复目标

### 通用视觉模型

视觉中的挑战

- 建模：不同类型输入，不同任务粒度，不同类型输出
- 数据：不同类型数据标注数量差异很大

<font size = 5>通用视觉模型的尝试</font>

<font size = 4.5 font color = blue>**closed-set classfication**: </font>alexnet, resnet, vit

<font color = blue font size = 4.5>**open-world recognition**:</font> clip, ALIGN

构建能够识别已知类别并处理未知类别的模型，在开放世界场景中，模型具有处理动态环境的能力

测试数据可能包含训练过程中**未见过**的类别。模型需要区分已知类别和**未知类别**，同时能够动态学习新的类别

<font color = blue font size = 4.5>**specialist Modles**</font>：（专家模型）

是指针对特定任务或领域优化的机器学习模型。这类模型设计的目标是通过聚焦于特定的数据分布和任务需求，**最大化性能**，而非处理广泛的多任务场景。它们通常在单一任务上表现出色，但可能在通用性方面有所局限。

<font size = 4.5 font color = blue>**generalist models**:</font> （通用模型）

是指设计和训练为支持多种任务和领域的机器学习模型。与专注于单一任务的专家模型不同，通用模型追求在多任务、多模态或多领域中的广泛适用性，其核心目标是提供一种具有通用性和迁移能力的解决方案。

<font size = 4.5 font color = blue>**representation learning**：</font>

表征学习的目标是找到一种变换，将高维、复杂、未处理的原始数据（如图像、文本、音频等）转化为低维、结构化、信息丰富的表示形式，使得这些表示可以更好地被机器理解和利用。如自监督学习，监督学习，无监督学习

<font size = 4.5 font color = blue>**promptable interface**：</font>

可提示接口，一种支持通过**提示（prompt）**与模型交互的接口或设计理念，通常用于自然语言处理（NLP）或多模态模型中。它允许用户通过构造提示或指令，灵活地引导模型执行特定任务，而不需要对模型进行额外的训练或调整。如<font color = red> **GPT、CLIP、DALL·E**</font> 等

- **核心特点**
- **自然语言作为接口**：
  - 用户可以用自然语言构造提示，比如 “Translate this to French: hello” 或 “What is the capital of France?”，模型根据提示解析任务并生成答案。
- **零样本或少样本任务泛化**：
  - 模型可以通过提示执行未明确训练过的任务（零样本），或者在少量示例的提示下进行任务泛化（少样本学习）。
- **高灵活性**：
  - Promptable interface 不局限于特定任务，例如分类、生成、翻译、总结等任务都可以通过构造不同的提示实现。

<center><font size = 5,font color = green>Open-world Generalist Interface</font></center>

![image-20241227114528626](https://s2.loli.net/2024/12/27/64pr3Dc5n9d2PlK.png)

### 构建语言和视觉的联系

![image-20241227114728958](https://s2.loli.net/2024/12/27/O9P5moxLpnujJeQ.png)

#### 图像分割

**分割任务**

- Generic segmentation：通用分割，能够处理多种分割任务的分割方法(semantic/instance/panoptic)
  - semaintic: 语义分割，对每一个像素进行分类，不区分同一类别的不同实例
  - instance：实例分割，不仅对像素进行类别标注，还要区分同一类别的不同实例
  - panoptic: 全景分割，试图对所有像素实现完整分割，确保每个像素都属于一个类别，并且如果属于某个实例，能够区分不同的实例。（不仅仅关注物体实例的划分，还要覆盖图像中的所有区域，包括背景，道路等）

- Referring segmentation：指向分割， (根据特定的文本短语分割图像)

**方法**

- clip初始化 v.s 从头训练
- 弱监督训练 v.s 全监督训练
- 两阶段训练 v.s 端到端训练

### 任务粒度

- **图像级:** image recognition, image-text retrieval（图像-文本检索）, image captioning（图像描述生成）, visual question answering, 等等
-  区域级: object detection（目标检测）, dense caption（密集图像描述生成：每个感兴趣的区域生成自然语言描述）, phrase grounding（短语定位：将给定的文本短语（通常是物体的描述）与图像中的相关区域进行对齐，定位出该文本短语所描述的图像区域。）, 等等
- 像素级: generic segmentation, referring segmentation, 等等

####  输出统一

输出统一（+模型统一）：将所有任务的输出都转换为序列 tokens（输出格式相同）

模型统一（输出不统一）：不同粒度的任务共享模型（输出的格式不同）

### 不同的提示

需要构建可提示的接口

- 便于in-context learning: 提供上下文，使得模型能够理解人类意图
- 交互式用户友好：人与人工智能的多轮交互对于复杂任务十分重要