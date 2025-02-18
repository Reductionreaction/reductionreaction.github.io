### 《深度学习平台与应用》作业五（20241221）

#### 1.列举几种常见的激活函数，并简要说明它们的作用。

| 激活函数 | sigmoid                             | tanh                            | relu                              | leaky relu                | elu                                                          | selu                                                         | gelu                  | maxout                      |
| -------- | ----------------------------------- | ------------------------------- | --------------------------------- | ------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | --------------------- | --------------------------- |
|          | $\frac{1}{e^{-x}+1}$                | $\frac{e^x-e^{-x}}{e^x+e^{-x}}$ | $\max{(0,x)}$                     | $\max{(0.01x,x)}$         | $\begin{cases}x,x>0\\ \alpha(\text{exp}(x)-1),x \leq 0\end{cases}$ | $\begin{cases}\lambda x,x>0\\ \lambda\alpha(\text{exp}(x)-1),x \leq 0\end{cases}$ | $x\Phi(x)$            | $\max {(W_1x+b,W_2x+b)}$    |
| **优点** | 将输出控制到0-1之间                 | 输出以0为中心                   | 不会梯度消失，收敛快              | 负数梯度不是0，收敛速度快 | 具有relu的优点，负数部分稳定性强                             | 具有Leakyrelu的优点，适合深度网络，具有自我规范的属性        | 0处可导，平滑易于训练 | 具有leakyrelu的优点，非线性 |
| **缺点** | 输出不能以0为中心，容易出现梯度消失 | 梯度消失                        | 负数梯度为0,输出不是以0为中心的。 | 输出不是以0为中心的       | 计算成本较高                                                 | 计算成本高                                                   | 负数部分较小时梯度为0 | 参数量翻倍                  |

#### 2.什么是反向传播？它的核心思想是什么？

反向传播是一种优化神经网络的训练方法，核心思想是通过梯度下降，通过链式法则，利用损失函数的梯度来更新权重，从而达到最小化损失函数的目的。

#### 3. Numpy 实现反向传播梯度

```python
import numpy as np

# 设置随机种子，方便复现
np.random.seed(42)

# 初始化参数
W1 = np.random.randn(2, 2)  # 输入层到隐藏层的权重
b1 = np.random.randn(2)  # 隐藏层的偏置
W2 = np.random.randn(2)  # 隐藏层到输出层的权重
b2 = np.random.randn()  # 输出层的偏置

# 输入数据和真实标签
x = np.array([1.0, -1.5])
y_true = 0.5

# 学习率
learning_rate = 0.01


# ReLU
def relu(x):
    #todo 输入x 返回relu(x)
    return np.where(x>0,x,0)


# ReLU 的梯度
def relu_derivative(x):
    return np.where(x > 0, 1, 0) 


# 均方损失函数和其导数
def mse_loss(y_pred, y_true):
    #todo 计算均方误差
    return (y_pred-y_true)**2/2


def mse_loss_derivative(y_pred, y_true):
    return y_pred - y_true  # 对 y_pred 求导


# 训练过程
epochs = 100
for epoch in range(epochs):
    # 前向传播
    
    z1 = np.dot(W1, x) + b1  
    h = relu(z1) 
    y_pred = np.dot(W2, h) + b2 

    # 计算损失
    loss = mse_loss(y_pred, y_true)

    # 反向传播
    d_loss_y_pred = mse_loss_derivative(y_pred, y_true)  # 损失对预测值的梯度
    #todo 
    d_loss_W2 =d_loss_y_pred *h  # 对 W2 的梯度
    d_loss_b2 =d_loss_y_pred  #  对 b2 的梯度

    d_loss_h = d_loss_y_pred *W2 # 损失对隐藏层输出 h 的梯度
    d_loss_z1 =d_loss_h*relu_derivative(z1)# 隐藏层的梯度 (通过 ReLU 计算)
    
    d_loss_W1 = np.outer(d_loss_h,x )# 对 W1 的梯度
    
    d_loss_b1 = d_loss_h# 对 b1 的梯度

    # 参数更新 (随机梯度下降)
    W1 -= learning_rate * d_loss_W1
    b1 -= learning_rate * d_loss_b1
    W2 -= learning_rate * d_loss_W2
    b2 -= learning_rate * d_loss_b2

    # 打印训练过程中的损失
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss}")

    # 打印最终的损失值
print(f"Final Loss: {loss}")
```

运行结果如下

```bash
Epoch 0: Loss = 0.025835351192849892
Epoch 10: Loss = 0.0020375628244268906
Epoch 20: Loss = 0.0001564739167320906
Epoch 30: Loss = 1.1920868579827328e-05
Epoch 40: Loss = 9.061439375630744e-07
Epoch 50: Loss = 6.883600510263259e-08
Epoch 60: Loss = 5.228286352452871e-09
Epoch 70: Loss = 3.9708406640054e-10
Epoch 80: Loss = 3.015781510371962e-11
Epoch 90: Loss = 2.2904231367201775e-12
Final Loss: 2.2510213012190517e-13
```

#### 4.什么是全连接网络（FC）或多层感知机（MLP）？它与线性分类器的主要区别是什么？

**全连接网络（FC）或多层感知机（MLP）**，是一种由多个层组成的前馈神经网络。在全连接网络中，除了输入层和输出层之外，每一层的每个神经元都与上一层的所有神经元相连接，并将其输出传递到下一层。具体地，第 l层的神经元的输出可以表示为：
$$
a^{(l)} = \sigma\left(\sum_{i} w_i x_i + b\right)
$$


其中，$w_i$  是权重，$x_i$ 是上一层的输入，b 是偏置，$\sigma$ 是激活函数（如ReLU、Sigmoid、Tanh等）。通过这种结构，网络能够学习到复杂的模式和特征。

与 **线性分类器** 的主要区别在于，线性分类器的决策边界是一个超平面，这意味着它只能解决线性可分的问题。而全连接网络（或MLP）通过在每一层应用非线性激活函数，能够构建非线性的决策边界，从而处理更复杂的分类任务。由于这些非线性变换，全连接网络（或MLP） 能够表示更复杂的模式，因此适用于解决更为复杂的分类问题。

#### 5.在梯度流的基本计算模式中，加法门(add gate)、乘法门(mul gate)和最大值门(max gate)分别具有什么特点？

这里假设只有两个输入$a,b$，上一级的梯度是$\frac{\partial L}{\partial f}$

|        | add gate                        | mul gate                         | max gate                                                     |
| ------ | ------------------------------- | -------------------------------- | ------------------------------------------------------------ |
| grad_a | $\frac{\partial L}{\partial f}$ | $\frac{\partial L}{\partial f}b$ | $\begin{cases}\frac{\partial L}{\partial f},a \geq b\\ 0,a<b\end {cases}$ |
| grad_b | $\frac{\partial L}{\partial f}$ | $\frac{\partial L}{\partial f}a$ | $\begin{cases}\frac{\partial L}{\partial f},b \geq a\\ 0,b<a\end {cases}$ |

