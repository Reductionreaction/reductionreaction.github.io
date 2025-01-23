---
layout: post
title: deep-learning
date: 2024-09-11
---

[cs231n](https://cs231n.stanford.edu/schedule.html)

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
distances = np.sum(np.abs(self.Xtr-X[i,:]),axis=1) #这是1-nn算法
```

时间复杂度：训练$O(1)$,预测$O(M*N*D+M*Nlogk)$

问题：k个邻居属于k个类，超参数（或模型）的选择，距离度量$L_1,L_2$

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
多分类SVM loss：

![image-20241219230433413](https://s2.loli.net/2024/12/19/MRaTWDYpHkf7SlQ.png)

1是松弛条件，可以换成别的。



## Transformer

self-attention：依赖一个输入

cross-attention: 依赖两个输入

CNN -> Attention -> CNN -> predict -> (loss)  

## 自监督学习

pretext Task

Downstream Task

如何进行评估？

- pretext性能：但是应用范围比较窄，重点还是在下游任务的性能上
- **特征质量**（in-domain上的性能--同分布的数据集）：评测时把自监督学习好的模型作为特征提取器，训练一层线性分类器，根据线性分类器的效果来评价自监督学习性能
- **鲁棒性和泛化性**（在不同数据集上的效果）
- 计算效率：训练时间和训练所需的资源
- 迁移学习和下游任务性能

