---
layout: post 

title: Auto-Driving
---



# 一些论文以及主要内容

[直接从自然记录的数据中提取流量基元，用于自动驾驶应用](https://ieeexplore.ieee.org/document/8260870)

- **主要研究目的**是如何通过有限数量的原始交通场景生成无线数量的新交通场景

[CommonRoad-CriMe：自动驾驶汽车临界度测量工具箱 |IEEE会议出版物 |IEEE Xplore（IEEE的Xplore）](https://ieeexplore.ieee.org/abstract/document/10186673)
- 一个工具箱,主要用于场景中的关键度度量
- [CommonRoad-克里姆 (tum.de)](https://commonroad.in.tum.de/tools/commonroad-crime): 这里是官方网址

[自动驾驶汽车安全关键变道场景的对抗性生成 |IEEE会议出版物 |IEEE Xplore（IEEE的Xplore）](https://ieeexplore.ieee.org/document/10422684)
- 对抗性生成方法, 但是好像没有开源的项目

![111](/imgs/2024-03-10/ShBYPGdZYRopzNzo.png)

## 56 为自动驾驶汽车的安全测试生成和表征场景
使用的平台[由 Omniverse 提供支持的 DRIVE Sim |英伟达 (nvidia.com)](https://www.nvidia.com/en-us/self-driving-cars/simulation/)
数据集 [Next Generation Simulation (NGSIM) Vehicle Trajectories and Supporting Data | Department of Transportation - Data Portal](https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj/about_data)

   使用 NVIDIA DRIVE Sim [3]，它支持逼真的驾驶模拟和集成 NVIDIA 的 AV 软件堆栈，用于测试感知和规划模块。通过在地图上以随机初始位置和随机速度生成自动驾驶汽车和其他车辆，并为每辆车（包括自动驾驶汽车）分配一个简单但现实的驾驶策略（用于高速公路驾驶）来模拟序列。该策略可以用一个简单的马尔可夫链来描述，其中车辆继续在同一车道上行驶，或进行机动（概率为 0.5）。机动涉及改变车道（概率为 0.6）或速度（概率为 0.4）。一旦开始机动，参与者就会尝试完成它并继续在车道上行驶，除非即将与另一辆车发生碰撞并执行规避机动（制动或转向调整）。
	这里主要是最小化成本函数
$$
\frac{\partial \gamma}{\partial t}=\frac{\partial \gamma}{\partial \mathbf{s}_{\mathbf{AT}}} \frac{\mathrm{d} \mathbf{s}_{\mathbf{A T}}}{\mathrm{d} t}+\frac{\partial \gamma}{\partial \mathbf{s}_{\mathbf{AV}}} \frac{\mathrm{d} \mathbf{s}_{\mathbf{AV}}}{\mathrm{d} t}< 0
	从而最小化自动驾驶车辆与攻击车辆的距离
ps: 没有具体实现代码
$$

## 60 拐角案例生成
仿真平台 HIGHWAY-ENV
基于马尔可夫决策过程建模驾驶环境，并应用深度 强化学习技术学习BVs的行为策略。
参考文献中
[学习碰撞：一种自适应安全关键场景生成方法 |IEEE会议出版物 |IEEE Xplore（IEEE的Xplore）](https://ieeexplore.ieee.org/abstract/document/9340696)
使用carla的文章和我们想做的比较相似,模型使用高斯过程,但是还是没有代码,不过描述的比较有细节

[GitHub - fzi-forschungszentrum-informatik/corner_case_ontology：存储库提供了模板本体和脚本，用于在模拟中创建与 OpenSCENARIO 兼容的极端情况场景。](https://github.com/fzi-forschungszentrum-informatik/corner_case_ontology?tab=readme-ov-file) 两年前的文章和代码, 长的很

## 64 使用自适应压力测试和反向算法在高保真仿真中查找故障
平台仍然是NVIDIA DRIVE Sim
使用马尔可夫过程建模, 提出了一种新的算法 AST 方法，该方法可以减少训练期间发现故障所需的高保真模拟步骤的数量，同时还可以学习克服与保真度相关的故障差异并拒绝虚假错误。故障首先在低保真度中发现，然后用作 BA 的专家演示，以发现高保真度的故障。
ps: 似乎离我们的项目有些远
