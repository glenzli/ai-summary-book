# 序章：范围、严格性和主线

## 本章目标

本章规定本书的对象、严格性等级和主线边界。String theory 同时使用微分几何、二维量子场论、表示论和高能物理语言；若不先固定 convention，很容易把同一个公式在不同归一化下混用。

## 依赖前置知识

需要经典力学、特殊相对论、量子力学、复变函数和基础量子场论。微分几何、Lie algebra、Riemann surface 和 supersymmetry 的技术细节将在附录中补充。

## 0.1 本书的对象

**定义 0.1.** 本书中的 string theory 指以一维扩展对象的量子化为核心的理论框架。其基本扰动论对象包括：

1. 世界面 $\Sigma$；
2. target spacetime $M$；
3. 映射 $X:\Sigma\to M$；
4. worldsheet metric $h_{ab}$；
5. 二维量子场论或共形场论；
6. genus expansion 给出的弦扰动论。

**约定 0.2.** 本书默认从 perturbative string theory 出发。非微扰 dualities、M-theory 和 AdS/CFT 在后半部分作为结构性原则和外部输入处理，不在开篇假设其完整数学存在性。

## 0.2 严格性等级

**定义 0.3.** 本书将陈述分为四类。

1. `P`：正文证明。
2. `S`：标准物理推导说明。
3. `E`：外部输入定理。
4. `C`：物理猜想或对偶性原则。

其中 `S` 只表示在已声明的世界面量子场论、路径积分、正规化或微扰
口径中给出可复核的物理推导，不表示相关对象已经获得完整的严格数学
构造，也不计入 `P` 类书内证明。

**原则 0.4.** 路径积分推导若依赖尚未完全构造的测度，应标为 formal path integral calculation 或外部输入。不能把形式 Gaussian integral 直接当作 Hilbert 空间定理使用。

## 0.3 主线

本书主线如下：

1. 从 point particle 到 string worldsheet。
2. 从 Nambu-Goto action 到 Polyakov action。
3. 从 conformal gauge 到二维 CFT。
4. 从 Virasoro constraints 到物理谱。
5. 从 gauge fixing 到 ghosts 和 BRST cohomology。
6. 从 vertex operators 到 scattering amplitudes。
7. 从 open strings 到 D-branes。
8. 从 worldsheet supersymmetry 到 spacetime supersymmetry。
9. 从 anomaly cancellation 到 consistent superstring theories。
10. 从 compactification 和 duality 到几何与 holography。

## 本章小结

本书不把 string theory 写成单一公式，而写成一条由 worldsheet QFT、CFT、BRST、geometry 和 duality 组成的结构链。每次进入新主线时，都必须声明哪些部分在本书内证明，哪些部分作为标准外部输入。

## 练习

**练习 0.1.** 说明为什么 string perturbation theory 的展开参数不是 $\hbar$，而是 worldsheet topology 或 string coupling。

**练习 0.2.** 举例说明同一个质量公式在不同 $\alpha'$ convention 下如何改变。
