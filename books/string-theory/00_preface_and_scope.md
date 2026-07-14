# 序章：对象、约定与阅读路径

同一个弦论公式可能同时被读成二维场论的 Ward identity、时空粒子的质量壳条件，
或某个低能有效作用的近似；若省略世界面 signature、target metric、正规序方案与
证明状态，这几种读法很快就会互相混淆。本书从映射 $X:\Sigma\to M$ 及其世界面
作用量出发，依次经过共形场论、量子约束、散射振幅、超弦、紧化和对偶性，并在每次
跨越数学定理、标准物理推导与非微扰猜想的边界时明确所用假设。读者应熟悉经典力学、
特殊相对论、量子力学、复变函数和基础量子场论；微分几何、Lie algebra、Riemann
surface 与 supersymmetry 的记号可随正文查阅附录。

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

## 0.3 从世界面到时空的推进

全书的论证次序由下列问题连接起来：

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

因此，贯穿后文的不是一条孤立公式，而是从 worldsheet QFT 到 CFT、BRST、
geometry 与 duality 的连续结构。一个结论只有在对象、归一化和适用层级都已固定后
才进入后续推导；需要大型外部理论时，正文会给出足以调用的精确版本，而不会把证明
路线写成已经完成的证明。

## 练习

**练习 0.1.** 说明为什么 string perturbation theory 的展开参数不是 $\hbar$，而是 worldsheet topology 或 string coupling。

**练习 0.2.** 举例说明同一个质量公式在不同 $\alpha'$ convention 下如何改变。
