# 附录 E：内部完整性矩阵

本附录说明本书哪些结论已经在正文内部证明，哪些结论被明确划为外部输入。它用于审稿时判断“内部完整”的范围。

## 内部证明闭包

| 主题 | 内部证明状态 |
|---|---|
| Hilbert 空间射线、转移概率 | 已证明 |
| Cauchy-Schwarz 与 Robertson 不确定性 | 已证明 |
| 有界算子伴随、投影分解 | 已证明 |
| 有限维谱定理 | 已证明 |
| 谱测度的一阶矩、算子定义域与二阶矩之区别 | 已证明并给出反例 |
| Born 规则有限维版本和 Luders 更新 | 已证明 |
| 有界 Hamiltonian 的 Schrodinger/Heisenberg 方程 | 已证明 |
| 一维方势阱、匹配条件 | 已推导 |
| 谐振子升降算符、能级、基态 | 已证明 |
| Schmidt 分解、偏迹公式与约化态纯度判别 | 已证明 |
| 自旋 $1/2$、Pauli 代数、Bloch 球 | 已证明 |
| 连续性方程和 Ehrenfest 定理 | 已证明于光滑核心 |
| 有限维非简并扰动分支的局部解析存在性 | 已由块分解与隐函数定理证明 |
| 相互作用图像、Dyson 级数、酉性、复合律及一阶截断余项 | 已证明于有限时间范数连续有界情形 |
| 自由传播子与 Lie--Trotter 单步分布核 | 已由 Fourier 变换和核复合证明 |
| Klein-Gordon 流守恒及其非正定性 | 已对光滑衰减解证明 |
| 长时间 sinc 核趋于能量 delta 分布 | 已在 Schwartz 测试函数上证明 |
| 两自旋 $1/2$ 的 singlet/triplet 分解 | 已证明 |
| 中心势径向方程和规范协变性 | 已证明 |
| 有限维 Naimark 扩张与一般量子仪器口径 | 扩张已构造，仪器已精确定义 |
| 等先验二元态区分的 Helstrom 公式 | 已由差算子的正负谱分解证明 |
| 氢型基态归一化与本征方程 | 已逐项计算；完整 Coulomb 谱仍为外部输入 |
| Gaussian 波包、自旋进动、Rabi 振荡 | 已完整计算 |

## 外部输入闭包

| 主题 | 外部输入原因 |
|---|---|
| 无界自伴算子的谱定理 | 需要完整泛函分析 |
| Stone 定理 | 需要强连续酉群生成元理论 |
| Kato-Rellich 与磁 Schrodinger 自伴性 | 需要无界扰动和二次型理论 |
| Stone-von Neumann 定理 | 需要 Weyl 表示理论 |
| Wigner 定理 | 需要射线几何和半线性算子理论 |
| 散射渐近完备性 | 需要深层谱与传播估计 |
| 绝热定理误差估计 | 需要投影族微分与谱隙分析 |
| 一般 Kraus/Stinespring 表示与 Lindblad 定理 | 需要算子代数和完全正映射理论 |
| 球谐与 Coulomb 完备性 | 需要 Sturm-Liouville 与特殊函数谱理论 |
| Wigner-Eckart 定理 | 需要 $\mathfrak{su}(2)$ 表示论 |
| Fourier sine 与 Hermite 完备性 | 需要 Sturm-Liouville/Fourier-Hermite 理论 |
| WKB 转折点连接公式 | 需要 Airy 模型和渐近匹配 |
| 光学定理与 partial wave 展开 | 需要连续谱散射归一化和球函数渐近分析 |
| Lindblad 与 Uhlmann 定理 | 需要完全正半群和纯化几何 |
| Kato 解析扰动理论 | 需要闭算子解析族理论 |

## 使用原则

正文中凡依赖第二表的结果，必须以“外部输入定理”或“说明/边界”标出。若后续要把某个外部输入内化为正文证明，应同步更新本附录、[THEOREM_DEPENDENCIES.md](THEOREM_DEPENDENCIES.md) 和 [D_external_theorem_index.md](D_external_theorem_index.md)。
