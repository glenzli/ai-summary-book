# 第十二章：合成同伦论、谱序列入口与近期形式化

## 本章目标

本章给出 HoTT 中合成同伦论的教材级地图：高阶同伦群、Eilenberg-Mac Lane 型、上同调、谱序列入口和近期 Cubical Agda 形式化结果。这里许多内容属于研究边界，本书不把它们作为基础定理使用。

## 依赖前置知识

本章依赖 HIT、截断、基本群、群和阿贝尔群的集合层定义。许多高级结果需要稳定同伦论和形式化库背景。

## 12.1 高阶 loop 与同伦群

**定义 12.1.** 定义迭代 loop space：
$$
\Omega^0(X,x_0)\coloneqq X,\qquad
\Omega^{n+1}(X,x_0)\coloneqq \Omega(\Omega^n(X,x_0),\mathsf{refl}^{(n)}).
$$

**定义 12.2.** 第 $n$ 个同伦群可定义为
$$
\pi_n(X,x_0)\coloneqq\|\Omega^n(X,x_0)\|_0
$$
并配备由路径复合诱导的群结构。对 $n\ge2$，该群应为阿贝尔群。

**定理 12.3（Eckmann-Hilton）.** 对 $n\ge2$，$\pi_n(X,x_0)$ 的群运算交换。

**证明.** 见附录 AC。二重 loop space 上纵向复合与横向复合有共同单位并满足 interchange law；Eckmann-Hilton 论证推出二重 loop 复合交换，再对 $\Omega^{n-2}(X,x_0)$ 应用该结论并下降到集合截断。$\square$

## 12.2 Eilenberg-Mac Lane 型与上同调

**输入 12.4.** 对阿贝尔群 $G$ 和 $n\ge0$，Eilenberg-Mac Lane 型 $K(G,n)$ 是满足
$$
\pi_n(K(G,n))\cong G
$$
且其他同伦群消失的指称类型。HoTT 中通常通过 HIT 或谱构造实现。本书采用附录 Y.1 的 EM 型塔输入。

**定义 12.5.** 合成上同调可写为
$$
H^n(X;G)\coloneqq\|X\to K(G,n)\|_0
$$
；带基点的约化版本为
$$
\widetilde H^n(X;G)\coloneqq\|X\to_\ast K(G,n)\|_0.
$$

**验证状态。** 定义、阿贝尔群结构、函子性、悬挂同构、球面计算和 cup product 的教材层证明核见附录 Y；Cubical Agda 对照路径见附录 S.4.4。

**定理 12.6（球面上同调，证明核 / 形式化入口）.** 对阿贝尔群 $G$，
$$
\widetilde H^k(\mathbb S^n;G)\cong
\begin{cases}
G,& k=n,\\
0,& k\ne n.
\end{cases}
$$

**证明.** 见附录 Y.12。证明使用约化上同调悬挂同构、$\mathbb S^{n+1}\simeq\Sigma\mathbb S^n$、维数公理和连通球面的约化 $H^0$ 消失。$\square$

## 12.3 近期形式化结果

**事实 12.7（合成上同调形式化，机器形式化 / 研究边界）.** Ljungström、Mörtberg 的 *Computational Synthetic Cohomology Theory in Homotopy Type Theory* 在 Brunerie 等早期合成整上同调工作的基础上，展示了在 Cubical Agda 中形式化合成上同调理论的路线，包括 Eilenberg-Mac Lane 空间、上同调运算、上同调环和计算案例。

**使用边界。** 本书只把该结果作为近期形式化方向的入口。若后续引用其中具体定理，必须核查 Cubical Agda 库路径、论文版本和所用基础。

**事实 12.8（Brunerie 数与高阶同伦计算，机器形式化 / 研究边界）.** HoTT 与 Cubical Agda 社区围绕球面高阶同伦群计算积累了形式化结果。由于版本和库路径变化较快，本书只记录方向，不把具体数值作为未经核查的教材定理。

## 12.4 谱序列与稳定同伦入口

**定义 12.9（谱，纲要）.** 谱可被视为一列基点类型 $E_n$ 与结构映射
$$
E_n\to\Omega E_{n+1}
$$
或等价数据。稳定同伦论在 HoTT 中可用 HIT 和 higher algebra 表达。

**研究边界 12.10.** 谱序列、稳定同伦范畴和高阶代数在 HoTT 中仍依赖活跃的形式化与理论开发。完整教材化需要单独成卷。

## 本章小结

HoTT 的合成同伦论已经远超圆的基本群，包括高阶同伦群、上同调、Eilenberg-Mac Lane 型和稳定方向。但本书把这些内容严格标为高级章节和研究边界，避免把未展开的形式化工程当作基础事实。

## 练习

**练习 12.1.** 写出 $\Omega^2(X,x_0)$ 的两个复合方向，并说明 Eckmann-Hilton 的核心相干。

**练习 12.2.** 解释为什么 $H^n(X;G)$ 的定义需要集合截断。

**练习 12.3.** 查找 Cubical Agda 库中一个 cohomology 相关模块，并记录其依赖。

**练习 12.4.** 说明为什么本章不能作为初学者的基础公理来源。
