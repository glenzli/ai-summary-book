# 第九章：椭圆曲线、复环面与 SYZ 的第一模型

椭圆曲线是少数能把镜像字典的三层结构同时画出来的例子：向量丛的 rank/degree 变成环面上圆的同调斜率，Ext 的 Euler 特征变成交数，而 theta 函数的乘法系数来自平坦环面中的三角形面积。只匹配第一层还只是对象命名，只匹配前两层仍未看到复合；第三层才触及范畴结构。本章在前八章的增强语言中逐层完成这些计算，并把完整 Polishchuk--Zaslow 等价诚实地保留为外部输入。所需几何背景仅是二维环面、椭圆曲线和线丛次数。

## 9.1 镜像数据

**定义 9.1.** B-side 椭圆曲线是光滑一维 proper genus-one $k$-variety $E$ 连同基点。其 B-side category 取
$$
\operatorname{Perf}(E)\simeq \mathrm D^b\operatorname{Coh}(E)
$$
的 dg enhancement。

**定义 9.2.** A-side 镜像取辛二维环面
$$
T^2=\mathbb R^2/\mathbb Z^2
$$
带面积形式 $\omega=A\,dx\wedge dy$（$A>0$），对象为带 grading、spin structure 和局部系统的有向直线斜率 Lagrangian circles。这里用 $A$ 表示实辛面积，避免与 B-side 椭圆曲线复模参数 $\tau$ 混淆。

**HMS 数据包 9.3.** 椭圆曲线 HMS 的标准形态为
$$
D^\pi\mathcal F(T^2)\simeq \operatorname{Perf}(E)
$$
或在更精细版本中给出适当 $A_\infty$ enhancement 的 quasi-equivalence。复结构参数、复化 Kähler 参数和 $B$-field 之间的匹配属于镜像映射的一部分。

## 9.2 对象字典

**定义 9.4.** 对互素整数 $(r,d)$，记 $L_{r,d}\subset T^2$ 为斜率 $d/r$ 的嵌入圆；若 $r=0$ 则解释为竖直圆。给它配备 rank-one local system $\xi$ 后得到 brane $(L_{r,d},\xi)$。

**例 9.5.** HMS 字典把 stable vector bundle 的 rank 和 degree 与 Lagrangian circle 的同调类对应：
$$
\operatorname{rank}=r,\qquad \deg=d,\qquad [L_{r,d}]=r[a]+d[b]\in H_1(T^2,\mathbb Z).
$$

**警告 9.6.** 例 9.5 是字典的一部分，不是独立证明。要证明范畴等价，还需比较 morphisms、composition 和高阶 $A_\infty$ 结构。

## 9.3 Morphism spaces

**命题 9.7.** 若两条直线斜率 Lagrangians $L_{r,d}$ 与 $L_{r',d'}$ 横截，则其交点数为
$$
|rd'-r'd|.
$$

**证明.** 两条嵌入圆代表 $H_1(T^2,\mathbb Z)$ 中的 primitive classes $(r,d)$ 与 $(r',d')$。代数交数为行列式
$$
\det\begin{pmatrix}r&r'\\ d&d'\end{pmatrix}=rd'-r'd.
$$
横截位置下几何交点数为该 determinant 的绝对值。证毕。

**命题 9.8.** 在横截、无 disk bubbling 且取平直代表元的二维环面模型中，
$$
\dim HF^\ast(L_{r,d},L_{r',d'})=|rd'-r'd|
$$
在适当 grading 下成立。

**证明.** 由命题 9.7，Floer cochain complex 有 $|rd'-r'd|$ 个生成元。
两条平直圆的所有横截交点具有相同的局部交号；选定两条 branes 的 grading
后，它们因而具有相同的模 $2$ 次数。Floer 微分次数为 $1$，故在这个
$\mathbb Z/2$-分次 complex 上必为零。无 disk bubbling 假设保证不存在改变
该链模型的 obstruction 项，于是 $HF^\ast=CF^\ast$，维数等于生成元数。
若改用非平直 Hamiltonian 扰动，则还需调用定理 3.18 的 continuation
quasi-isomorphism，结论仍保持。证毕。

**解释 9.9.** B-side 上，stable bundles $V_{r,d}$、$V_{r',d'}$ 的 Ext 维数由 Riemann-Roch 和稳定性控制：
$$
\chi(V_{r,d},V_{r',d'})=rd'-r'd
$$
到符号约定。命题 9.7 是 A-side 的数值匹配。

## 9.4 三角形计数与 theta 函数

**定义 9.10.** 三条横截 Lagrangian circles $L_0,L_1,L_2$ 的乘法
$$
\mu^2:CF^\ast(L_1,L_2)\otimes CF^\ast(L_0,L_1)\to CF^\ast(L_0,L_2)
$$
由以三条圆为边界的 holomorphic triangles 计数给出。

**解释 9.11.** 在平坦环面上，holomorphic triangles 可提升到 $\mathbb R^2$ 中的仿射三角形。其面积给出 Novikov 权重，按所有 lift 求和得到 theta series。这与 B-side 椭圆曲线上的 theta 函数乘法相匹配。

**外部输入定理 9.12（Polishchuk-Zaslow 椭圆曲线 HMS）.** 椭圆曲线与镜像二维环面之间存在 HMS 等价，且对象、morphisms 和乘法由斜率 Lagrangians、stable bundles 与 theta 函数乘法对应。
来源：Polishchuk-Zaslow, *Categorical Mirror Symmetry: The Elliptic Curve*；后续由 Kreussler 和 Polishchuk 等工作补强不同增强和横截限制。

## 9.5 SYZ 视角

**定义 9.13.** SYZ 口径把镜像对看作 dual torus fibrations。对二维环面，投影
$$
T^2\to S^1
$$
的 fibers 是 circles，镜像由取对偶局部系统参数得到。

**解释 9.14.** 在椭圆曲线例子中，Lagrangian section 对应 line bundle，fiber 上的 local system 参数对应点或平移数据。这个模型是高维 SYZ 的最低维影子。

行列式 $rd'-r'd$ 同时控制 B-side Euler 特征与 A-side 有向交数，平坦环面中三角形的 Novikov 和又与 theta 乘法相接，这三层匹配解释了椭圆曲线 HMS 为何远强于一张对象表。外部输入定理 9.12 承担把这些局部计算组织成增强等价的责任；SYZ 对偶圆纤维则提示，在高维情形中对象字典会受到奇异纤维与圆盘修正的干扰。

## 练习

**练习 9.1.** 计算 $L_{2,1}$ 与 $L_{3,5}$ 在 $T^2$ 中的交点数。

**练习 9.2.** 用 Riemann-Roch 写出两个向量丛之间 Euler characteristic 的表达式，并与 determinant 公式比较。

**练习 9.3.** 在 $\mathbb R^2$ 中画出三条不同斜率直线围成的三角形，并说明其投影如何贡献 $\mu^2$。

**练习 9.4.** 解释 SYZ dual torus fibration 在二维环面中的具体含义。
