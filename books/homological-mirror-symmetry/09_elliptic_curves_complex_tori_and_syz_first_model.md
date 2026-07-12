# 第九章：椭圆曲线、复环面与 SYZ 的第一模型

## 本章目标

本章把 HMS 的第一个标准例子写成可检查的数据包：椭圆曲线与其镜像二维环面。重点是对象字典、morphism 计算、斜率与次数、三角形计数和 theta 函数的作用。完整证明作为外部输入登记。

## 依赖前置知识

需要前八章的增强范畴和 Fukaya 范畴语言。需要知道椭圆曲线、线丛的次数和二维环面的基本拓扑。

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

**命题 9.8.** 在横截且无 disk bubbling 的二维 exact-local 模型中，
$$
\dim HF^\ast(L_{r,d},L_{r',d'})=|rd'-r'd|
$$
在适当 grading 下成立。

**证明路线（外部输入）.** Floer cochain 由命题 9.7 的交点生成。二维直线环面模型中可选择使差分消失或按 grading 分离的情形；更一般情形需计算 strips 并使用 continuation。完整陈述依赖椭圆曲线 HMS 文献中的模型选择。证毕。

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

## 本章小结

椭圆曲线 HMS 是最早可计算的标准例子。它把 stable bundles 的 rank/degree 与 Lagrangian circles 的斜率联系起来，把 Ext 维数与交点数联系起来，把 theta 函数乘法与 holomorphic triangle 计数联系起来。完整等价仍是外部输入定理，但本章给出了其数据包和必要计算。

## 练习

**练习 9.1.** 计算 $L_{2,1}$ 与 $L_{3,5}$ 在 $T^2$ 中的交点数。

**练习 9.2.** 用 Riemann-Roch 写出两个向量丛之间 Euler characteristic 的表达式，并与 determinant 公式比较。

**练习 9.3.** 在 $\mathbb R^2$ 中画出三条不同斜率直线围成的三角形，并说明其投影如何贡献 $\mu^2$。

**练习 9.4.** 解释 SYZ dual torus fibration 在二维环面中的具体含义。
