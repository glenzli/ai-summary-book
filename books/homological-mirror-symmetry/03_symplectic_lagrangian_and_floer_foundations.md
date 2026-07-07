# 第三章：辛流形、Lagrangian brane 与 exact Floer 口径

## 本章目标

本章建立 HMS 的 A-side 基础对象：辛流形、Lagrangian submanifolds、brane data、exact Floer cochains 和 analytic inputs 的边界。完整 Fukaya category 的构造将在后续章节展开；本章只建立可检查的 exact 入口。

## 依赖前置知识

需要熟悉光滑流形、微分形式、切丛、向量丛、基本同调代数和第一章的 $A_\infty$ 语言。

## 3.1 辛流形与 Lagrangian

**定义 3.1.** 一个辛流形（symplectic manifold）是偶维光滑流形 $M$ 和闭的非退化 $2$-形式 $\omega\in\Omega^2(M)$ 的二元组 $(M,\omega)$。非退化指映射
$$
T_pM\to T_p^\vee M,\qquad v\mapsto \omega_p(v,-)
$$
对每个 $p\in M$ 都是线性同构。

**定义 3.2.** 子流形 $L\subset M$ 称为 Lagrangian，若
$$
\omega|_L=0,\qquad \dim L=\frac12\dim M.
$$

**命题 3.3.** 若 $L\subset(M,\omega)$ 是 Lagrangian，则 $T_pL$ 是辛向量空间 $(T_pM,\omega_p)$ 中的极大 isotropic 子空间。

**证明.** 条件 $\omega|_L=0$ 表示 $\omega_p$ 在 $T_pL$ 上恒为零，所以 $T_pL$ isotropic。任意 $2n$ 维辛向量空间中的 isotropic 子空间维数不超过 $n$：因为 $W\subset W^\omega$，而非退化性给出 $\dim W+\dim W^\omega=2n$，故 $\dim W\le n$。这里 $\dim T_pL=n$，所以 $T_pL$ 是极大 isotropic。证毕。

**定义 3.4.** exact symplectic manifold 是 $(M,\omega)$ 连同 $1$-形式 $\lambda$，满足 $\omega=d\lambda$。Lagrangian $L\subset M$ 称为 exact，若存在函数 $f_L:L\to k$ 或 $\mathbb R$ 语境中的实值函数，使得
$$
\lambda|_L=df_L.
$$

**例 3.5.** 余切丛 $T^\ast Q$ 带 tautological $1$-形式 $\lambda_{\mathrm{can}}$，其微分 $d\lambda_{\mathrm{can}}$ 给出标准 exact symplectic structure。零截面 $Q\subset T^\ast Q$ 是 exact Lagrangian，因为 $\lambda_{\mathrm{can}}$ 在零截面上为零。

## 3.2 Brane data

只给出 Lagrangian 子流形不足以定义分次 Floer cochains。还需要 orientation、grading、spin/Pin 结构和局部系统等数据。

**定义 3.6.** 在本书的 exact Calabi-Yau 入口中，一个 Lagrangian brane 是数据
$$
\mathbb L=(L,\alpha_L,\mathfrak s_L,E_L)
$$
其中：

1. $L\subset M$ 是 exact Lagrangian；
2. $\alpha_L$ 是相对于固定 Calabi-Yau volume form 或 Maslov cover 的 grading；
3. $\mathfrak s_L$ 是用于 orientation lines 的 spin 或 Pin 结构；
4. $E_L$ 是有限秩 $k$-局部系统。

**警告 3.7.** 不同文献对 brane data 的最低要求不同。若 $M$ 不 exact、$L$ 不 exact、Maslov class 不消失，或采用 monotone/Novikov/obstructed 口径，则必须修改定义 3.6。本书后续章节会在进入一般情形前重新声明假设。

## 3.3 横截交与 Floer cochains

**定义 3.8.** 两个 Lagrangians $L_0,L_1\subset M$ 横截相交，若对每个 $p\in L_0\cap L_1$，
$$
T_pL_0+T_pL_1=T_pM.
$$

**引理 3.9.** 若 $L_0,L_1$ 是紧子流形且横截相交，则 $L_0\cap L_1$ 是有限集合。

**证明.** 横截性推出 $L_0\cap L_1$ 是维数
$$
\dim L_0+\dim L_1-\dim M=0
$$
的光滑子流形，因此是离散集合。它又是紧集 $L_0\cap L_1$，因为 $L_0,L_1$ 在 Hausdorff 流形中闭且紧。紧离散空间有限。证毕。

**定义 3.10.** 设 $\mathbb L_0,\mathbb L_1$ 是横截 exact branes。Floer cochain space 定义为
$$
CF^\ast(\mathbb L_0,\mathbb L_1)
=\bigoplus_{p\in L_0\cap L_1}
\operatorname{Hom}_k((E_{L_0})_p,(E_{L_1})_p)\otimes o_p,
$$
其中 $o_p$ 是由 grading 与 orientation data 决定的 orientation line，次数由 Maslov index 给出。

**解释 3.11.** 若忽略局部系统和 orientation lines，定义 3.10 只是把相交点生成的向量空间分次化。严格 Floer theory 的难点不在生成元，而在微分、乘法和高阶复合由 holomorphic curves 的计数给出，并且这些计数必须有 compactness、orientation 和 transversality 支撑。

## 3.4 Floer 微分的 exact 版本

**定义 3.12.** 取与 $\omega$ 相容的 almost complex structure $J$。从 $p\in L_0\cap L_1$ 到 $q\in L_0\cap L_1$ 的 Floer strip 是映射
$$
u:\mathbb R\times[0,1]\to M
$$
满足
$$
\partial_su+J(u)\partial_tu=0,\qquad
u(s,0)\in L_0,\quad u(s,1)\in L_1,
$$
以及渐近条件
$$
\lim_{s\to-\infty}u(s,t)=p,\qquad
\lim_{s\to+\infty}u(s,t)=q.
$$

**定义 3.13.** 若相关 moduli spaces 已经完成 regularization、orientation 和 compactification，则 Floer 微分定义为
$$
\mu^1(p)=\sum_q \#\mathcal M^0(p,q)\,q,
$$
其中 $\mathcal M^0(p,q)$ 是除去 $\mathbb R$-平移后维数为 $0$ 的 Floer strips 模空间。

**外部输入定理 3.14（exact Floer 微分）.** 在 exact、横截、紧性和 regularity 假设下，$\mu^1$ 满足
$$
(\mu^1)^2=0.
$$

**证明草图.** 一维 Floer strip 模空间的紧化边界由两层 broken strips 组成。带符号计数边界为零，而边界点的代数和正是 $(\mu^1)^2$ 的系数。exactness 排除非平凡 disk bubbling，因为能量由 action 差控制，边界 bubbling 会违反 exact Lagrangian 的面积恒等式。完整证明需要 Gromov compactness、gluing、orientation 和 transversality。证毕。

**警告 3.15.** 定理 3.14 是外部输入。若没有 exactness 或 monotonicity，disk bubbling 可能产生 obstruction term $\mu^0$，此时 $\mu^1$ 不一定平方为零，必须引入 bounding cochains 或 curved $A_\infty$ 结构。

## 3.5 Hamiltonian 扰动与不变量性

**定义 3.16.** Hamiltonian 函数 $H:M\to\mathbb R$ 定义 Hamiltonian vector field $X_H$：
$$
\omega(X_H,-)=dH.
$$
其流若存在，记为 $\varphi_H^t$。

**定义 3.17.** 两个 branes 的 Floer cochains 在非横截时通常通过 Hamiltonian perturbation 定义：
$$
CF^\ast(L_0,L_1):=CF^\ast(\varphi_H^1(L_0),L_1),
$$
其中 $H$ 选择到使交点横截。

**外部输入定理 3.18（continuation invariance）.** 合适 exact 假设下，不同 Hamiltonian perturbations 和 almost complex structures 得到的 Floer complexes 在同伦范畴中 quasi-isomorphic。

**解释 3.19.** continuation invariance 是 Fukaya category 成为几何不变量的基础。没有它，Floer complex 只是依赖辅助选择的链复形，不能作为 HMS 的 A-side。

## 3.6 从 Floer cochains 到 Fukaya category

本章只定义了对象和 morphism complexes 的入口。Fukaya category 还需要高阶复合
$$
\mu^d:
CF^\ast(L_{d-1},L_d)\otimes\cdots\otimes CF^\ast(L_0,L_1)
\to CF^\ast(L_0,L_d)[2-d],
$$
这些复合由 holomorphic $(d+1)$-gons 的零维模空间计数定义。其 $A_\infty$ 方程来自一维 polygon 模空间的边界分解。第四章将把这个构造写成定理链，并明确列出 analytic inputs。

## 本章小结

A-side 的最低入口是带 brane data 的 Lagrangian，而不是裸 Lagrangian 子流形。Floer cochains 由横截交点生成，Floer 微分由 holomorphic strips 的计数给出。$\mu^1{}^2=0$、不变量性和高阶 $A_\infty$ 方程都依赖深层分析输入；本书在未建立分析基础时必须把它们标为外部输入。

## 练习

**练习 3.1.** 证明辛向量空间中 isotropic 子空间的维数不超过总维数的一半。

**练习 3.2.** 在 $T^\ast S^1$ 中写出零截面和一个横截 Hamiltonian 扰动的交点，并描述 Floer cochain 的生成元。

**练习 3.3.** 用 exactness 计算 holomorphic strip 的能量，并说明为什么 action 差控制能量。

**练习 3.4.** 解释若存在 disk bubbling，为什么 $\mu^1{}^2=0$ 的边界分解会失败。
