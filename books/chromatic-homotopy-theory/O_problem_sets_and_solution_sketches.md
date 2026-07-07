# 附录 O：综合习题与解题提示

本附录为正式教材配套习题。每组题都要求使用正文定义，而不是只复述术语。解题提示不替代完整证明，但指出应调用的定义、命题和外部输入。

## O.1 稳定局部化

**习题 O.1.** 设 $E$ 为谱。证明 $E$-acyclic 谱构成 localizing tensor ideal。

**提示.** 使用 $E\otimes-$ 保持 colimits 和 cofiber 序列；tensor ideal 性用结合律
$$
E\otimes(X\otimes Y)\simeq(E\otimes X)\otimes Y.
$$

**习题 O.2.** 设 $X\to Y$ 是 $E$-equivalence，即其 cofiber 为 $E$-acyclic。证明对任意 $E$-local 谱 $Z$，映射
$$
F(Y,Z)\to F(X,Z)
$$
是等价。

**提示.** 令 $C=\operatorname{cofib}(X\to Y)$。对 fiber/cofiber 序列取 $F(-,Z)$，再用 $F(C,Z)\simeq0$。

## O.2 形式群和 Landweber exactness

**习题 O.3.** 对乘法形式群 $F_m(X,Y)=X+Y+XY$，直接计算 $[3]_{F_m}(X)$。

**提示.** 使用
$$
1+\operatorname{series}_m^{F_m}(X)=(1+X)^m.
$$
所以三重级数为 $3X+3X^2+X^3$。

**习题 O.4.** 说明为什么 $E(n)_*$ 满足 Landweber exactness 中 $v_n$ 单射条件。

**提示.** 在 $E(n)_*$ 中 $v_n$ 已经可逆；可逆元乘法自动为同构，因而为单射。

## O.3 Morava K/E 与 type

**习题 O.5.** 证明若 $K(n)_*X=0$，则 $K(n)_*(X\otimes Y)=0$。

**提示.** 调用 Morava Kunneth 外部输入：
$$
K(n)_*(X\otimes Y)\cong K(n)_*X\otimes_{K(n)_*}K(n)_*Y.
$$

**习题 O.6.** 证明球谱 $\mathbb S_{(p)}$ 为 type $0$，Moore spectrum $M(p)$ 不是 type $0$。

**提示.** 对球谱计算 $H\mathbb Q_*$；对 $M(p)$ 使用 cofiber 序列
$$
\mathbb S_{(p)}\xrightarrow{p}\mathbb S_{(p)}\to M(p)
$$
并张量 $H\mathbb Q$。

## O.4 Chromatic tower

**习题 O.7.** 设 fracture square 对 $X$ 成立。证明若右下角为零，则 $L_nX\simeq L_{n-1}X\times L_{K(n)}X$。

**提示.** 在稳定 infinity-范畴中，pullback over terminal object is product；零谱既是初对象也是终对象。

**习题 O.8.** 证明若 $X$ 有理化为零，高度 $1$ fracture square 中仍不能推出 $L_1X\simeq L_{K(1)}X$。

**提示.** 写出 square：
$$
\begin{CD}
L_1X @>>> L_{K(1)}X\\
@VVV @VVV\\
0 @>>> L_0L_{K(1)}X.
\end{CD}
$$
左上角是右上角到右下角的 fiber。

## O.5 Morava descent

**习题 O.9.** 解释为什么 $\mathbb G_n$ 的 cohomology 必须写作 $H_c^s$。

**提示.** $\mathbb G_n$ 是 profinite group，Morava module 带拓扑和连续 semilinear action。普通离散 cochains 忘记拓扑。

**习题 O.10.** 从
$$
L_{K(n)}X\simeq(E_n\otimes X)^{h\mathbb G_n}
$$
推出 Morava descent spectral sequence 的 abutment。

**提示.** 对连续 homotopy fixed point spectral sequence 写
$$
H_c^s(\mathbb G_n;\pi_t(E_n\otimes X))\Rightarrow\pi_{t-s}(E_n\otimes X)^{h\mathbb G_n}.
$$

## O.6 Telescope 和 redshift

**习题 O.11.** 给出一个陈述，说明 $T(n)$ 和 $K(n)$ 混用会导致什么错误。

**提示.** 参考第七章。正确答案应包含“2023 年后高度至少 $2$ 不可默认相同”。

**习题 O.12.** 将“$K$-theory raises height by one”改写成可检查定理模板。

**提示.** 必须指定 $R$ 的 ring structure、高度定义、$K(R)$ 的版本、局部化类型和结论类型。

## O.7 tmf 和高度二

**习题 O.13.** 区分 weak elliptic cohomology datum、sheaf of $\mathbb E_\infty$-rings 和 $TMF$。

**提示.** 使用第八章 8.14。答案应说明第 1 层不自动推出第 2 层。

**习题 O.14.** 说明为什么 supersingular locus 控制 $K(2)$-local tmf。

**提示.** 椭圆曲线形式群高度只有 $1$ 或 $2$；$K(2)$ 只看高度 $2$ 信息。

## O.8 Semiadditivity

**习题 O.15.** 证明有限群 $G$ 下 norm map 为等价当且仅当 Tate object 为零。

**提示.** Tate object 定义为 norm map 的 cofiber。

**习题 O.16.** 在 0-semiadditive 范畴中计算三点集合的 cardinality。

**提示.** 常值图的 colimit 和 limit 都是三重 biproduct；comparison 给出 $3\cdot\operatorname{id}_{\mathbbm 1}$。

## O.9 Picard 和 Gross-Hopkins

**习题 O.17.** 若 $X$ 是 $K(n)$-local invertible spectrum，说明为什么 $(E_n)_*X$ 应是 invertible Morava module。

**提示.** 使用 $X\otimes Y\simeq\mathbb S_{K(n)}$，再张量 $E_n$ 并调用 Kunneth/完备性外部输入。

**习题 O.18.** 写出 Gross-Hopkins duality 公式模板，并指出每一项需要的 convention。

**提示.** 使用附录 L：
$$
I_n^{GH}\simeq\Sigma^aS\langle\det\rangle\otimes P.
$$

## O.10 Equivariant 和 motivic

**习题 O.19.** 解释 genuine $G$-spectra 与 naive $G$-spectra 的区别。

**提示.** 答案必须提到 representation spheres、geometric fixed points 或 transfers。

**习题 O.20.** 说明 complex realization 不保守会给 motivic chromatic theory 带来什么风险。

**提示.** 若 realization 后为零不能推出 motivic 对象为零，则不能把拓扑检测结果反向搬回 motivic category。

## O.11 Adams-Novikov 计算

**习题 O.21.** 解释为什么 Adams-Novikov $E_2$ 页是 Hopf algebroid comodule Ext。

**提示.** $BP_*BP$ 编码 $BP$ cooperations；$BP_*X$ 是 comodule，不只是 $BP_*$-module。

**习题 O.22.** 给出两个 filtered abelian groups，它们的 associated graded 都是 $\mathbb Z/2\oplus\mathbb Z/2$，但总群不同。

**提示.** 比较 $\mathbb Z/4$ 与 $\mathbb Z/2\oplus\mathbb Z/2$。这就是 hidden additive extension 的最小模型。

## 本附录小结

这些习题覆盖本书目前的主要定义链。若读者能完成 O.1-O.22，则说明已经掌握本书基础层和前沿接口的严格使用方式；若要进入计算层，还必须补充 Ravenel 和现代稳定 stems 表格。
