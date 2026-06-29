# 附录 BN：Steenrod 代数、Ext 与 Adams 计算接口

附录 AV 给出 Adams 谱序列模板；本附录补上其代数输入：Steenrod operations、Steenrod algebra、$\mathcal A$-module、Ext 以及低维 Adams 类的解释。这里的具体计算主要是经典稳定同伦论外部输入；本书给出 HoTT/谱接口和可审查的代数形态。

## BN.1 Mod 2 cohomology operations

**输入 BN.1（Steenrod squares）。** 对每个 $i\ge0$，存在自然上同调运算
$$
\mathsf{Sq}^i:H^n(X;\mathbb F_2)\to H^{n+i}(X;\mathbb F_2).
$$
它们满足：

1.  $\mathsf{Sq}^0=\mathsf{id}$；
2.  自然性；
3.  Cartan formula
    $$
    \mathsf{Sq}^k(x\smile y)=\sum_{i+j=k}\mathsf{Sq}^i(x)\smile\mathsf{Sq}^j(y);
    $$
4.  instability：若 $i>n$ 则 $\mathsf{Sq}^i(x)=0$，且 $\mathsf{Sq}^n(x)=x^2$；
5.  Adem relations。

**验证状态.** 对 HoTT 而言，这是 cohomology operations 的高级输入。附录 AM 给出 cup product 的几何来源；完整 Steenrod operation 构造需要更高相干的 EM 型乘法和 diagonal。

## BN.2 Steenrod algebra

**定义 BN.2（$\mathcal A_2$）。** Mod 2 Steenrod algebra $\mathcal A_2$ 是由 $\mathsf{Sq}^i$ 生成并模 Adem relations 的分次代数。乘法为 operations 的复合。

**定义 BN.3（unstable module）。** 一个 unstable $\mathcal A_2$-module 是分次 $\mathbb F_2$-向量空间 $M^\ast$，带 $\mathsf{Sq}^i:M^n\to M^{n+i}$ 作用，满足 Adem、$\mathsf{Sq}^0=1$ 和 instability 条件。

**命题 BN.4（cohomology is unstable module，证明架构）。** 对类型 $X$，$H^\ast(X;\mathbb F_2)$ 是 unstable $\mathcal A_2$-module。

**证明架构.** 运算由 BN.1 给出；自然性、Adem 和 instability 是输入公理或高级构造定理；加法相容由 cohomology operation 的定义给出。

## BN.3 Odd primes

**输入 BN.5（odd primary operations）。** 对奇素数 $p$，Steenrod algebra $\mathcal A_p$ 由 Bockstein $\beta$ 和 reduced powers
$$
\mathcal P^i:H^n(X;\mathbb F_p)\to H^{n+2i(p-1)}(X;\mathbb F_p)
$$
生成，并满足对应 Adem relations、Cartan formula 和 instability 条件。

**使用边界.** 本书默认在 Adams 低维例子中采用 $p=2$。奇素数版本仅登记接口。

## BN.4 Ext groups

**定义 BN.6（graded Ext）。** 对 $\mathcal A_p$-modules $M,N$，定义
$$
\mathsf{Ext}^{s,t}_{\mathcal A_p}(M,N)
$$
为 graded module category 中 Hom 的右导出函子，其中 $s$ 是分辨率次数，$t$ 是内部次数。

**定义 BN.7（Adams $E_2$ page）。** 对 spectrum $X$，
$$
E_2^{s,t}(X)\coloneqq
\mathsf{Ext}^{s,t}_{\mathcal A_p}(H^\ast(X;\mathbb F_p),\mathbb F_p).
$$
对于 sphere spectrum $\mathbb S$，
$$
H^\ast(\mathbb S;\mathbb F_p)\cong\mathbb F_p
$$
集中在零次，因此
$$
E_2^{s,t}(\mathbb S)=\mathsf{Ext}^{s,t}_{\mathcal A_p}(\mathbb F_p,\mathbb F_p).
$$

## BN.5 Adams convergence

**定理 BN.8（classical Adams spectral sequence，外部输入）。** 对适当 connective finite type spectrum $X$，存在谱序列
$$
E_2^{s,t}=
\mathsf{Ext}^{s,t}_{\mathcal A_p}(H^\ast(X;\mathbb F_p),\mathbb F_p)
\Rightarrow
\pi_{t-s}(X)^{\wedge}_p
$$
或相应 $p$-complete stable homotopy groups 的 associated graded。

**HoTT 接口.** 附录 AZ 提供 spectra 和 filtered convergence 口径；附录 AV 登记 Adams 谱序列模板。本定理的来源构造需要 Adams resolution、Eilenberg-Mac Lane spectrum 和 Steenrod algebra action。

## BN.6 Low-dimensional named classes

**事实 BN.9（低维 Adams 类，经典计算入口）。** 在 $p=2$ 的 sphere Adams 谱序列中，常用记号包括：

$$
h_0\in\mathsf{Ext}^{1,1}_{\mathcal A_2}(\mathbb F_2,\mathbb F_2),
$$
检测 $2$-adic filtration 中的 multiplication by $2$；
$$
h_1\in\mathsf{Ext}^{1,2},
$$
检测 stable Hopf map $\eta\in\pi_1^S$；
$$
h_2\in\mathsf{Ext}^{1,4},
$$
检测 $\nu\in\pi_3^S$；
$$
h_3\in\mathsf{Ext}^{1,8},
$$
检测 $\sigma\in\pi_7^S$。

**边界.** 这些 detection statements 是经典稳定同伦论计算，不是本书当前对象语言中的证明。若完整展开，需要固定 Adams chart convention、stem $t-s$、filtration $s$ 和 differential convention。

## BN.7 Example: projective space formula

**事实 BN.10（$\mathbb RP^\infty$ 上的平方公式）。** 经典公式为
$$
H^\ast(\mathbb RP^\infty;\mathbb F_2)\cong\mathbb F_2[t],\qquad |t|=1,
$$
并且
$$
\mathsf{Sq}^i(t^k)=\binom{k}{i}t^{k+i}
$$
其中二项式系数取模 $2$。

**HoTT 边界.** 要在 HoTT 中证明该公式，需要构造 $\mathbb RP^\infty$ 的 HIT 或 classifying type $B\mathbb Z/2$，证明其 cohomology ring，并构造 Steenrod squares。

## BN.8 本附录关闭的缺口

本附录把 Adams 谱序列的代数端从符号推进到具体结构：Steenrod operations、Steenrod algebra、unstable module、Ext、Adams convergence 和低维检测类。剩余义务是 Steenrod operations 的 HoTT/cubical 构造、Adams resolution 的 spectrum-level 构造、Ext 计算和 Adams differential 的具体例子。
