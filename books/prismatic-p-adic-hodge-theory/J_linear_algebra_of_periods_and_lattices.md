# 附录 J：周期环、滤过向量空间与 lattice 的线性代数

## 本附录目标

本附录给出第四、十二章所需的线性代数细节：filtered vector spaces、semilinear Frobenius、$B$-admissibility 映射、lattice 和 rationalization。周期环本身仍为外部输入，但相关线性代数在本书内部说明。

## J.1 Semilinear maps

**定义 J.1.** 令 $\sigma:A\to A$ 为环同态。$A$-module $M$ 上的 $\sigma$-semilinear map 是加法映射 $F:M\to M$，满足
$$
F(am)=\sigma(a)F(m).
$$
其 linearization 为
$$
A\otimes_{A,\sigma}M\to M,\qquad a\otimes m\mapsto aF(m).
$$

**命题 J.2.** $\sigma$-semilinear map $F:M\to M$ 与 $A$-linear map $A\otimes_{A,\sigma}M\to M$ 自然等价。

**证明.** 给定 $F$，上式定义 $A$-linear map。反向，给定 $g:A\otimes_{A,\sigma}M\to M$，令 $F(m)=g(1\otimes m)$。则
$$
F(am)=g(1\otimes am)=g(\sigma(a)\otimes m)=\sigma(a)g(1\otimes m)=\sigma(a)F(m).
$$
二者互逆。证毕。

## J.2 Filtered vector spaces

**定义 J.3.** 一个递减滤过向量空间是向量空间 $D$ 与子空间族 $\operatorname{Fil}^iD$，满足
$$
\operatorname{Fil}^{i+1}D\subseteq\operatorname{Fil}^iD.
$$
若存在 $a\ll0$ 和 $b\gg0$ 使得 $\operatorname{Fil}^aD=D$ 且 $\operatorname{Fil}^bD=0$，称 filtration exhaustive and separated in finite range。

**定义 J.4.** Associated graded 为
$$
\operatorname{gr}^iD=\operatorname{Fil}^iD/\operatorname{Fil}^{i+1}D.
$$

**命题 J.5.** 若 $D$ 的 filtration finite，且每个 $\operatorname{gr}^iD$ 有限维，则 $D$ 有限维，且
$$
\dim D=\sum_i\dim\operatorname{gr}^iD.
$$

**证明.** 对 filtration 长度归纳。长度为 1 时结论直接成立。一般情形取短正合列
$$
0\to\operatorname{Fil}^{i+1}D\to\operatorname{Fil}^{i}D\to\operatorname{gr}^iD\to0
$$
并使用维数可加性。证毕。

## J.3 $B$-admissibility map

**定义 J.6.** 令 $B$ 为带 $G_K$-作用的 period ring，$E=B^{G_K}$。对 representation $V$ 定义
$$
D_B(V)=(B\otimes_{\mathbf Q_p}V)^{G_K}.
$$
Comparison map 为
$$
\alpha_{B,V}:B\otimes_E D_B(V)\to B\otimes_{\mathbf Q_p}V.
$$

**外部输入命题 J.7（period invariants 的维数界）.** 设 $B$ 满足 Fontaine 的 regularity hypotheses，$E=B^{G_K}$ 为域。则对每个有限维 $\mathbf Q_p$-representation $V$，
$$
\dim_E D_B(V)\le \dim_{\mathbf Q_p}V
$$
成立。若比较映射 $\alpha_{B,V}$ 为同构，则等号成立：两侧分别是秩
$\dim_ED_B(V)$ 与 $\dim_{\mathbf Q_p}V$ 的自由 $B$-模，而同构保持秩。

**说明 J.8.** 不等式本身依赖 period ring 的 admissibility formalism，故作为外部输入；比较映射为同构时的等号则是上述自由 $B$-模秩比较的书内形式后果。

## J.4 Lattices

**定义 J.9.** 令 $V$ 为有限维 $\mathbf Q_p$-向量空间。一个 lattice 是有限生成 $\mathbf Z_p$-submodule $T\subset V$，使得
$$
T\otimes_{\mathbf Z_p}\mathbf Q_p=V.
$$

**命题 J.10.** 若 $T$ 是 lattice，则存在 $n$ 使得
$$
p^n\mathbf Z_p^r\subseteq T\subseteq p^{-n}\mathbf Z_p^r
$$
在某个同构 $V\simeq\mathbf Q_p^r$ 下成立。

**证明.** 取 $T$ 的 $\mathbf Z_p$-生成元和 $\mathbf Q_p$-基。由于 $T$ 张成 $V$，可选 $r$ 个元素构成 $\mathbf Q_p$-基；有限多个生成元在该基下的坐标有有界 $p$-adic 分母，得到右包含。反向，所选 $r$ 个基向量的 $\mathbf Z_p$-span 与标准 lattice 可相差有限个 $p$-幂尺度，得到左包含。证毕。

**推论 J.11.** Rational representation 不唯一决定 lattice。

**证明.** 在 $V=\mathbf Q_p$ 中，$\mathbf Z_p$ 和 $p\mathbf Z_p$ 都是 lattice，且张量 $\mathbf Q_p$ 后都等于 $V$。证毕。

## 本附录小结

Classical comparison theorem 的目标不是裸向量空间，而是带 filtration、Frobenius、monodromy 和 lattice 的线性代数对象。Prismatic theory 的积分强度正体现在保留 lattice 和 torsion 信息。

## 练习

**练习 J.1.** 对 $\sigma=\phi_A$，把 Frobenius-semilinear map 写成 linearized map。

**练习 J.2.** 给出一个二维 filtered vector space，并计算 associated graded。

**练习 J.3.** 在 $V=\mathbf Q_p^2$ 中给出两个不同 lattice。
