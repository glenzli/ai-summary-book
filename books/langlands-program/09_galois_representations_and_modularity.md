# 第九章：Galois 表示与模性定理

## 本章目标

本章把第六章的 Hecke eigenform、第七章的 `GL(2)` 自守表示和第八章的椭圆曲线 Galois 表示放入同一语言：二维 $\ell$-adic Galois 表示。我们定义 ramification、导子、奇性、残余表示和模性，并陈述 Deligne 表示、椭圆曲线模性定理和模性提升定理的接口形式。

## 依赖前置知识

需要第五章的局部参数、第六章的 Deligne 表示、第七章的自守表示和第八章的 Tate module。Galois cohomology、deformation rings、Hecke algebras 和 Taylor-Wiles patching 本章只作为外部输入出现。附录 T 给出 Deligne 表示的模曲线上同调来源；附录 U 给出 p-adic Hodge 和 automorphy lifting 的高维接口。

收口归一化回指：本章是算术 Frobenius、几何 Frobenius、Tate twist、Deligne 表示和自守 Satake 参数的比较点；统一 convention 见 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 2、5、6、7 节。

## 9.1 $\ell$-adic Galois 表示

**定义 9.1.** 设 $E/\mathbb Q_\ell$ 为有限扩张。一个二维 $\ell$-adic Galois 表示是连续同态
$$
\rho:G_\mathbb Q\to\operatorname{GL}_2(E),
$$
其中 $G_\mathbb Q$ 带 Krull 拓扑，$\operatorname{GL}_2(E)$ 带 $\ell$-adic 拓扑。

若存在有限集合 $S$，包含 $\ell$ 和无穷素位，使得对每个素数 $p\notin S$，惯性群 $I_p$ 在 $\rho$ 下作用平凡，则称 $\rho$ 几乎处处非分歧。

**定义 9.2.** 若 $\rho$ 在 $p\ne\ell$ 处非分歧，则定义算术 Frobenius trace
$$
a_p(\rho)=\operatorname{tr}\rho(\operatorname{Frob}_p^{\operatorname{arith}})
$$
和 characteristic polynomial
$$
P_p(\rho,X)=\det(1-\rho(\operatorname{Frob}_p^{\operatorname{arith}})X).
$$

**定义 9.3.** $\rho$ 的局部 L 因子在 $p\ne\ell$ 处定义为
$$
L_p(\rho,s)=
\det\left(1-\rho(\operatorname{Frob}_p^{\operatorname{arith}})p^{-s}\mid V^{I_p}\right)^{-1},
$$
其中 $V=E^2$。若 $\rho$ 在 $p$ 处非分歧，则
$$
L_p(\rho,s)=P_p(\rho,p^{-s})^{-1}.
$$

**注 9.4.** 这里使用算术 Frobenius，以便直接比较第六章 Deligne 表示和第八章椭圆曲线表示。若转到第五章的几何 Frobenius 局部 Langlands 参数，需要取逆或使用对偶/Tate twist 调整。

## 9.2 奇表示和几何条件

**定义 9.5.** 设 $c\in G_\mathbb Q$ 为复共轭元。二维 $\ell$-adic 表示
$$
\rho:G_\mathbb Q\to\operatorname{GL}_2(E)
$$
称为奇的（odd），若
$$
\det\rho(c)=-1.
$$

**定义 9.6.** 表示 $\rho$ 称为 Hodge-Tate，若其限制到 $G_{\mathbb Q_\ell}$ 是 Hodge-Tate 表示。若 Hodge-Tate 权为 $\{0,k-1\}$，则说 $\rho$ 具有权 $k$ 的 Hodge-Tate 型。

**猜想 9.7（Fontaine-Mazur 预期，二维接口）.** 不可约、奇、几乎处处非分歧、几何的二维 $\ell$-adic 表示预期来自模形式。这里“几何”包括在 $\ell$ 处 de Rham，并在几乎所有素数处非分歧。该陈述是 Fontaine-Mazur 猜想在 `GL(2)/\mathbb Q` 方向的形式之一，只有许多重要情形已知。

本书不会把 Fontaine-Mazur 猜想当作已证明定理使用；它只解释模性定理应处于何种大图景。

## 9.3 残余表示

设 $\rho:G_\mathbb Q\to\operatorname{GL}_2(E)$ 为连续表示，$\mathcal O_E$ 为整数环，$\lambda$ 为极大理想，剩余域为 $k_E$。

**外部输入定理 9.8（稳定格）.** 若 $\rho$ 连续且 $G_\mathbb Q$ 紧，则存在 $G_\mathbb Q$-稳定的 $\mathcal O_E$-格
$$
\Lambda\subset E^2.
$$

**定义 9.9.** 选择稳定格 $\Lambda$ 后，约化得到
$$
\rho_{\Lambda,\lambda}:G_\mathbb Q\to\operatorname{GL}_2(k_E).
$$
其半单化的同构类与稳定格选择无关，记为
$$
\overline\rho:G_\mathbb Q\to\operatorname{GL}_2(k_E)
$$
并称为 $\rho$ 的残余表示（residual representation）。

**定义 9.10.** 残余表示 $\overline\rho$ 称为绝对不可约，若
$$
\overline\rho\otimes_{k_E}\overline{k_E}
$$
不可约。

**注 9.11.** 模性提升定理通常不是直接从 $\overline\rho$ 的模性推出任意 lift 的模性；还需要局部变形条件、不可约性、奇性、最小或非最小 ramification 条件，以及 Hecke 代数和变形环之间的比较。

## 9.4 模形式给出的 Galois 表示

**外部输入定理 9.12（Deligne 表示，模性方向）.** 设
$$
f(q)=\sum_{n\ge1}a_nq^n
$$
为归一化 cuspidal Hecke eigenform，权 $k\ge2$，级 $N$，nebentypus $\varepsilon$，系数域为 $E_f$。对 $E_f$ 的每个有限素位 $\lambda$，存在连续半单表示
$$
\rho_{f,\lambda}:G_\mathbb Q\to\operatorname{GL}_2(E_{f,\lambda})
$$
满足对所有 $p\nmid N\operatorname{char}(\lambda)$，
$$
\operatorname{tr}\rho_{f,\lambda}(\operatorname{Frob}_p^{\operatorname{arith}})=a_p,
$$
$$
\det\rho_{f,\lambda}(\operatorname{Frob}_p^{\operatorname{arith}})
=\varepsilon(p)p^{k-1}.
$$
并且 $\rho_{f,\lambda}$ 在 $p\nmid N\operatorname{char}(\lambda)$ 处非分歧。

**定义 9.13.** 若二维 $\ell$-adic 表示 $\rho:G_\mathbb Q\to\operatorname{GL}_2(E)$ 与某个 $\rho_{f,\lambda}$ 在系数扩张和半单化后同构，则称 $\rho$ 是模的（modular），或称 $\rho$ 来自模形式 $f$。

**命题 9.14.** 若 $\rho$ 来自 $f$，则对几乎所有 $p$，
$$
L_p(\rho,s)=
\left(1-a_pp^{-s}+\varepsilon(p)p^{k-1-2s}\right)^{-1}.
$$

**证明.** 对 $p\nmid N\ell$，由定理 9.12，$\rho$ 非分歧且
$$
\operatorname{tr}\rho(\operatorname{Frob}_p^{\operatorname{arith}})=a_p,\qquad
\det\rho(\operatorname{Frob}_p^{\operatorname{arith}})=\varepsilon(p)p^{k-1}.
$$
因此
$$
\det(1-\rho(\operatorname{Frob}_p^{\operatorname{arith}})X)
=1-a_pX+\varepsilon(p)p^{k-1}X^2.
$$
令 $X=p^{-s}$ 即得结论。$\square$

## 9.5 椭圆曲线表示的模性

设 $E/\mathbb Q$ 为椭圆曲线。第八章定义了
$$
\rho_{E,\ell}:G_\mathbb Q\to\operatorname{GL}_2(\mathbb Z_\ell)
$$
和
$$
V_\ell(E)=T_\ell(E)\otimes_{\mathbb Z_\ell}\mathbb Q_\ell.
$$

**定义 9.15.** 椭圆曲线 $E/\mathbb Q$ 称为模的，若存在权 $2$、级 $\Gamma_0(N_E)$ 的归一化 newform $f_E$，使得
$$
L(E,s)=L(f_E,s).
$$
等价地，对几乎所有素数 $p$，
$$
a_p(E)=a_p(f_E).
$$

**命题 9.16.** 若对某个素数 $\ell$，表示 $V_\ell(E)$ 是模的且来自权 $2$ newform $f$，并且坏素数处局部因子相容，则 $E$ 是模的。

**证明.** 若 $V_\ell(E)$ 来自 $f$，则对所有 $p\nmid N_E N_f\ell$，有
$$
a_p(E)=\operatorname{tr}\rho_{E,\ell}(\operatorname{Frob}_p^{\operatorname{arith}})
=
\operatorname{tr}\rho_{f,\lambda}(\operatorname{Frob}_p^{\operatorname{arith}})
=a_p(f).
$$
好素数处 Euler 因子相等。若进一步假设坏素数处局部因子相容，则完整 Euler 乘积相等，即
$$
L(E,s)=L(f,s).
$$
这就是定义 9.15 的模性。$\square$

**外部输入定理 9.17（椭圆曲线模性定理）.** 每条椭圆曲线 $E/\mathbb Q$ 都是模的。更精确地，存在权 $2$、级 $N_E$ 的 newform $f_E$，使得对所有素数 $p$，局部 L 因子相容：
$$
L_p(E,s)=L_p(f_E,s).
$$

**注 9.18.** Wiles 和 Taylor-Wiles 证明了半稳定椭圆曲线情形；这已经足以结合 Ribet 降层推出费马大定理。一般情形由后续工作完成。本书在第九十章使用的是半稳定版本。

## 9.6 模性提升定理的接口

模性定理的证明核心不是直接构造 $f_E$，而是证明某些 Galois 表示的 lift 必须来自 Hecke eigenform。下面只给出接口形式。

**定义 9.19.** 设
$$
\overline\rho:G_\mathbb Q\to\operatorname{GL}_2(k)
$$
为连续残余表示。一个 lift 是连续表示
$$
\rho:G_\mathbb Q\to\operatorname{GL}_2(\mathcal O)
$$
使得 $\rho\bmod\mathfrak m_\mathcal O$ 的半单化同构于 $\overline\rho$。

**外部输入定理 9.20（模性提升，接口形式）.** 设 $\rho:G_\mathbb Q\to\operatorname{GL}_2(\mathcal O)$ 是连续、奇、几何二维 $\ell$-adic 表示，并在每个 ramified prime 和 $\ell$-adic prime 处满足某个已固定 Taylor-Wiles 型局部变形问题的条件。若残余表示 $\overline\rho$ 绝对不可约且已知是模的，并且这些局部变形条件与相应 Hecke 侧局部类型匹配，则 $\rho$ 是模的。

本定理的精确版本有许多变体：minimal、semistable、ordinary、potentially Barsotti-Tate、potentially semistable 等。每个版本都需要具体局部变形环和 Hecke 代数的比较。

**外部输入定理 9.21（$R=T$ 原理，接口形式）.** 在 Taylor-Wiles 方法中，给定残余表示 $\overline\rho$ 和局部变形条件，可构造 universal deformation ring $R$ 与相应 Hecke algebra $T$。当残余表示、局部变形环、Taylor-Wiles primes 和 Hecke 模满足所选版本的 patching 与数值判据假设时，有同构
$$
R\cong T.
$$
该同构把 Galois 侧的 lift 与自守侧的 Hecke eigenforms 识别起来。

**注 9.22.** $R=T$ 不是一个单独形式定理，而是一类定理的共同结构。证明涉及 Galois cohomology、Selmer 群、Taylor-Wiles primes、patching 和 commutative algebra。附录 K 给出 deformation functor、Selmer tangent space、Hecke algebra 和 patching 的接口，但仍把完整 Taylor-Wiles 证明作为外部输入。

**注 9.22.1.** 附录 U 从 p-adic Hodge 和 Shimura variety/cohomology 角度记录更高维 automorphy lifting 的共同接口。本章的二维半稳定情形是该方法的历史核心样本，但不是一般定理的全部范围。

## 9.7 半稳定模性和费马大定理

**外部输入定理 9.23（半稳定模性定理）.** 每条半稳定椭圆曲线 $E/\mathbb Q$ 都是模的。

**命题 9.24.** 半稳定模性定理是第九十章费马大定理应用所需的模性输入。

**证明.** 第九十章从假设的 Fermat 反例构造 Frey 曲线 $E_{a,b,p}$。外部输入定理 90.5 说明该曲线半稳定。由半稳定模性定理 9.23，$E_{a,b,p}$ 是模的。随后 Ribet 降层把其模 $p$ 表示降到权 $2$、级 $2$ 的 newform；而 $S_2(\Gamma_0(2))=0$ 给出矛盾。因此第九十章只需要半稳定模性，而不需要完整椭圆曲线模性定理。$\square$

## 9.8 与 Langlands 主线的关系

本章中的对象与 Langlands 纲领的对应关系如下：

1. Galois 侧：二维 $\ell$-adic 表示 $\rho:G_\mathbb Q\to\operatorname{GL}_2(E)$。
2. 自守侧：`GL(2,\mathbb A_\mathbb Q)` 的 cuspidal automorphic representation $\pi_f$。
3. 经典桥梁：归一化 Hecke eigenform $f$。
4. 局部相容：几乎所有 $p$ 处，Frobenius trace 等于 Hecke eigenvalue。
5. L 函数相容：$L(\rho,s)=L(f,s)=L(\pi_f,s)$，其中三者均使用相同坏素数 Euler 因子、同一 Frobenius 方向和 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 中的 L 函数变量约定。

因此，椭圆曲线模性是二维 Galois 表示与 `GL(2)` 自守表示对应的核心实例，而模性提升定理是证明这类对应的主要机制之一。

## 9.9 本章小结

本章定义了二维 $\ell$-adic Galois 表示、残余表示、奇性和模性。Deligne 定理从 Hecke eigenform 构造 Galois 表示；椭圆曲线模性定理则反向说明椭圆曲线的 Tate module 表示来自权 $2$ newform。模性提升定理和 $R=T$ 原理提供了证明模性的结构性方法，但本书在此阶段只把它们作为外部输入。

## 练习

**练习 9.1.** 设 $\rho$ 在 $p\ne\ell$ 处非分歧。证明定义 9.3 的局部因子只依赖于 Frobenius 共轭类。

**练习 9.2.** 设 $\rho$ 来自权 $k$、nebentypus $\varepsilon$ 的 eigenform。证明命题 9.14 的 Euler 因子公式。

**练习 9.3.** 对椭圆曲线 $E/\mathbb Q$，解释为什么 $\det\rho_{E,\ell}=\chi_\ell$ 与权 $2$ 模形式的 determinant 公式相容。

**练习 9.4.** 说明“$\overline\rho$ 模”与“$\rho$ 模”不是同一个命题，并列出模性提升还需要哪些类型的额外条件。

**练习 9.5.** 用命题 9.24 的语言重述费马大定理证明中模性定理的精确使用位置。
