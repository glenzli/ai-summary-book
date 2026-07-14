# 第七章：Nygaard filtration、syntomic cohomology 与 Tate twists

Frobenius 在 prismatic complex 上并非无条件可除，Nygaard filtration 正是记录“除以多少个 prism ideal 后仍保持积分”的层级。Syntomic complex 再把该滤过与 divided Frobenius 的固定点组合起来，从而接近 étale 与 motivic Tate twist。这里最危险的不是计算长度，而是指标、twist 和 fiber 的移位一旦错位，后续乘法与比较都会整体偏移。本章沿第三章的 Frobenius、第五章的 $A_{\inf}$ 与 $\mu$、第四章的 Tate twists 建立统一约定；relative Nygaard 及 BMS2 的 syntomic、products 和 nearby-cycles 定理精确引用，oriented module 公式只作为能逐项核对的模型。

## 7.1 Nygaard filtration 的基本形式

**定义 7.1（naive Nygaard condition, oriented case）.** 令 $(A,d)$ 为 oriented prism，令 $M$ 为带 Frobenius-semilinear map $\varphi_M:M\to M$ 的 $A$-module。定义 naive Nygaard 子模
$$
N^{\ge i}_{\mathrm{naive}}M=\{x\in M\mid \varphi_M(x)\in d^iM\}.
$$

**警告 7.2.** 定义 7.1 只是离散、无高阶导出问题时的模型公式。对 prismatic cohomology complex，Nygaard filtration 必须在 filtered derived category 中定义，不能逐项套用 naive 子模公式。

**外部输入定理 7.3（relative Nygaard theorem）.** 令 $(A,I)$ 为 bounded
prism，$X=\operatorname{Spf}(R)$ affine smooth over $A/I$，并记
$$
C=R\Gamma_\Delta(X/A),\qquad
C^{(1)}=C\widehat\otimes_{A,\phi_A}^LA.
$$
Bhatt--Scholze 在 Frobenius twist $C^{(1)}$ 上构造递减 Nygaard filtration
$\operatorname{Fil}_N^iC^{(1)}$，满足
$$
\operatorname{gr}_N^iC^{(1)}
\simeq \tau^{\le i}\overline\Delta_{R/A}\{i\},
$$
并给出 Frobenius factorization
$$
C^{(1)}\xrightarrow{\widetilde\varphi}
L\eta_I C\longrightarrow C,
$$
其中 $\widetilde\varphi$ 是同构。来源为 Bhatt--Scholze, Theorem 1.16
（正文 Theorem 15.3）。

**说明 7.4.** Filtration 位于 completed Frobenius twist $C^{(1)}$，graded
piece 带 $\tau^{\le i}$ 与 $\{i\}$。把它写在未扭曲的 $C$ 上、删除
truncation，或把 $\{i\}$ 改成 $\{-i\}$，都会改变 theorem。

## 7.2 Syntomic complexes

**定义 7.5（BMS2 quasisyntomic syntomic fibre）.** 令 $S$ 为
quasisyntomic $\mathbf Z_p$-algebra，$\widehat\Delta_S$ 为 BMS2 的
Nygaard-complete object，$\widehat\Delta_S\{i\}$ 为其 Breuil--Kisin
twist。对 $i\ge0$，定义
$$
\mathbf Z_p(i)(S)
=
\operatorname{fib}\left(
\varphi-\operatorname{can}:
\mathcal N^{\ge i}\widehat\Delta_S\{i\}
\longrightarrow
\widehat\Delta_S\{i\}
\right).
$$
这里两张 map 都落在同一个 twisted target；文献中的 $\varphi-1$ 把
$\operatorname{can}$ 简写为 $1$。Modulo $p$ 后得到
$$
\mathbf Z/p(i)(S)
=
\operatorname{hofib}\left(
\varphi-\operatorname{can}:
\mathcal N^{\ge i}\widehat\Delta_S\{i\}/p
\to
\widehat\Delta_S\{i\}/p
\right).
$$
一般地记
$$
\mathbf Z/p^r(i)=\mathbf Z_p(i)\otimes_{\mathbf Z_p}^L\mathbf Z/p^r.
$$
这是 BMS2, Theorem 1.12 (5) 的 graded $TC$ construction。在 formal scheme 上须先 sheafify，再取 derived global
sections。

**警告 7.6.** 定义 7.5 是 quasisyntomic/Nygaard-complete construction，
不是把任意 relative complex $R\Gamma_\Delta(X/A)$ 代入 fibre 的许可。
$\mathbf Z_p(i)$、其 derived mod-$p^r$ reduction，以及 nearby cycles 的
$\tau^{\le i}$ comparison 是三个不同层级。

**外部输入定理 7.7（BMS2 syntomic comparisons）.** 设 $i\ge0$。

1. 若 $S$ smooth over a perfect field $k$ of characteristic $p$，则在
   pro-etale site 上
   $$
   \mathbf Z_p(i)\simeq W\Omega^i_{S,\log}[-i].
   $$
2. 若 $S$ 是 smooth $\mathcal O_C$-algebra 的 $p$-adic completion，其中
   $C/\mathbf Q_p$ algebraically closed and complete，则对每个 $r\ge1$，
   在 pro-etale site 上有 compatible in $r$ 的拟同构
   $$
   \mathbf Z/p^r(i)
   \simeq\tau^{\le i}R\psi_*\mathbf Z/p^r(i).
   $$

若把 continuous $\mathbf Z_p(i)$ 定义为 finite-level tower 的 derived
inverse limit，则对这些 compatible maps 取 $R\varprojlim_r$ 得到相应
$\mathbf Z_p$-level statement；这里不是在 sheaf cohomology groups 上取
ordinary inverse limit。

来源为 BMS2, Theorem 1.15（mixed-characteristic proof 为 Theorem 10.1）。这里没有 properness 假设，也没有未说明的 torsion
假设；mixed-characteristic target 必须保留 $\tau^{\le i}$。

## 7.3 Tate twists 的积分问题

**说明 7.8.** Rational Tate twist $\mathbf Q_p(i)$ 在 classical theory 中相对简单；integral Tate twist $\mathbf Z_p(i)$ 需要处理 torsion、Bockstein、Frobenius divisibility 和 filtrations。Prismatic cohomology 的 Nygaard filtration 提供了统一控制这些问题的结构。

**命题 7.9（形式层必要条件）.** 若某个 complex $C(i)$ 要作为 $\mathbf Z_p(i)$ 的 integral prismatic model，则至少应满足：

1. after inverting $p$，与 $\mathbf Q_p(i)$ 的 rational comparison 相容；
2. modulo $p^n$ 后与 etale motivic 或 syntomic model 相容；
3. cup product 下有 $C(i)\otimes C(j)\to C(i+j)$；
4. Frobenius normalization 中的 twist convention 与 Tate twist convention 一致。

**证明.** 第一项确保 rational $p$-adic Hodge theory 不被改变；第二项确保积分 torsion 信息正确；第三项是 Tate twists 的张量结构要求；第四项确保 Frobenius fixed construction 得到正确的 Galois character。缺少任一项，都无法把 $C(i)$ 作为 $\mathbf Z_p(i)$ 的积分模型。证毕。

## 7.4 与 BMS 的关系

**外部输入定理 7.10（multiplicativity）.** BMS2 的 THH/$TC^-$/TP
filtrations 是 complete、exhaustive、decreasing 且 multiplicative；其
$TC$ graded pieces 是定义 7.5 的 $\mathbf Z_p(i)$。因此有 products
$$
\mathbf Z_p(i)\otimes_{\mathbf Z_p}^L\mathbf Z_p(j)
\longrightarrow\mathbf Z_p(i+j).
$$
来源为 BMS2, Theorem 1.12 (2), (5)。

**说明 7.11.** Bhatt--Scholze 的 $C^{(1)}$、$\tau^{\le i}$、$\{i\}$
Nygaard formula 与 BMS2 的 fibre/products/nearby-cycles formulas 已分别绑定
numbered statements。两套 Nygaard-complete constructions 的比较是深输入，
不能仅由相似记号视为定义相等。

## 7.5 前沿接口

**研究边界 7.12.** 2025 年 Carmeli-Feng 使用 perfectoid geometry 与 prismatic cohomology 构造 syntomic Steenrod algebra 和 spectral syntomic cohomology，并组织为 spectral prismatic $F$-gauges。该方向说明 Nygaard/syntomic 结构已超出传统 comparison theorem，进入 operations and duality 层面。本书只记录其位置，不把新结果纳入正文定理。

## 7.6 Naive Nygaard filtration 的形式性质

**命题 7.13.** 在定义 7.1 的 naive oriented 模型中，
$$
N^{\ge i+1}_{\mathrm{naive}}M\subseteq N^{\ge i}_{\mathrm{naive}}M.
$$

**证明.** 若 $x\in N^{\ge i+1}_{\mathrm{naive}}M$，则 $\varphi(x)\in d^{i+1}M$。因为 $d^{i+1}M\subseteq d^iM$，所以 $x\in N^{\ge i}_{\mathrm{naive}}M$。证毕。

**命题 7.14.** 若 $M$、$N$ 带 Frobenius，且 $\varphi_{M\otimes N}(m\otimes n)=\varphi_M(m)\otimes\varphi_N(n)$，则
$$
N^{\ge i}_{\mathrm{naive}}M\otimes N^{\ge j}_{\mathrm{naive}}N
\to
N^{\ge i+j}_{\mathrm{naive}}(M\otimes N).
$$

**证明.** 若 $\varphi_M(m)\in d^iM$ 且 $\varphi_N(n)\in d^jN$，则
$$
\varphi_{M\otimes N}(m\otimes n)
\in d^iM\otimes d^jN
\subset d^{i+j}(M\otimes N).
$$
故张量落在 $N^{\ge i+j}$ 中。证毕。

**警告 7.15.** 命题 7.14 只证明 naive 模型中的乘法相容性。Derived Nygaard filtration 的乘法相容需要外部输入或独立 filtered derived category 论证。

## 7.7 Divided Frobenius 与 syntomic fiber

Nygaard filtration 是 prismatic cohomology 中控制 Frobenius 可除性和
syntomic information 的结构。Relative Nygaard filtration 位于 completed
Frobenius twist；BMS2 syntomic complex 是 twisted divided Frobenius 与
canonical map 的 homotopy fibre。Characteristic-$p$ 和 mixed-characteristic
comparisons 的 hypotheses、shift、finite-level coefficient 与 derived-limit
边界已分别写明。

## 练习

**练习 7.1.** 在 oriented prism $(A,d)$ 的 naive 模型中，证明 $N^{\ge i+1}_{\mathrm{naive}}M\subseteq N^{\ge i}_{\mathrm{naive}}M$。

**练习 7.2.** 解释定理 7.3 为什么把 Nygaard filtration 放在
$C^{(1)}$ 而不是 $C$ 上，并说明
$C^{(1)}\simeq L\eta_I C\to C$ 如何记录 Frobenius 的 $I$-divisibility。

**练习 7.3.** 写出定义 7.5 中两张 maps 的共同 target 与 coefficient
category，并指出为什么不能把 $\widehat\Delta_S$ 无条件替换成 relative
$R\Gamma_\Delta(X/A)$。
