# 第五章：$A_{\inf}$、Breuil-Kisin theory 与 BMS 积分比较

## 本章目标

本章说明 Bhatt-Morrow-Scholze integral $p$-adic Hodge theory 在 prismatic theory 出现前解决的问题：构造 $A_{\inf}$-cohomology 和 Breuil-Kisin 型上同调，使同一个积分对象同时控制 de Rham、crystalline 和 etale specialization。Prismatic cohomology 后来把这些对象解释为特定 prism 上的 cohomology。

## 依赖前置知识

依赖第一章的 perfectoid interface、第二章的 perfect prism 与 Breuil-Kisin prism，以及第四章的 classical comparison interface。需要熟悉 derived category、almost mathematics 和 pro-etale site 的基本语言；大型构造作为外部输入。

## 5.1 Fontaine's $A_{\inf}$

**约定 5.1.** 令 $C/\mathbf Q_p$ 为代数闭完备非阿基米德域，$\mathcal O_C$ 为整数环。记
$$
A_{\inf}=A_{\inf}(\mathcal O_C)=W(\mathcal O_C^\flat).
$$
Fontaine map 记为
$$
\theta:A_{\inf}\to\mathcal O_C.
$$
记 $k=\mathcal O_C/\mathfrak m_C$；Witt functoriality 给出
$A_{\inf}\to W(k)$。

**定义 5.2.** 选取 $\xi\in A_{\inf}$ 生成 $\ker(\theta)$。若 $\mathcal O_C$ 中选定兼容的 $p$-power roots of unity $\epsilon=(1,\zeta_p,\zeta_{p^2},\ldots)$，则记
$$
\mu=[\epsilon]-1.
$$

**警告 5.3.** $\xi$ 和 $\mu$ 的选择依赖生成元或 roots of unity。最终公式中的 twists、Nygaard filtration 和 syntomic maps 不得在未说明选择的情况下混用。

## 5.2 $L\eta$ functor

**定义 5.4（chain-level $\eta_f$）.** 令 $f\in A$ 为 nonzerodivisor，
$K^\bullet$ 为逐项 $f$-torsionfree 的 cochain complex。把
$K^\bullet$ 嵌入 $K^\bullet[1/f]$，并定义
$$
(\eta_fK)^i
=\{x\in f^iK^i\mid dx\in f^{i+1}K^{i+1}\}.
$$
For $i<0$，$f^iK^i$ 表示 $K^i[1/f]$ 中的 fractional submodule。
Differential 是 $K[1/f]$ 的 differential 的限制。若写 $x=f^iy$，则
条件正是 $dy\in fK^{i+1}$，所以 differential 的确落在下一项。

在这个 representative 上有自然同构
$$
H^i(\eta_fK)
\cong
\bigl(H^i(K)/H^i(K)[f]\bigr)\otimes_A f^iA.
$$
证明如下。Cocycles 恰为 $f^iZ^i(K)$。映射
$[z]\mapsto[f^iz]$ 是满射；若 $fz=dy$，则
$f^iz=d(f^{i-1}y)$，且 $f^{i-1}y\in(\eta_fK)^{i-1}$，所以
$f$-torsion classes 落入 kernel。反之，若 $f^iz=d(f^{i-1}y)$，逐项
$f$-torsionfreeness 给出 $fz=dy$，故 kernel 正是 $H^i(K)[f]$。

**警告 5.5.** $L\eta_f$ 不是 ordinary truncation、derived tensor product
或 exact functor。例如 $L\eta_p(\mathbf Z/p)=0$，而
$L\eta_p(\mathbf Z/p^2)\simeq\mathbf Z/p$。它保留某个理想的 derived
completeness，但一般不与另一个理想的 completion 交换。

**外部输入定理 5.6（derived $L\eta$ package）.** BMS1 证明：

1. chain-level $\eta_f$ 在 strongly K-flat representatives 上下降为
   $L\eta_f:D(A)\to D(A)$，并与 filtered colimits、canonical truncations
   相容（Corollary 6.5）；
2. $L\eta_f$ 带 natural lax symmetric monoidal structure
   （Proposition 6.7）；
3. $L\eta_fK\otimes_A^LA/f$ 由 $H^*(K\otimes_A^LA/f)$ 及对应 Bockstein
   differential 组成的 complex 计算（Proposition 6.12）；
4. $L\eta_I$ 保 derived $J$-completeness（Lemma 6.19），但一般不与
   $J$-completion 交换；当 $J=I$ 且 $I$ invertible 时，它与 derived
   $I$-completion 交换（Lemma 6.20）。

这些结果的 locator 为 `BMS1-LETA`。$A\Omega$ 上的 Frobenius 还使用 BMS
构造中的额外 comparison maps，不能把它说成任意 $L\eta_f$ 的形式性质。

## 5.3 BMS 的 $A_{\inf}$-cohomology

**外部输入定义 5.7.** 令 $\mathfrak X$ 为 proper smooth $p$-adic formal
scheme over $\mathcal O_C$，generic fibre 为 $X$。BMS 构造一个 perfect
complex
$$
R\Gamma_{A_{\inf}}(\mathfrak X)
$$
in $D(A_{\inf})$，带 $\phi_{A_{\inf}}$-semilinear map $\varphi$，并诱导
$\phi_{A_{\inf}}$-semilinear 拟同构
$$
\varphi:
R\Gamma_{A_{\inf}}(\mathfrak X)[1/\xi]
\xrightarrow{\sim}
R\Gamma_{A_{\inf}}(\mathfrak X)[1/\phi(\xi)].
$$
其 $A_{\inf}$-linearization 是
$$
(\phi_{A_{\inf}}^*R\Gamma_{A_{\inf}}(\mathfrak X))[1/\phi(\xi)]
\xrightarrow{\sim}
R\Gamma_{A_{\inf}}(\mathfrak X)[1/\phi(\xi)].
$$
该 complex 可通过 pro-etale period
sheaves 和 $L\eta_\mu$ 构造；来源为 `BMS1-AINF`。

**外部输入定理 5.8（BMS integral comparison）.** 在定义 5.7 的假设
下，令 $C_{A_{\inf}}=R\Gamma_{A_{\inf}}(\mathfrak X)$。BMS1, Theorem
1.8（正式证明汇总为 Theorem 14.3；locator `BMS1-AINF`）给出：

1. **crystalline specialization**：
   $$
   C_{A_{\inf}}\otimes_{A_{\inf}}^LW(k)
   \simeq R\Gamma_{\mathrm{crys}}(\mathfrak X_k/W(k));
   $$
2. **de Rham specialization**：
   $$
   C_{A_{\inf}}\otimes_{A_{\inf},\theta}^L\mathcal O_C
   \simeq R\Gamma_{\mathrm{dR}}(\mathfrak X/\mathcal O_C);
   $$
3. **$A_{\mathrm{crys}}$-specialization**：
   $$
   C_{A_{\inf}}\otimes_{A_{\inf}}^LA_{\mathrm{crys}}
   \simeq
   R\Gamma_{\mathrm{crys}}(\mathfrak X_{\mathcal O_C/p}/A_{\mathrm{crys}});
   $$
4. **etale localization**：
   $$
   C_{A_{\inf}}\otimes_{A_{\inf}}A_{\inf}[1/\mu]
   \simeq
   R\Gamma_{\mathrm{et}}(X,\mathbf Z_p)
   \otimes_{\mathbf Z_p}A_{\inf}[1/\mu].
   $$

此外，每个 $H^i(C_{A_{\inf}})$ 都是定义 12.7 意义下的
Breuil-Kisin-Fargues module。

**说明 5.9（积分与反演边界）.** 前三项是 derived integral base change，
不能把 $\otimes^L$ 静默改为 ordinary tensor。第四项先 invert $\mu$，但
其结论仍是 $\mathbf Z_p$-cohomology 延标后的 complex，而不是一个裸
$\mathbf Q_p$-vector space 等式。进一步延标到 $B_{\mathrm{crys}}$ 并
invert $p$ 才得到 classical rational crystalline comparison。对单个
cohomology group 去掉 derived Tor 项需要命题 5.16 中的额外 torsion 假设。

## 5.4 Prismatic reinterpretation

**外部输入定理 5.10（$A_{\inf}$ comparison 的 Frobenius twist）.** 令
$(A_{\inf},I)$ 为 perfect prism，其中 $I=\ker\theta=(\xi)$。对 proper
smooth $\mathfrak X/\mathcal O_C$，有自然 $\varphi$-equivariant 拟同构
$$
R\Gamma_{A_{\inf}}(\mathfrak X)
\simeq
\phi_{A_{\inf}}^\ast
R\Gamma_\Delta(\mathfrak X/A_{\inf}).
$$
来源为 Bhatt-Scholze, Theorem 17.2 与其全局化（locator
`BS-COMP-AINF`）。因 $A_{\inf}$ perfect，$\phi_{A_{\inf}}$ 是同构，故
$\phi^*$ 是 autoequivalence；但它改变 $A_{\inf}$-module structure，不能
从公式中删去。

**形式推论 5.11.** 定理 5.10 与第三章的 comparison theorem 组合后，BMS 的 de Rham、crystalline 和 etale specialization 可由 prismatic comparison theorem 统一解释。

**证明.** 置
$C=R\Gamma_\Delta(\mathfrak X/A_{\inf})$。定理 5.10 把 BMS complex
识别为 $C$ 的 Frobenius pullback。若 $A_{\inf}\to B$ 是某个
specialization map，则
$$
(\phi_{A_{\inf}}^*C)\otimes_{A_{\inf}}^LB
\simeq C\otimes_{A_{\inf},\,A_{\inf}\xrightarrow{\phi}A_{\inf}\to B}^LB.
$$
因此在 $W(k)$ 与 $A_{\mathrm{crys}}$ 出口，prismatic side 实际使用的是
Frobenius-composed coefficient map，而不是把未扭曲的 $C$ 沿裸 coefficient
map 直接 base change。保留这一 typing 后，分别应用定理 3.7、3.9、3.11
与 3.13，得到 de Rham、crystalline、etale 与相应 base-change exits。
把这些 resulting maps 识别为定理 5.8 的 canonical BMS maps，仍使用定理
5.10 所引用的外部兼容性；这不是对 BMS comparison 的书内重证。证毕。

## 5.5 Breuil-Kisin cohomology

**约定 5.12.** 令 $K/\mathbf Q_p$ 为完全离散赋值域，剩余域 $k$ 完美，uniformizer 为 $\pi$。设
$$
\mathfrak S=W(k)[[u]],\qquad \phi(u)=u^p,
$$
并令 $E(u)$ 为 $\pi$ 的 Eisenstein polynomial。

**外部输入定义 5.13.** 对 proper smooth $p$-adic formal scheme
$\mathfrak X/\mathcal O_K$，BMS2 构造 perfect $\mathfrak S$-complex
$$
R\Gamma_{\mathfrak S}(\mathfrak X)
$$
及 $\phi_{\mathfrak S}$-semilinear Frobenius。其 linearization
$$
\phi_{\mathfrak S}^\ast R\Gamma_{\mathfrak S}(\mathfrak X)[1/E(u)]
\xrightarrow{\sim}
R\Gamma_{\mathfrak S}(\mathfrak X)[1/E(u)]
$$
为同构，且每个 cohomology group 是定义 12.4 意义下的 Breuil-Kisin
module。构造来自 relative topological Hochschild homology。在 affine
smooth $\operatorname{Spf}(R)/\mathcal O_K$ 情形，local object 是一个
$(p,u)$-complete $E_\infty$-$\mathfrak S$-algebra
$$
\widehat{\Prism}_{R/\mathfrak S}.
$$
来源为 BMS2, Theorem 1.2 与 `BMS2-BKLOCAL`。

**外部输入定理 5.14（Breuil-Kisin/prismatic comparison）.** 在
Breuil-Kisin prism $(\mathfrak S,(E(u)))$ 上有自然 $\varphi$-equivariant
拟同构
$$
R\Gamma_\Delta(\mathfrak X/\mathfrak S)
\simeq
R\Gamma_{\mathfrak S}(\mathfrak X).
$$
选择 $\pi$ 的兼容 $p$-power roots 后，BMS2 的 maps 可写为：

1. 一个 $\varphi$-compatible embedding
   $\iota_{\inf}:\mathfrak S\to A_{\inf}$；在 BMS2 的 convention 中，它在
   $W(k)$ 上为 Frobenius，并把 $u$ 送到 $[\pi^\flat]^p$。它给出
   $$
   R\Gamma_{\mathfrak S}(\mathfrak X)\otimes_{\mathfrak S,\iota_{\inf}}^LA_{\inf}
   \simeq R\Gamma_{A_{\inf}}(\mathfrak X_{\mathcal O_C});
   $$
2. 若 $\widetilde\theta:\mathfrak S\to\mathcal O_K$ 由 $u\mapsto\pi$
   给出，并令 $\theta_{\mathrm{BK}}=\widetilde\theta\circ\phi_{\mathfrak S}$，则
   $$
   R\Gamma_{\mathfrak S}(\mathfrak X)
   \otimes_{\mathfrak S,\theta_{\mathrm{BK}}}^L\mathcal O_K
   \simeq R\Gamma_{\mathrm{dR}}(\mathfrak X/\mathcal O_K);
   $$
3. 若 $\iota_0:\mathfrak S\to W(k)$ 在 $W(k)$ 上为 Frobenius 且
   $u\mapsto0$，则
   $$
   R\Gamma_{\mathfrak S}(\mathfrak X)\otimes_{\mathfrak S,\iota_0}^LW(k)
   \simeq R\Gamma_{\mathrm{crys}}(\mathfrak X_k/W(k));
   $$
4. 沿 $\iota_{\inf}$ 再 invert $\mu$，得到
   $R\Gamma_{\mathrm{et}}(X_C,\mathbf Z_p)$ 延标到
   $A_{\inf}[1/\mu]$ 的 comparison。

来源为 BMS2, Theorem 1.2 与 Bhatt--Scholze, Proposition 15.7 及
§15.2 末段的 Breuil--Kisin/prismatic identification；locators 为
`BMS2-BK`、`BMS2-BKLOCAL` 和 `BS-COMP-BK`。

**说明 5.15.** 这说明 prismatic site 不是只重写 $A_{\inf}$ 情形，而是把 crystalline prism、perfect prism、Breuil-Kisin prism 和 $q$-crystalline prism 放入同一形式。BMS2 的 THH construction 还解释了 Nygaard filtration、syntomic sheaves 和 Breuil-Kisin twists 的来源；这些结构在第七章和第十一章中通过 BMS2-SYN locator 使用。

## 5.6 积分结构与 torsion 控制

**命题 5.16（ordinary Tor-dimension one 的基变换短正合列）.** 设
$C\in D(A)$ 为 bounded complex，设 $B$ 为离散 $A$-algebra，并假设对每个
离散 $A$-module $N$ 都有
$$
\operatorname{Tor}_i^A(N,B)=0\qquad(i>1).
$$
换言之，按 cohomological convention，$B$ 的 ordinary Tor-amplitude 包含于
$[-1,0]$。则对每个整数 $n$ 有自然短正合列
$$
0\to H^n(C)\otimes_AB
\to H^n(C\otimes_A^LB)
\to \operatorname{Tor}_1^A(H^{n+1}(C),B)
\to0.
$$

**证明.** $C$ bounded，故 hyper-Tor spectral sequence 强收敛：
$$
E_2^{i,j}=\operatorname{Tor}_{-i}^A(H^j(C),B)\Rightarrow H^{i+j}(C\otimes_A^L B).
$$
Tor-dimension 假设使 $E_2^{i,j}$ 只可能在 $i=0,-1$ 两列非零，因而
不存在长度至少二的非零 differential，谱序列在 $E_2$ 退化。总次数
$n$ 的两级过滤的 associated graded 分别是
$H^n(C)\otimes_AB$ 与
$\operatorname{Tor}_1^A(H^{n+1}(C),B)$，于是得到所述自然短正合列。
该列一般不自然分裂。它给出 torsion 传播的精确形式机制，但不替代 BMS
积分比较定理中对具体 $A\to B$ 的 torsion 控制。证毕。

特别地，若 $B=A/(f)$ 且 $f$ 是 $A$ 的 nonzerodivisor，则
$0\to A\xrightarrow{f}A\to B\to0$ 是长度一自由 resolution，且
$$
\operatorname{Tor}_1^A(M,B)=M[f].
$$
因此
$$
0\to H^n(C)/fH^n(C)
\to H^n(C\otimes_A^LA/(f))
\to H^{n+1}(C)[f]\to0.
$$
只有在 $H^{n+1}(C)[f]=0$ 时，degree $n$ 的 cohomology 才与 ordinary
base change 交换。

## 5.7 BMS object 的四个出口

**说明 5.17.** 对 proper smooth $\mathfrak X/\mathcal O_C$，$R\Gamma_{A_{\inf}}(\mathfrak X)$ 应被看成一个有四个出口的对象：

| 出口 | 操作 | 目标 |
| --- | --- | --- |
| de Rham | $\otimes_{A_{\inf},\theta}^L\mathcal O_C$ | $R\Gamma_{\mathrm{dR}}(\mathfrak X/\mathcal O_C)$ |
| crystalline | $\otimes_{A_{\inf}}^LW(k)$ | $R\Gamma_{\mathrm{crys}}(\mathfrak X_k/W(k))$ |
| etale | $\otimes A_{\inf}[1/\mu]$ | $R\Gamma_{\mathrm{et}}(X,\mathbf Z_p)\otimes A_{\inf}[1/\mu]$ |
| lattice | $H^i$ 与 semilinear Frobenius | BKF module；finite free 需第十二章的 torsion 条件 |

**命题 5.18.** 若一个 construction 只给出 de Rham 出口，则它不足以替代 $A_{\inf}$-cohomology。

**证明.** $A_{\inf}$-cohomology 的作用是同时控制 de Rham、crystalline、etale 和 integral lattice 信息。只给出 de Rham 出口会遗忘 Frobenius、torsion 和 etale comparison 所需的结构。因此它不能替代完整对象。证毕。

## 本章小结

BMS 理论在 prismatic theory 前已经构造了强大的积分 cohomology object。BMS1 给出 $A_{\inf}$-cohomology 和 $A\Omega$，BMS2 用 THH/TC refinement 产生 Breuil-Kisin cohomology、Nygaard filtration 和 syntomic sheaves。Prismatic theory 的作用不是废弃 BMS，而是把 $A_{\inf}$ 和 Breuil-Kisin cohomology 解释为特定 prism 上的统一 cohomology，并把 comparison theorem 放入一个 site-theoretic 框架。

## 练习

**练习 5.1.** 说明为什么 $R\Gamma_{A_{\inf}}(\mathfrak X)$ 不能只看成 $\mathbf Z_p$-complex。

**练习 5.2.** 比较 perfect prism $(A_{\inf},\ker\theta)$ 与 Breuil-Kisin prism $(\mathfrak S,(E(u)))$ 的 quotient ring。

**练习 5.3.** 写出形式推论 5.11 中用到的两个外部输入定理，并说明它们分别属于 integral layer 还是 prismatic layer。
