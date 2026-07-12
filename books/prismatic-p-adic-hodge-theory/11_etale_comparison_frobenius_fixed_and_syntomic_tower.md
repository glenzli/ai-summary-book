# 第十一章：Etale comparison、Frobenius fixed points 与 syntomic tower

## 本章目标

本章写清楚 étale comparison 的操作顺序和 coefficient category：先在
perfect prism 上 modulo $p^r$ 并 invert prism ideal，再把 semilinear
Frobenius 视为 $\mathbf Z/p^r$-linear endomorphism并取 derived fibre。
Syntomic comparison 则使用 Nygaard-divided Frobenius 与明确的 truncation。

## 依赖前置知识

依赖第三章的 finite-level étale comparison、第七章与附录 F 的 Nygaard
convention、附录 A 的 derived inverse limit，以及 derived category 中 fibre
triangle 的基本性质。

## 11.1 Perfect prism 上的 etale comparison

**约定 11.1.** 令 $(A,(d))$ 为 oriented perfect prism，
$R=A/(d)$ 为对应 perfectoid ring。令 $X$ 为 $p$-adic formal scheme over
$R$，并记其 adic generic fibre 为 $X_\eta$。Orientation 只用于把
$[1/I]$ 写成 $[1/d]$。

**外部输入定理 11.2（Bhatt-Scholze finite-level comparison）.** 对每个
$r\ge1$ 有 $D(\mathbf Z/p^r)$ 中的自然拟同构
$$
R\Gamma_{\mathrm{et}}(X_\eta,\mathbf Z/p^r)
\simeq
\left((R\Gamma_\Delta(X/A)/p^r)[1/d]\right)^{\varphi=1}.
$$
右侧的 $/p^r$ 是 derived cofiber，$[1/d]$ 是 localization，且
$$
C^{\varphi=1}=\operatorname{fib}(\varphi-1:C\to C).
$$
虽然 $\varphi$ 对 $A$ semilinear，它固定 $\mathbf Z_p$，故在 modulo $p^r$
并 restriction of scalars 后是 $\mathbf Z/p^r$-linear，所以上式的 fibre
确实位于 $D(\mathbf Z/p^r)$。来源为 Bhatt-Scholze, Theorem 9.1（locator
`BS-COMP-ETALE`）。

若连续 étale cohomology 定义为
$$
R\Gamma_{\mathrm{et}}(X_\eta,\mathbf Z_p)
:=R\varprojlim_rR\Gamma_{\mathrm{et}}(X_\eta,\mathbf Z/p^r),
$$
则对定理 11.2 的 compatible system 取 $R\varprojlim_r$ 得到
$\mathbf Z_p$-level comparison。该结论是 derived inverse limit，不是先在
cohomology groups 上取 ordinary inverse limit。

**警告 11.3.** 定理 11.2 的 $[1/d]$ 不等于 $[1/p]$，而 $/p^r$ 也不是
rationalization。省略其中任一操作，或把 derived fixed complex 换成
$H^n$ 上的 pointwise invariants，都会改变 theorem。

## 11.2 Frobenius fixed points

**定义 11.4.** 令 $S$ 为环，$C\in D(S)$，$u:C\to C$ 为 $S$-linear
endomorphism。定义
$$
C^{u=1}:=\operatorname{fib}(C\xrightarrow{u-1}C)\in D(S).
$$
若 $u$ 只对更大系数环 semilinear，必须先指定一个被 $u$ 固定的 subring
$S$ 并 restriction of scalars；否则 $u-1$ 不是该大系数环上的 linear map。

**命题 11.5.** 若 $C=M[0]$ 集中在 degree $0$，则有自然短正合列
$$
0\to H^0(C^{u=1})\to M\xrightarrow{u-1}M
\to H^1(C^{u=1})\to0,
$$
其中 $H^0(C^{u=1})=\ker(u-1)$，
$H^1(C^{u=1})=\operatorname{coker}(u-1)$。

**证明.** Fibre triangle
$$
C^{u=1}\longrightarrow C\xrightarrow{u-1}C
$$
给出 cohomology 长正合列。因 $H^j(C)=0$ 对 $j\ne0$，长正合列只剩
所述五项，并分别识别 kernel 与 cokernel。证毕。

**警告 11.6.** Ordinary fixed module 只给出 $H^0$。若 $u-1$ 不满射，
$H^1$ 非零；derived comparison 不能丢掉该 obstruction。

## 11.3 Syntomic tower

**定义 11.7（BMS2 quasisyntomic model）.** 令 $S$ 为 quasisyntomic
$\mathbf Z_p$-algebra，$\widehat\Delta_S$ 为 BMS2 的 Nygaard-complete
prismatic complex。对 $i\ge0$，定义
$$
\mathbf Z_p(i)(S)
=\operatorname{fib}\left(
\varphi_i-\operatorname{can}_i:
\mathcal N^{\ge i}\widehat\Delta_S\{i\}
\longrightarrow
\widehat\Delta_S\{i\}
\right).
$$
$\varphi_i$ 是 divided Frobenius，$\operatorname{can}_i$ 是进入同一
twisted target 的 canonical map。常见的 “$\varphi_i-1$” 记号只有在该
canonical map 已固定时才合法。定义
$$
\mathbf Z/p^r(i)=\mathbf Z_p(i)\otimes_{\mathbf Z_p}^L\mathbf Z/p^r.
$$
在 formal scheme 上先 sheafify 此 construction，再取 derived global
sections；不能把任意 relative prismatic complex 无条件代入该公式。

**外部输入定理 11.8（BMS2 syntomic comparisons）.** 设 $i\ge0$。

1. 若 $S$ smooth over a perfect field $k$ of characteristic $p$，则在
   pro-etale site 上有
   $$
   \mathbf Z_p(i)\simeq W\Omega^i_{S,\log}[-i].
   $$
2. 若 $S$ 是 smooth $\mathcal O_C$-algebra 的 $p$-adic completion，其中
   $C/\mathbf Q_p$ algebraically closed and complete，令
   $\psi:X_{\eta,\mathrm{et}}\to X_{\mathrm{et}}$ 为 nearby-cycles map。
   则对每个 $r\ge1$ 有 compatible in $r$ 的 sheaf-level 拟同构
   $$
   \mathbf Z/p^r(i)
   \simeq
   \tau^{\le i}R\psi_*\mathbf Z/p^r(i).
   $$

来源为 BMS2, Theorem 1.15、Corollary 8.21 与 Theorem 10.1（locator
`BMS2-SYN`）。这些是外部输入；本书不重证 quasisyntomic descent、Nygaard
identification 或 nearby-cycles comparison。

**警告 11.9.** Mixed-characteristic theorem 的 truncation 正是
$\tau^{\le i}$，不是未指定的 “适当范围”。去掉 truncation 会得到更强且
未被该来源证明的陈述。

## 11.4 Cup products

**外部输入定理 11.10.** BMS2 的 motivic/Nygaard filtrations 是
multiplicative，因而 syntomic complexes 有自然 products
$$
\mathbf Z_p(i)\otimes_{\mathbf Z_p}^L\mathbf Z_p(j)
\longrightarrow\mathbf Z_p(i+j),
$$
并与 finite-level reduction 相容。来源为 BMS2, Theorem 1.12 (2), (5)
及 `BMS2-SYN`。

**命题 11.11（multiplicative comparison 的形式后果）.** 若一族
comparison maps $c_i:\mathbf Z_p(i)\to E(i)$ 是 graded $E_\infty$-algebra
map，且每个 $c_i$ 是拟同构，则诱导的 cohomology isomorphisms 保 cup
products。

**证明.** Graded $E_\infty$-map 的定义给出交换图
$$
\begin{array}{ccc}
\mathbf Z_p(i)\otimes^L\mathbf Z_p(j)&\longrightarrow&\mathbf Z_p(i+j)\\
\downarrow c_i\otimes c_j&&\downarrow c_{i+j}\\
E(i)\otimes^LE(j)&\longrightarrow&E(i+j).
\end{array}
$$
取 cohomology 后，横向 maps 正是 cup products；纵向 maps 为同构，故
products 相容。证毕。

## 11.5 Derived fixed points 的长正合列

**命题 11.12.** 对定义 11.4 的 $(C,u)$，存在长正合列
$$
\cdots\to H^n(C^{u=1})\to H^n(C)
\xrightarrow{u-1}H^n(C)
\to H^{n+1}(C^{u=1})\to\cdots.
$$

**证明.** 对 fibre triangle
$C^{u=1}\to C\xrightarrow{u-1}C$ 应用 cohomology functors 即得。证毕。

**推论 11.13.** 若 $u-1$ 在每个 $H^n(C)$ 上为同构，则
$C^{u=1}\simeq0$。

**证明.** 命题 11.12 中每个 kernel 与 cokernel 均为零，故
$H^n(C^{u=1})=0$ 对所有 $n$ 成立。证毕。

## 本章小结

Etale comparison 是 modulo $p^r$、invert prism ideal 与 derived
Frobenius fibre 的 finite-level theorem；$\mathbf Z_p$ 版本通过 derived
inverse limit 得到。Syntomic nearby-cycles theorem 精确带
$\tau^{\le i}$。这些 operations 与 invert $p$ 的 rational comparison
彼此不同。

## 练习

**练习 11.1.** 对 $M[0]$ 写出
$\operatorname{fib}(M\xrightarrow{u-1}M)$ 的 cohomology。

**练习 11.2.** 说明为什么 semilinear $\varphi$ 的 $\varphi-1$ 通常不是
$A$-linear，并指出定理 11.2 中可使用的固定 coefficient ring。

**练习 11.3.** 说明定理 11.8 (2) 去掉 $\tau^{\le i}$ 后为何不是同一
comparison statement。
