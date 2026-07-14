# 第十章：局部-整体相容性和降层

一个全局模形式同时产生所有素数处的局部表示，也产生一个可限制到各分解群的 Galois 表示。局部--整体相容性要求这两种局部化经过 Weil--Deligne 与局部 Langlands 后给出同一参数。降层定理利用这种相容性反向读取 residual conductor：若某些素数处的分歧在模 $\lambda$ 后消失，残余表示便可能来自更低级的 newform。Frey 曲线论证的关键正是把这种级数下降推到不存在的 $S_2(\Gamma_0(2))$。

本章调用第五、七、八、九章的局部参数、自守表示、导子和残余表示。局部--整体相容性、Serre 模性与 Ribet 降层均作为外部输入；椭圆曲线的约化和局部导子计算见附录 AD。所有 Frobenius、Tate twist 与局部因子转换按 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 2、6、7 节执行。

## 10.1 局部-整体相容性的形式

设 $f$ 为归一化 cuspidal newform，权 $k\ge2$，级 $N$，nebentypus $\varepsilon$。设
$$
\rho_{f,\lambda}:G_\mathbb Q\to\operatorname{GL}_2(E_{f,\lambda})
$$
为 Deligne 表示，$\pi_f=\otimes_v'\pi_{f,v}$ 为第七章中的 unitary cuspidal automorphic representation。固定 $\lambda\mid\ell$ 及同构
$\iota:\overline{\mathbb Q}_\ell\simeq\mathbb C$；$\operatorname{WD}_q$ 使用几何 Frobenius，且
$|\operatorname{Fr}_q|=q^{-1}$。

**外部输入定理 10.1（局部-整体相容性，$q\ne\ell$）.** 对每个素数 $q\ne\ell$，有 Frobenius-semisimple Weil-Deligne 表示同构
$$
\operatorname{rec}_{\mathbb Q_q,2}(\pi_{f,q})
\cong
\iota\,\operatorname{WD}_q
(\rho_{f,\lambda}^{\vee}|_{G_{\mathbb Q_q}})^{\mathrm{F\text{-}ss}}
\otimes|\cdot|^{(k-1)/2}.
$$
这里对偶、几何 Frobenius 和 norm twist 都是公式的一部分。特别地，对 $q\nmid N\ell$，二者均非分歧，并满足算术侧公式
$$
\operatorname{tr}\rho_{f,\lambda}(\operatorname{Frob}_q^{\operatorname{arith}})=a_q(f),
\qquad
\det\rho_{f,\lambda}(\operatorname{Frob}_q^{\operatorname{arith}})=\varepsilon(q)q^{k-1}.
$$

**注 10.2.** 对偶把 $\rho(\operatorname{Fr}_q)$ 的逆特征值变回 arithmetic Frobenius 的 classical roots；再张量
$|\cdot|^{(k-1)/2}$，把它们乘以 $q^{-(k-1)/2}$，得到 $\pi_f$ 的 unitary Satake roots。只说“至多差一个 twist”不能确定局部因子变量。

**外部输入定理 10.2.1（$q=\ell$ 的 p-adic Hodge 边界）.** 表示
$\rho_{f,\lambda}|_{G_{\mathbb Q_\ell}}$ 是 de Rham，按本书算术编号的 Hodge-Tate 权为
$\{0,k-1\}$；它是 potentially semistable，故可由
$D_{\operatorname{pst}}$ 产生 Weil-Deligne 数据。该数据与 $\pi_{f,\ell}$ 的相容性属于
p-adic Hodge 局部-整体相容定理，而不是定理 10.1 中 $q\ne\ell$ 的 Grothendieck monodromy 构造。本章的 Ribet 降层链不使用本定理。

**推论 10.3.** 对 $q\nmid N\ell$，
$$
L_q(\rho_{f,\lambda},s)=L_q(f,s)
=L_q(s-(k-1)/2,\pi_f,\operatorname{Std}).
$$
等价地，
$$
L_q(s,\pi_f,\operatorname{Std})
=L_q(s+(k-1)/2,\rho_{f,\lambda}).
$$

**证明.** Galois/classical 局部因子由
$$
1-a_q(f)X+\varepsilon(q)q^{k-1}X^2
$$
在 $X=q^{-s}$ 处给出。$\pi_f$ 的 unitary Satake roots 是该多项式两根分别乘
$q^{-(k-1)/2}$；因此其标准因子在变量 $u$ 处等于 classical 因子在
$s=u+(k-1)/2$ 处的值。令 $u=s-(k-1)/2$ 即得。$\square$

## 10.2 残余表示的导子

设
$$
\overline\rho:G_\mathbb Q\to\operatorname{GL}_2(k)
$$
为连续、奇、绝对不可约残余表示，其中 $k$ 为特征 $p$ 的有限域。

**定义 10.4.** 对素数 $q\ne p$，局部导子指数 $n_q(\overline\rho)$ 定义为 $\overline\rho|_{G_{\mathbb Q_q}}$ 的 Artin 导子指数。Prime-to-$p$ Serre conductor 定义为
$$
N(\overline\rho)=\prod_{q\ne p}q^{n_q(\overline\rho)}.
$$

**定义 10.5.** 若存在权 $k$、级 $N$、nebentypus $\varepsilon$ 的归一化 eigenform $f$ 和 $E_f$ 的素位 $\lambda\mid p$，使
$$
\overline\rho\cong\overline\rho_{f,\lambda}
$$
半单同构，则称 $\overline\rho$ 是模的，且来自 $f$。

**外部输入定理 10.6（Serre 模性定理，接口形式）.** 每个连续、奇、绝对不可约的二维残余表示
$$
\overline\rho:G_\mathbb Q\to\operatorname{GL}_2(\overline{\mathbb F}_p)
$$
都来自某个模形式。更精确地，Serre 的 recipe 预测并定理化了最小权、最小级和 nebentypus，其中 prime-to-$p$ 级由 $N(\overline\rho)$ 给出。

该定理由 Khare-Wintenberger 及相关工作证明。本书在费马应用中主要使用 Ribet 降层的历史版本，而不是完整 Serre 模性定理。

## 10.3 降层的基本思想

降层比较两个 level：

1. 自守侧：newform 的级 $N$。
2. Galois 侧：残余表示 $\overline\rho_{f,\lambda}$ 的 Artin/Serre 导子 $N(\overline\rho)$。

若某个素数 $q\mid N$ 出现在 $f$ 的级中，但模 $p$ 表示 $\overline\rho_{f,\lambda}$ 在 $q$ 处的 ramification 比 $\rho_{f,\lambda}$ 更小，则预期 $\overline\rho_{f,\lambda}$ 应来自一个不含 $q$ 的较低级 newform。Ribet 降层定理正是这种预期的核心形式。

**定义 10.7.** 设 $f$ 为 newform，$\lambda\mid p$。若另一个 newform $g$ 满足
$$
\overline\rho_{f,\lambda}\cong\overline\rho_{g,\lambda'}
$$
对某个 $\lambda'\mid p$ 成立，则称 $f$ 和 $g$ 模 $p$ Galois 同余。

## 10.4 Ribet 降层定理

**外部输入定理 10.8（Ribet 单素数降层，平方自由接口）.** 设 $p\ge3$，$f$ 为权 $2$、trivial nebentypus 的 newform，级
$N=qM$，其中 $\gcd(pq,M)=1$、$p\ne q$ 且 $q\parallel N$。设残余表示
$$
\overline\rho_{f,\lambda}:G_\mathbb Q\to\operatorname{GL}_2(k_\lambda)
$$
绝对不可约，并假设它在 $q$ 处非分歧。则存在权 $2$、trivial nebentypus、级
$M'\mid M$ 的 newform $g$ 和 $\lambda'\mid p$，使得
$$
\overline\rho_{g,\lambda'}\cong\overline\rho_{f,\lambda}
$$
且在所有 $r\nmid Np$ 处有 $a_r(g)\equiv a_r(f)\pmod{\lambda'}$。

**注 10.9.** 定理 10.8 有意只陈述 trivial nebentypus、$q\parallel N$ 的 one-prime 版本。$q^2\mid N$、非平凡 nebentypus、$q=p$ 和一般 Serre weight 需要 Carayol、Diamond、Edixhoven 等精化，不能由本定理字面推出。费马应用实际调用的是下一个半稳定最优级版本。

## 10.5 半稳定椭圆曲线的降层口径

设 $E/\mathbb Q$ 为半稳定椭圆曲线，$p\ge5$，并考虑
$$
\overline\rho_{E,p}:G_\mathbb Q\to\operatorname{GL}_2(\mathbb F_p).
$$

**外部输入定理 10.10（Ribet epsilon theorem，半稳定最优级版本）.** 设 $E/\mathbb Q$ 为 modular semistable elliptic curve，$p\ge5$。假设
$\overline\rho_{E,p}$ 绝对不可约；在 $p$ 处取由 $E[p]$ 给出的 finite-flat/Serre-weight-$2$ 局部类型，并令
$$
N(\overline\rho_{E,p})=\prod_{q\ne p}q^{n_q(\overline\rho_{E,p})}.
$$
则 $\overline\rho_{E,p}$ 来自权 $2$、trivial nebentypus、级
$N(\overline\rho_{E,p})$ 的 Hecke eigenform；取其 primitive constituent 后得到级整除该整数的 newform。在 Frey 情形，局部计算把最优级精确确定为 $2$。

**证明路线（外部输入）.** 模性先把 $\overline\rho_{E,p}$ 放入某个权 $2$ Hecke eigensystem。Ribet 的 epsilon theorem 及其局部精化比较 newvector level 与残余 Artin conductor，并逐个删除 lift 中存在而残余导子中不存在的素因子；$p$ 处 finite-flat 条件保持 Serre weight $2$。这些删除步骤依赖 Ihara lemma、模曲线 Jacobian 和 Hecke 代数，本书不重证。

## 10.6 Frey 曲线的导子下降

设存在 primitive Fermat 反例
$$
a^p+b^p=c^p,\qquad p\ge5.
$$
Frey 曲线为
$$
E_{a,b,p}:\quad y^2=x(x-a^p)(x+b^p).
$$

**外部输入定理 10.11（Frey 曲线的局部导子计算）.** Frey 曲线 $E=E_{a,b,p}$ 满足：

1. $E$ 半稳定。
2. 除 $2$ 外，坏素数都整除 $abc$。
3. 对每个奇素数 $q\ne p$ 且 $q\mid abc$，模 $p$ 表示 $\overline\rho_{E,p}$ 在 $q$ 处的导子指数为 $0$，从而 Ribet 降层可从 prime-to-$p$ level 中删除这些 $q$。
4. 在素数 $p$ 处，$\overline\rho_{E,p}$ 满足相应的有限平坦或 Serre weight $2$ 局部条件。
5. 在素数 $2$ 处，局部计算使最终剩余级为 $2$。
6. $\overline\rho_{E,p}$ 绝对不可约，因而满足定理 10.10 的全局不可约性假设。

**注 10.12.** 第 3 项是 Frey 曲线构造的关键：曲线本身在 $q\mid abc$ 处有乘法坏约化，故 $q$ 出现在 $N_E$ 中；但当 $q\ne p$ 时，由于判别式在这些素数处的指数含有 $p$ 的倍数，模 $p$ 表示的 ramification 降低，从而 prime-to-$p$ 残余导子不含这些 $q$。素数 $p$ 处不是用同一个“删除 $q$”论证处理，而是进入 Serre weight 和局部有限平坦条件。第 6 项同样是独立深输入：它不能由导子等式形式推出，标准证明使用 Frey 曲线的特殊性质及有理 isogeny/残余表示结果。

**注 10.12.1.** 附录 AD.7 将 Frey 曲线的局部导子输入拆为判别式计算、奇素数处乘法约化、$v_q(\Delta_E)$ 被 $p$ 整除、以及 prime-to-$p$ residual conductor 降到 $2$ 四个步骤；命题 AD.21 只承担定理 10.11(1)--(3)、(5) 的这部分局部结论。$p$ 处 Serre weight 条件和 10.11(6) 的绝对不可约性是另外的 Frey-Serre-Ribet 外部输入，不能从 AD.21 推出。

**Frey--Ribet 论证的逻辑链 10.A.** 费马应用需要本章提供如下转换：

| 输入 | 来源 | 本章作用 |
|---|---|---|
| Frey 曲线半稳定和局部导子计算 | 附录 AD 与定理 10.11 | 确定可删除的奇坏素数和最终 residual level |
| 半稳定椭圆曲线的模性 | 第九章定理 9.23 | 把 $\overline\rho_{E,p}$ 放入模形式来源 |
| Ribet 降层条件 | 定理 10.8 和 10.10 | 从原始级降到权 $2$、级 $2$ |
| $S_2(\Gamma_0(2))=0$ | 附录 W 与第 90 章 | 把级 $2$ newform 结论转为矛盾 |

**命题 10.13（Frey-Ribet 级数结论）.** 接受半稳定模性定理、Frey 曲线局部导子计算和 Ribet 降层定理，则 Fermat 反例给出的模 $p$ 表示
$$
\overline\rho_{E_{a,b,p},p}
$$
来自权 $2$、级 $2$ 的 newform。

**证明.** 由外部输入定理 10.11，$E=E_{a,b,p}$ 半稳定，且 $\overline\rho_{E,p}$ 绝对不可约。由半稳定模性定理，$E$ 模，因此 $\overline\rho_{E,p}$ 来自某个权 $2$、级 $N_E$ 的 newform。由 Frey 曲线局部导子计算，所有奇素数 $q\ne p$ 且 $q\mid abc$ 都满足 Ribet 降层删除条件；定理 10.10 的不可约性假设由 10.11(6) 保证，故可删除这些 prime-to-$p$ 素数。素数 $p$ 处的局部条件保证残余表示具有正确的 Serre weight $2$，素数 $2$ 处的局部计算把最优剩余级确定为 $2$。于是 $\overline\rho_{E,p}$ 来自权 $2$、级 $2$ 的 newform。$\square$

## 10.7 级 `2` 的矛盾

**命题 10.14.** 若存在权 $2$、级 $2$ 的 newform，则
$$
S_2(\Gamma_0(2))\ne0.
$$

**证明.** Newform 是 $S_2(\Gamma_0(2))$ 的非零元素。因此存在 newform 直接推出该空间非零。$\square$

结合第九十章的引理 $S_2(\Gamma_0(2))=0$，命题 10.13 给出 Fermat 反例不存在的核心矛盾。

## 10.8 从局部相容性到降层

局部-整体相容性把模形式的局部 Hecke 数据、Galois 表示的局部 ramification 和自守表示的局部分量对应起来。Ribet 降层说明，如果残余表示在某些坏素数处 ramification 降低，则它来自更低 level 的 newform。Frey 曲线恰好制造了这种导子下降：曲线本身有很多坏素数，但模 $p$ 表示的 residual conductor 删除了它们，最终降到级 $2$，与 $S_2(\Gamma_0(2))=0$ 矛盾。

## 练习

**练习 10.1.** 设 $f$ 为权 $2$ newform，$q\nmid Np$。写出 $\rho_{f,\lambda}$ 在 $q$ 处的 characteristic polynomial，并由此计算局部 Euler 因子。

**练习 10.2.** 说明为什么 residual representation 的导子可能小于其某个 lift 的导子。

**练习 10.3.** 用定理 10.8 的语言解释“降层”删除的是 newform 的 level 中的素因子，而不是改变残余表示本身。

**练习 10.4.** 假设 Frey 曲线的所有局部输入成立，复述命题 10.13 的证明。

**练习 10.5.** 解释为什么级 $2$ newform 的不存在性足以结束费马大定理的 Langlands 型证明。
