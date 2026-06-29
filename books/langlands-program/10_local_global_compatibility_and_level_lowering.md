# 第十章：局部-整体相容性和降层

## 本章目标

本章解释 `GL(2)/\mathbb Q` 中模形式、Galois 表示和局部表示之间的相容性，并陈述 Ribet 降层定理的接口形式。重点是说明：为什么一个来自高 level newform 的残余表示，在满足局部条件时也来自较低 level；以及该机制如何把 Frey 曲线的模性降到级 $2$。

## 依赖前置知识

需要第五章的局部参数，第七章的 `GL(2)` 自守表示，第八章的椭圆曲线导子，第九章的残余表示和模性。本章把局部-整体相容性、Serre 导子、Serre 模性定理和 Ribet 降层作为外部输入。附录 AD 给出本章使用的椭圆曲线约化类型、Tate algorithm、Ogg conductor formula 和 Frey 曲线局部导子接口。

收口归一化回指：本章比较 $\rho_{f,\lambda}|_{G_{\mathbb Q_q}}$、Weil-Deligne 参数、局部 `GL(2)` 表示和 residual conductor；所有 Frobenius、Tate twist 和局部因子转换见 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 2、6、7 节。

## 10.1 局部-整体相容性的形式

设 $f$ 为归一化 cuspidal newform，权 $k\ge2$，级 $N$，nebentypus $\varepsilon$。设
$$
\rho_{f,\lambda}:G_\mathbb Q\to\operatorname{GL}_2(E_{f,\lambda})
$$
为 Deligne 表示，$\pi_f=\otimes_v'\pi_{f,v}$ 为第七章中的 cuspidal automorphic representation。

**外部输入定理 10.1（局部-整体相容性，接口形式）.** 对每个素数 $q$，$\rho_{f,\lambda}|_{G_{\mathbb Q_q}}$ 关联的 Frobenius-semisimple Weil-Deligne 表示与 $\pi_{f,q}$ 在局部 Langlands 对应下的参数相容，至多差一个由权和归一化决定的 Tate twist。特别地，对 $q\nmid N\ell$，二者均非分歧，并满足
$$
\operatorname{tr}\rho_{f,\lambda}(\operatorname{Frob}_q^{\operatorname{arith}})=a_q(f),
$$
$$
\det\rho_{f,\lambda}(\operatorname{Frob}_q^{\operatorname{arith}})=\varepsilon(q)q^{k-1}.
$$

**注 10.2.** 本书在 Galois 表示章节使用算术 Frobenius，而第五章局部 Langlands 参数默认使用几何 Frobenius。因此完整相容性必须说明取逆、对偶和 Tate twist 的 convention。本章只在需要比较 trace 和 Euler 因子时使用算术 Frobenius 口径。

**推论 10.3.** 对 $q\nmid N\ell$，
$$
L_q(\rho_{f,\lambda},s)=L_q(f,s)=L_q(\pi_f,s)
$$
在第六、七、九章采用的算术归一化下成立。

**证明.** 三个局部因子都由同一个二次多项式
$$
1-a_q(f)X+\varepsilon(q)q^{k-1}X^2
$$
在 $X=q^{-s}$ 处给出。$\square$

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

**外部输入定理 10.8（Ribet 降层，接口形式）.** 设 $p\ge3$，$f$ 为权 $2$ newform，级 $N$，$\lambda\mid p$。设残余表示
$$
\overline\rho_{f,\lambda}:G_\mathbb Q\to\operatorname{GL}_2(k_\lambda)
$$
绝对不可约。若素数 $q\ne p$ 满足：

1. $q\mid N$；
2. $\overline\rho_{f,\lambda}$ 在 $q$ 处的导子指数小于 $f$ 的局部级指数；
3. 满足 Ribet 定理所需的局部最小性和奇性条件；

则存在较低级 $N'$ 的权 $2$ newform $g$，其中 $N'$ 从 $N$ 中删除相应的 $q$-因子，并且
$$
\overline\rho_{g,\lambda'}\cong\overline\rho_{f,\lambda}
$$
对某个 $\lambda'\mid p$ 成立。

**注 10.9.** 定理 10.8 是接口陈述，不是完整技术陈述。完整版本需要区分 $q\parallel N$、$q^2\mid N$、nebentypus、局部表示类型、有限平坦条件和 $p$ 处条件。费马应用只需要半稳定 Frey 曲线给出的特殊情形。

## 10.5 半稳定椭圆曲线的降层口径

设 $E/\mathbb Q$ 为半稳定椭圆曲线，$p\ge5$，并考虑
$$
\overline\rho_{E,p}:G_\mathbb Q\to\operatorname{GL}_2(\mathbb F_p).
$$

**外部输入定理 10.10（半稳定椭圆曲线的降层接口）.** 假设 $E$ 模，且 $\overline\rho_{E,p}$ 绝对不可约。若某个坏素数 $q\ne p$ 满足局部条件使得 $\overline\rho_{E,p}$ 在 $q$ 处的导子指数小于 $E$ 的导子指数，则 $\overline\rho_{E,p}$ 来自删除 $q$ 后的较低级权 $2$ newform。对一组这样的素数反复应用，可把级降到 $\overline\rho_{E,p}$ 的 prime-to-$p$ Serre conductor。

**证明草图.** 由 $E$ 模，存在权 $2$ newform $f_E$，级 $N_E$，使 $\rho_{E,p}$ 与 $\rho_{f_E,\lambda}$ 相容。于是 $\overline\rho_{E,p}$ 来自 $f_E$。对每个满足导子下降条件的 $q\mid N_E$，应用 Ribet 降层定理 10.8，得到去掉该 $q$-因子的较低级 newform，残余表示不变。重复有限次后得到最小 prime-to-$p$ 级。$\square$

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

**注 10.12.** 第 3 项是 Frey 曲线构造的关键：曲线本身在 $q\mid abc$ 处有乘法坏约化，故 $q$ 出现在 $N_E$ 中；但当 $q\ne p$ 时，由于判别式在这些素数处的指数含有 $p$ 的倍数，模 $p$ 表示的 ramification 降低，从而 prime-to-$p$ 残余导子不含这些 $q$。素数 $p$ 处不是用同一个“删除 $q$”论证处理，而是进入 Serre weight 和局部有限平坦条件。

**注 10.12.1.** 附录 AD.7 将本节的 Frey 曲线局部输入拆为判别式计算、奇素数处乘法约化、$v_q(\Delta_E)$ 被 $p$ 整除、以及 residual conductor 降到 $2$ 四个步骤。命题 AD.21 说明这些步骤正推出定理 10.11 的接口形式。

**命题 10.13（Frey-Ribet 级数结论）.** 接受半稳定模性定理、Frey 曲线局部导子计算和 Ribet 降层定理，则 Fermat 反例给出的模 $p$ 表示
$$
\overline\rho_{E_{a,b,p},p}
$$
来自权 $2$、级 $2$ 的 newform。

**证明.** 由外部输入定理 10.11，$E=E_{a,b,p}$ 半稳定。由半稳定模性定理，$E$ 模，因此 $\overline\rho_{E,p}$ 来自某个权 $2$、级 $N_E$ 的 newform。由 Frey 曲线局部导子计算，所有奇素数 $q\ne p$ 且 $q\mid abc$ 都满足 Ribet 降层删除条件；反复应用定理 10.10 删除这些 prime-to-$p$ 素数。素数 $p$ 处的局部条件保证残余表示具有正确的 Serre weight $2$，素数 $2$ 处的局部计算保留级 $2$。于是 $\overline\rho_{E,p}$ 来自权 $2$、级 $2$ 的 newform。$\square$

## 10.7 级 `2` 的矛盾

**命题 10.14.** 若存在权 $2$、级 $2$ 的 newform，则
$$
S_2(\Gamma_0(2))\ne0.
$$

**证明.** Newform 是 $S_2(\Gamma_0(2))$ 的非零元素。因此存在 newform 直接推出该空间非零。$\square$

结合第九十章的引理 $S_2(\Gamma_0(2))=0$，命题 10.13 给出 Fermat 反例不存在的核心矛盾。

## 10.8 本章小结

局部-整体相容性把模形式的局部 Hecke 数据、Galois 表示的局部 ramification 和自守表示的局部分量对应起来。Ribet 降层说明，如果残余表示在某些坏素数处 ramification 降低，则它来自更低 level 的 newform。Frey 曲线恰好制造了这种导子下降：曲线本身有很多坏素数，但模 $p$ 表示的 residual conductor 删除了它们，最终降到级 $2$，与 $S_2(\Gamma_0(2))=0$ 矛盾。

## 练习

**练习 10.1.** 设 $f$ 为权 $2$ newform，$q\nmid Np$。写出 $\rho_{f,\lambda}$ 在 $q$ 处的 characteristic polynomial，并由此计算局部 Euler 因子。

**练习 10.2.** 说明为什么 residual representation 的导子可能小于其某个 lift 的导子。

**练习 10.3.** 用定理 10.8 的语言解释“降层”删除的是 newform 的 level 中的素因子，而不是改变残余表示本身。

**练习 10.4.** 假设 Frey 曲线的所有局部输入成立，复述命题 10.13 的证明。

**练习 10.5.** 解释为什么级 $2$ newform 的不存在性足以结束费马大定理的 Langlands 型证明。
