# 第二十二章：函数域类比和数论-几何桥梁

函数域 $K=\mathbb F_q(X)$ 同时是一维整体域和代数曲线的函数域，因此数论与几何两种语言可以真正相遇。Adelic 双商描述 $G$-bundle 的有限域点，Hecke correspondence 几何化 Hecke 算子，Frobenius trace 又把层送回函数。若把有限域换成代数闭域，trace function 消失，但层范畴及其 Hecke 作用仍然存在，这便解释了几何 Langlands为何比函数域 Langlands保留更多结构。本章沿这条桥梁比较 Drinfeld--Lafforgue、shtukas、几何 Satake 和范畴化对应，同时指出数域不能原样复制这套几何模型的原因。

Sheaf--function dictionary、Grothendieck--Lefschetz trace formula 与函数域 `GL(n)` 定理均作为外部输入；shtuka 和 excursion operators 见附录 S，Fargues--Fontaine 曲线与 local shtukas 见附录 AC。几何/算术 Frobenius、trace function 与 Tate twist 约定见 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 2、6、9 节。

## 22.1 函数域的双重身份

设 $X/\mathbb F_q$ 为光滑射影几何连通曲线，函数域为
$$
K=\mathbb F_q(X).
$$

**定义 22.1.** 函数域 Langlands 的数论侧对象包括：

1. $G(\mathbb A_K)$ 的自守表示或自守函数；
2. $G(K)\backslash G(\mathbb A_K)$ 上的 Hecke eigenfunctions；
3. $G_K=\operatorname{Gal}(\overline K/K)$ 的 $\ell$-adic 表示或 L 参数。

几何侧对象包括：

1. $\operatorname{Bun}_G(X)$ 上的 sheaves；
2. Hecke correspondence；
3. $\widehat G$-local systems on $X$。

**命题 22.2.** Weil uniformization 诱导 $\operatorname{Bun}_G(\mathbb F_q)$ 的点集与 adelic 双商
$$
G(K)\backslash G(\mathbb A_K)/G(\mathcal O_{\mathbb A})
$$
之间的标准对应。

**证明路线（外部输入）.** 第十八章的 Weil uniformization 给出该双商描述。一个 $G$-bundle 在 generic point 上平凡化后，由各闭点处的相对位置给出 adele 数据；改变 generic trivialization 对应左乘 $G(K)$，改变局部平凡化对应右乘 $G(\mathcal O_{\mathbb A})$。$\square$

## 22.2 Sheaf-Function Dictionary

**外部输入定理 22.3（sheaf-function dictionary）.** 设 $\mathcal X/\mathbb F_q$ 为局部有限型且满足 Grothendieck-Lefschetz trace formula 所需有限性条件的代数栈，$\mathcal F$ 为 constructible $\ell$-adic complex。对每个 $x\in\mathcal X(\mathbb F_q)$，定义
$$
f_{\mathcal F}(x)=\sum_i(-1)^i\operatorname{tr}(\operatorname{Frob}_x\mid H^i(\mathcal F_{\bar x})).
$$
该过程把 sheaves 上的六 functor 操作转化为函数上的拉回、推前和卷积操作；对非 proper 映射或非有限型栈，必须分别验证 compact support、收敛和 automorphism 权重条件。

**注 22.4.** 对 stack，需要除以 automorphism group 或使用 groupoid cardinality 修正。严格公式依赖 Grothendieck-Lefschetz trace formula for stacks。

**命题 22.5.** Hecke eigensheaf 的 Frobenius trace 给出 Hecke eigenfunction。

**证明.** 这是第二十章命题 20.11 的全局函数域版本。Hecke correspondence 的 sheaf push-pull 在 trace 下变成 Hecke 算子的函数求和；本征同构
$$
\mathsf H_V(\mathcal F)\cong\mathcal F\boxtimes V_{\mathcal E}
$$
在 trace 下给出 Hecke eigenvalue
$$
\operatorname{tr}(\operatorname{Frob}_x\mid V_{\mathcal E,x}).
$$
$\square$

**Frobenius 迹约定 22.A.** 把几何对象转成函数时采用以下约定：

| 几何侧 | 函数侧 | 归一化提醒 |
|---|---|---|
| $\ell$-adic complex $\mathcal F$ | trace function $f_{\mathcal F}$ | 使用同一 Frobenius 方向；若换成算术 Frobenius，需要整体取逆约定 |
| proper pushforward | 对纤维的加权求和 | stack 情形需除以 automorphism group |
| convolution of sheaves | Hecke algebra convolution | Haar 测度和 $q^{1/2}$ 正规化必须与经典 Satake 一致 |
| local system $\mathcal E$ | Frobenius eigenvalue system | 对应 Galois 表示的 characteristic polynomial |
| Tate twist | 函数乘以 $q$ 的幂 | 与 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 和几何 Satake 的半 Tate twist 对齐 |

## 22.3 Drinfeld 和 Lafforgue 的函数域定理

**外部输入定理 22.6（Drinfeld, `GL(2)` over function fields）.** 对函数域上的 `GL(2)`，Drinfeld 证明了相应的全局 Langlands 对应，将 rank-two $\ell$-adic Galois 表示与 cuspidal automorphic representations 联系起来。

**外部输入定理 22.7（Lafforgue, `GL(n)` over function fields）.** 对函数域上的 `GL(n)`，Lafforgue 证明了 cuspidal automorphic representations 与不可约 $\ell$-adic Galois representations 之间的对应，满足几乎所有位置的 Frobenius-Satake 相容性。

**注 22.8.** Drinfeld-Lafforgue 理论不是几何 Langlands 范畴等价本身，但它使用曲线、shtukas 和几何方法证明函数域全局 Langlands。它是数论 Langlands 与几何方法之间最强的桥梁之一。

**注 22.8.1.** 附录 S 将本节结果拆成 `GL(2)`、`GL(n)` 和一般还原群函数域参数化三层：Drinfeld、Laurent Lafforgue 和 V. Lafforgue。第十四章使用的是其中 `GL(n)` 层；一般 $G$ 的 excursion operator 形式不等同于完整 Arthur packet 分类。

## 22.4 Shtukas 的角色

**定义 22.9.** 在本书所需的接口层面，$G$-shtuka 是曲线 $X/\mathbb F_q$ 上的 $G$-bundle，连同若干点处的 Hecke 修改和 Frobenius pullback 之间的同构。对 `GL(n)`，它可表述为向量丛及其 Frobenius-twisted Hecke 修改数据。

**注 22.10.** Shtukas 同时包含 Hecke correspondence 和 Frobenius。它们的上同调携带 Galois 作用与 Hecke 作用，是 Drinfeld-Lafforgue 证明函数域 Langlands 的几何核心。

**外部输入定理 22.11（shtuka cohomology 的接口）.** Shtuka 模空间的 $\ell$-adic cohomology 同时带有 Hecke algebra 作用和 Galois group 作用。通过 excursion operators、Hecke 作用的谱分解或 `GL(n)` 情形中的 cuspidal 分解提取这些上同调，可构造函数域 Langlands 对应中的 Galois 表示。

## 22.5 数域情形为什么更难

数域没有对应的代数曲线 $X/\mathbb F_q$，因此不能直接用 $\operatorname{Bun}_G(X)$ 和 Frobenius trace 替代 adelic 商。

**注 22.12.** 数域仍有 Arakelov geometry、Shimura varieties、locally symmetric spaces 和 perfectoid spaces 等几何替代物，但它们不提供一个简单的“数域曲线”使几何 Langlands 原样适用。

**注 22.12.1.** Fargues-Fontaine 曲线提供的是 $p$-adic 局部域的几何化，而不是数域本身的全局曲线替代物。附录 AC 中的 local shtukas 和 Fargues-Scholze 谱作用解释局部 LLC 的几何来源；本章的函数域桥梁则依赖全局曲线 $X/\mathbb F_q$、Frobenius trace 和 adelic quotient。两者都使用曲线和 shtuka 型对象，但对应的基域、谱侧和自动侧不同。

**例 22.13.** 对 `GL(2)/\mathbb Q`，模曲线和 Shimura curves 的上同调构造二维 Galois 表示。这是第九章 Deligne 表示和第十章模性提升的几何背景。对更高维和一般群，需要 Shimura varieties 或 locally symmetric spaces 的上同调，并带来 torsion、boundary cohomology 和 endoscopy 等问题。

## 22.6 几何 Langlands 对数论的反馈

几何 Langlands 提供的不只是类比，还提供结构性工具：

1. 几何 Satake 给出对偶群的几何构造。
2. Hecke eigensheaves 范畴化 Hecke eigenfunctions。
3. Shtukas 提供函数域 Galois 表示构造。
4. Perverse sheaves 和 trace formula 的几何化推动 fundamental lemma 的证明。
5. Derived stacks 和 spectral categories 澄清 packet、singular support 和 Eisenstein series 的范畴结构。

**外部输入定理 22.14（Ngô 支持定理与 fundamental lemma，接口）.** Hitchin fibration 的几何研究和支持定理是 Ngô 证明 fundamental lemma 的核心输入之一，从而反过来支撑 endoscopy 和稳定 trace formula。

**注 22.15.** 这说明几何 Langlands 与数论 Langlands 不是两条完全分离的路线；几何方法已经成为数论端oscopy 和函数域 Langlands 的关键组成部分。

## 22.7 函数域的双重语言

函数域是数论和几何之间的桥梁。Adelic 双商可解释为 $\operatorname{Bun}_G$ 的点，Hecke 算子可解释为 Hecke correspondence，Hecke eigenfunctions 可由 Hecke eigensheaves 取 Frobenius trace 得到。Drinfeld 和 Lafforgue 的工作用曲线和 shtukas 证明了函数域 `GL(n)` Langlands。数域没有完全相同的几何模型，但 Shimura varieties、trace formula、perfectoid methods 和几何 Langlands 的范畴思想持续影响数域 Langlands。

## 练习

**练习 22.1.** 说明函数域 $K=\mathbb F_q(X)$ 如何同时具有 adelic 和曲线几何描述。

**练习 22.2.** 写出 sheaf-function dictionary 中 Frobenius trace 的公式。

**练习 22.3.** 解释 Hecke eigensheaf 取 trace 后为何给出 Hecke eigenfunction。

**练习 22.4.** 说明 shtuka 为什么同时携带 Hecke 作用和 Galois 作用。

**练习 22.5.** 列出数域情形不能直接复制函数域几何证明的两个原因。
