# 附录 Z：局部调和分析、Harish-Chandra Characters 和 Plancherel 接口

**收口归一化回指。** 本附录涉及 Haar measure、卷积、Fourier transform、characters 和 Plancherel measure；与迹公式、局部因子和 spherical transform 比较时使用 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 3、4、6、8 节。

## Z.1 测试函数和表示范畴

设 $F$ 为非 Archimedean 局部域，$G/F$ 为 connected reductive group。令
$$
G=G(F).
$$

**定义 Z.1.** Hecke 测试函数空间为
$$
C_c^\infty(G)
$$
即 $G$ 上 compactly supported locally constant functions。它在卷积下成为代数。若 $\pi$ 是 smooth representation，定义
$$
\pi(f)v=\int_G f(g)\pi(g)v\,dg.
$$

**命题 Z.2.** 若 $\pi$ 是 admissible smooth representation，则 $\pi(f)$ 在每个开紧不变量空间上为有限秩算子。

**证明.** 取开紧子群 $J$ 使 $f$ 左右 $J$-不变。则 $\pi(f)$ 的像包含在 $\pi^J$ 中，因为左 $J$-不变性给出
$$
\pi(j)\pi(f)=\pi(f).
$$
Admissibility 给出 $\dim\pi^J<\infty$，故 $\pi(f)$ 有有限维像。$\square$

## Z.2 Matrix coefficients 和 temperedness

**定义 Z.3.** 若 $\pi$ 为 smooth representation，$v\in\pi$，$\lambda\in\pi^\vee$，matrix coefficient 为
$$
m_{v,\lambda}(g)=\lambda(\pi(g)v).
$$

**定义 Z.4.** 不可约 admissible representation $\pi$ 称为 tempered，若其 matrix coefficients modulo center 满足 Harish-Chandra 意义下的 $L^{2+\epsilon}$ 衰减条件。称为 square-integrable modulo center，若 matrix coefficients modulo center 属于 $L^2$。

**外部输入定理 Z.5（Harish-Chandra tempered criterion）.** 对 reductive $p$-adic group，不可约 admissible representation 的 temperedness 可由 matrix coefficients 衰减、exponents of Jacquet modules 或 Plancherel support 等价刻画。

**命题 Z.6.** Square-integrable modulo center 的不可约表示是 tempered。

**证明.** 若 matrix coefficients modulo center 属于 $L^2$，则由 Hölder 不等式和 Harish-Chandra 的增长控制，它们满足 $L^{2+\epsilon}$ 型 tempered 条件。严格推出使用定理 Z.5 的等价刻画。$\square$

## Z.3 Characters as distributions

**定义 Z.7.** 对 admissible representation $\pi$，其 distribution character 定义为
$$
\Theta_\pi(f)=\operatorname{tr}\pi(f),\qquad f\in C_c^\infty(G).
$$

**外部输入定理 Z.8（Harish-Chandra character theorem）.** 若 $\pi$ 不可约 admissible，则 $\Theta_\pi$ 由 $G$ 上 regular semisimple locus 的 locally constant conjugation-invariant function 表示，并在全群上为 locally integrable invariant distribution。

**命题 Z.9.** 若两个不可约 admissible 表示的 characters 相等，则它们同构。

**证明路线（外部输入）.** Hecke algebra 在 admissible category 上分离不可约对象。若 $\Theta_\pi=\Theta_{\pi'}$，则所有 compactly supported bi-$J$-invariant Hecke operators 的 trace 在 $\pi^J$ 与 $(\pi')^J$ 上相同。对足够多 $J$，这些 traces 决定相应 Hecke algebra 的简单模 character。由 admissible 表示的 Bernstein 分解与 character 线性无关性，得到 $\pi\simeq\pi'$。完整证明依赖 Harish-Chandra character 线性无关性。$\square$

## Z.4 Plancherel 分解

**外部输入定理 Z.10（local Plancherel formula）.** 群 $G$ 的右正则表示在 $L^2(G)$ 上可按 tempered dual 分解：
$$
L^2(G)\simeq \int_{\widehat G_{\operatorname{temp}}}^{\oplus}\pi\otimes\pi^\vee\,d\mu_{\operatorname{Pl}}(\pi),
$$
并且对 $f\in C_c^\infty(G)$ 有 Plancherel 公式
$$
f(1)=\int_{\widehat G_{\operatorname{temp}}}\Theta_\pi(f)\,d\mu_{\operatorname{Pl}}(\pi)
$$
在固定 Haar 测度并取相应 Plancherel measure normalization 后成立。

**命题 Z.11.** Plancherel support 只含 tempered representations。

**证明.** 这是定理 Z.10 的内容之一：右正则表示的谱分解支撑在 tempered dual 上。若非 tempered 表示出现于 Plancherel support，则其 matrix coefficients 衰减不足以出现在 $L^2(G)$ 的正则分解中，违背 Harish-Chandra Plancherel theorem。$\square$

## Z.5 Bernstein 中心和 Paley-Wiener

**外部输入定理 Z.12（Bernstein decomposition）.** Smooth representations of $G$ 的范畴分解为 Bernstein blocks：
$$
\operatorname{Rep}(G)=\prod_{\mathfrak s}\operatorname{Rep}(G)_{\mathfrak s},
$$
其中 $\mathfrak s$ 由 cuspidal support modulo unramified twists 给出。Bernstein center 作用在每个不可约表示上为标量。

**外部输入定理 Z.13（local Paley-Wiener theorem）.** Hecke algebra $C_c^\infty(G)$ 的 Fourier transform
$$
f\mapsto(\pi\mapsto \operatorname{tr}\pi(f))
$$
的像可由 Bernstein varieties 上的正则性、支撑和相容条件刻画。

**命题 Z.14.** Trace formula 中用测试函数分离局部表示依赖 Paley-Wiener 型定理。

**证明.** 要在 global trace formula 的谱侧隔离指定局部表示 $\pi_v$，需要选择 $f_v\in C_c^\infty(G_v)$，使 $\operatorname{tr}\sigma_v(f_v)$ 对目标表示非零而对其他 Bernstein components 或 packets 有控制。定理 Z.13 正是说明哪些 spectral-side functions 来自 compactly supported test functions。因此分离局部谱不能只靠形式线性代数，还依赖 local Paley-Wiener。$\square$

## Z.6 Orbital integrals 和局部字符展开

**定义 Z.15.** 对 regular semisimple $\gamma\in G$，orbital integral 为
$$
O_\gamma(f)=\int_{G_\gamma\backslash G}f(x^{-1}\gamma x)\,dx.
$$

**外部输入定理 Z.16（local character expansion）.** Harish-Chandra character 在半单元素附近可按 nilpotent orbits 的 Fourier transforms 展开。特别地，在单位元附近，
$$
\Theta_\pi(\exp X)=\sum_{\mathcal O}c_{\mathcal O}(\pi)\widehat\mu_{\mathcal O}(X)
$$
在 $0$ 的充分小邻域和标准 good-characteristic hypotheses 下成立。

**命题 Z.17.** Local character expansion 是稳定字符和 endoscopy 的局部技术输入之一。

**证明路线（外部输入）.** Endoscopic character identities 比较的是 packets 的稳定字符组合。要验证或归一化这些分布，需知道 characters 在 regular semisimple locus 和 singular neighborhoods 的行为。定理 Z.16 把 singular behavior 化为 nilpotent orbital integrals 的 Fourier transforms，使 transfer factors 和稳定组合可被局部计算控制。$\square$

## Z.7 与 Langlands 纲领的接口

**命题 Z.18.** 局部 LLC 的 tempered 性相容可表述为 Plancherel/Harish-Chandra 语言。

**证明.** LLC 预期 bounded Langlands parameters 对应 tempered representations。Harish-Chandra 理论从表示侧定义 temperedness，Plancherel theorem 说明 tempered dual 是正则表示的谱支撑。因而 LLC 的 tempered 相容性可解释为：bounded Galois 参数正好对应局部调和分析中出现于 Plancherel 分解的表示类型。$\square$

**命题 Z.19.** Trace formula 的谱侧需要 Harish-Chandra characters 才能从形式和变成分布等式。

**证明.** Trace formula 谱侧含有 $\operatorname{tr}\pi(f)$。对 admissible 表示，该 trace 由 character distribution $\Theta_\pi$ 给出。Harish-Chandra 定理保证这些 distributions 由局部可积函数表示，并满足线性无关和局部展开性质。没有这些结果，谱侧只是形式求和，不能与几何侧 orbital integrals 作为 distributions 比较。$\square$

## 练习

**练习 Z.1.** 证明 $\pi(f)$ 的像落在某个开紧不变量空间中。

**练习 Z.2.** 解释 square-integrable modulo center 与 tempered 的关系。

**练习 Z.3.** 说明 character distribution 为什么是 trace formula 谱侧的局部输入。

**练习 Z.4.** 解释 Plancherel formula 为什么只支撑在 tempered dual 上。

**练习 Z.5.** 说明 Paley-Wiener theorem 在构造测试函数时的作用。
