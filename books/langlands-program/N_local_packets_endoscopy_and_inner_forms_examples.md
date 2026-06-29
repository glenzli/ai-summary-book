# 附录 N：局部 Packets、内形式和 Endoscopy 例子

本附录补充第十二、十六、十七章中 L-packet、component group、inner form 和 endoscopy 的具体模型。目的不是给出完整局部 Langlands 分类，而是提供可计算例子，说明为什么一般还原群不能像 `GL(n)` 那样只用单个表示描述一个参数。

**收口归一化回指。** 本附录涉及 L-packets、enhancement、inner twist、endoscopic transfer 和 transfer factor；与稳定迹公式和 Arthur 分类比较时使用 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 2、4、8 节。

## N.1 Component Group 和增强参数

设 $F$ 为局部域，$G/F$ 为 connected reductive group，${}^LG=\widehat G\rtimes W_F$。

**定义 N.1.** 对 Langlands 参数
$$
\varphi:W_F'\to{}^LG
$$
定义 centralizer
$$
S_\varphi=\operatorname{Cent}_{\widehat G}(\operatorname{im}\varphi).
$$
常用 component group 为
$$
\mathcal S_\varphi=\pi_0(S_\varphi/Z(\widehat G)^{W_F}).
$$

**定义 N.2.** 增强参数是二元组
$$
(\varphi,\rho),
$$
其中 $\rho\in\operatorname{Irr}(\mathcal S_\varphi)$，并需满足与内形式或 pure inner twist 相关的 relevance 条件。

**命题 N.3.** 对 $G=\operatorname{GL}_n$，$\mathcal S_\varphi$ 在 LLC 所需意义下平凡，因此 L-packet 为单元素。

**证明草图.** $\widehat G=\operatorname{GL}_n(\mathbb C)$。任意 semisimple Weil-Deligne 参数对应一个 $n$ 维表示。其 centralizer 是各同构不可约 summands 的 multiplicity spaces 上的 general linear groups 的乘积，连通。商去中心后 component group 仍平凡。因此没有 packet 内部离散标签。$\square$

## N.2 Tori

设 $T/F$ 为 torus。

**外部输入定理 N.4（局部 Langlands for tori）.** 有自然对应
$$
\operatorname{Hom}_{\operatorname{cont}}(T(F),\mathbb C^\times)
\longleftrightarrow
H^1(W_F,\widehat T)
$$
或等价的 admissible homomorphisms $W_F\to{}^LT$ 的共轭类。该对应由局部类域论和 Tate-Nakayama duality 给出。

**命题 N.5.** 若 $T=\mathbb G_m$, 则 N.4 退化为局部类域论：
$$
F^\times{}^\vee\simeq\operatorname{Hom}_{\operatorname{cont}}(W_F,\mathbb C^\times).
$$

**证明.** 此时 $\widehat T=\mathbb C^\times$，且 split Galois action 平凡。L 参数就是一维 Weil character。局部 reciprocity map
$$
F^\times\to W_F^{\operatorname{ab}}
$$
给出二者对应。$\square$

## N.3 `SL(2)` 的 Packet 现象

设 $G=\operatorname{SL}_2$。其对偶群为
$$
\widehat G=\operatorname{PGL}_2(\mathbb C).
$$

**注 N.6.** `SL(2)` 的局部 packets 可通过限制 `GL(2)` 的表示来观察。若 $\widetilde\pi$ 是 $\operatorname{GL}_2(F)$ 的 irreducible admissible representation，则其限制到 $\operatorname{SL}_2(F)$ 通常分解为有限直和。组成这些 summands 的集合与同一个 projective Langlands 参数相关。

**外部输入定理 N.7（`SL(2)` packet 的限制模型）.** 对合适的 $\widetilde\pi\in\operatorname{Irr}(\operatorname{GL}_2(F))$，限制
$$
\widetilde\pi|_{\operatorname{SL}_2(F)}
$$
是有限长度且 multiplicity-free。其不可约 summands 构成与 projective parameter
$$
W_F'\to\operatorname{PGL}_2(\mathbb C)
$$
相关的 L-packet 的一个模型。Packet 大小由某个 component group 或 self-twist 群控制。

**例 N.8.** 若 $\widetilde\pi$ 没有非平凡 self-twist，即没有非平凡 character $\chi$ 使
$$
\widetilde\pi\otimes(\chi\circ\det)\simeq\widetilde\pi,
$$
则限制到 $\operatorname{SL}_2(F)$ 的 packet 通常较小；有 self-twist 时 packet 可能含多个 summands。

**注 N.9.** 这说明一般群的 LLC 不能只陈述为“参数与表示一一对应”。同一个参数可对应多个表示；还需 component group character 标记 packet 内成员。

## N.4 Inner Forms

设 $D/F$ 为四元数除代数，$D^\times$ 是 $\operatorname{GL}_2$ 的内形式。

**外部输入定理 N.10（Jacquet-Langlands，局部接口）.** $\operatorname{GL}_2(F)$ 的 discrete series representations 与 $D^\times$ 的 irreducible smooth representations 之间有局部 Jacquet-Langlands 对应。二者共享同一二维 Weil-Deligne 参数，但出现在不同 inner forms 上。

**命题 N.11.** 内形式迫使 enhanced LLC 记录额外 relevance 数据。

**证明草图.** 同一 Langlands 参数可能在 split group 和非 split inner form 上都有相关表示，也可能只在某些 inner forms 上出现。仅给出 $\varphi$ 不能说明表示位于哪个 $G'(F)$。Rigid inner twist、Kottwitz 符号或类似数据用于标记 inner form；component group 的 character 再标记 packet 内成员。$\square$

## N.5 Endoscopic Data

**定义 N.12.** Endoscopic datum 的简化形式由三元组
$$
(H,s,\eta)
$$
组成，其中 $H/F$ 为 quasi-split reductive group，$s\in\widehat G$ 为 semisimple element，且
$$
\eta:{}^LH\to{}^LG
$$
为 L homomorphism，使 $\widehat H$ 与 $\operatorname{Cent}_{\widehat G}(s)^\circ$ 相关。

**外部输入定理 N.13（endoscopic transfer，局部接口）.** 给定 endoscopic datum 和 transfer factor，可在合适测试函数空间上定义 matching orbital integrals。稳定 orbital integrals 的匹配诱导 stable distributions 和 characters 之间的转移。

**定义 N.14.** 对一个 L-packet $\Pi_\varphi(G)$，stable character 形式上是
$$
S\Theta_\varphi=\sum_{\pi\in\Pi_\varphi}a_\pi\Theta_\pi
$$
的线性组合，其中 $a_\pi$ 由 packet normalization 和 component group character 决定。

**注 N.15.** Endoscopy 的基本思想是：单个 $\Theta_\pi$ 通常不稳定，但 packet 的加权和可能稳定；这些稳定分布才能与 endoscopic group 的稳定分布比较。

## N.6 Fundamental Lemma 的位置

**外部输入定理 N.16（fundamental lemma，局部函数匹配）.** 对非分歧 endoscopic data，单位元测试函数的稳定 orbital integrals 在 transfer factor 归一化下匹配。Ngô 的证明建立了该命题的核心情形。

**命题 N.17.** Fundamental lemma 是稳定 trace formula 比较的局部输入。

**证明.** Trace formula 比较要求几乎所有非分歧位置选择单位元球函数。若这些局部单位元不能匹配，则全局测试函数的几何侧不能逐位置比较。Fundamental lemma 保证在几乎所有好位置局部匹配成立，从而使剩余有限坏位置的 transfer 问题可单独处理。$\square$

## N.7 Packet、Endoscopy 和 Arthur Multiplicity

**命题 N.18.** Arthur multiplicity formula 需要 packet 内部标签，而不只需要 Langlands 参数。

**证明草图.** 离散谱中的 multiplicity 取决于全局 component group character 与局部 packet 标签的配对。若只知道 $\psi$ 而不知道每个 $\pi_v$ 在 $\Pi_{\psi_v}$ 中对应哪个 component group representation，就无法决定 restricted tensor product 是否出现在离散谱中以及重数是多少。$\square$

**注 N.19.** 第十七章的 multiplicity formula 接口正是这种现象的全局版本。

## N.8 本附录小结

本附录展示：

1. `GL(n)` packet 为单点是特殊现象。
2. Tori 的 LLC 由局部类域论和 Tate-Nakayama 给出。
3. `SL(2)` 已展示 packet 可含多个表示。
4. Inner forms 需要 relevance 或 rigid inner twist 数据。
5. Endoscopy 比较 stable packet characters，而不是单个 character。
6. Fundamental lemma 是稳定 trace formula 比较的局部基础。

## 练习

**练习 N.1.** 解释为什么 `GL(n)` centralizer 连通性导致 packet 为单点。

**练习 N.2.** 对 $T=\operatorname{Res}_{E/F}\mathbb G_m$，结合附录 G 写出其 L 群，并说明 torus LLC 的参数侧对象。

**练习 N.3.** 解释为什么限制 $\operatorname{GL}_2(F)$ 表示到 $\operatorname{SL}_2(F)$ 会自然产生 packet 现象。

**练习 N.4.** 说明 Jacquet-Langlands 局部对应为什么是“同一参数分布在不同内形式上”的例子。

**练习 N.5.** 解释 stable character 为什么通常是 packet 中多个 characters 的线性组合。
