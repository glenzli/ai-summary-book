# 第十二章：局部 Langlands 猜想

## 本章目标

本章给出局部 Langlands 猜想（local Langlands conjecture, LLC）的精确定式。第五章已经定义 Weil 群和 Weil-Deligne 参数，第十一章已经定义对偶群和 L 群；本章把这些参数与局部群 $G(F)$ 的不可约表示联系起来。核心点是：一般还原群的局部 Langlands 不是单个参数对应单个表示，而是参数对应 L-packet；若要得到真正的一一对应，通常必须加入 component group 的表示以及内形式数据。

## 依赖前置知识

需要第四章的光滑表示和 Hecke 代数，第五章的 Weil-Deligne 参数，第十一章的 L 群、L 同态和 Satake 参数。需要知道局部域上的不可约可容许表示、抛物诱导、tempered representation 和 essentially square-integrable representation 的基本定义。附录 Z 给出 temperedness、characters 和 Plancherel 的局部调和分析口径，附录 AA 给出 depth、parahoric 和 hyperspecial 的结构口径，附录 AC 给出 Fargues-Fontaine 曲线、local Shimura varieties 和 Fargues-Scholze semisimple LLC 的几何接口，附录 AE 给出 `GL(2)` 局部 LLC 的 principal series、Steinberg 和 supercuspidal 例子。本章把 `GL(n)` 局部 Langlands、Archimedean 局部 Langlands、一般还原群的 enhanced LLC 和 endoscopic character identities 作为外部输入或猜想。

收口归一化回指：本章所有 LLC 陈述均默认几何 Frobenius、归一化抛物诱导和归一化 Satake 参数；具体 convention 见 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 2、4、6 节。

## 12.1 表示侧：不可约可容许表示

本章固定局部域 $F$。若 $F$ 非 Archimedean，设 $\mathcal O_F$ 为整数环，剩余域基数为 $q$，Weil 群 $W_F$ 采用第五章的几何 Frobenius 归一化。

**定义 12.1.** 设 $G/F$ 为 connected reductive group。

1. 若 $F$ 非 Archimedean，记
   $$
   \operatorname{Irr}(G(F))
   $$
   为 $G(F)$ 的不可约可容许复光滑表示的同构类集合。
2. 若 $F$ Archimedean，记 $\operatorname{Irr}(G(F))$ 为不可约可容许 $(\mathfrak g,K)$-modules 或相应 admissible Fréchet representations 的 Harish-Chandra 同构类集合；本章只使用其 Langlands 分类接口。

**定义 12.2.** 设 $Z_G$ 为 $G$ 的中心。若 $\pi\in\operatorname{Irr}(G(F))$，则其中心特征为同态
$$
\omega_\pi:Z_G(F)\to\mathbb C^\times
$$
满足
$$
\pi(z)v=\omega_\pi(z)v,\qquad z\in Z_G(F),\ v\in V_\pi.
$$

**命题 12.3.** 若 $F$ 非 Archimedean 且 $\pi$ 不可约光滑，则中心 $Z_G(F)$ 在 $\pi$ 上通过一个 character 作用。

**证明.** 对每个 $z\in Z_G(F)$，算子 $\pi(z)$ 与所有 $\pi(g)$ 交换。由于 $\pi$ 不可约，Schur 引理给出 $\pi(z)$ 是标量。映射 $z\mapsto\pi(z)$ 与乘法相容，因此得到 character $\omega_\pi$。$\square$

**定义 12.4.** 若 $F$ 非 Archimedean，表示 $\pi\in\operatorname{Irr}(G(F))$ 称为：

1. spherical，若对某个 hyperspecial maximal compact subgroup $K\subset G(F)$ 有 $\pi^K\ne0$；
2. tempered，若其矩阵系数满足 Harish-Chandra 的 tempered 增长条件；
3. essentially square-integrable，若存在 character $\chi:G(F)\to\mathbb C^\times$ 使 $\pi\otimes\chi^{-1}$ 的矩阵系数在 $G(F)/Z_G(F)$ 上平方可积。

**注 12.5.** 本章不重新建立 tempered 表示理论。后续使用的事实是：tempered 性质应在 LLC 下对应参数的 boundedness，而 essentially square-integrable 性质应对应离散参数。这些对应是 LLC 的相容性要求，不是定义。

## 12.2 参数侧：L 参数集合

设 $G/F$ 为 connected reductive group，其局部 L 群为
$$
{}^LG=\widehat G\rtimes W_F.
$$

**定义 12.6.** 若 $F$ 非 Archimedean，$G$ 的 Langlands 参数是同态
$$
\varphi:W_F'\to{}^LG,\qquad W_F'=W_F\times\operatorname{SL}_2(\mathbb C),
$$
满足：

1. 复合 $W_F'\to{}^LG\to W_F$ 为自然投影；
2. $\varphi|_{\operatorname{SL}_2(\mathbb C)}$ 是复代数群同态到 $\widehat G$；
3. $\varphi(W_F)$ 由 semisimple 元素给出，并满足通常的连续性条件；
4. 若 $G$ 非 quasi-split 或考虑内形式，则参数还须满足相应 relevance 条件。

参数按 $\widehat G$-共轭取等价类。等价类集合记为
$$
\Phi(G/F).
$$

**定义 12.7.** 若 $F$ Archimedean，Langlands 参数是 admissible homomorphism
$$
\varphi:W_F\to{}^LG
$$
的 $\widehat G$-共轭类。其 admissibility 条件包括：覆盖 $W_F$，在 $\mathbb C^\times\subset W_\mathbb R$ 上代数或半代数，并与 parabolic relevance 条件相容。

**注 12.8.** 非 Archimedean 情形引入 $\operatorname{SL}_2(\mathbb C)$ 是为了记录 Weil-Deligne 表示中的 monodromy 算子 $N$。Archimedean 情形的参数域按经典局部 Langlands 采用 $W_\mathbb R$ 或 $W_\mathbb C$，不再额外加入同一个 Weil-Deligne $\operatorname{SL}_2$。

**定义 12.9.** 设 $F$ 非 Archimedean。参数 $\varphi$ 称为：

1. unramified，若 $\varphi$ 在惯性群 $I_F$ 上平凡，并且在 $\operatorname{SL}_2(\mathbb C)$ 上平凡；
2. bounded，若 $\varphi(W_F)$ 在 $\widehat G$ 中的投影模去 $Z(\widehat G)$ 后有相对紧闭包；
3. discrete，若 $\varphi$ 的像不包含在任何 proper Levi subgroup 的 L 群中，等价地，
   $$
   \operatorname{Cent}_{\widehat G}(\operatorname{im}\varphi)/Z(\widehat G)^{W_F}
   $$
   是有限群。

**注 12.10.** 对非 split 群，“包含在 Levi subgroup 的 L 群中”必须理解为 ${}^LG$ 中与 $W_F$ 作用相容的 L-Levi subgroup。完整定义依赖 parabolic subgroup 和 Levi subgroup 的 L 群构造。

**命题 12.11.** 若 $G=\operatorname{GL}_n$，则 $\Phi(G/F)$ 等同于 $n$ 维 Frobenius-semisimple Weil-Deligne 表示的同构类集合。

**证明.** 第十一章给出
$$
{}^L\operatorname{GL}_n=\operatorname{GL}_n(\mathbb C)\times W_F.
$$
覆盖 $W_F$ 的参数 $\varphi:W_F'\to{}^LG$ 等价于同态
$$
\phi:W_F'\to\operatorname{GL}_n(\mathbb C).
$$
将 $W_F'$ 的 $\operatorname{SL}_2(\mathbb C)$ 部分换成 nilpotent 算子 $N$，得到 Frobenius-semisimple Weil-Deligne 表示；反向构造由 Jacobson-Morozov 型对应给出。第五章已经说明这两种模型等价。$\square$

## 12.3 Component group 与增强参数

一般还原群的 L-packet 可能含有多个表示。区分 packet 内部元素需要参数的 centralizer。

**定义 12.12.** 设 $\varphi\in\Phi(G/F)$。定义
$$
S_\varphi=\operatorname{Cent}_{\widehat G}(\operatorname{im}\varphi),
$$
以及 component group
$$
\mathcal S_\varphi=\pi_0\left(S_\varphi/Z(\widehat G)^{W_F}\right).
$$
这里 $Z(\widehat G)^{W_F}$ 表示在 Weil 作用下不变的中心部分。

**定义 12.13.** 一个增强参数（enhanced parameter）是二元组
$$
(\varphi,\rho),
$$
其中 $\varphi\in\Phi(G/F)$，而 $\rho$ 是有限群 $\mathcal S_\varphi$ 的不可约复表示，满足与 $G$ 的内形式或强内形式相容的 relevance 条件。

**注 12.14.** 对 quasi-split 群并固定 Whittaker datum 后，通常期望 generic 表示对应 $\mathcal S_\varphi$ 的平凡表示。对内形式，$\rho$ 可能必须具有指定的中心 character；这就是 enhanced LLC 的一部分。

**命题 12.15.** 对 $G=\operatorname{GL}_n$，每个 L-packet 至多含一个表示的预期与 $\mathcal S_\varphi$ 平凡相容。

**证明.** 由命题 12.11，参数是 $n$ 维 Weil-Deligne 表示。把该 Weil-Deligne 表示按 indecomposable objects 作 Krull-Schmidt 分解。其自同构群是若干 multiplicity spaces 的一般线性群与一个 unipotent 群的扩张，因此为连通复代数群。于是
$$
\pi_0(S_\varphi)=1.
$$
由于 $Z(\widehat G)^{W_F}$ 也连通，商的 component group 仍平凡。因此 enhanced parameter 没有额外有限群表示可选。$\square$

**注 12.16.** 对 $\operatorname{SL}_n$、orthogonal groups 和 symplectic groups，centralizer 可以非连通；这正是 L-packet 含多个表示的来源之一。

**注 12.16.1.** 附录 N 以 tori、$\operatorname{SL}_2$ 和 quaternion algebra 内形式为模型例子，说明 component group、内形式和 stable character identity 如何改变 packet 的形状。本章的猜想陈述只给出一般口径，具体例子应与该附录合读。

## 12.4 局部 Langlands 猜想：packet 形式

**猜想 12.17（局部 Langlands，packet 形式）.** 设 $G/F$ 为 connected reductive group。存在一个映射
$$
\operatorname{LL}_G:\operatorname{Irr}(G(F))\to\Phi(G/F),
$$
其纤维
$$
\Pi_\varphi(G)=\operatorname{LL}_G^{-1}(\varphi)
$$
称为参数 $\varphi$ 的 L-packet，并满足：

1. $\operatorname{Irr}(G(F))$ 是 L-packets 的不交并：
   $$
   \operatorname{Irr}(G(F))=\bigsqcup_{\varphi\in\Phi(G/F)}\Pi_\varphi(G).
   $$
2. 每个 $\Pi_\varphi(G)$ 是有限集合；对不 relevant 的参数，该 packet 为空。
3. 若 $G=\operatorname{GL}_n$，则每个非空 packet 只有一个元素。
4. 对 $G=\mathbb G_m$，该对应等于局部类域论。

**注 12.18.** 对一般 $G$，猜想 12.17 只是 coarse form。它把 packet 作为一个集合给出，但没有说明 packet 内每个表示如何命名，也没有说明内形式之间如何同时参数化。Enhanced LLC 用定义 12.13 的增强参数给出更精细的形式。

**猜想 12.19（增强局部 Langlands）.** 固定 rigid inner twist 数据，例如一组内形式 $G'$ 及其 rigid cocycle 或 Kottwitz 参数。存在自然双射
$$
\bigsqcup_{G'}\operatorname{Irr}(G'(F))
\longleftrightarrow
\left\{(\varphi,\rho):\varphi\in\Phi(G/F),\ \rho\in\operatorname{Irr}(\mathcal S_\varphi)\text{ satisfying relevance}\right\},
$$
其中 $G'$ 遍历与 $G$ 相关的内形式或纯内形式。对固定的 $G'$，右侧由内形式对应的 character 条件切出。

**注 12.20.** 猜想 12.19 的完整陈述需要 Kottwitz 符号、Galois cohomology、rigid inner forms 和 transfer factor normalization。本书在本章只固定接口：packet 内部由 component group 的表示控制。

**注 12.20.1.** 若只研究 quasi-split group 的 coarse packet，容易误以为 LLC 是单群上的集合分解。附录 N 的 Jacquet-Langlands 和 endoscopy 例子说明：内形式与 packet 内部参数必须同时出现，稳定 trace formula 才能看到正确的谱侧分布。

## 12.5 LLC 应满足的相容性

局部 Langlands 猜想不是任意分组；它由一系列相容性条件刻画。

**条件 12.21（中心特征）.** 若 $\pi\in\Pi_\varphi(G)$，则 $\omega_\pi$ 应由参数 $\varphi$ 在 $Z_G$ 的对偶数据上诱导。换言之，中心嵌入
$$
Z_G\hookrightarrow G
$$
在 L 群侧给出相应的对偶映射，$\varphi$ 经该映射应恢复 $Z_G(F)$ 的 character。

**注 12.22.** 对 $G=\operatorname{GL}_n$，条件 12.21 具体化为
$$
\omega_\pi\leftrightarrow\det\varphi_\pi
$$
其中右侧通过局部类域论看作 $F^\times$ 的 character。

**条件 12.23（L 因子和 epsilon 因子）.** 对每个有限维 L 群表示
$$
r:{}^LG\to\operatorname{GL}(V),
$$
应有局部因子
$$
L(s,\pi,r)=L(s,r\circ\varphi),
$$
$$
\varepsilon(s,\pi,r,\psi)=\varepsilon(s,r\circ\varphi,\psi),
$$
其中 $\psi:F\to\mathbb C^\times$ 为非平凡加法特征。右侧由线性 Weil-Deligne 表示的局部因子定义。

**条件 12.24（非分歧相容性）.** 若 $G/F$ unramified，$K\subset G(F)$ 为 hyperspecial maximal compact subgroup，且 $\pi^K\ne0$，则 $\varphi_\pi$ 为 unramified 参数，并且
$$
\varphi_\pi(\operatorname{Fr}_F)
$$
在 L 群中的半单共轭类等于第十一章的 Satake parameter。

**条件 12.25（tempered 与离散）.** 参数 $\varphi$ bounded 当且仅当 packet $\Pi_\varphi(G)$ 中的表示为 tempered。参数 $\varphi$ discrete 时，对应 packet 应由离散系列或 essentially square-integrable modulo center 的表示组成，具体陈述需按 $Z_G(F)$ 的非紧性修正。

**条件 12.26（抛物诱导相容性）.** 若参数 $\varphi$ 通过某个 proper Levi subgroup $M$ 的 L 群
$$
{}^LM\hookrightarrow{}^LG
$$
分解，则 $\Pi_\varphi(G)$ 应由 $M(F)$ 上相应 packet 的表示经归一化抛物诱导和 Langlands quotient 构造得到。

**条件 12.27（对偶与函子性）.** 取 contragredient 表示 $\pi^\vee$ 应对应于由 Chevalley involution 作用在 $\widehat G$ 上得到的 dual parameter。若给定 L 同态
$$
\xi:{}^LH\to{}^LG,
$$
则局部函子性预期把 $\Pi_{\varphi_H}(H)$ 转移到 $\Pi_{\xi\circ\varphi_H}(G)$。

**条件 12.28（稳定性和 endoscopy）.** 固定 Whittaker datum 与 transfer factor normalization 后，对每个参数 $\varphi$，由 component group 参数化指定的 packet 线性组合应给出稳定分布 character。Endoscopic transfer 应由 component group character 和 transfer factors 控制。

**注 12.29.** 条件 12.28 是一般 reductive 群 LLC 的技术核心之一。没有稳定 character identity，packet 的内部参数化通常不是唯一规范的。

## 12.6 基本例子

### 12.6.1 Tori

设 $T/F$ 为 torus。其对偶群 $\widehat T$ 是复 torus，L 群为
$$
{}^LT=\widehat T\rtimes W_F.
$$

**外部输入定理 12.30（局部 Langlands for tori）.** 局部类域论和 Tate-Nakayama 对偶给出自然双射
$$
\operatorname{Hom}_{\operatorname{cont}}(T(F),\mathbb C^\times)
\longleftrightarrow
H^1(W_F,\widehat T),
$$
右侧可解释为 $T$ 的一维 L 参数集合。对 $T=\mathbb G_m$，该定理退化为第三、五章的局部类域论。

**注 12.31.** Tori 的 packet 均为单元素，但非 split torus 已经要求 L 群中的 Galois 作用；这说明第十一章的半直积不是只为非 Abel 群服务。

### 12.6.2 `GL(n)`

**外部输入定理 12.32（局部 Langlands for `GL(n)`）.** 设 $F$ 为非 Archimedean 局部域。存在唯一的自然双射
$$
\operatorname{Irr}(\operatorname{GL}_n(F))
\longleftrightarrow
\Phi(\operatorname{GL}_n/F),
$$
满足：

1. $n=1$ 时为局部类域论；
2. 中心特征对应 determinant；
3. Rankin-Selberg 局部 $L$ 因子、$\varepsilon$ 因子和 $\gamma$ 因子相容；
4. 非分歧表示与 Satake 参数相容；
5. tempered 表示对应 bounded 参数；
6. essentially square-integrable 表示对应 indecomposable Weil-Deligne 数据；在 `GL(n)` 情形这等价于参数为 discrete。

**注 12.33.** 该定理由 Harris-Taylor、Henniart 及相关工作建立；Scholze 给出了另一种几何证明路线。本书后续在 `GL(n)` 章节把定理 12.32 作为外部输入，而不重建其证明。

**注 12.33.1.** 对 `GL(2)`，附录 AE 给出定理 12.32 的可计算影子：principal series、Steinberg twists 和 supercuspidals 分别对应可约半单参数、带 monodromy 的 special parameters 和不可约二维 Weil 参数。

**命题 12.34.** 对 `GL(n)`，packet 形式 LLC 与定理 12.32 的双射形式等价。

**证明.** 命题 12.15 给出 `GL(n)` 的 component group 平凡，因此 packet 内没有额外有限群表示参数。猜想 12.17 的每个非空 packet 至多一个元素。定理 12.32 说明每个 $n$ 维 Frobenius-semisimple Weil-Deligne 参数都来自唯一不可约可容许表示。因此 packet 形式退化为双射形式。$\square$

### 12.6.3 非分歧主级数

设 $G/F$ 为 split connected reductive group，$B=TU$ 为 split Borel subgroup。

**定义 12.35.** 一个非分歧 character
$$
\chi:T(F)\to\mathbb C^\times
$$
是指 $\chi$ 在 maximal compact subgroup $T(\mathcal O_F)$ 上平凡。由归一化抛物诱导得到的表示
$$
I_B^G(\chi)=\operatorname{Ind}_{B(F)}^{G(F)}(\chi)
$$
称为非分歧主级数表示。

**命题 12.36.** 非分歧主级数表示的 spherical constituent 的 L 参数由 $\chi$ 经局部类域论得到的 $\widehat T$-半单共轭类给出，并经嵌入
$$
\widehat T\hookrightarrow\widehat G
$$
成为 $G$ 的非分歧参数。

**证明草图.** 非分歧 character $\chi$ 由一致化元在各 cocharacter 方向上的取值决定。局部类域论把这些取值解释为 $W_F/I_F$ 上的 $\widehat T$-值参数。Satake 同构把 spherical Hecke character 识别为 $\widehat G$ 中的 Weyl 轨道。归一化诱导的 spherical constituent 与该 Hecke character 对应，因此其参数为 $\widehat T$ 中元素在 $\widehat G$ 中的半单共轭类。完整证明依赖 Satake 同构和归一化诱导的 spherical vector 计算。$\square$

### 12.6.4 Packet 大于一的现象

**例 12.37.** 对 $G=\operatorname{SL}_2$，对偶群为
$$
\widehat G=\operatorname{PGL}_2(\mathbb C).
$$
某些参数 $\varphi:W_F'\to\operatorname{PGL}_2(\mathbb C)$ 的 centralizer 非连通，因此 $\mathcal S_\varphi$ 非平凡。对应的 L-packet 可以含有多个 $\operatorname{SL}_2(F)$ 的不可约表示。

**注 12.38.** 这个例子说明：若把 LLC 写成“表示等于 Galois 表示”，则对一般 $G$ 已经在形式上错误。正确说法是：参数先给出 packet，packet 内部还需要 component group 表示、Whittaker normalization 和内形式数据来区分。

## 12.7 Archimedean 局部 Langlands

**外部输入定理 12.39（Archimedean LLC）.** 设 $F=\mathbb R$ 或 $\mathbb C$，$G/F$ 为 connected reductive group。Langlands 分类给出 $\operatorname{Irr}(G(F))$ 与 admissible homomorphisms
$$
\varphi:W_F\to{}^LG
$$
以及相应 component group 表示之间的对应。该对应满足中心特征、infinitesimal character、temperedness 和 parabolic induction 的相容性。

**注 12.40.** 对 $F=\mathbb C$，$W_\mathbb C=\mathbb C^\times$，参数基本由代数 character 与其复共轭行为给出。对 $F=\mathbb R$，$W_\mathbb R$ 的非平凡分量记录 Cartan involution 和实形式信息。

**命题 12.41.** Archimedean LLC 中 tempered 参数应对应 tempered representations。

**证明草图.** Archimedean Langlands 分类把不可约可容许表示表示为标准模的唯一不可约商。Tempered 表示恰对应诱导数据中实部为零的情形。参数侧 boundedness 对应 infinitesimal character 的实部条件和 $W_F$ 像的相对紧性。完整证明属于 Harish-Chandra 和 Langlands 分类理论。$\square$

## 12.8 一般还原群的已知性与本书口径

一般 reductive 群的 LLC 由多个层次组成：

1. Coarse packet decomposition。
2. Enhanced parameterization。
3. Inner forms 的同时参数化。
4. Endoscopic character identities。
5. 与局部因子、稳定 trace formula 和全局 Arthur 参数的相容性。

**外部输入定理 12.42（若干已知情形，接口表述）.** 下列情形有成熟的局部 Langlands 理论或足够完整的构造可供本书后续引用：

1. Tori，由局部类域论和 Tate-Nakayama 对偶给出。
2. `GL(n)`，由 Harris-Taylor、Henniart 等工作给出。
3. Archimedean reductive groups，由 Langlands 分类和后续 Vogan 形式给出。
4. 许多 quasi-split classical groups，通过 Arthur、Mok 及相关 endoscopic classification 给出。

**注 12.43.** 定理 12.42 不是“所有 reductive groups 的 LLC 已完全定理化”的声明。一般 reductive 群的最精确版本涉及 enhanced parameters、rigid inner forms、wild ramification 和稳定 character identities；本书在需要使用具体群时会单独声明所需外部输入。

**注 12.43.1.** Fargues-Scholze 的几何化给出 $p$-adic reductive groups 的 semisimple 参数化和谱作用框架，见附录 AC.6。它解释了为什么局部 LLC 可以被看作 Fargues-Fontaine 曲线上 sheaves on $\operatorname{Bun}_G$ 的谱分解；但它不替代本章猜想 12.19 中所有 enhanced packet、内形式标号和 endoscopic character identities。

## 12.9 与全局 Langlands 的接口

设 $K$ 为整体域，$G/K$ 为 connected reductive group，$\pi=\otimes_v'\pi_v$ 为 $G(\mathbb A_K)$ 的自守表示。全局 Langlands 纲领要求每个局部分量 $\pi_v$ 有局部参数
$$
\varphi_{\pi_v}:W_{K_v}'\to{}^LG_v.
$$

**定义 12.44.** 设 $r:{}^LG\to\operatorname{GL}(V)$ 为 L 群表示。若 $\pi_v$ 的局部参数为 $\varphi_v$，定义局部标准因子
$$
L(s,\pi_v,r)=L(s,r\circ\varphi_v).
$$
在几乎所有非分歧位置，$\pi_v$ spherical，$\varphi_v$ 由 Satake 参数给出。

**命题 12.45.** 若几乎所有 $\pi_v$ 非分歧，则形式 Euler 乘积
$$
L^S(s,\pi,r)=\prod_{v\notin S}L(s,\pi_v,r)
$$
由几乎所有位置的 Satake 参数决定，其中 $S$ 包含所有 ramified 位置和 Archimedean 位置。

**证明.** 对 $v\notin S$，$\pi_v$ spherical。由条件 12.24，$\varphi_v(\operatorname{Fr}_v)$ 等于 Satake parameter。局部因子 $L(s,\pi_v,r)$ 是
$$
r(\varphi_v(\operatorname{Fr}_v))
$$
在惯性不变量上的 characteristic polynomial 的倒数。因此每个局部因子由 Satake parameter 决定，乘积也由这些参数决定。$\square$

**注 12.46.** 全局章节将研究 $L^S(s,\pi,r)$ 的收敛、解析延拓和函数方程。局部 LLC 只提供局部因子的定义和相容性，不单独证明全局解析性质。

**收口精修 12.A（LLC 状态边界）.** 本章把局部 Langlands 分成以下可调用层次：

| 层次 | 本书使用方式 | 状态 |
|---|---|---|
| Tori 和 `GL(1)` | 作为类域论模型 | 定理级输入 |
| `GL(n)` | 定义局部因子、Satake 参数和 local-global compatibility 的标准接口 | 外部输入定理 |
| Archimedean groups | 解释无穷处参数和代数性 | 外部输入定理 |
| Classical groups | 进入 Arthur packets 和 standard transfer | Arthur-Mok 等外部输入 |
| 一般 enhanced LLC | packet、component group、inner form 和 endoscopy 的总框架 | 猜想或已知特殊情形，不能当作全体已证 |

## 12.10 本章小结

局部 Langlands 猜想把局部表示论组织为参数理论。对 `GL(n)`，参数与不可约可容许表示一一对应；对一般还原群，一个参数给出有限 L-packet，packet 内部通常由 component group 的不可约表示和内形式数据区分。LLC 的内容不仅是集合分解，还包括中心特征、局部因子、非分歧 Satake 参数、tempered 性、离散性、抛物诱导、对偶、函子性和 endoscopy 的相容性。后续全局章节会把这些局部参数作为自守 L 函数和全局函子性的局部输入。

## 练习

**练习 12.1.** 对 $G=\mathbb G_m$，说明猜想 12.17 如何退化为局部类域论。

**练习 12.2.** 对 $G=\operatorname{GL}_n$，用 Schur 引理证明中心特征对应 determinant 的必要性。

**练习 12.3.** 设 $\varphi$ 为 `GL(n)` 的参数，且对应 Weil-Deligne 表示为不可约表示的直和。计算 $S_\varphi$ 的连通分量群。

**练习 12.4.** 解释为什么非分歧 spherical 表示的参数必须在惯性群和 $\operatorname{SL}_2(\mathbb C)$ 上平凡。

**练习 12.5.** 设 $\xi:{}^LH\to{}^LG$ 为 L 同态。写出局部函子性对参数和 L-packets 的预期作用。

**练习 12.6.** 说明为什么一般 reductive 群的 LLC 不能只表述为 $\operatorname{Irr}(G(F))$ 与 $\Phi(G/F)$ 的双射。

**练习 12.7.** 设 $\pi=\otimes_v'\pi_v$ 为自守表示。说明局部 LLC 如何为 $L^S(s,\pi,r)$ 的每个非分歧局部因子提供定义。
