# 附录 S：函数域、Shtukas、Excursion Operators 和 Lafforgue 接口

收口归一化回指：本附录比较函数域 Satake 参数、Galois Frobenius、shtuka cohomology 和 sheaf-function dictionary；统一 convention 见 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 2、6、9 节。

## S.1 函数域 Langlands 的基本对象

设 $X/\mathbb F_q$ 为光滑射影几何连通曲线，函数域为
$$
K=\mathbb F_q(X).
$$
设 $G$ 为 $K$ 上 split connected reductive group。函数域 Langlands 的自守侧可写为
$$
G(K)\backslash G(\mathbb A_K)/K_N
$$
上的函数，其中 $K_N$ 是某个 level subgroup。

**命题 S.1.** 当 $G$ split 且 $K_N$ 是由有限闭点集 $N$ 上的 level structure 与 $X\setminus N$ 上 hyperspecial level 给出的开紧子群时，双商
$$
G(K)\backslash G(\mathbb A_K)/K_N
$$
可解释为带 level structure 的 $G$-bundles 模栈 $\operatorname{Bun}_{G,N}$ 的 $\mathbb F_q$-点集合。

**证明路线（外部输入）.** 一个 adelic double coset 给出在 $X$ 的泛点平凡化的 $G$-bundle，并在各闭点由 $G(K_x)$ 与 $G(\mathcal O_x)$ 的相对位置粘合。商去 $G(K)$ 改变泛点平凡化，商去 $K_N$ 改变局部 level 平凡化。反向由 Beauville-Laszlo 型粘合恢复 adelic 数据。完整证明需要 $G$-bundles 的代数栈理论。$\square$

## S.2 Hecke correspondences 和 Frobenius

**定义 S.2.** 对闭点 $x\in X$ 和表示 $V\in\operatorname{Rep}(\widehat G)$，Hecke correspondence 参数化三元组
$$
(\mathcal E,\mathcal E',\beta)
$$
其中 $\mathcal E,\mathcal E'$ 为 $G$-bundles，$\beta$ 是 $X\setminus\{x\}$ 上的同构，其相对位置由 $V$ 控制。

**命题 S.3.** Hecke correspondence 在函数迹下给出自守函数上的 Hecke 算子。

**证明路线（外部输入）.** Hecke correspondence 的两个投影给出拉回-推出算子。对定义在有限域上的 sheaf 取 Frobenius trace，Grothendieck-Lefschetz 公式把 cohomological correspondence 的 trace 变成有限集合上的求和。这正是 Hecke 算子按修改点计数的公式。$\square$

## S.3 Shtukas

**定义 S.4.** 在本附录的接口层面，一个带 $r$ 条腿的 $G$-shtuka 是数据
$$
(\mathcal E_0,\ldots,\mathcal E_r;x_1,\ldots,x_r;\beta_i;\iota)
$$
其中 $x_i\in X$，$\beta_i$ 是在 $x_i$ 处的 Hecke 修改，最后给出 Frobenius 同构
$$
\iota:\mathcal E_r\simeq{}^\tau\mathcal E_0.
$$
相对位置由 $\widehat G$ 的表示数据控制。

**外部输入定理 S.5（Shtuka stacks）.** 带 level 和 Harder-Narasimhan 截断的 $G$-shtuka 构成 Deligne-Mumford stacks 或 Artin stacks，具有适合定义 $\ell$-adic cohomology 的有限型截断。其交叉上同调携带 Galois 群、Hecke 代数和腿置换的作用。

**注 S.6.** Shtuka 是函数域中 Shimura variety 的替代物。它同时带 Hecke 修改和 Frobenius，因此能把自守侧 Hecke 作用与 Galois 侧 Frobenius 作用放在同一 cohomology 中。

## S.4 Drinfeld 和 Lafforgue 的 `GL(n)` 定理

**外部输入定理 S.7（Drinfeld for `GL(2)`）.** 对函数域 $K$，`GL(2)` 的 cuspidal automorphic representations 与连续、不可约、几乎处处非分歧且 determinant 与中心特征匹配的二维 $\ell$-adic Galois representations 之间存在 Langlands 对应，满足几乎所有位置的 Frobenius-Satake 相容性。

**外部输入定理 S.8（Laurent Lafforgue for `GL(n)`）.** 对函数域 $K$，cuspidal automorphic representations of $\operatorname{GL}_n(\mathbb A_K)$ 与不可约 $n$ 维 $\ell$-adic Galois representations 之间存在相容对应，满足：

1. 几乎所有非分歧位置的 Hecke Satake polynomial 等于 Frobenius characteristic polynomial。
2. 中心 character 与 determinant 相容。
3. 局部 L 因子和 epsilon 因子满足预期相容性。
4. 对有限阶 twisting 和 contragredient 操作相容。

**命题 S.9.** 定理 S.8 是第十四章数域 `GL(n)` 猜想的函数域定理版本。

**证明.** 第十四章的数域完整全局 `GL(n)` Langlands 要求 cuspidal automorphic representations 与 $n$ 维 Galois representations 相互对应，并在几乎所有位置满足 Satake-Frobenius 相容。定理 S.8 在 $K=\mathbb F_q(X)$ 情形给出这些条件。差异在于函数域没有 Archimedean 位置，且证明使用 shtuka cohomology 而非 Shimura varieties 或 $p$-adic Hodge theory。$\square$

## S.5 V. Lafforgue 和一般还原群

**定义 S.10.** 对有限集合 $I$、函数
$$
f\in\mathcal O(\widehat G\backslash \widehat G^I/\widehat G)
$$
和 Galois 元素族 $(\gamma_i)_{i\in I}$，excursion operator 形式上记为
$$
S_{I,f,(\gamma_i)}.
$$
它作用在 cuspidal automorphic forms 空间上，并应记录 Langlands 参数在 $(\gamma_i)$ 上的 $\widehat G$-共轭不变量。

**外部输入定理 S.11（V. Lafforgue excursion operators）.** 对函数域上的 split reductive group $G$，cuspidal automorphic forms 空间存在交换的 excursion operator 代数。其 characters 给出 semisimple global Langlands parameters
$$
\sigma:\operatorname{Gal}(\overline K/K)\to{}^LG(\overline{\mathbb Q}_\ell)
$$
的 $\widehat G$-共轭类，并满足 Hecke eigenvalues 与 Frobenius conjugacy classes 的非分歧相容。

**命题 S.12.** Excursion operators 比单个 Hecke eigenvalue 系统更接近一般 $\widehat G$-参数。

**证明.** 对 `GL(n)`，标准表示的 characteristic polynomial 在足够多位置可恢复半单共轭类。对一般 $\widehat G$，单个表示的 trace 不一定分离 $\widehat G$-共轭类。函数
$$
\mathcal O(\widehat G\backslash \widehat G^I/\widehat G)
$$
对所有有限集合 $I$ 同时记录多个 Galois 元素之间的关系，足以通过 invariant theory 描述半单参数的共轭类。因此 excursion operators 给出比单点 Hecke 特征更完整的参数代数。$\square$

## S.6 与几何 Langlands 的桥梁

**命题 S.13.** Sheaf-function dictionary 把几何 Hecke eigensheaf 的 Frobenius trace 变成函数域自守 Hecke eigenfunction。

**证明路线（外部输入）.** 若 $\mathcal F$ 是 $\operatorname{Bun}_G$ 上的 Hecke eigensheaf，且本征值为 $\widehat G$-local system $\mathcal E$，则 Hecke correspondence 上的同构给出 sheaf 层面的本征关系。对 $\mathbb F_q$-点取 Frobenius trace，Grothendieck-Lefschetz 公式把 sheaf 关系变成函数等式。右侧本征值变为 $\mathcal E$ 在闭点 Frobenius 上的 trace。$\square$

**注 S.14.** 这说明几何 Langlands 不是数域 Langlands 的直接证明，而是函数域上“sheaf 级别结构”的增强。Shtukas 和 excursion operators 则把这种几何结构投影回 Galois 参数。

## S.7 局限和边界

**命题 S.15.** 函数域结果不能直接推出数域 Langlands。

**证明.** 函数域证明使用曲线 $X/\mathbb F_q$、Frobenius endomorphism、$\ell$-adic sheaf cohomology 和 shtuka stacks。这些对象在数域上没有同型替代物；数域缺少一个带绝对 Frobenius 的几何曲线，其闭点对应全部有限素数并能形成同样的 shtuka cohomology。因此函数域定理提供结构模型和类比，而不是数域定理的形式推论。$\square$

## 练习

**练习 S.1.** 说明 $\operatorname{Bun}_G(\mathbb F_q)$ 与 adelic 双商的关系。

**练习 S.2.** 解释 shtuka 中 Frobenius 同构的作用。

**练习 S.3.** 说明 Lafforgue 定理中 Satake polynomial 与 Frobenius characteristic polynomial 的相容性。

**练习 S.4.** 解释 excursion operators 为什么需要多个 Galois 元素。

**练习 S.5.** 说明函数域定理为何不能直接推出数域定理。
