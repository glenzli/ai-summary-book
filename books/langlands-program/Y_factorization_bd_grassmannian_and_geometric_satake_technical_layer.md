# 附录 Y：Factorization、Beilinson-Drinfeld Grassmannian 和几何 Satake 技术层

收口归一化回指：本附录支撑几何 Satake 的 tensor convention、fusion 和 sheaf-function 比较；与经典 Satake 的 $q$-因子和 Tate twists 见 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 4、9 节。

## Y.1 Ran space 和 factorization 的语言

设 $X/k$ 为光滑曲线，$G/k$ 为 split connected reductive group。

**定义 Y.1.** Ran space $\operatorname{Ran}(X)$ 的点可理解为 $X$ 的非空有限子集。更精确地，它是有限集合 $I$ 上 $X^I$ 的 colimit，转移映射由有限集合的满射诱导。

**定义 Y.2.** 一个 factorization object 粗略地是在所有 $X^I$ 上给出对象 $\mathcal F_I$，并在互不相交的点集上给出乘法同构
$$
\mathcal F_{I\sqcup J}|_{(X^I\times X^J)_{\operatorname{disj}}}
\simeq
\mathcal F_I\boxtimes\mathcal F_J
$$
满足结合性和对称性相容。

**注 Y.3.** Factorization 是几何 Langlands 中“多个 Hecke 修改在不同点彼此独立”的范畴化表达。

## Y.2 Beilinson-Drinfeld Grassmannian

**定义 Y.4.** Beilinson-Drinfeld Grassmannian $\operatorname{Gr}_{G,I}$ over $X^I$ 参数化：

1. 点族 $(x_i)_{i\in I}$；
2. 一个 $G$-bundle $\mathcal E$ on $X$；
3. 在 $X\setminus\{x_i\}_{i\in I}$ 上的平凡化。

当 $I=\{1\}$ 且在形式圆盘局部化时，得到 affine Grassmannian
$$
\operatorname{Gr}_G=G((t))/G[[t]].
$$

**外部输入定理 Y.5（BD Grassmannian 的几何性质）.** $\operatorname{Gr}_{G,I}$ 是 ind-proper ind-scheme 或相应 ind-stack，带有 factorization structure。其纤维在点互异处为各单点 affine Grassmannian 的乘积，在点碰撞处编码 convolution。

**命题 Y.6.** 点互异时 BD Grassmannian 分解为单点 Grassmannian 的乘积。

**证明草图.** 若点 $(x_i)$ 两两不同，则在每个 $x_i$ 的形式邻域上的修改互不干涉。一个在所有 $x_i$ 处的 $G$-bundle 修改等价于分别在每个 $x_i$ 处给出修改。Beauville-Laszlo 粘合把这些局部修改拼成全局对象，得到乘积分解。$\square$

## Y.3 Convolution 和 fusion

**定义 Y.7.** 单点 affine Grassmannian 上的 convolution diagram 为
$$
\operatorname{Gr}_G\times\operatorname{Gr}_G
\xleftarrow{p}
G((t))\times^{G[[t]]}\operatorname{Gr}_G
\xrightarrow{m}
\operatorname{Gr}_G.
$$
对 sheaves $\mathcal A,\mathcal B$，定义
$$
\mathcal A*\mathcal B=m_!p^!(\mathcal A\boxtimes\mathcal B)
$$
或按 sheaf theory 选择相应 $*$、$!$ 版本。

**外部输入定理 Y.8（fusion equals convolution）.** BD Grassmannian 上当两个点相碰时的 nearby cycles 或 specialization 给出单点 affine Grassmannian 上的 convolution product。该结构使 Satake category 成为 symmetric monoidal category。

**命题 Y.9.** Factorization 解释了 convolution 的交换性。

**证明草图.** 两个 Hecke 修改在不同点时可交换，因为它们作用在 disjoint formal discs 上。沿 $X^2$ 中去掉对角线的开集交换两个点，得到自然对称同构。将该同构延拓到对角线的 specialization，就是 convolution product 的 commutativity constraint。完整证明依赖 nearby cycles 和 perversity 保持。$\square$

## Y.4 几何 Satake 的精确接口

**外部输入定理 Y.10（geometric Satake, factorization form）.** 存在 tensor equivalence
$$
\operatorname{Sat}_G:\operatorname{Perv}_{G[[t]]}(\operatorname{Gr}_G)
\xrightarrow{\sim}
\operatorname{Rep}(\widehat G)
$$
并且 factorization/fusion 结构给出右侧张量范畴的对称结构。Schubert variety $\overline{\operatorname{Gr}}^\lambda$ 的 IC sheaf 对应 $\widehat G$ 的最高权 $\lambda$ 不可约表示。

**命题 Y.11.** 几何 Satake 恢复对偶群的 root datum。

**证明草图.** Affine Grassmannian 的 $G[[t]]$-orbits 由 dominant coweights $\lambda\in X_*(T)^+$ 参数化。IC sheaves 的 simple objects 因而由这些 $\lambda$ 标号。Convolution 的最高项规则与表示张量积的最高权规则一致。Tannakian formalism 从该 symmetric tensor category 恢复一个 reductive group，其 dominant weights 为 $X_*(T)^+$，故该群的 root datum 为 $G$ 的 dual root datum。$\square$

## Y.5 Hecke action 的 factorization

**定义 Y.12.** 对 $V\in\operatorname{Rep}(\widehat G)$，几何 Satake 给出 sheaf $\mathcal S_V$ on $\operatorname{Gr}_G$。Hecke functor
$$
\mathsf H_V:\mathcal D(\operatorname{Bun}_G)\to\mathcal D(\operatorname{Bun}_G\times X)
$$
由 Hecke stack 和 kernel $\mathcal S_V$ 给出。

**命题 Y.13.** Hecke functors 满足张量相容：
$$
\mathsf H_V\circ\mathsf H_W\simeq\mathsf H_{V\otimes W}
$$
在 factorization 意义下成立。

**证明草图.** 两次 Hecke 修改对应 convolution diagram；几何 Satake 把 convolution $\mathcal S_V*\mathcal S_W$ 识别为 $\mathcal S_{V\otimes W}$。BD Grassmannian 的 factorization 保证当修改点分离时为外积，当点碰撞时为 convolution。由此得到 Hecke functor 的张量相容。$\square$

## Y.6 Beilinson-Drinfeld Grassmannian 与 Hecke eigensheaves

**命题 Y.14.** Hecke eigensheaf 条件是 factorization-compatible 的本征条件。

**证明.** 对每个 $V$，Hecke eigensheaf 给出同构
$$
\mathsf H_V(\mathcal F)\simeq\mathcal F\boxtimes V_{\mathcal E}.
$$
对 $V,W$ 同时作用时，左侧由命题 Y.13 化为 $\mathsf H_{V\otimes W}(\mathcal F)$；右侧由 local system 的张量函子性质化为
$$
\mathcal F\boxtimes (V_{\mathcal E}\otimes W_{\mathcal E}).
$$
因此本征同构必须与 factorization 和 tensor product 相容。$\square$

## Y.7 与 categorical geometric Langlands 的关系

**外部输入定理 Y.15（factorization 在 categorical GL 中的作用）.** 范畴化几何 Langlands 等价应与 Hecke factorization、Eisenstein/constant term functors、Verdier duality 和 spectral action 相容。BD Grassmannian 和 factorization sheaves 是构造 Hecke action 与 spectral action 的基础输入。

**命题 Y.16.** 若忽略 factorization，只保留单个 Hecke 算子，则无法恢复完整 $\widehat G$-local system。

**证明.** 单个表示 $V$ 的 Hecke eigenvalue 只给出 associated local system $V_{\mathcal E}$。完整 $\widehat G$-local system 等价于 tensor functor
$$
\operatorname{Rep}(\widehat G)\to\operatorname{Loc}(X),
$$
需要对所有 $V$ 的张量相容数据。Factorization 正是保证这些 Hecke eigenspaces 对 tensor product、对称性和多点修改相容的结构。因此忽略 factorization 会丢失 Tannakian 重构所需的数据。$\square$

## 练习

**练习 Y.1.** 解释 Ran space 为什么适合记录多点 Hecke 修改。

**练习 Y.2.** 说明点互异时 BD Grassmannian 分解为 affine Grassmannian 的乘积。

**练习 Y.3.** 解释 fusion 如何给出 convolution 的交换约束。

**练习 Y.4.** 用 Tannakian 语言说明几何 Satake 如何恢复 $\widehat G$。

**练习 Y.5.** 说明 Hecke eigensheaf 条件为什么必须对所有 $V\in\operatorname{Rep}(\widehat G)$ 张量相容。
