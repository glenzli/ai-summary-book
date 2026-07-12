# 附录 Q：Bernstein-Zelevinsky 理论、Langlands 商和 `GL(n)` 局部分类

收口归一化回指：本附录使用归一化抛物诱导，并把 `GL(n)` 局部分类与 WD 参数和局部因子比较；相关 convention 见 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 2、4、6、8 节。

## Q.1 基本设定

设 $F$ 为非 Archimedean 局部域，$G_n=\operatorname{GL}_n(F)$。所有表示均指复向量空间上的 smooth representation。若 $P=MN$ 是标准抛物子群，$M\simeq\prod_iG_{n_i}$，归一化抛物诱导记为
$$
\operatorname{Ind}_P^{G_n}(\sigma_1\boxtimes\cdots\boxtimes\sigma_r).
$$
为简洁，常写作
$$
\sigma_1\times\cdots\times\sigma_r.
$$

**定义 Q.1.** 不可约 smooth representation $\pi$ 称为 cuspidal，若它的所有 proper parabolic Jacquet modules 为 $0$。称为 essentially square-integrable，若存在 character $\chi$ 使 $\pi\otimes(\chi\circ\det)$ 的 matrix coefficients modulo center 平方可积。

**外部输入定理 Q.2（Jacquet module 和 cuspidal support）.** 每个不可约 smooth representation $\pi$ of $G_n$ 有有限 cuspidal support，即存在若干不可约 cuspidal representations $\rho_i$ of $G_{n_i}$，使 $\pi$ 是
$$
\rho_1\times\cdots\times\rho_r
$$
的某个 subquotient。Cuspidal support 在重排后唯一。

## Q.2 Segments 和 essentially square-integrable 表示

设 $\nu(g)=|\det g|_F$。

**定义 Q.3.** 令 $\rho$ 为 $G_m$ 的不可约 cuspidal representation，$a,b\in\mathbb Z$ 且 $a\le b$。Segment 定义为
$$
\Delta=[\rho\nu^a,\rho\nu^{a+1},\ldots,\rho\nu^b].
$$
其长度为 $b-a+1$，degree 为 $m(b-a+1)$。

**外部输入定理 Q.4（Zelevinsky segment representation）.** 对每个 segment $\Delta$，诱导表示
$$
\rho\nu^b\times\rho\nu^{b-1}\times\cdots\times\rho\nu^a
$$
有唯一不可约 quotient，记为 $L(\Delta)$；诱导表示
$$
\rho\nu^a\times\rho\nu^{a+1}\times\cdots\times\rho\nu^b
$$
有唯一不可约 subrepresentation，记为 $Z(\Delta)$。其中相应的 essentially square-integrable 表示可由这些 segment 数据给出，具体 convention 依赖 Zelevinsky 或 Langlands 归一化。

**注 Q.5.** 文献中 $Z(\Delta)$、$L(\Delta)$ 和 $\delta(\Delta)$ 的符号不完全一致。本书只使用两件事：segment 是局部分类的原子；它在 LLC 下对应 $\rho$ 的参数张量上一个 special representation。

## Q.3 Multisegments 和 Zelevinsky 分类

**定义 Q.6.** Multisegment 是 segment 的有限 multiset：
$$
\mathfrak m=\{\Delta_1,\ldots,\Delta_r\}.
$$
其 degree 为各 segment degree 的和。

**外部输入定理 Q.7（Zelevinsky 分类）.** `GL(n)` 的不可约 smooth representations 的同构类由 degree $n$ 的 multisegments 参数化。更精确地，对每个 multisegment $\mathfrak m$ 可构造不可约表示 $Z(\mathfrak m)$，且
$$
\mathfrak m\mapsto Z(\mathfrak m)
$$
是 multisegments 与 $\operatorname{Irr}(G_n)$ 之间的双射。

**命题 Q.8.** Zelevinsky 分类与 cuspidal support 的唯一性相容。

**证明.** Multisegment $\mathfrak m=\{\Delta_i\}$ 展开后给出一族 cuspidal twists $\rho\nu^j$。构造 $Z(\mathfrak m)$ 时只使用这些 cuspidal twists 的抛物诱导和 subquotient。定理 Q.2 说明不可约表示的 cuspidal support 唯一；定理 Q.7 说明 multisegment 唯一。因此展开后的 cuspidal support 与表示本身相容。$\square$

## Q.4 Langlands 分类和标准模

**定义 Q.9.** 若 $\delta_i$ 是 essentially square-integrable representations of $G_{n_i}$，且实数
$$
e(\delta_1)>\cdots>e(\delta_r)
$$
按中心 character 的实指数严格递减排列，则
$$
I(\delta_1,\ldots,\delta_r)
=\delta_1\times\cdots\times\delta_r
$$
称为 standard module。

**外部输入定理 Q.10（Langlands quotient theorem for `GL(n)`）.** 每个 standard module $I(\delta_1,\ldots,\delta_r)$ 有唯一不可约 quotient，记为
$$
J(\delta_1,\ldots,\delta_r).
$$
每个不可约 smooth representation of $G_n$ 唯一地以这种方式出现。

**命题 Q.11.** 对 `GL(n)`，局部 Langlands 对应的集合双射形式与 Langlands quotient 分类相容。

**证明路线（外部输入）.** 每个 essentially square-integrable $\delta(\Delta)$ 在 LLC 下对应某个不可约 Weil-Deligne 参数 $\phi_\rho$ 张量 special representation $\operatorname{Sp}_\ell$ 并带 unramified twist。Standard module 的诱导对应参数直和：
$$
\phi=\bigoplus_i\phi_{\delta_i}.
$$
Langlands quotient theorem 给出唯一不可约 quotient，LLC 把该 quotient 对应到 Frobenius-semisimple Weil-Deligne 参数 $\phi$。完整相容性是 `GL(n)` LLC 的核心定理之一。$\square$

## Q.5 Tempered、generic 和 Whittaker 模型

**外部输入定理 Q.12（Tempered classification for `GL(n)`）.** 不可约 tempered representations of $G_n$ 正好是若干 unitary essentially square-integrable representations 的归一化抛物诱导的不可约表示。其 LLC 参数在 $W_F$ 上 bounded。

**外部输入定理 Q.13（Whittaker uniqueness）.** 对 $G_n$，相对于非退化 character $\psi$ of maximal unipotent subgroup，任一不可约 smooth representation $\pi$ 满足
$$
\dim\operatorname{Hom}_{N_n(F)}(\pi,\psi)\le1.
$$
若该空间非零，称 $\pi$ 为 generic。

**外部输入定理 Q.14（Generic standard module criterion）.** `GL(n)` 的 generic irreducible representations 可由相应 standard modules 和 multisegment 的非退化条件刻画。特别地，irreducible generic standard module 的 Whittaker model 与 Rankin-Selberg 和 Langlands-Shahidi 局部因子的归一化相容。

**注 Q.15.** 本书不在正文中使用 multisegment 的完整 linking criterion；需要精确判别 generic 性时，应引用 Bernstein-Zelevinsky 或 Zelevinsky-Tadic 分类。

## Q.6 局部因子和 LLC

**外部输入定理 Q.16（`GL(n)` 局部因子相容）.** 若 $\pi$ 对应 Weil-Deligne 参数 $\phi_\pi$，则标准 L 因子、epsilon 因子和 Rankin-Selberg 因子满足
$$
L(s,\pi)=L(s,\phi_\pi),
\qquad
\varepsilon(s,\pi,\psi)=\varepsilon(s,\phi_\pi,\psi),
$$
以及
$$
L(s,\pi\times\pi')=L(s,\phi_\pi\otimes\phi_{\pi'}).
$$

**命题 Q.17.** 若 $\pi$ 是非分歧主级数，其 Langlands quotient 参数与附录 P 的 Satake parameter 给出同一非分歧 L 因子。

**证明.** 非分歧主级数由非分歧 characters $\chi_i$ 给出。附录 P 的 Satake parameter 是
$$
\operatorname{diag}(\chi_1(\varpi),\ldots,\chi_n(\varpi)).
$$
`GL(n)` LLC 把 $\pi$ 对应到惯性平凡、Frobenius 半单部分为同一对角矩阵的 Weil-Deligne 参数。故标准 L 因子两种定义均为
$$
\prod_{i=1}^n(1-\chi_i(\varpi)q^{-s})^{-1}.
$$
$\square$

## Q.7 对一般 Langlands 主线的接口

**命题 Q.18.** `GL(n)` 的 packet 为单元素与 Bernstein-Zelevinsky 分类相容。

**证明.** Zelevinsky 分类把每个不可约表示唯一写成 multisegment 数据；`GL(n)` LLC 把相应表示送到一个 $n$ 维 Weil-Deligne 参数。反向由定理 Q.10 和定理 Q.16 的 LLC 相容性恢复唯一 Langlands quotient。由于 centralizer component group 在第十二章命题 12.15 中为平凡，同一参数不会产生多个 packet 成员。$\square$

**命题 Q.19.** `GL(n)` converse theorem 中的候选局部表示必须至少满足 Langlands quotient 分类给出的可容许性和局部因子相容性。

**证明.** Converse theorem 的输入是 restricted tensor product $\Pi=\otimes_v'\Pi_v$。若某个 $\Pi_v$ 不是不可约可容许 representation 或不能放入 Langlands quotient 分类，则局部 Whittaker、Rankin-Selberg 积分和 $\gamma$ 因子无法按标准理论定义。若局部因子与预期参数不相容，则全局 twists 的函数方程不会匹配目标 L 群同态。因此局部分类和局部因子相容性是 converse theorem 应用的前置条件。$\square$

## 练习

**练习 Q.1.** 对 `GL(2)`，列出 principal series、Steinberg twist 和 supercuspidal 三类不可约表示在 LLC 中的大致参数形状。

**练习 Q.2.** 解释为什么 cuspidal support 只在重排后唯一。

**练习 Q.3.** 对 segment $[\rho,\rho\nu]$，说明它的 degree 是 $2\deg\rho$。

**练习 Q.4.** 说明 Langlands quotient theorem 为什么使 LLC 能从 essentially square-integrible 原子扩展到所有 `GL(n)` 表示。

**练习 Q.5.** 解释非分歧主级数的 Satake 参数和 Weil-Deligne 参数为何给出同一 Euler 因子。
