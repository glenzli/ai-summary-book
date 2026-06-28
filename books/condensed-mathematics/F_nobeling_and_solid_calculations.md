# 附录 F：Nöbeling 定理与 solid 计算

## F.0 目标

第十二章引用 Nöbeling 定理来说明 solid 对象具有大量乘积型投射生成元。本附录整理相关计算，并给出可直接检查的特殊情形。一般 Nöbeling 定理的完整证明很长，本书仍按引用定理处理。

## F.1 局部常值整数函数

设 $S$ 是 profinite 集合。记

$$
C(S,\mathbb Z)
$$

为连续函数 $S\to\mathbb Z$ 构成的阿贝尔群，其中 $\mathbb Z$ 赋予离散拓扑。

**命题 F.1.** 若

$$
S=\varprojlim_i S_i
$$

是有限离散集合的逆极限，则

$$
C(S,\mathbb Z)\cong\varinjlim_i C(S_i,\mathbb Z).
$$

**证明.** 任一连续函数 $f:S\to\mathbb Z$ 的像是紧子集，因此是有限集。对每个 $n\in f(S)$，集合 $f^{-1}(n)$ 是开闭集。有限多个开闭集给出 $S$ 的有限开闭划分，因此该划分由某个有限商 $S_i$ 拉回。于是 $f$ 在 $S_i$ 上因子化。反向映射由拉回给出。证毕。

**命题 F.2.** 若 $S$ 是 metrizable profinite 空间，则 $C(S,\mathbb Z)$ 是自由阿贝尔群。

**证明.** 取可数 cofinal 的有限商塔

$$
S\to\cdots\to S_n\to S_{n-1}\to\cdots\to S_0.
$$

于是

$$
C(S,\mathbb Z)=\varinjlim_n C(S_n,\mathbb Z).
$$

每个 $C(S_n,\mathbb Z)$ 是有限秩自由阿贝尔群。若 $q:S_{n+1}\to S_n$ 是满射，则拉回

$$
q^*:C(S_n,\mathbb Z)\hookrightarrow C(S_{n+1},\mathbb Z)
$$

把一个函数变成在每个纤维上常值的函数。对每个纤维选一个点，可把 $C(S_{n+1},\mathbb Z)$ 分解为“纤维常值部分”与“纤维内差分部分”的直和，因此 $q^*$ 的像是直和因子。

递归选择每一步的补空间，可得

$$
C(S,\mathbb Z)\cong C(S_0,\mathbb Z)\oplus\bigoplus_{n\ge0} Q_n
$$

其中每个 $Q_n$ 是有限秩自由阿贝尔群。因此 $C(S,\mathbb Z)$ 自由。证毕。

## F.2 Nöbeling 定理

**定理 F.3（Nöbeling）.** 对任意 profinite 集合 $S$，阿贝尔群 $C(S,\mathbb Z)$ 是自由阿贝尔群。

**证明说明.** 非 metrizable 情形不能只用可数塔。Nöbeling/Bergman 的证明把 $S$ 嵌入某个 Cantor cube

$$
\{0,1\}^I
$$

并对有限支撑的坐标条件作良序归纳，构造一组由开闭条件函数生成的基。关键点是控制无限坐标集带来的相容性，避免一般“自由群的滤过并仍自由”这一错误命题。本书不重写该长证明。

## F.3 整值测度与 $\mathbb Z^\square[S]$

**定义 F.4.** $S$ 上的整值测度群定义为

$$
M(S,\mathbb Z)=\operatorname{Hom}(C(S,\mathbb Z),\mathbb Z).
$$

若 $\mu\in M(S,\mathbb Z)$，$f\in C(S,\mathbb Z)$，写作

$$
\int_S f\,d\mu=\mu(f).
$$

**命题 F.5.** 若 $S=\varprojlim_iS_i$，则

$$
M(S,\mathbb Z)\cong\varprojlim_i\mathbb Z[S_i].
$$

**证明.** 由命题 F.1，

$$
C(S,\mathbb Z)=\varinjlim_iC(S_i,\mathbb Z).
$$

因此

$$
\operatorname{Hom}(C(S,\mathbb Z),\mathbb Z)
\cong
\varprojlim_i\operatorname{Hom}(C(S_i,\mathbb Z),\mathbb Z).
$$

有限集合 $S_i$ 上有自然配对

$$
\mathbb Z[S_i]\times C(S_i,\mathbb Z)\to\mathbb Z,
\qquad
\left(\sum n_s[s],f\right)\mapsto\sum n_sf(s),
$$

给出

$$
\mathbb Z[S_i]\cong\operatorname{Hom}(C(S_i,\mathbb Z),\mathbb Z).
$$

代入即得结论。证毕。

**命题 F.6.** 若 $T$ 是 profinite 测试对象，则

$$
\mathbb Z^\square[S](T)
\cong
\operatorname{Hom}(C(S,\mathbb Z),C(T,\mathbb Z)).
$$

**证明.** 由定义

$$
\mathbb Z^\square[S]=\varprojlim_i\mathbb Z[\underline{S_i}].
$$

对有限离散 $S_i$，自由凝聚阿贝尔群 $\mathbb Z[\underline{S_i}]$ 在 $T$ 上的取值是局部常值函数

$$
C(T,\mathbb Z[S_i]).
$$

有限自由性给出

$$
C(T,\mathbb Z[S_i])
\cong
\operatorname{Hom}(C(S_i,\mathbb Z),C(T,\mathbb Z)).
$$

对 $i$ 取逆极限，并使用 $C(S,\mathbb Z)=\varinjlim_iC(S_i,\mathbb Z)$，得到所需同构。证毕。

## F.4 乘积型表示

**推论 F.7.** 对任意 profinite 集合 $S$，存在集合 $I$ 使

$$
\mathbb Z^\square[S]\cong\prod_I\underline{\mathbb Z}.
$$

**证明.** 由 Nöbeling 定理，取基

$$
C(S,\mathbb Z)\cong\bigoplus_I\mathbb Z.
$$

对任意 profinite 测试对象 $T$，命题 F.6 给出

$$
\mathbb Z^\square[S](T)
\cong
\operatorname{Hom}\left(\bigoplus_I\mathbb Z,C(T,\mathbb Z)\right)
\cong
\prod_I C(T,\mathbb Z)
\cong
\left(\prod_I\underline{\mathbb Z}\right)(T).
$$

这些同构自然于 $T$，因此给出凝聚阿贝尔群同构。证毕。

## F.5 例子

**例 F.8（有限集合）.** 若 $S$ 有 $n$ 个点，则

$$
C(S,\mathbb Z)\cong\mathbb Z^n,\qquad
M(S,\mathbb Z)\cong\mathbb Z^n,
$$

且

$$
\mathbb Z^\square[S]\cong\mathbb Z[\underline S].
$$

**例 F.9（Cantor 集）.** 令 $S=\{0,1\}^{\mathbb N}$。有限前缀给出有限商 $S_n=\{0,1\}^n$。命题 F.2 的证明给出 $C(S,\mathbb Z)$ 的可数自由基。因此

$$
\mathbb Z^\square[S]\cong\prod_{\mathbb N}\underline{\mathbb Z}.
$$

这里的同构依赖基的选择，不是典范同构。

**例 F.10（不可数 Cantor cube）.** 若 $S=\{0,1\}^{I}$ 且 $I$ 不可数，则 $C(S,\mathbb Z)$ 仍自由，但通常不是可数秩。此时必须使用 Nöbeling 定理，而不能用可数塔证明。

## F.6 solid 性的计算用法

若 $A$ 是固体阿贝尔群，则定义 12.4 给出

$$
\operatorname{Hom}(\mathbb Z^\square[S],A)\cong A(S).
$$

结合推论 F.7，可把左侧改写为

$$
\operatorname{Hom}\left(\prod_I\underline{\mathbb Z},A\right).
$$

因此 solid 对象的核心性质可以读作：$A$ 能以与所有 profinite 测度兼容的方式接收从乘积型自由对象来的态射。

## F.7 不应误读的地方

1. Nöbeling 定理不是说所有 torsion-free 阿贝尔群都自由。
2. $\mathbb Z^\square[S]\cong\prod_I\underline{\mathbb Z}$ 中的 $I$ 依赖所选的 $C(S,\mathbb Z)$ 基，通常不具典范性。
3. solidification 不是普通拓扑群完备化；它是凝聚阿贝尔群范畴中的左伴随。
4. 对实数或一般 Banach 空间，solid 不是最终答案，后续需要 analytic/liquid 结构。

## 练习

**练习 F.1.** 对有限满射 $q:T\to S$，显式构造 $q^*:C(S,\mathbb Z)\to C(T,\mathbb Z)$ 的一个直和补。

**练习 F.2.** 用命题 F.2 证明 $C(\mathbb Z_p,\mathbb Z)$ 自由。

**练习 F.3.** 对 $S=\{0,1\}^{\mathbb N}$，写出前三级有限商带来的自由群分解。

**练习 F.4.** 证明推论 F.7 中的同构不是典范的：它依赖 $C(S,\mathbb Z)$ 的基。
