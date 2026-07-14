# 第十章：凝聚模

一个凝聚环 $R$ 只有在其作用态射也留在 sheaf 范畴中时才能支撑模论。把普通
$R(S)$-模逐点拼在一起并不够，因为相对张量积和余等化子仍须满足粘合。正确的对象是
$\mathbf{CondAb}$ 内部的 $R$-模；正确的自由模则应由一个泛性质表示测试空间上的
截面，而不是由未经 sheafification 的逐点自由构造给出。

第九章的对称幺半结构提供作用态射与相对张量积，第七章的自由凝聚群以及第六章的 ED
正合取值给出投射性机制。对任意测试空间 $S$，对象

$$
R[\underline E]
$$

的同类构造 $R[\underline S]$ 将表示 $M\mapsto M(S)$；当 $S=E$ 极不连通时它
投射，并且这些对象足以逐步覆盖任意凝聚 $R$-模。这为下一章的派生张量提供实际可用的
投射分解。

## 10.1 凝聚模的定义

设 $R$ 是凝聚环。

**定义 10.1.** 一个凝聚 $R$-模是凝聚阿贝尔群 $M$，配备作用态射

$$
\alpha:R\otimes M\to M,
$$

满足通常的结合律和单位律：

$$
(rs)m=r(sm),\qquad 1m=m.
$$

凝聚 $R$-模范畴记为

$$
\mathbf{Mod}_R(\mathbf{CondAb})
$$

或简写为

$$
\mathbf{CondMod}_R.
$$

**注 10.2.** 若 $R$ 是环值 sheaf，则凝聚 $R$-模也可理解为 $R$-模值 sheaf。但张量积定义提醒我们：模结构应放在 sheaf 范畴的张量结构中理解。

## 10.2 拓扑模给出的例子

**定义 10.3.** 设 $R$ 是拓扑环，$M$ 是拓扑 $R$-模，即 $M$ 是拓扑阿贝尔群，并且作用映射

$$
R\times M\to M
$$

连续。定义

$$
\underline M(S)=\operatorname{Cont}(S,M).
$$

**命题 10.4.** $\underline M$ 是凝聚 $\underline R$-模。

**证明.** 第四章给出 $\underline M$ 的凝聚阿贝尔群结构，第九章给出 $\underline R$ 的凝聚环结构。作用映射逐点定义：

$$
(\rho\cdot m)(s)=\rho(s)m(s),
$$

其中 $\rho:S\to R$，$m:S\to M$ 连续。由于原作用 $R\times M\to M$ 连续，复合

$$
S\xrightarrow{(\rho,m)}R\times M\to M
$$

连续。模公理逐点成立。证毕。

## 10.3 自由凝聚 $R$-模

设 $X\in\mathbf{CondSet}$。定义

$$
R[X]=R\otimes \mathbb Z[X].
$$

其中 $\mathbb Z[X]$ 是第七章的自由凝聚阿贝尔群。

**命题 10.5.** $R[X]$ 是由凝聚集合 $X$ 生成的自由凝聚 $R$-模。即对任意 $M\in\mathbf{CondMod}_R$，有自然双射

$$
\operatorname{Hom}_{\mathbf{CondMod}_R}(R[X],M)
\cong
\operatorname{Hom}_{\mathbf{CondSet}}(X,U(M)),
$$

其中 $U$ 是忘却到凝聚集合的函子。

**证明.** 使用两次伴随。首先，扩张标量函子

$$
R\otimes -:\mathbf{CondAb}\to\mathbf{CondMod}_R
$$

左伴随于忘却函子 $\mathbf{CondMod}_R\to\mathbf{CondAb}$。因此

$$
\operatorname{Hom}_{\mathbf{CondMod}_R}(R\otimes \mathbb Z[X],M)
\cong
\operatorname{Hom}_{\mathbf{CondAb}}(\mathbb Z[X],U_{\mathbf{Ab}}(M)).
$$

再由第七章命题 7.2，

$$
\operatorname{Hom}_{\mathbf{CondAb}}(\mathbb Z[X],U_{\mathbf{Ab}}(M))
\cong
\operatorname{Hom}_{\mathbf{CondSet}}(X,U(M)).
$$

证毕。

## 10.4 可表自由模

取 $X=\underline S$，得到

$$
R[\underline S]=R\otimes \mathbb Z[\underline S].
$$

**命题 10.6.** 对任意 $M\in\mathbf{CondMod}_R$，有自然同构

$$
\operatorname{Hom}_{\mathbf{CondMod}_R}(R[\underline S],M)
\cong
M(S).
$$

**证明.** 由命题 10.5 与 Yoneda 引理：

$$
\operatorname{Hom}_{\mathbf{CondMod}_R}(R[\underline S],M)
\cong
\operatorname{Hom}_{\mathbf{CondSet}}(\underline S,U(M))
\cong
M(S).
$$

证毕。

## 10.5 极不连通空间给出投射模

**定理 10.7.** 若 $E$ 是极不连通紧 Hausdorff 空间，则

$$
R[\underline E]
$$

是 $\mathbf{CondMod}_R$ 中的投射对象。

**证明.** 由命题 10.6，

$$
\operatorname{Hom}_{\mathbf{CondMod}_R}(R[\underline E],-)
\cong
(-)(E).
$$

凝聚 $R$-模中的短正合列按底层凝聚阿贝尔群判断。第六章定理 6.11 说明在极不连通 $E$ 上取值是正合的。因此 Hom 函子正合，$R[\underline E]$ 投射。证毕。

## 10.6 足够多投射模

**定理 10.8.** $\mathbf{CondMod}_R$ 有足够多的投射对象。更具体地，每个 $M\in\mathbf{CondMod}_R$ 都存在满射

$$
\bigoplus_\alpha R[\underline{E_\alpha}]
\to M,
$$

其中 $E_\alpha$ 是极不连通紧 Hausdorff 空间。

**证明.** 与第七章定理 7.7 相同。对所有二元组 $(E,m)$ 取直和，其中 $E$ 遍历极不连通紧 Hausdorff 空间的代表集合，$m\in M(E)$。每个 $m$ 通过命题 10.6 给出态射

$$
R[\underline E]\to M.
$$

把这些态射求和得到 $\Phi$。对任意 $S$ 和 $x\in M(S)$，取极不连通覆盖 $E\to S$；限制 $x|_E\in M(E)$ 正是某个生成元的像。因此 $\Phi$ 在 sheaf 意义下局部满射，故为满射。源是投射对象的直和，仍投射。证毕。

## 10.7 相对张量积

设 $M,N\in\mathbf{CondMod}_R$。定义相对张量积

$$
M\otimes_R N
$$

为 $\mathbf{CondAb}$ 中的余等化子：

$$
M\otimes R\otimes N
\rightrightarrows
M\otimes N
\to
M\otimes_R N.
$$

两条箭头分别由 $R$ 对 $M$ 的作用和 $R$ 对 $N$ 的作用给出。

若 $R$ 是交换凝聚环，则 $M\otimes_R N$ 自然也是凝聚 $R$-模。

**注 10.9.** 相对张量积一般不是逐点相对张量积。预层层面的逐点构造仍需 sheafification 或等价地在 sheaf 范畴中取余等化子。

## 10.8 自由模仍由测试空间控制

内部作用态射定义了凝聚 $R$-模，拓扑 $R$-模则通过连续映射给出实例。伴随计算

$$
\operatorname{Hom}_R(R[\underline S],M)\cong M(S)
$$

把自由模与测试取值联系起来；当 $S$ 极不连通时，右侧正合，故自由模投射。利用 ED
覆盖可得到足够多投射对象，而相对张量积由 sheaf 范畴中的余等化子给出。现在
$-\otimes_RN$ 已有定义域、值域与投射分解，下一章可以严格取其左导出函子。

## 练习

**练习 10.1.** 写出凝聚 $R$-模定义中的结合律和单位律交换图。

**练习 10.2.** 设 $R$ 是离散环，$M$ 是离散 $R$-模，$S$ 是有限离散空间。证明

$$
\underline M(S)\cong M^S
$$

作为 $\underline R(S)$-模成立。

**练习 10.3.** 证明命题 10.6 的同构关于 $M$ 自然。

**练习 10.4.** 证明定理 10.8 中构造的 $\Phi$ 是 sheaf 意义下的满射。

**练习 10.5.** 写出相对张量积 $M\otimes_R N$ 的泛性质。
