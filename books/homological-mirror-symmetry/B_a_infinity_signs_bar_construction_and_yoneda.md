# 附录 B：$A_\infty$ 符号、bar construction 与 Yoneda embedding

## B.1 Bar coalgebra

**定义 B.1.** 对分次空间 $A$，reduced tensor coalgebra 为
$$
T^c(sA)=\bigoplus_{d\ge1}(sA)^{\otimes d}.
$$
其 coproduct 为 deconcatenation：
$$
\Delta(sa_d\otimes\cdots\otimes sa_1)
=\sum_i (sa_d\otimes\cdots\otimes sa_{i+1})\otimes
(sa_i\otimes\cdots\otimes sa_1).
$$

**定义 B.2.** $A_\infty$ 结构是次数 $+1$ coderivation $b:T^c(sA)\to T^c(sA)$，满足 $b^2=0$。Taylor component $b_d:(sA)^{\otimes d}\to sA$ desuspend 后给出 $\mu^d$。

## B.2 低阶方程

**命题 B.3.** 在本书约定下，$b^2=0$ 的低阶含义为：

1. $\mu^1\mu^1=0$；
2. $\mu^1\mu^2$ 等于 $\mu^2$ 对两个输入分别作用 $\mu^1$ 的带符号和；
3. $\mu^2(\mu^2(-,-),-)$ 与 $\mu^2(-,\mu^2(-,-))$ 的差由 $\mu^1\mu^3$ 和 $\mu^3$ 中插入 $\mu^1$ 的项控制。

**证明.** 按 tensor length 分解 $b^2=0$。length $1$ 给出第一项；length $2$ 给出微分与二元复合相容；length $3$ 给出 associator 与 $\mu^3$ 边界的关系。具体 Koszul 符号由 suspension 元素穿过输入时产生。证毕。

## B.3 $A_\infty$ functor

**定义 B.4.** $A_\infty$ functor 是 coalgebra morphism
$$
F:T^c(sA)\to T^c(sB)
$$
满足
$$
F b_A=b_B F.
$$
其 Taylor components $F^d$ desuspend 后次数为 $1-d$。

**命题 B.5.** $A_\infty$ functor 的一阶部分 $F^1$ 是 cochain map。

**证明.** 取方程 $F b_A=b_B F$ 的 tensor length $1$ 部分，只出现 $F^1\mu_A^1$ 与 $\mu_B^1F^1$。因此 $F^1$ 与微分相容。证毕。

## B.4 Yoneda

**定义 B.6.** 对 $A_\infty$ category $\mathcal A$，右 Yoneda module 为
$$
Y_X(-)=\operatorname{hom}_{\mathcal A}(-,X).
$$

**外部输入定理 B.7.** Yoneda embedding
$$
Y:\mathcal A\to\operatorname{Mod}(\mathcal A)
$$
cohomologically fully faithful。

**解释 B.8.** 该定理是把对象转化为 modules 的基础。Morita theory、perfect modules 和 split-generation 都依赖它。

## 本附录小结

Bar coalgebra 约定把所有 $A_\infty$ 符号压缩为 $b^2=0$。正文使用该约定；需要低阶公式时，先展开 tensor length，再处理 Koszul 符号。
