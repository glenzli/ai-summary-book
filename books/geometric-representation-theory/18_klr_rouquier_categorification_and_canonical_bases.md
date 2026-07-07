# 第十八章：KLR/Rouquier categorification 与 canonical bases

## 本章目标

本章介绍 Khovanov-Lauda-Rouquier algebras、cyclotomic quotients、Grothendieck groups 和 canonical bases 的关系。

## 依赖前置知识

需要 Kac-Moody algebra、graded algebras、projective modules 和 Grothendieck groups。

## 18.1 KLR algebra 的数据

**定义 18.1.** 固定 symmetrizable Cartan datum，顶点集为 $I$。对 $\nu\in\mathbb N[I]$，KLR algebra $R(\nu)$ 是由 idempotents $e(\mathbf i)$、多项式生成元 $y_r$ 和 braid 生成元 $\psi_r$ 给出的 graded algebra，关系由 Cartan datum 和多项式 $Q_{ij}(u,v)$ 决定。

**警告 18.2.** KLR relations 的完整列表较长，且存在 simply-laced、symmetric、symmetrizable 等多个版本。本章只登记结构角色，最终公式需单独附录。

**定义 18.3.** 在 simply-laced 情形中，$R(\nu)$ 的 idempotents 由序列
$$
\mathbf i=(i_1,\ldots,i_m),\qquad \sum_r\alpha_{i_r}=\nu
$$
标号。生成元包括
$$
e(\mathbf i),\qquad y_1,\ldots,y_m,\qquad \psi_1,\ldots,\psi_{m-1}.
$$
它们满足正交幂等元关系、$y_r$ 交换关系、$\psi_r e(\mathbf i)=e(s_r\mathbf i)\psi_r$，以及二次关系
$$
\psi_r^2 e(\mathbf i)=
\begin{cases}
0,& i_r=i_{r+1},\\
(y_{r+1}-y_r)e(\mathbf i),& i_r\text{ 与 }i_{r+1}\text{ 相邻},\\
e(\mathbf i),& i_r\text{ 与 }i_{r+1}\text{ 不相邻}.
\end{cases}
$$
braid relation 在相邻颜色相互作用时带有修正项。

**警告 18.4.** 上述公式只是 simply-laced normalization。非 simply-laced 情形必须用 $Q_{ij}(u,v)$ 统一写法，不能直接套用该二次关系。

**定义 18.5.** induction product 定义为
$$
M\circ N=R(\nu+\nu')\otimes_{R(\nu)\otimes R(\nu')}(M\otimes N).
$$
restriction functor 由相应 idempotent 截取给出。

**命题 18.6.** induction product 在 Grothendieck group 上给出结合乘法。

**证明.** 三个权重 $\nu,\nu',\nu''$ 的 induction 可通过嵌入
$$
R(\nu)\otimes R(\nu')\otimes R(\nu'')\subset R(\nu+\nu'+\nu'')
$$
一次完成。两种加括号方式都是张量到同一个大代数：
$$
R(\nu+\nu'+\nu'')\otimes_{R(\nu)\otimes R(\nu')\otimes R(\nu'')}(M\otimes N\otimes P).
$$
张量积的结合约束给出自然同构，因此 Grothendieck group 上乘法结合。$\square$

## 18.2 $\mathfrak{sl}_2$ 情形

**例 18.7.** 对 $\mathfrak{sl}_2$，顶点集只有一个元素 $i$。权重 $\nu=n\alpha_i$ 时，所有序列都等于 $(i,\ldots,i)$，因此 idempotent 只有一个。KLR algebra $R(n)$ 由 $y_1,\ldots,y_n$ 和 $\psi_1,\ldots,\psi_{n-1}$ 生成，并满足 nilHecke algebra 关系：
$$
\psi_r^2=0,\qquad
\psi_r\psi_{r+1}\psi_r=\psi_{r+1}\psi_r\psi_{r+1},
$$
以及
$$
\psi_r y_{r+1}-y_r\psi_r=1,\qquad
y_{r+1}\psi_r-\psi_r y_r=1.
$$

**命题 18.8.** $R(1)\simeq E[y_1]$，其 finitely generated graded projective modules 的 split Grothendieck group 是 rank-one free $\mathbb Z[q,q^{-1}]$-module。

**证明.** 当 $n=1$ 时没有 braid 生成元，只有多项式生成元 $y_1$，故 $R(1)=E[y_1]$。graded polynomial ring 上的 finitely generated graded projective modules 在本情形中由 Quillen-Suslin 定理给出为 graded free modules 的直和项；其 split Grothendieck group由 $[R(1)\langle m\rangle]$ 生成，grading shift 对应乘以 $q^m$，故为 rank-one free。若不调用 Quillen-Suslin，可把该结论限制为 free projectives 子范畴中的计算。$\square$

## 18.3 Categorification theorem

**外部输入定理 18.9.** Khovanov-Lauda 和 Rouquier 的 categorification theorem：KLR algebras 的 finitely generated graded projective modules 的 Grothendieck group categorifies $U_q^-(\mathfrak g)$ 的 integral form。

**外部输入定理 18.10.** 在 symmetric 情形中，Varagnolo-Vasserot 和 Rouquier 识别 indecomposable projective modules 与 Lusztig canonical basis。cyclotomic KLR algebras categorify integrable highest weight modules。

**资料入口 18.11.** Kang-Kashiwara, arXiv:1102.4677 证明 cyclotomic KLR algebras 对所有 symmetrizable Kac-Moody algebras categorify highest weight modules。

## 18.4 与 quiver varieties 的关系

**外部输入定理 18.12.** KLR algebras 可通过 quiver varieties、perverse sheaves 或 Ext algebras 几何实现，在这些模型中 canonical basis 对应 simple perverse sheaves 或 indecomposable projectives。

**边界说明 18.13.** “canonical basis = simple objects”需要说明使用 lower global basis、upper global basis、projective modules、simple modules 还是 IC sheaves。不同 convention 下会出现 duality。

## 本章小结

本章给出 KLR algebra 的结构角色、simply-laced 关系模板、$\mathfrak{sl}_2$ 低阶情形、induction product 和 Grothendieck group 乘法。核心范畴化和 canonical basis theorem 均为外部输入。

## 练习

**练习 18.1.** 在 $\mathfrak{sl}_2$ 情形中描述 $R(n)$ 的生成元类型。

**练习 18.2.** 证明 restriction functor 与 induction functor 在有限维代数情形中形成伴随对的标准条件。

**练习 18.3.** 解释 projective Grothendieck group 和 finite-dimensional module Grothendieck group 为什么通常互为配对。
