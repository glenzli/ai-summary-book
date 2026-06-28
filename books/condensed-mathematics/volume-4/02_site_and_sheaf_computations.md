# 第二章：站点、覆盖与 sheaf 条件的计算

## 本章目标

本章给出 sheaf 条件和站点比较的可计算模板。

## 2.1 二元覆盖模板

设

$$
U=U_1\cup U_2
$$

对应覆盖 $\{U_1\to U,U_2\to U\}$。sheaf 条件为

$$
F(U)\to F(U_1)\times F(U_2)
\rightrightarrows
F(U_1\times_UU_2).
$$

具体地，截面 $s_1\in F(U_1)$、$s_2\in F(U_2)$ 可粘合，当且仅当

$$
s_1|_{U_1\times_UU_2}=s_2|_{U_1\times_UU_2}.
$$

## 2.2 有限覆盖模板

对有限覆盖 $\{U_i\to U\}_{i=1}^n$，sheaf 条件为等化子

$$
F(U)\to\prod_iF(U_i)
\rightrightarrows
\prod_{i,j}F(U_i\times_UU_j).
$$

## 2.3 可表 sheaf 检查

若 $F=h_T$，其中

$$
h_T(S)=\operatorname{Hom}(S,T),
$$

则 sheaf 条件表示：从 $U$ 到 $T$ 的连续映射等价于在覆盖块 $U_i$ 上给出连续映射并在交上相容。

这是紧 Hausdorff 站点中可表对象为 sheaf 的基本原因。

## 2.4 站点比较模板

若 $\mathcal D\subset\mathcal C$ 是基子站点：

1. 每个 $U\in\mathcal C$ 可由 $\mathcal D$ 对象覆盖。
2. 纤维积可再由 $\mathcal D$ 对象覆盖。
3. 覆盖拓扑由 $\mathcal C$ 限制得到。

则

$$
\operatorname{Sh}(\mathcal C)\simeq\operatorname{Sh}(\mathcal D).
$$

## 2.5 本章小结

sheaf 计算的核心是等化子。站点比较的核心是共同细化。

## 练习

**练习 2.1.** 对三个开集覆盖写出 sheaf 条件。

**练习 2.2.** 证明可表 sheaf 的二元覆盖粘合。

**练习 2.3.** 解释共同细化在站点比较中的作用。
