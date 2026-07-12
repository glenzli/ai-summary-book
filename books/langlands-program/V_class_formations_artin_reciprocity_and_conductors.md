# 附录 V：Class Formations、Artin Reciprocity 和导子接口

收口归一化回指：本附录支撑 `GL(1)` 链的 reciprocity、ray class conductor 和 finite Hecke character 比较；Frobenius 与 conductor convention 见 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 2、6、8 节。

## V.1 Class formation 的抽象形状

类域论在本书中作为 `GL(1)` Langlands 的基础外部输入。本附录记录其 cohomological formulation，避免把 Artin reciprocity 当作无结构黑箱。

**定义 V.1.** 一个 class formation 的数据包括：

1. 一个 profinite group $G$；
2. 对每个开子群 $H\subset G$ 给出一个 $H$-module $C_H$；
3. restriction、corestriction、norm maps
   $$
   C_H\to C_{H'},\qquad C_{H'}\to C_H
   $$
   对 $H'\subset H$；
4. 一个 fundamental class
   $$
   u_{H'/H}\in H^2(H/H',C_{H'})
   $$
   满足 functorial compatibility；
5. Tate cohomology 条件，使得 cup product with $u_{H'/H}$ 给出预期同构。

**外部输入定理 V.2（class formation axioms for local and global fields）.** 局部域和数域的 ideles/ideles class groups 构成 class formations。局部情形的 $C_H$ 可取 multiplicative group，整体情形的 $C_H$ 可取 idele class group。

**注 V.3.** Class formation 的意义在于：reciprocity map 不是先验构造的偶然同构，而是由 fundamental class 和 Tate cohomology 控制的规范对象。

## V.2 局部 Artin 映射

设 $F$ 为非 Archimedean 局部域。

**外部输入定理 V.4（局部 reciprocity）.** 存在唯一连续同态
$$
\operatorname{rec}_F:F^\times\to W_F^{\operatorname{ab}}
$$
满足：

1. $\operatorname{rec}_F(\varpi)$ 为几何 Frobenius 在非分歧商中的像；
2. 对任意有限 Abel 扩张 $E/F$，诱导同构
   $$
   F^\times/N_{E/F}E^\times\simeq\operatorname{Gal}(E/F);
   $$
3. 对有限扩张 $F'/F$ 与 norm/corestriction 相容。

**命题 V.5.** 局部非分歧扩张由 valuation quotient 控制。

**证明.** 对非分歧有限 Abel 扩张 $E/F$，norm map 在单位群上满射：
$$
N_{E/F}\mathcal O_E^\times=\mathcal O_F^\times.
$$
同时
$$
v_F(N_{E/F}x)=[E:F]\,v_E(x)
$$
在 value group 上给出 $[E:F]\mathbb Z\subset\mathbb Z$。因此
$$
F^\times/N_{E/F}E^\times\simeq\mathbb Z/[E:F]\mathbb Z,
$$
由一致化元类生成。局部 reciprocity 把它送到由几何 Frobenius 生成的 Galois 群。$\square$

## V.3 全局 Artin 映射

设 $K$ 为数域，idele class group 为
$$
C_K=K^\times\backslash\mathbb A_K^\times.
$$

**外部输入定理 V.6（全局 reciprocity）.** 存在连续同态
$$
\operatorname{rec}_K:C_K\to G_K^{\operatorname{ab}}
$$
使得对每个有限 Abel 扩张 $L/K$，诱导同构
$$
C_K/N_{L/K}C_L\simeq\operatorname{Gal}(L/K),
$$
并与所有局部 reciprocity maps 相容。其像在 $G_K^{\operatorname{ab}}$ 中稠密，且诱导
$$
\widehat{C_K}\xrightarrow{\sim}G_K^{\operatorname{ab}}.
$$
因此本定理给出的是所有有限 Abel 商及 profinite completion 上的同构，不是
$C_K\xrightarrow{\sim}G_K^{\operatorname{ab}}$。

**外部输入定理 V.7（global product formula for local symbols）.** 对 $a\in K^\times$，所有局部 Artin 符号的乘积为 $1$：
$$
\prod_v\operatorname{rec}_{K_v}(a)=1
$$
在任意有限 Abel 商中成立。等价地，全局 reciprocity map 在对角嵌入的 $K^\times$ 上平凡。

**命题 V.8.** 全局 reciprocity 从局部 reciprocity 拼合时必须商去 $K^\times$。

**证明.** 局部 reciprocity maps 给出
$$
\prod_v K_v^\times\to G_K^{\operatorname{ab}}
$$
的候选乘积。若 $a\in K^\times$，它在每个 $K_v^\times$ 中给出局部元素。定理 V.7 说明这些局部 Artin 符号乘积为 $1$。因此该乘积同态在对角 $K^\times$ 上平凡，下降到 idele class group $C_K$。$\square$

## V.4 Norm subgroup theorem

**外部输入定理 V.9（Norm subgroup theorem）.** 数域 $K$ 的 idele class group $C_K$ 的开有限指数子群恰为某个有限 Abel 扩张 $L/K$ 的 norm subgroup
$$
N_{L/K}C_L\subset C_K.
$$
并且
$$
C_K/N_{L/K}C_L\simeq\operatorname{Gal}(L/K).
$$

**命题 V.10.** 有限阶 Hecke characters 与有限 Abel 扩张的 characters 等价。

**证明.** 有限阶 Hecke character
$$
\chi:C_K\to\mathbb C^\times
$$
的 kernel 是开有限指数子群。由定理 V.9，存在有限 Abel 扩张 $L/K$ 使 $\ker\chi$ 含某个 norm subgroup，并且 $\chi$ 通过
$$
C_K/N_{L/K}C_L\simeq\operatorname{Gal}(L/K)
$$
分解。反过来，任何 finite character of $\operatorname{Gal}(L/K)$ 与全局 reciprocity 复合给出有限阶 Hecke character。$\square$

## V.5 Ray class groups 和导子

令 $\mathfrak m=\mathfrak m_f\mathfrak m_\infty$ 为 modulus。

**定义 V.11.** Ray class group 可写为
$$
\operatorname{Cl}_{\mathfrak m}(K)
=I^{\mathfrak m}/P_{1,\mathfrak m},
$$
其中 $I^{\mathfrak m}$ 为与 $\mathfrak m_f$ 互素的 fractional ideals 群，$P_{1,\mathfrak m}$ 由满足局部 congruence 和符号条件的 principal ideals 生成。

**外部输入定理 V.12（ray class field）.** 对每个 modulus $\mathfrak m$，存在有限 Abel 扩张 $K_{\mathfrak m}/K$，使
$$
\operatorname{Gal}(K_{\mathfrak m}/K)\simeq\operatorname{Cl}_{\mathfrak m}(K),
$$
且其 ramification 由 $\mathfrak m$ 控制。每个有限 Abel 扩张嵌入某个 ray class field。

**定义 V.13.** 有限阶 Hecke character $\chi:C_K\to\mathbb C^\times$ 的 conductor 是最小 modulus $\mathfrak f(\chi)$，使 $\chi$ 在相应 principal congruence ideles 上平凡。

**命题 V.14.** Dirichlet character 的 conductor 是定义 V.13 在 $K=\mathbb Q$ 情形的特例。

**证明.** 对 $K=\mathbb Q$，有限 idele 单位商
$$
\widehat{\mathbb Z}^\times/(1+N\widehat{\mathbb Z})
\simeq(\mathbb Z/N\mathbb Z)^\times.
$$
一个 Dirichlet character modulo $N$ 等价于在该商上的 character。最小的 $N$ 使 character 通过该商分解，正是经典 Dirichlet conductor；这与 Hecke character 在 principal congruence subgroup 上平凡的最小 modulus 一致。$\square$

## V.6 `GL(1)` Langlands 的重述

**命题 V.15.** 有限阶 `GL(1)` 全局 Langlands 是全局类域论的 character 形式。

**证明.** Galois 侧对象为有限像 character
$$
\rho:G_K\to\mathbb C^\times.
$$
由于像 Abel，$\rho$ 通过 $G_K^{\operatorname{ab}}$ 分解。由全局 reciprocity 得到 Hecke character
$$
\chi_\rho=\rho\circ\operatorname{rec}_K:C_K\to\mathbb C^\times.
$$
反过来，有限阶 Hecke character $\chi$ 通过有限商
$C_K/\ker\chi$ 分解，因而唯一延拓到 $\widehat{C_K}$；再用定理 V.6 的同构
$\widehat{C_K}\simeq G_K^{\operatorname{ab}}$ 得到有限像 Galois character。两个方向在同一有限 Abel 商上互逆。局部相容性来自定理 V.6 与局部 reciprocity 的相容性。$\square$

**命题 V.16.** 非有限阶 Hecke quasi-character 属于 Weil 侧而非 profinite Galois 侧。

**证明.** Profinite group $G_K$ 的连续像仍为 profinite；其在 $\mathbb C^\times$ 中又是紧子群，故落在 $S^1$。$S^1$ 的闭子群只有有限循环群或 $S^1$，而 profinite image 是 totally disconnected，不能等于 $S^1$，所以该像必有限。一般 Hecke quasi-character 可含 $|\cdot|^s$ 这样的非有限阶连续因子，其在 ideles 上自然存在，但不能作为普通 profinite Galois group 的连续复表示解释。Weil group 允许这类非紧实参数，因此 `GL(1)` 的一般 quasi-character 应放在 Weil 参数侧。$\square$

## 练习

**练习 V.1.** 用局部 reciprocity 计算非分歧 character 的 Frobenius 值。

**练习 V.2.** 证明全局 reciprocity 必须在 $K^\times$ 上平凡。

**练习 V.3.** 对 $K=\mathbb Q$，把 Dirichlet characters 写成 finite idele class characters。

**练习 V.4.** 解释 norm subgroup theorem 与有限 Abel 扩张分类的关系。

**练习 V.5.** 说明为什么一般 Hecke quasi-character 不应写成 profinite Galois character。
