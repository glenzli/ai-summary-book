# 第十八章：Hochschild invariants、closed-open maps 与 categorical enumerative checks

候选对象字典即使给出相同的 $K_0$，也可能在 morphism complexes、pairing 或高阶复合上失败。Hochschild chains 把所有循环可复合态射同时纳入一个复形，因而比逐对象维数更接近范畴整体；HKR 又把 B-side Hochschild 同调展开为 Hodge 上同调。A-side 的 open-closed/closed-open maps 则把这些范畴不变量接到量子或辛上同调。本章从第一章的 Morita 理论和第十四章的 string maps 出发，给出可计算的必要检验，并精确说明 Todd 修正、properness 与 Calabi--Yau 假设出现在哪里。

## 18.1 Hochschild invariants

**定义 18.1.** 对小、严格含单位 $A_\infty$ category $\mathcal A$，采用附录
B 的 suspension 约定，其 Hochschild chain module 为 cyclic bar construction
$$
CC_\bullet(\mathcal A)=
\bigoplus_{d\ge0}\ \bigoplus_{X_0,\ldots,X_d}
\operatorname{hom}_{\mathcal A}(X_d,X_0)\otimes
s\operatorname{hom}_{\mathcal A}(X_{d-1},X_d)\otimes\cdots\otimes
s\operatorname{hom}_{\mathcal A}(X_0,X_1).
\tag{18.1}
$$
Hochschild differential 把各 $b_r$ 插入连续输入块，并包含跨越 cyclic cut
的插入；$b^2=0$ 蕴含该 differential 平方为零。其 homology 记为
$HH_\ast(\mathcal A)$。Hochschild cochains 是从 bar tensors 到 morphism
spaces 的 compatible multilinear maps，带由 $A_\infty$ 结构诱导的微分；
其 cohomology 记为 $HH^\ast(\mathcal A)$，并带 cup product 与 Gerstenhaber
bracket。

**例 18.1A.** 把 $k$ 看成单对象 dg category，则
$$
HH_0(k)\cong k,\qquad HH_i(k)=0\quad(i\ne0).
$$
归一化 bar complex 的正长度部分含单位输入并可由插入单位的退化算子收缩；
只剩长度零的 $k$。这个例子说明 Hochschild chains 虽含任意长度 cyclic
tensors，却会按范畴关系消去冗余长度。

**外部输入定理 18.2（Morita invariance）.** Hochschild homology 和 cohomology 在 Morita equivalence 下不变。

**推论 18.3.** 若 HMS 以 Morita equivalence 形式成立：
$$
\mathcal A_A\simeq_{\mathrm{Morita}}\mathcal B_B,
$$
则
$$
HH_\ast(\mathcal A_A)\cong HH_\ast(\mathcal B_B),\qquad
HH^\ast(\mathcal A_A)\cong HH^\ast(\mathcal B_B).
$$

**证明.** 直接由定理 18.2 应用于 HMS Morita equivalence。证毕。

## 18.2 HKR 与 Hodge 检查

**外部输入定理 18.4（HKR）.** 若 $X$ 是特征零域上的光滑 proper variety，则存在 Hochschild-Kostant-Rosenberg 型同构
$$
HH_i(\operatorname{Perf}(X))\cong
\bigoplus_{p-q=i}H^q(X,\Omega_X^p)
$$
在适当修正后与乘法、Todd class 和 Mukai pairing 相容。

**警告 18.5.** HKR 的乘法相容不是裸同构自动给出的，需要 Todd class 修正。HMS 文献中比较 pairings 时必须说明采用的规范化。

**例 18.5A（椭圆曲线的 HKR 分次）.** 设 $E$ 是特征零代数闭域上的
elliptic curve。由
$$
h^0(E,\mathcal O_E)=h^1(E,\mathcal O_E)
=h^0(E,\Omega_E^1)=h^1(E,\Omega_E^1)=1
$$
和定理 18.4，得到
$$
HH_1(\operatorname{Perf}E)\cong k,\qquad
HH_0(\operatorname{Perf}E)\cong k^2,\qquad
HH_{-1}(\operatorname{Perf}E)\cong k.
\tag{18.2}
$$
这里 $HH_1$ 来自 $H^0(\Omega_E^1)$，$HH_0$ 来自
$H^0(\mathcal O_E)\oplus H^1(\Omega_E^1)$，$HH_{-1}$ 来自
$H^1(\mathcal O_E)$。这给第九章椭圆曲线对象字典之外的一个全范畴检查。

**命题 18.6.** 若 $X,Y$ 为 Calabi-Yau mirror pair 且 HMS 成立，则 A-side Fukaya category 的 Hochschild homology 维数与 $X$ 的 Hodge numbers 的相应组合匹配。

**证明.** HMS 给出 $HH_\ast(\mathcal F(Y))\cong HH_\ast(\operatorname{Perf}(X))$。由 HKR 外部输入，右边维数等于 Hodge cohomology 的相应直和维数。证毕。

## 18.3 Closed-open map 与 quantum/symplectic cohomology

**定义 18.7.** closed-open map
$$
\mathcal{CO}:SH^\ast(M)\to HH^\ast(\mathcal W(M))
$$
把 closed-string operations 映到 open-string category 的 Hochschild cochains。compact monotone 情况下 $SH^\ast$ 替换为 $QH^\ast$ 或其幂等分量。

**外部输入定理 18.8（Ganatra 的非退化 wrapped 情形）.** 设 $M$ 是
Liouville manifold，$\mathcal W$ 是一组 exact Lagrangians 生成的 wrapped
Fukaya category，并满足 Ganatra 所规定的 non-degeneracy 条件，即有足够多
Lagrangians 使相应 open-closed 类命中单位。在其 wrapped analytic、duality
和 finiteness 假设下，自然映射
$$
HH_\bullet(\mathcal W)\longrightarrow SH^{\bullet+n}(M),\qquad
SH^\bullet(M)\longrightarrow HH^\bullet(\mathcal W)
\tag{18.3}
$$
是同构，并与 ring/module structures 相容。
来源：Ganatra, *Symplectic cohomology and duality for the wrapped Fukaya
category*, arXiv:1304.7312。本定理不对任意 Weinstein manifold 无条件断言
同构；non-degeneracy 正是不可删除的生成性输入。

**解释 18.9.** B-side 上，$HH^\ast(\operatorname{Perf}(X))$ 与 polyvector fields 相关。镜像对称预期把 A-side quantum/symplectic cohomology 与 B-side polyvector/Hochschild cohomology 匹配。

## 18.4 Categorical enumerative checks

**定义 18.10.** 一个 categorical enumerative check 是从 HMS 等价推出的数值或结构匹配，例如：

- Euler pairing 与 intersection pairing 匹配；
- Serre functor 与 Calabi-Yau dimension 匹配；
- Hochschild homology 与 Hodge diamond 匹配；
- disk potential critical values 与 quantum cohomology eigenvalues 匹配；
- open-closed map 命中单位与生成性匹配。

**命题 18.11.** 若某个候选 HMS 数据包无法通过 Euler pairing 检查，则不存在保持给定对象字典的增强等价。

**证明.** 增强等价保持 morphism complexes 的 quasi-isomorphism type，因此保持其上同调维数和 Euler pairing。若对象字典下 Euler pairing 不匹配，则不存在这样的等价。证毕。

**警告 18.12.** 通过所有已知 categorical checks 不推出 HMS。它们是必要条件，不是充分条件。

Hochschild Morita 不变性把 HMS 转化为可计算的 Hodge 与 closed-string 约束，Euler pairing 和 Serre 函子又能快速排除错误的对象字典；Todd 修正提醒我们，向量空间同构还不等于乘法或 pairing 相容。不过这些条件全都只是等价的后果。真正建立 HMS 仍须构造生成性和增强函子，下一章的奇点模型会展示为何选择正确的范畴本身就是首要问题。

## 练习

**练习 18.1.** 证明 Morita 等价保持 Hochschild homology 的形式原因。

**练习 18.2.** 对椭圆曲线，写出 HKR 右边的各项维数。

**练习 18.3.** 解释 Todd class 修正在 HKR pairing 比较中的作用。

**练习 18.4.** 给出一个候选 HMS 字典，并列出三项 categorical checks。
