# 第十章：Harish-Chandra bimodules、primitive ideals 与 characteristic cycles

一个 simple $U(\mathfrak g)$-模可以先由 annihilator 变成双边理想，再由 PBW filtration 变成 $\mathfrak g^*$ 中的代数簇；经 localization 后，同一对象又在 $T^*\mathcal B$ 中留下 characteristic cycle。Harish--Chandra bimodule 是连接左右 $U(\mathfrak g)$-作用的代数载体，primitive ideal 记录不可约表示看不见的算子，associated variety 与 moment map 则把这些非交换数据送入 nilpotent cone。$\mathfrak{sl}_2$ 的 Casimir 只给出中心 character，$\mathcal O_X$ 的零截面 characteristic cycle 则展示 multiplicity 如何从 good filtration 读出；一般 primitive-orbit 对应仍属于 Joseph 理论的外部输入。

## 10.1 Harish-Chandra bimodules

**定义 10.1.** 一个 $U(\mathfrak g)$-bimodule $M$ 称为 Harish-Chandra bimodule，若：

1. $M$ 作为 left 和 right $U(\mathfrak g)$-module 均有限生成；
2. adjoint action
$$
x\cdot m=xm-mx
$$
对每个 $m\in M$ 生成有限维 $\mathfrak g$-submodule；
3. $Z(\mathfrak g)$ 的左右作用满足适当 locally finite 或中心 character 条件。

**例 10.2.** $U(\mathfrak g)$ 本身在左右乘法下是 bimodule。其 adjoint action 局部有限性可由 PBW filtration 和 $\mathfrak g$ 在 tensor powers 上的有限维性推出。

**命题 10.3.** 若 $V$ 是有限维 $\mathfrak g$-module，则 $U(\mathfrak g)\otimes V$ 带自然 Harish-Chandra bimodule 结构。右作用在 $U(\mathfrak g)$ 因子上；左作用由 coproduct 给出，在 Lie algebra 生成元上为
$$
x\cdot(u\otimes v)=xu\otimes v+u\otimes xv,\qquad
(u\otimes v)\cdot x=ux\otimes v.
$$
于是 commutator adjoint action 同时看见 $U(\mathfrak g)$ 的 adjoint action 和 $V$ 的给定作用。

**证明.** 左右有限生成来自 $U(\mathfrak g)$ 因子。对 $u\otimes v$，adjoint action 落在由 PBW degree 不超过 $\deg u$ 的有限维 $\mathfrak g$-submodules 与有限维 $V$ 张成的空间中；这是因为 $\mathfrak g$ 对 $U(\mathfrak g)$ 的 adjoint action 保持 PBW filtration，且每个 filtered piece 是有限维 $\mathfrak g$-module 的商。中心条件在自由 bimodule 情形中由 $U(\mathfrak g)$ 的中心作用给出。$\square$

## 10.2 Primitive ideals

**定义 10.4.** $U(\mathfrak g)$ 的 primitive ideal 是某个 simple $U(\mathfrak g)$-module $L$ 的 annihilator：
$$
\operatorname{Ann}_{U(\mathfrak g)}(L)=\{u\in U(\mathfrak g)\mid uL=0\}.
$$

**命题 10.4.1.** Primitive ideal 是 two-sided ideal。

**证明.** 若 $u\in\operatorname{Ann}(L)$，则对任意 $a,b\in U(\mathfrak g)$ 和 $\ell\in L$，
$$
(aub)\ell=a\,u\,(b\ell)=a\cdot0=0.
$$
因此 $aub\in\operatorname{Ann}(L)$，annihilator 对左右乘法封闭。$\square$

**外部输入定理 10.5.** Primitive ideals 与 highest weight modules、Weyl group combinatorics、nilpotent orbit closures 和 associated varieties 有深层关系。Joseph theory 给出 primitive ideals 的一系列结构定理。  
来源：Dixmier、Joseph、Borho-Brylinski、Borho-MacPherson。

**定义 10.6.** 对 finitely generated $U(\mathfrak g)$-module $M$，取 good filtration，定义 associated variety
$$
\operatorname{AV}(M)\subset\mathfrak g^\ast
$$
为 $\operatorname{gr}M$ 作为 $S(\mathfrak g)$-module 的 support。

**外部输入定理 10.7.** 若 $M$ 有中心 character 且属于适当有限生成范畴，则 $\operatorname{AV}(M)$ 包含于 nilpotent cone。对 primitive quotient，associated variety 与 nilpotent orbit closure 密切相关。

Associated variety 只记支撑而不记各不可约分支的重数。D-module 一侧的 characteristic cycle 保留这些重数，因此是比较 localization 前后微局部信息的更细对象。

**例 10.7.1.** 对 $\mathfrak{sl}_2$ 的有限维 simple module $L(n)$，Casimir element 在 $L(n)$ 上以标量作用。因而
$$
C-c_n\in\operatorname{Ann}_{U(\mathfrak{sl}_2)}(L(n))
$$
对某个标量 $c_n$ 成立。这给出 $L(n)$ 的中心 character。完整 annihilator 还包含使最高权表示有限维的额外关系。

## 10.3 Characteristic cycles

**定义 10.8.** 若 $\mathcal M$ 是 holonomic $\mathcal D_X$-module，则其 characteristic cycle 是
$$
\operatorname{CC}(\mathcal M)=\sum_i m_i[\Lambda_i],
$$
其中 $\Lambda_i$ 是 $\operatorname{Char}(\mathcal M)$ 的 irreducible Lagrangian components，$m_i$ 是由 good filtration 定义的 multiplicities。

**外部输入定理 10.9.** 在 Riemann-Hilbert correspondence 下，regular holonomic D-modules 的 characteristic cycles 与 constructible sheaves 的 singular support 和 characteristic cycles 相容。  
来源：Kashiwara-Schapira。

**命题 10.10.** 若 $\mathcal M=\mathcal O_X$，则
$$
\operatorname{CC}(\mathcal O_X)=[T_X^\ast X].
$$

**证明.** 第七章例 7.10 已计算 characteristic variety 为 zero section。对应 good filtration 的 associated graded module 是 zero section 上的 structure sheaf，generic rank 为 $1$，因此 cycle multiplicity 为 $1$。$\square$

## 10.4 与 localization 的关系

**外部输入定理 10.11.** Beilinson-Bernstein localization 把 primitive ideals、highest weight modules 和 flag variety 上的 holonomic D-modules 联系起来；特征 variety 经 moment map 投影到 nilpotent cone 后给出 associated variety 的几何解释。

**几何机制 10.12.** 该关系需要同时使用：

1. localization equivalence；
2. $\mathcal D_\lambda$-module 的 characteristic variety；
3. moment map $T^\ast\mathcal B\to\mathcal N$；
4. good filtration 与 associated graded 的相容性。

从 primitive annihilator 到 associated variety 的过程把非交换理想压到 $\mathfrak g^*$ 中，而 characteristic cycle 进一步在 cotangent bundle 中保留 Lagrangian 分支及其重数。Localization 与 moment map 使两条路径相遇于 nilpotent cone，但一般的 orbit-closure 结论需要 Joseph 与 microlocal 理论。下一章给出另一种压缩几何的方法：用多项式环上的双模纯代数地重建 Hecke category。

## 练习

**练习 10.1.** 证明 primitive ideal 是 two-sided ideal。

**练习 10.2.** 对 $\mathfrak{sl}_2$ 的有限维 simple module，计算其 annihilator 的中心 character。

**练习 10.3.** 说明 associated variety 的定义为什么需要 good filtration 独立性定理。
