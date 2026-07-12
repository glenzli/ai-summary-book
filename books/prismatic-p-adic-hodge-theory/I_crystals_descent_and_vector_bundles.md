# 附录 I：Crystals、descent 与 vector bundles

## 本附录目标

本附录补齐第六、十三章使用的 crystals 和 vector bundles 语言。重点是区分 sheaf condition、descent condition、crystal rigidity 和 finite locally free condition。

## I.1 Sheaves 与 crystals 的差异

**定义 I.1.** 令 $(\mathcal C,\mathcal O)$ 为 ringed site。一个 $\mathcal O$-module sheaf $\mathcal F$ 是 sheaf condition 下的线性对象。它只要求对覆盖满足 descent。

**定义 I.2.** 若 $\mathcal C$ 的态射表示某种 thickening direction，则 crystal 是 $\mathcal O$-module sheaf $\mathcal E$，满足对指定态射 $u:T'\to T$，canonical pullback map
$$
u^\ast\mathcal E(T)\to\mathcal E(T')
$$
为同构。

**警告 I.3.** Sheaf condition 是覆盖方向的 glueing；crystal condition 是 thickening 方向的 rigidity。二者逻辑独立，不能互相替代。

## I.2 Prismatic crystals

**定义 I.4.** 在 prismatic site $(X/A)_\Delta$ 上，一个 prismatic crystal 是 $\mathcal O_\Delta$-module sheaf $\mathcal E$，使得对每个 morphism of prism probes
$$
(B,IB)\to(B',IB')
$$
都有 base-change isomorphism
$$
\mathcal E(B,IB)\otimes_BB'\xrightarrow{\sim}\mathcal E(B',IB').
$$
这里使用引理 2.4A：在固定 base prism $(A,I)$ 的 relative site 上，probe
ideal 不是独立的 $J$，而必为 $IB$。Absolute site 的写法见定义 6.1。

**命题 I.5.** 若 $\mathcal E$ 是 relative prismatic crystal，且
$\mathcal E(B,IB)$ 在某个覆盖上 finite projective，则在该覆盖的 pullback
probes 上仍为 finite projective。

**证明.** Crystal condition 给出
$$
\mathcal E(B',IB')\cong\mathcal E(B,IB)\otimes_BB'.
$$
Finite projective modules 在任意 base change 下保持 finite projective。证毕。

## I.3 Descent for finite projective modules

**外部输入定理 I.6（faithfully flat descent）.** 若 $A\to B$ faithfully flat，则 finite projective $A$-modules 范畴等价于 finite projective $B$-modules 加 descent datum 的范畴。

**说明 I.7.** Prismatic site 的 covers 使用 completed faithfully flat maps。严格版本需要 completed faithfully flat descent；本书将其作为外部输入技术使用。

**命题 I.8（descent datum 的 cocycle 条件）.** 对 $A\to B$，一个 descent datum 是 $B$-module $M_B$ 上的同构
$$
\alpha:p_1^\ast M_B\xrightarrow{\sim}p_2^\ast M_B
$$
over $B\otimes_AB$，满足 over $B\otimes_AB\otimes_AB$ 的 cocycle 条件
$$
p_{23}^\ast\alpha\circ p_{12}^\ast\alpha=p_{13}^\ast\alpha.
$$

**证明.** 这是 descent datum 的定义展开。Cocycle 条件保证从三重交叠 glueing 时结果与路径无关。证毕。

## I.4 $F$-crystals

**定义 I.9.** 令 $\mathcal I_\Delta$ 为 absolute prismatic site 上的 prism
ideal sheaf。一个 vector-bundle-valued prismatic $F$-crystal 是 finite
locally free prismatic crystal $\mathcal E$，配有同构
$$
\varphi_{\mathcal E}^{\mathrm{lin}}:
\phi^*\mathcal E[1/\mathcal I_\Delta]
\xrightarrow{\sim}
\mathcal E[1/\mathcal I_\Delta].
$$
它等价于 localized $\phi$-semilinear Frobenius。若该同构由积分映射
$\phi^*\mathcal E\to\mathcal E$ 诱导，则称 $\mathcal E$ effective。此处
$\mathcal I_\Delta(B,J)=J$ 随 absolute probe 变化；relative site 上才可在
probe $(B,IB)$ 写成 $IB$。这与定义 6.4 及 Bhatt--Scholze, Definition 4.1
（locator `BS-FCRYS`）一致。

**警告 I.10.** 对 $F$-crystal 的任何例子，都必须检查 Frobenius
linearization 的 source/target 和被反演的 probe ideal。Localized
isomorphism 不推出 integral isomorphism，effectivity 也不是一般
$F$-crystal 自动具有的性质；只写“带 Frobenius”不够。

## I.5 Rational crystals and boundary objects

**定义框架 I.11.** Rational Hodge-Tate prismatic crystal 通常是在 $\overline{\mathcal O}_\Delta$ 或 Hodge-Tate specialization 后 rationalized 的 crystal-like 对象。它不应与 integral prismatic $F$-crystal 混用。

**命题 I.12.** 若一个对象只在 rationalized category 中定义，则不能无条件推出 integral lattice。

**证明.** Rationalization 是 localization，会杀掉 torsion 并遗忘 lattice 选择。不同 integral lattices 可有相同 rationalization。因此 rational object 不唯一决定 integral lattice。证毕。

## 本附录小结

Crystals 是带 rigidity 的 sheaves；vector bundle crystals 还要求 finite locally free；$F$-crystals 再加入 Frobenius isogeny。带系数和非阿贝尔边界必须先说明使用的是哪一种 crystal。

## 练习

**练习 I.1.** 写出 sheaf condition 和 crystal condition 的对象方向差异。

**练习 I.2.** 证明 finite projective modules 在 base change 下保持 finite projective。

**练习 I.3.** 给出两个不同 lattice 有同一 rationalization 的例子。
