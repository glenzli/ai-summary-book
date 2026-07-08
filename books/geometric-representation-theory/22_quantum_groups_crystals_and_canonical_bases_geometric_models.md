# 第二十二章：Quantum groups、crystals 与 canonical bases 的几何模型

## 本章目标

本章整理 quantum groups、canonical bases、crystals 和几何模型之间的关系。它把第十七章 quiver varieties、第十八章 KLR categorification 和第二十一章 CoHA 接到统一的 canonical basis 主题上。

## 依赖前置知识

需要 Kac-Moody algebra、KLR algebras、quiver varieties 和基本表示论。

## 22.1 Quantum group 和 integral form

**定义 22.1.** 给定 symmetrizable Cartan datum，quantum group $U_q(\mathfrak g)$ 是由生成元
$$
E_i,F_i,K_i^{\pm1}
$$
和 quantum Serre relations 定义的 $\mathbb Q(q)$-代数。其负半部记为 $U_q^-(\mathfrak g)$。

**定义 22.2.** Lusztig integral form 是 $U_q(\mathfrak g)$ 中由 divided powers
$$
E_i^{(n)}=\frac{E_i^n}{[n]_{q_i}!},\qquad
F_i^{(n)}=\frac{F_i^n}{[n]_{q_i}!}
$$
以及 Cartan integral part（$K_i^{\pm1}$ 和 quantum binomial coefficients）生成的 $\mathbb Z[q,q^{-1}]$-子代数。若只讨论正半部或负半部，则相应 integral form 只需对应的 divided powers。

**外部输入定理 22.3.** Lusztig canonical basis 和 Kashiwara global crystal basis 给出 $U_q^-(\mathfrak g)$ 及 integrable highest weight modules 的特殊基，并满足 positivity、bar-invariance 和 crystal limit 性质。

## 22.2 Crystals

**定义 22.4.** crystal 是集合 $B$ 连同 maps
$$
e_i,f_i:B\to B\sqcup\{0\},\qquad
\operatorname{wt}:B\to P,
$$
以及 $\varepsilon_i,\varphi_i$ 等函数，满足 Kashiwara axioms。它记录 $q\to0$ 极限下的表示组合结构。

**命题 22.5.** 若一个 integrable representation 有 crystal basis，则每个 $f_i$ 沿 $i$-string 给出有限链。

**证明.** integrability 意味着每个 $\mathfrak{sl}_2$-方向局部有限，故对任意向量反复施加 lowering operator 只能产生有限长 string。crystal basis 是该结构的 $q\to0$ 记录，因此 $f_i$ 迭代在有限步后到达 $0$。$\square$

## 22.3 几何模型

**例 22.6.** 对 $\mathfrak{sl}_2$，最高权为 $n$ 的 irreducible representation 的 crystal 为
$$
B(n)=\{b_0,b_1,\ldots,b_n\},
$$
其中
$$
\operatorname{wt}(b_r)=n-2r,\qquad
f(b_r)=b_{r+1}\ (r<n),\qquad f(b_n)=0,
$$
$$
e(b_r)=b_{r-1}\ (r>0),\qquad e(b_0)=0.
$$
这是长度为 $n+1$ 的一条 string。

**命题 22.7.** $B(n)$ 满足 $\mathfrak{sl}_2$ crystal 的基本 string 条件。

**证明.** 对每个 $b_r$，可向左施加 $e$ 共 $r$ 次，向右施加 $f$ 共 $n-r$ 次。因此
$$
\varepsilon(b_r)=r,\qquad \varphi(b_r)=n-r.
$$
权满足
$$
\varphi(b_r)-\varepsilon(b_r)=n-2r=\langle h,\operatorname{wt}(b_r)\rangle.
$$
同时 $e$ 与 $f$ 在非端点处互逆，端点送到 $0$。这正是 $\mathfrak{sl}_2$ crystal axioms 的内容。$\square$

**外部输入定理 22.8.** Lusztig 用 quiver varieties 和 perverse sheaves 构造 canonical bases；Nakajima quiver varieties 给出 highest weight representations 的几何模型；KLR algebras 的 indecomposable projectives 在合适情形中对应 canonical basis。

**边界说明 22.9.** “canonical basis 的几何实现”不是单一陈述。必须说明是：

1. Lusztig perverse sheaf model；
2. Nakajima quiver variety homology model；
3. KLR projective module model；
4. cluster/DT/CoHA model；
5. Satake/MV cycle model。

**表 22.10.** 几何模型与基的对应。

| 模型 | 几何/代数对象 | 基向量候选 | 外部输入 |
| --- | --- | --- | --- |
| Lusztig perverse sheaves | quiver representation varieties | simple perverse sheaves | Lusztig canonical basis |
| Nakajima quiver varieties | Lagrangian varieties $\mathfrak L(v,w)$ | irreducible components / homology classes | Nakajima theorem |
| KLR algebras | projective module categories | indecomposable projectives | KLR/Rouquier theorem |
| MV cycles | affine Grassmannian | MV cycle classes | geometric Satake |
| CoHA | critical cohomology of stacks | BPS classes | KS/DT theory |

该表用于定位，不表示这些基在没有额外定理时自动相同。

## 本章小结

本章固定 quantum groups、crystals 和 canonical bases 的基本口径，补充 $\mathfrak{sl}_2$ crystal 的完整计算，并把多个几何模型统一放入外部输入框架。

## 练习

**练习 22.1.** 对 $\mathfrak{sl}_2$ 写出 $U_q(\mathfrak{sl}_2)$ 的生成元和关系。

**练习 22.2.** 画出最高权为 $n$ 的 $\mathfrak{sl}_2$-module 的 crystal graph。

**练习 22.3.** 比较 KLR indecomposable projectives 和 simple modules 分别对应哪一种 dual canonical basis convention。

**练习 22.4.** 对 $B(2)$ 画出 crystal graph，并验证 $\varepsilon,\varphi$ 与 weight 的关系。
