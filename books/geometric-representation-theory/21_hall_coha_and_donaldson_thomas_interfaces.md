# 第二十一章：Hall algebras、cohomological Hall algebras 与 Donaldson-Thomas 接口

## 本章目标

本章介绍 Hall algebra 和 cohomological Hall algebra 的几何表示论接口。它们把 quiver 表示、stack 上的卷积、Donaldson-Thomas invariants 和 quantum groups 联系起来。

## 依赖前置知识

需要 quiver representations、stack quotient、Borel-Moore homology 和卷积 correspondence。

## 21.1 Hall algebra 的范畴来源

**定义 21.1.** 令 $\mathcal A$ 为 finitary abelian category，即 $\operatorname{Hom}$ 和 $\operatorname{Ext}^1$ 有限，并且每个对象只有有限多个子对象。Hall algebra 的基由同构类 $[M]$ 给出，乘法为
$$
[M]\ast[N]=\sum_{[E]} g_{M,N}^E [E],
$$
其中 $g_{M,N}^E$ 是短正合列
$$
0\to N\to E\to M\to0
$$
的适当计数。

**命题 21.2.** Hall multiplication 的结合性来自短正合列的二步滤过计数。

**证明.** $([L]\ast[M])\ast[N]$ 计数带有二步滤过
$$
0\subset N\subset E_1\subset E
$$
且 $E_1/N\simeq M$、$E/E_1\simeq L$ 的对象 $E$。$[L]\ast([M]\ast[N])$ 计数同一类二步滤过，只是先记录 $E_1$ 还是先记录 $E/E_0$ 的中间扩张。二者由同一有限 groupoid 的基数给出，因此相等。严格版本需要除以 automorphism groups。$\square$

## 21.2 CoHA 的几何卷积

**定义 21.3.** 对 quiver $Q$，令 $\mathfrak M_v$ 为 dimension vector $v$ 的表示 stack。cohomological Hall algebra 形式上为
$$
\operatorname{CoHA}(Q)=\bigoplus_v H_\ast^{BM}(\mathfrak M_v)
$$
或其 vanishing cycle/mixed Hodge 版本。乘法由短正合列 stack correspondence 定义。

**定义 21.4.** 令 $\mathfrak E_{v,w}$ 为短正合列 stack，其点为
$$
0\to M_w\to M_{v+w}\to M_v\to0.
$$
有 correspondence
$$
\mathfrak M_v\times\mathfrak M_w
\xleftarrow{\ p\ }
\mathfrak E_{v,w}
\xrightarrow{\ q\ }
\mathfrak M_{v+w}.
$$
CoHA 乘法的基本形式为
$$
a\ast b=q_\ast p^\ast(a\boxtimes b)
$$
加上 Euler class、Tate twist、vanishing cycle 或 orientation 修正，具体依模型而定。

**命题 21.5.** 若 $p,q$ 的 pull-push formalism 在所选同调理论中成立，则 CoHA 乘法结合。

**证明.** 三重乘法由二步滤过 stack 控制：
$$
0\subset M_u\subset M_{w+u}\subset M_{v+w+u}.
$$
两种加括号方式分别先记录下层短正合列或上层短正合列，但对应同一个二步滤过 stack。base change 把两次 pull-push 化为沿该 stack 的单次 pull-push，因此得到相同乘积。若存在 vanishing cycle 或 orientation data，需额外使用其对短正合列拼接的相容性。$\square$

**外部输入定理 21.6.** Kontsevich-Soibelman 和后续工作构造了带 potential 的 quiver CoHA，并把它与 Donaldson-Thomas theory、BPS invariants 和 Yangian/quantum group 结构联系起来。

**警告 21.7.** CoHA 的定义高度依赖采用 ordinary cohomology、Borel-Moore homology、critical CoHA、vanishing cycles 还是 mixed Hodge modules。没有 potential 和 orientation data 时，不得直接引用 DT 结论。

## 21.3 与本书主线的关系

**边界说明 21.8.** CoHA 与本书其他对象的接口包括：

1. quiver varieties 的 Steinberg-type correspondences；
2. KLR algebras 和 shuffle algebras；
3. Coulomb branches 和 Yangians；
4. Donaldson-Thomas invariants；
5. cluster varieties 和 wall crossing。

每个接口都是独立研究方向，本章只建立词汇和依赖位置。

## 21.4 最小 Hall 代数计算

**例 21.9.** 令 $\mathcal A$ 为有限域 $\mathbb F_q$ 上有限维向量空间范畴。简单对象为一维空间 $S$。在未 twisted 的 Hall algebra 中，
$$
[S]\ast[S]=(q+1)[\mathbb F_q^2],
$$
因为 $\mathbb F_q^2$ 中一维子空间数为 $q+1$。

**证明.** 短正合列
$$
0\to S\to E\to S\to0
$$
在向量空间范畴中总是分裂，所以 $E\simeq\mathbb F_q^2$。这样的短正合列由 $E$ 中作为 subobject 的一维子空间决定。$\mathbb P^1(\mathbb F_q)$ 有 $q+1$ 个点，因此系数为 $q+1$。若采用除以 automorphism groups 的 groupoid cardinality 归一化，系数会相应改变。$\square$

## 本章小结

本章给出 Hall algebra 和 CoHA 的卷积来源，写出短正合列 stack correspondence、结合性证明和有限域向量空间的最小计算，并说明它们与 quiver、DT 和 quantum groups 的接口。核心 CoHA 定理作为外部输入。

## 练习

**练习 21.1.** 对有限域上一维向量空间范畴，计算最简单的 Hall product。

**练习 21.2.** 写出 quiver 表示短正合列 stack 的 correspondence。

**练习 21.3.** 说明 ordinary CoHA 与 critical CoHA 的输入数据差异。

**练习 21.4.** 在有限域向量空间范畴中计算 $[S]\ast[\mathbb F_q^2]$ 的未 twisted Hall 系数。
