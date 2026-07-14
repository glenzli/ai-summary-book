# 第二十一章：Hall algebras、cohomological Hall algebras 与 Donaldson-Thomas 接口

把两个表示相乘，可以不在张量积中进行，而去计数所有以一个为子对象、另一个为商的扩张。Hall algebra 用有限域上的点数实现这一想法；CoHA 则把点数替换为表示 stack 上短正合列 correspondence 的同调 pull--push。结合律的共同来源不是形式符号，而是同一个对象的二步滤过。零箭头 quiver 已包含完整的最低阶计算：所有扩张分裂，但中间子空间的选择给出 Gaussian binomial coefficient。先算出 $[V_a]*[V_b]$，再把同一旗标计数提升到 Borel--Moore homology，便能清楚区分 ordinary Hall、CoHA 与还需 potential、vanishing cycles 和 orientation data 的 Donaldson--Thomas 版本。

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

这些联系都需要额外的比较定理。短正合列 correspondence 的存在与结合性本身既不产生 potential，也不自动给出 Yangian、KLR 或 Coulomb-branch action。

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

**命题 21.9.1（Gaussian Hall 乘法）.** 仍取未 twisted、按 subobjects 计数的 normalization，并记 $V_a=\mathbb F_q^a$。则
$$
[V_a]*[V_b]
=\binom{a+b}{b}_q[V_{a+b}],
$$
其中
$$
\binom{a+b}{b}_q
=\prod_{r=0}^{b-1}
\frac{q^{a+b}-q^r}{q^b-q^r}
$$
是 Gaussian binomial coefficient。

**证明.** 向量空间范畴中的短正合列全部分裂，所以中间项只能是 $V_{a+b}$。给定中间项后，一条
$$
0\to V_b\to V_{a+b}\to V_a\to0
$$
由其像 $U\subset V_{a+b}$ 决定，其中 $\dim U=b$。有序线性无关 $b$-元组的数目为
$$
(q^{a+b}-1)(q^{a+b}-q)\cdots(q^{a+b}-q^{b-1}),
$$
而同一个 $b$-维子空间含有
$$
(q^b-1)(q^b-q)\cdots(q^b-q^{b-1})
$$
个有序基。两者相除即得公式。$\square$

**推论 21.9.2（结合律的数值影子）.** 对 $a,b,c\ge0$，
$$
\binom{a+b}{b}_q\binom{a+b+c}{c}_q
=\binom{b+c}{c}_q\binom{a+b+c}{b+c}_q.
$$

**证明.** 两侧都计数 flags
$$
0\subset U_c\subset U_{b+c}\subset V_{a+b+c},
\qquad
\dim U_c=c,\qquad \dim U_{b+c}=b+c.
$$
先选 $U_c$ 时，剩余选择是在商 $V_{a+b+c}/U_c$ 中选 $b$-维子空间；先选 $U_{b+c}$ 时，再在其中选 $c$-维子空间。两种次序分别给出等式两边。$\square$

例 21.9 是 $a=b=1$ 的特例，而
$$
[S]*[V_2]=(q^2+q+1)[V_3]
$$
对应 $\mathbb P^2(\mathbb F_q)$ 的点数。CoHA 将这种有限点计数替换为同一 subspace/extension correspondence 的 fundamental classes、Euler classes 与 cohomological degrees。

Gaussian coefficient 显示，即使所有扩张都分裂，Hall 乘法仍会记住子对象的几何；结合恒等式正是二步旗标的两种计数。CoHA 沿同一短正合列 stack 作 pull--push，但 potential、vanishing cycles 与 orientation data 会改变所用同调理论，不能从 ordinary Hall 公式省略。下一章把 quiver variety、KLR projectives 与 CoHA 等模型重新放到 canonical basis 与 crystal 的同一问题下比较。

## 练习

**练习 21.1.** 对有限域上一维向量空间范畴，计算最简单的 Hall product。

**练习 21.2.** 写出 quiver 表示短正合列 stack 的 correspondence。

**练习 21.3.** 说明 ordinary CoHA 与 critical CoHA 的输入数据差异。

**练习 21.4.** 在有限域向量空间范畴中计算 $[S]\ast[\mathbb F_q^2]$ 的未 twisted Hall 系数。

**练习 21.5.** 用先选 $U_c$ 再选 $U_{b+c}/U_c$ 的方法重新证明推论 21.9.2，并解释两个 Gaussian factors 各自参数化什么。
