# 第六章：liquid 函数分析例子

## 本章目标

本章整理 liquid 向量空间的基本例子和误读风险。重点不是重建全部函数分析，而是说明
Banach、Fréchet、分布和 Dolbeault 复形进入凝聚/liquid 语言时，哪些结构必须被保留。
本章沿用第一卷固定的 \(\kappa\)-测试层级，并固定 \(0<p\le1\)。

## 6.1 Banach 空间

Banach 空间 $V$ 的候选 liquid 对象就是凝聚化：

$$
\underline V(S)=\operatorname{Cont}(S,V),
\qquad S\in\mathbf{CHaus}_\kappa.
$$

凝聚化是构造，liquid 是该凝聚对象满足的额外性质：对每个
\(S\in\mathbf{ProFin}_\kappa\)，Dirac 映射应诱导同构

$$
\operatorname{Hom}_{\mathbb R}(\mathcal M_{<p}[S],\underline V)
\xrightarrow{\sim}
\operatorname{Hom}_{\mathbb R}(\mathbb R[\underline S],\underline V)
\cong \underline V(S).
$$

这里 \(\mathcal M_{<p}=\bigcup_{0<q<p}\mathcal M_q\)；把它换成单个
\(\mathcal M_p\) 或普通 Radon measures 会改变命题。

**外部输入定理 6.1.1（经典完备空间的 membership）.** 每个实 Banach 空间的
\(\underline V\) 对所有 \(0<p\le1\) 都是 \(p\)-liquid。更一般地，\(p\)-Banach
空间以及 complete locally \(p\)-convex 空间的凝聚化都是 \(p\)-liquid。

**来源与边界.** 这是第二卷输入 D.6，即 CS26 Theorem 2.14、Lemma 2.16 及其后的
inverse-limit 推论。该定理不构造另一个 \(V_{\mathrm{liq}}\)，也不声称每个 Banach
空间之间的连续满射在凝聚化后都是 epimorphism。

**例 6.1.2。** 若 $S$ 是有限离散集合，则

$$
\underline V(S)\cong V^S.
$$

若 $S$ 是无限 profinite 集合，则 $\underline V(S)$ 是连续映射空间，不是所有集合映射 $S\to V$。

**证明。** 有限离散空间上的每个映射都连续，所以得到 $V^S$。对一般 profinite $S$，凝聚化按定义取连续映射；连续性要求 $S$ 的紧全不连通拓扑与 $V$ 的范数拓扑相容。例如映射必须把足够小的 clopen 分块送入 $V$ 的小邻域，远强于集合映射条件。证毕。

## 6.2 Fréchet 空间

全纯函数空间 $\mathcal O(U)$ 常为 Fréchet 空间。第三卷中，Dolbeault 复形的项应放入 liquid 范畴，以保留拓扑和连续性。

设

$$
V=\varprojlim_nV_n
$$

是 Banach 空间的可数逆极限表示，并赋予逆极限拓扑。对紧 Hausdorff $S$，自然映射

$$
\operatorname{Cont}(S,V)\to
\varprojlim_n\operatorname{Cont}(S,V_n)
$$

是同构。

**证明。** 映射 $f:S\to V$ 与坐标映射 $f_n:S\to V_n$ 相容，且每个 $f_n$ 连续。反过来，给定相容连续族 $(f_n)$，由逆极限的集合性质得到唯一映射 $f:S\to V$。逆极限拓扑的定义说明 $f$ 连续当且仅当所有坐标 $f_n$ 连续。证毕。

这个命题解释了 Fréchet 空间与凝聚测试对象相容的基本原因：紧 Hausdorff 参数族可以逐 Banach 层检查。

**推论 6.2.1.** 每个实 Fréchet 空间 \(V\) 的凝聚化对所有
\(0<p\le1\) 都是 \(p\)-liquid。

**证明.** 取定义拓扑的递增可数半范数族，把相应商空间完备化为 Banach 空间
\(V_n\)。Fréchet 完备性与 Hausdorff 性给 \(V\cong\varprojlim_nV_n\)。上面的逐测试
对象同构给

$$
\underline V\cong\varprojlim_n\underline{V_n}.
$$

输入定理 6.1.1 说明每个 \(\underline{V_n}\) 是 \(p\)-liquid，而 CS26 Theorem 3.11
说明 liquid 对象对逆极限封闭，故 \(\underline V\) 也是 \(p\)-liquid。证毕。

## 6.3 分布空间

分布空间通常是某种对偶空间，适合用 liquid 或解析结构处理。关键不是选择一个范数，而是控制测试对象上的测度和连续线性泛函。

例如复流形 $X$ 上的 Dolbeault 复形

$$
0\to\mathcal A_X^{p,0}\xrightarrow{\bar\partial}
\mathcal A_X^{p,1}\xrightarrow{\bar\partial}\cdots
$$

不是单纯的代数复形：每一项带有自然 locally convex topology，$\bar\partial$ 是连续线性算子。

**命题 6.3.1（凝聚化保持复形结构）。** 若 $V^\bullet$ 是拓扑向量空间复形，且每个微分 $d^n:V^n\to V^{n+1}$ 连续，则

$$
\underline{V^\bullet}(S)=\operatorname{Cont}(S,V^\bullet)
$$

定义凝聚向量空间复形。

**证明。** 对每个紧 Hausdorff $S$，连续映射的复合仍连续，所以 $d^n$ 诱导

$$
\operatorname{Cont}(S,V^n)\to
\operatorname{Cont}(S,V^{n+1}).
$$

因为 $d^{n+1}\circ d^n=0$ 在 $V^\bullet$ 中成立，逐点复合后仍为零。对 $S$ 的反变函子性来自连续映射的预合成。证毕。

## 6.4 Exactness 与 cohomology 的边界

逐项 membership 不保证凝聚化保持 cokernel。设

$$
0\longrightarrow V'\longrightarrow V\xrightarrow{q}V''\longrightarrow0
$$

是底层向量空间正合的连续线性映射列，并假设 \(V'\) 带有 \(\ker q\) 的子空间拓扑。
则左端 kernel 在凝聚化后仍正确；右端 cokernel 正确当且仅当：对每个
\(S\in\mathbf{ProFin}_\kappa\) 和连续 \(f:S\to V''\)，存在有限联合满射覆盖
\(S_i\to S\)，使 \(f|_{S_i}\) 有连续提升到 \(V\)。这是 sheaf epimorphism 的定义，
不是由“\(V,V''\) 都 liquid”推出的性质。

**充分条件 6.4.1.** 若 \(q\) 有连续截面（不要求线性），则上述局部提升条件成立，
所以凝聚短正合列正合。

**证明.** 对任意 \(f:S\to V''\)，截面 \(s\) 给出全局连续提升 \(s\circ f\)。证毕。

应用到 Fréchet 复形 \(V^\bullet\) 时，要比较

$$
H^q(\underline{V^\bullet})
\quad\text{与}\quad
\underline{\ker d^q/\operatorname{im}d^{q-1}},
$$

必须对 \(V^{q-1}\twoheadrightarrow\operatorname{im}d^{q-1}\) 和
\(\ker d^q\twoheadrightarrow H^q_{\mathrm{top}}(V^\bullet)\) 检查上述局部提升。
闭值域只保证 quotient 是 Hausdorff Fréchet 空间；连续 Hodge/Green splitting 才给出
本书使用的充分条件。

## 6.5 Liquid 判别与 completion

\(\mathcal M_{<p}[S]\) 的 Hom 判别控制对象对测度参数族的响应。这与 Banach 完备化
不同：Banach 完备化修补 Cauchy 列；liquidification 是 analytic ring 所定义的反射
局部化；而 \(V\mapsto\underline V\) 只是经典空间的凝聚化。三者类型不同。

## 6.6 风险点

1. liquid 不是 Banach 空间范畴。
2. Banach/Fréchet 的凝聚化确实 liquid，但连续满射的 exactness 仍需局部提升。
3. 张量积需要使用 analytic 或 liquid 张量，而非普通代数张量。
4. 完备性、核性与 exactness 的关系需要在相应范畴中陈述，不能只凭经典函数分析直觉迁移。

## 练习

**练习 6.1.** 说明 Banach completion 与 liquid localization 的区别。

**练习 6.2.** 写出 $p$-liquid 判别式，并指出其中的测试对象。

**练习 6.3.** 解释为什么 Dolbeault 复形需要拓扑向量空间结构。

**练习 6.4.** 设 $V=\varprojlim_nV_n$ 是 Fréchet 空间表示。证明紧 Hausdorff $S$ 上连续映射 $S\to V$ 等价于相容的连续映射族 $S\to V_n$。

**练习 6.5.** 证明有连续截面的满射在凝聚化后是 epimorphism，并指出证明中为何不需要
截面线性。
