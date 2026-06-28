# 第六章：liquid 函数分析例子

## 本章目标

本章整理 liquid 向量空间的基本例子和误读风险。重点不是重建全部函数分析，而是说明 Banach、Fréchet、分布和 Dolbeault 复形进入凝聚/liquid 语言时，哪些结构必须被保留。

## 6.1 Banach 空间

Banach 空间 $V$ 可先凝聚化：

$$
\underline V(S)=\operatorname{Cont}(S,V),
\qquad S\in\mathbf{CHaus}.
$$

但 liquid 结构不是单纯凝聚化，而是要求对 $\mathcal M_{<p}[S]$ 的 Hom 判别。凝聚化只说明“怎样在紧 Hausdorff 测试对象上取连续族”；liquid 条件进一步控制测度对象和连续线性泛函的同调行为。

**例 6.1.1。** 若 $S$ 是有限离散集合，则

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

是 Banach 空间的可数逆极限表示。对紧 Hausdorff $S$，自然映射

$$
\operatorname{Cont}(S,V)\to
\varprojlim_n\operatorname{Cont}(S,V_n)
$$

通常是同构。

**证明。** 映射 $f:S\to V$ 与坐标映射 $f_n:S\to V_n$ 相容，且每个 $f_n$ 连续。反过来，给定相容连续族 $(f_n)$，由逆极限的集合性质得到唯一映射 $f:S\to V$。逆极限拓扑的定义说明 $f$ 连续当且仅当所有坐标 $f_n$ 连续。证毕。

这个命题解释了 Fréchet 空间与凝聚测试对象相容的基本原因：紧 Hausdorff 参数族可以逐 Banach 层检查。

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

## 6.4 liquid 判别的数学含义

在 liquid 理论中，常见判别形式是比较

$$
\operatorname{Hom}(\mathcal M_{<p}[S],V)
$$

随 profinite $S$ 变化的行为。直观上，$\mathcal M_{<p}[S]$ 是受增长条件约束的测度对象；Hom 判别要求 $V$ 对这些测度测试对象表现良好。

这与 Banach 完备化不同。Banach 完备化修补 Cauchy 列；liquid localization 修补的是一个范畴中相对于指定测试对象和张量行为的同调缺陷。

## 6.5 风险点

1. liquid 不是 Banach 空间范畴。
2. 连续线性映射必须在凝聚/liquid 意义下处理。
3. 张量积需要使用 analytic 或 liquid 张量，而非普通代数张量。
4. 完备性、核性与 exactness 的关系需要在相应范畴中陈述，不能只凭经典函数分析直觉迁移。

## 练习

**练习 6.1.** 说明 Banach completion 与 liquid localization 的区别。

**练习 6.2.** 写出 $p$-liquid 判别式，并指出其中的测试对象。

**练习 6.3.** 解释为什么 Dolbeault 复形需要拓扑向量空间结构。

**练习 6.4.** 设 $V=\varprojlim_nV_n$ 是 Fréchet 空间表示。证明紧 Hausdorff $S$ 上连续映射 $S\to V$ 等价于相容的连续映射族 $S\to V_n$。
