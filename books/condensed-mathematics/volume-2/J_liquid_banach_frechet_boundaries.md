# 附录 J：Liquid、Banach 与 Fréchet 的边界

## J.0 目标

第五章说明 liquid 向量空间不是 Banach 空间的改名。本附录把这个边界写成更可检查的形式：

1. Banach/Fréchet 空间如何给出凝聚向量空间。
2. 为什么 Banach 范畴不适合作为派生同调代数的主场。
3. 哪些经典完备空间确实满足 liquid 条件。
4. 在复几何中使用 liquid 时必须检查哪些类型。

本附录不构造 $\mathcal M_{<p}$；它精确引用“\(p\)-Banach 因而 Fréchet 对象
\(p\)-liquid”的输入定理，并把 object membership 与 exactness 分开。

## J.1 拓扑向量空间的凝聚化

设 $V$ 是 Hausdorff 拓扑实向量空间。定义凝聚集合

$$
\underline V(S)=\operatorname{Cont}(S,V),
\qquad S\in\mathbf{CHaus}_\kappa.
$$

若加法和数乘连续，则 $\underline V$ 是通常拓扑实数凝聚环
\(\underline{\mathbb R}\) 上的凝聚模。

**命题 J.1.** 若 $V$ 是拓扑实向量空间，则 $S\mapsto\operatorname{Cont}(S,V)$ 满足有限覆盖的 sheaf 条件。

**证明.** 对紧 Hausdorff 空间的有限闭满射覆盖，连续映射到 Hausdorff 空间可由覆盖上的连续映射粘合，并且唯一性由覆盖满射给出。加法和数乘逐点定义，并由 $V$ 中运算连续性保证仍为连续映射。证毕。

**边界 J.2.** 命题 J.1 本身只给凝聚对象；一般拓扑向量空间是否 liquid 仍需
\(\mathcal M_{<p}[S]\) 的 Hom 判别。对 Banach/Fréchet 空间，这个缺口由下面的输入
定理 J.7 闭合；不能把该特例推广成“任意拓扑向量空间都 liquid”。

## J.2 Banach 范畴的同调问题

令 $\mathbf{Ban}$ 为 Banach 空间和连续线性映射的范畴。

**事实 J.3.** $\mathbf{Ban}$ 不是普通意义下的阿贝尔范畴。

**说明.** 连续线性映射的像可能非闭。若 $f:V\to W$ 的像非闭，则 cokernel 使用 $W/\overline{\operatorname{im} f}$ 才是 Hausdorff Banach 空间，而代数 coimage 与 image 的拓扑不一致。因而 image/coimage 比较不能满足阿贝尔范畴公理。

**例 J.4（非闭像风险）.** 存在紧算子 $T:\ell^2\to\ell^2$，例如

$$
T(x_1,x_2,\ldots)=\left(x_1,\frac{x_2}{2},\frac{x_3}{3},\ldots\right),
$$

其像不是闭子空间。

**证明.** 向量

$$
y=(1,1/2,1/3,\ldots)
$$

属于 $\ell^2$。若 $y=T(x)$，则 $x=(1,1,1,\ldots)$，不属于 $\ell^2$，故 $y$ 不在像中。另一方面，截断向量

$$
y^{(N)}=(1,1/2,\ldots,1/N,0,0,\ldots)
$$

属于像，并且 $y^{(N)}\to y$ 于 $\ell^2$。故像非闭。证毕。

**意义 J.5.** 若直接在 Banach 范畴中做 cokernel、Ext、Tor，会遇到 exactness 与拓扑闭包纠缠的问题。liquid/analytic 范畴的目的之一，是把这些函数空间放进适合导出范畴的环境。

## J.3 Fréchet 空间与 Dolbeault 复形

紧复流形 $X$ 上的光滑形式空间

$$
\Gamma(X,\mathcal A^{p,q})
$$

是 Fréchet 空间。Dolbeault 微分

$$
\bar\partial:\Gamma(X,\mathcal A^{p,q})
\to
\Gamma(X,\mathcal A^{p,q+1})
$$

是连续线性映射。

**命题 J.6.** 忘记拓扑后，Dolbeault 复形仍是复向量空间复形；但 Fredholm 性、闭像、核的有限维性和对偶配对的连续性都不是纯代数向量空间结构能表达的性质。

**证明.** 复向量空间结构只记录加法、数乘和线性微分。Fredholm 性要求 kernel、cokernel 有限维且像闭；闭像和连续对偶依赖拓扑。若忘记拓扑，无法区分同一个代数向量空间上的不同 Fréchet 拓扑，也无法表述算子的连续性。证毕。

## J.4 Liquid 类型检查

在本书中，声称某个函数空间进入 liquid 框架时，至少应写明：

1. 使用哪个 analytic ring，例如 $(\mathbb R,\mathcal M_{<p})$。
2. 该函数空间作为凝聚 $\mathbb R$-模的构造。
3. 微分算子是否是该 analytic 模范畴中的态射。
4. 张量积、Hom 和 dual 是否在 liquid/analytic 范畴中计算。
5. 忘记 liquid 结构后是否恢复经典 Fréchet 或 Banach 空间上的算子。

**外部输入定理 J.7（Banach/liquid 接口）.** 固定 \(0<p\le1\)。每个
\(p\)-Banach 空间 \(V\) 的凝聚化 \(\underline V\) 都 \(p\)-liquid；任意此类对象的
逆极限仍 \(p\)-liquid。来源为 CS26 Theorem 2.14、Lemma 2.16 及其后的逆极限推论。

**命题 J.8（Fréchet/liquid 接口）.** 每个实 Fréchet 空间的凝聚化对所有
\(0<p\le1\) 都 \(p\)-liquid；每个连续线性映射给出
\(\mathbf{Liquid}_p\) 中的态射。

**证明.** Fréchet 空间是其 Banach seminorm completions 的可数逆极限。普通 Banach
空间通过范数的 \(p\) 次方成为 \(p\)-Banach，故由 J.7 及逆极限稳定性得到第一项。
连续线性映射与测试空间预合成，逐 \(S\) 给出凝聚模态射；因
\(\mathbf{Liquid}_p\subset\mathbf{CondAb}\) 是满子范畴，它也是 liquid 态射。证毕。

**边界 J.9（对象属于不等于序列正合）.** J.8 只判断各项对象。若
\(E\twoheadrightarrow F\) 是 Fréchet 空间满射，要让
\(\underline E\to\underline F\) 成为 sheaf epimorphism，仍须验证第五章定义 5.8 的
profinite 局部提升；闭像只保证 quotient Hausdorff。连续 splitting 是充分条件。

## J.5 练习

**练习 J.1.** 证明紧 Hausdorff 空间上连续函数的局部粘合仍连续。

**练习 J.2.** 验证例 J.4 中 $T$ 是紧算子。

**练习 J.3.** 说明闭像条件为什么不能在纯代数向量空间范畴中表达。

**练习 J.4.** 对 Dolbeault 复形的一项，列出把它放入 liquid 范畴时需要检查的类型信息。

**练习 J.5.** 对一个 Fréchet 复形，分别列出“每项 \(p\)-liquid”和“cohomology 与
凝聚化交换”所需的假设。
